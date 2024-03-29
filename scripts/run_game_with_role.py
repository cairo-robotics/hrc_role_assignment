from gym import spaces
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pygame
from pygame import K_UP, K_LEFT, K_RIGHT, K_DOWN, K_SPACE, K_s
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE
import matplotlib
matplotlib.use('TkAgg')
from os import listdir, environ
from os.path import isfile, join
import re
import time

from overcooked_role_assignment.agents.agent_utils import DummyPolicy
from overcooked_role_assignment.agents.base_agent import OAIAgent
from overcooked_role_assignment.agents.hrl import HierarchicalRL
from overcooked_role_assignment.common.arguments import get_arguments
from overcooked_role_assignment.common.subtasks import Subtasks
from overcooked_role_assignment.gym_environments.base_overcooked_env import OvercookedGymEnv
from overcooked_role_assignment.agents.agent_utils import load_agent, DummyAgent
from overcooked_role_assignment.common.state_encodings import ENCODING_SCHEMES
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action, OvercookedState, OvercookedGridworld
# from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.planning.planners import MediumLevelActionManager
# from scripts.train_agents import get_bc_and_human_proxy

from llm_interface import GPTRolePrompter
from role_assignment import RoleAssigner


no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

valid_counters = [(5, 3)]
one_counter_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': valid_counters,
    'counter_drop': valid_counters,
    'counter_pickup': [],
    'same_motion_goals': True
}


class App:
    """Class to run an Overcooked Gridworld game, leaving one of the agents as fixed.
    Useful for debugging. Most of the code from http://pygametutorials.wikidot.com/tutorials-basic."""
    def __init__(self, args, agent=None, teammate=None, layout=None, fps=5, p_idx=0, role_mask=None):
        self._running = True
        self._display_surf = None
        self.args = args
        self.layout_name = layout or 'asymmetric_advantages'

        self.env = OvercookedGymEnv(layout_name=self.layout_name, args=args, ret_completed_subtasks=True, is_eval_env=True, horizon=400)
        
        # TODO: move the mask setting to outside
        # subtask_weights = np.zeros(Subtasks.NUM_SUBTASKS)
        # subtask_weights[Subtasks.SUBTASKS_TO_IDS['get_onion_from_dispenser']] = 1
        # subtask_weights[Subtasks.SUBTASKS_TO_IDS['put_onion_in_pot']] = 1
        # subtask_weights[Subtasks.SUBTASKS_TO_IDS['unknown']] = 1
        # self.env.set_subtask_weights(subtask_weights)

        self.env.set_teammate(teammate)
        self.env.reset(p_idx=p_idx)
        self.env.teammate.set_idx(self.env.t_idx, self.layout_name, False, True, False)

        if role_mask is not None:
            self.env.set_subtask_weights(role_mask)

        self.grid_shape = self.env.grid_shape
        self.agent = agent
        self.human_action = None
        self.fps = fps
        self.score = 0
        self.curr_tick = 0
        self.data_path = args.base_dir / args.data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.tile_size = 100

        self.collect_trajectory = False
        if self.collect_trajectory:
            self.trajectory = []
            trial_file = re.compile('^.*\.[0-9]+\.pickle$')
            trial_ids = []
            for file in listdir(self.data_path):
                if isfile(join(self.data_path, file)) and trial_file.match(file):
                    trial_ids.append(int(file.split('.')[-2]))
            self.trial_id = max(trial_ids) + 1 if len(trial_ids) > 0 else 1

    def on_init(self):
        pygame.init()
        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.env.state, grid=self.env.env.mdp.terrain_mtx, hud_data={"timestep": 0, "orders": self.env.state.all_orders})

        self.surface_size = surface.get_size()
        self.x, self.y = (1920 - self.surface_size[0]) // 2, (1080 - self.surface_size[1]) // 2
        self.grid_shape = self.env.mdp.shape
        self.hud_size = self.surface_size[1] - (self.grid_shape[1] * self.tile_size)
        environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.x, self.y)

        self.window = pygame.display.set_mode(self.surface_size, HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        self._running = True


    def on_event(self, event):
        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']
            action = None

            if pressed_key == K_UP:
                action = Direction.NORTH
            elif pressed_key == K_RIGHT:
                action = Direction.EAST
            elif pressed_key == K_DOWN:
                action = Direction.SOUTH
            elif pressed_key == K_LEFT:
                action = Direction.WEST
            elif pressed_key == K_SPACE:
                action = Action.INTERACT
            elif pressed_key == K_s:
                action = Action.STAY
            else:
                action = Action.STAY
            self.human_action = Action.ACTION_TO_INDEX[action]

        if event.type == pygame.QUIT:
            self._running = False

    def step_env(self, agent_action):
        prev_state = self.env.state

        obs, reward, done, info = self.env.step(agent_action)

        # pygame.image.save(self.window, f"screenshots/screenshot_{self.curr_tick}.png")

        # Log data to send to psiturk client
        curr_reward = sum(info['sparse_r_by_agent'])
        self.score += curr_reward
        transition = {
            "state" : json.dumps(prev_state.to_dict()),
            "joint_action" : json.dumps(self.env.get_joint_action()), # TODO get teammate action from env to create joint_action json.dumps(joint_action.item()),
            "reward" : curr_reward,
            "time_left" : max((1200 - self.curr_tick) / self.fps, 0),
            "score" : self.score,
            "time_elapsed" : self.curr_tick / self.fps,
            "cur_gameloop" : self.curr_tick,
            "layout" : self.env.env.mdp.terrain_mtx,
            "layout_name" : self.layout_name,
            "trial_id" : 100, # TODO this is just for testing self.trial_id,
            "dimension": (self.x, self.y, self.surface_size, self.tile_size, self.grid_shape, self.hud_size),
            "timestamp": time.time()
        }
        if self.collect_trajectory:
            self.trajectory.append(transition)
        return done

    def on_render(self, pidx=None):
        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.env.state, grid=self.env.env.mdp.terrain_mtx, hud_data={"timestep": self.curr_tick, "orders": self.env.state.all_orders})
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        # save = input('press y to save')
        # if save.lower() == 'y':
        # pygame.image.save(self.window, "screenshot.png")

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False
        sleep_time = 1000 // (self.fps or 5)

        on_reset = True
        while (self._running):
            if self.agent == 'human':
                # while self.human_action is None and self.fps is None:
                for event in pygame.event.get():
                    self.on_event(event)
                pygame.event.pump()
                action = self.human_action if self.human_action is not None else Action.ACTION_TO_INDEX[Action.STAY]
            else:
                obs = self.env.get_obs(self.env.p_idx, on_reset=False)
                action = self.agent.predict(obs, state=self.env.state, deterministic=False)[0]
                # pygame.time.wait(sleep_time)

            done = self.step_env(action)
            self.human_action = None
            pygame.time.wait(sleep_time)
            self.on_render()
            self.curr_tick += 1

            if done:
                self._running = False

        self.on_cleanup()
        print(f'Trial finished in {self.curr_tick} steps with total reward {self.score}')

    def save_trajectory(self):
        df = pd.DataFrame(self.trajectory)
        df.to_pickle(self.data_path / f'{self.layout_name}.{self.trial_id}.pickle')

    @staticmethod
    def combine_df(data_path):
        trial_file = re.compile('^.*\.[0-9]+\.pickle$')
        df = pd.concat([pd.read_pickle(data_path / f) for f in listdir(data_path) if trial_file.match(f)])
        print(f'Combined df has a length of {len(df)}')
        df.to_pickle(data_path / f'all_trials.pickle')

    @staticmethod
    def fix_files_df(data_path):
        trial_file = re.compile('^.*\.[0-9]+\.pickle$')
        for f in listdir(data_path):
            if trial_file.match(f):
                df = pd.read_pickle(data_path / f)
                def joiner(list_of_lists):
                    for i in range(len(list_of_lists)):
                        list_of_lists[i] = ''.join(list_of_lists[i])
                    return str(list_of_lists)
                df['layout'] = df['layout'].apply(joiner)
                df.to_pickle(data_path / f)

# Just for testing
class HumanManagerHRL(OAIAgent):
    def __init__(self, worker, args):
        super(HumanManagerHRL, self).__init__('hierarchical_rl', args)
        self.worker = worker
        self.curr_subtask_id = 11
        self.prev_pcs = None

    def get_distribution(self, obs, sample=True):
        if obs['player_completed_subtasks'] is not None:
            # Completed previous subtask, set new subtask
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {obs["player_completed_subtasks"]}')
            next_st = input("Enter next subtask (0-10): ")
            self.curr_subtask_id = int(next_st)
        obs['curr_subtask'] = self.curr_subtask_id
        return self.worker.get_distribution(obs, sample=sample)

    def predict(self, obs, state=None, episode_start=None, deterministic: bool=False):
        print(obs['player_completed_subtasks'])
        if np.sum(obs['player_completed_subtasks']) == 1:
            comp_st = np.argmax(obs["player_completed_subtasks"], axis=0)
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {Subtasks.IDS_TO_SUBTASKS[comp_st]}')
            doable_st = [Subtasks.IDS_TO_SUBTASKS[idx] for idx, doable in enumerate(obs['subtask_mask']) if doable == 1]
            print('DOABLE SUBTASKS:', doable_st)
            next_st = input("Enter next subtask (0-10): ")
            self.curr_subtask_id = int(next_st)
        obs['curr_subtask'] = self.curr_subtask_id
        obs.pop('player_completed_subtasks')
        obs.pop('teammate_completed_subtasks')
        return self.worker.predict(obs, state=state, episode_start=episode_start, deterministic=True)

# Just for testing
class HumanPlayer(OAIAgent):
    def __init__(self, name, args):
        super(HumanPlayer, self).__init__(name, args)
        self.policy = DummyPolicy(spaces.Dict({'visual_obs': spaces.Box(0,1,(1,))}))

    def get_distribution(self, obs, sample=True):
        key = input(f'{self.name} enter action:')
        if key == 'w':
            action = Direction.NORTH
        elif key == 'd':
            action = Direction.EAST
        elif key == 's':
            action = Direction.SOUTH
        elif key == 'a':
            action = Direction.WEST
        elif key == ' ':
            action = Action.INTERACT
        else:
            action = Action.STAY
        return np.array([Action.ACTION_TO_INDEX[action]])

    def predict(self, obs, state=None, episode_start=None, deterministic: bool=False):
        return self.get_distribution(obs)

def create_mask_from_task_list(task_list):
    mask = np.zeros(Subtasks.NUM_SUBTASKS)
    for subtask in Subtasks.HUMAN_READABLE_ST:
        if subtask in task_list:
            mask[Subtasks.HR_SUBTASKS_TO_IDS[subtask]] = 1.0
    # unsure must always at least be an option
    mask[Subtasks.SUBTASKS_TO_IDS['unknown']] = 1.0
    return mask

SAMPLE_GPT_OUTPUT = {'Division 1': {'Chef': ['Grabbing an onion from dispenser', 'Putting onion in pot', 'Grabbing dish from dispenser', 'Placing dish closer to pot', 'Serving the soup'], 'Sous Chef': ['Grabbing a tomato from dispenser', 'Putting tomato in pot', 'Grabbing dish from counter', 'Getting the soup', 'Grabbing soup from counter', 'Placing soup closer']}, 'Division 2': {'Prep Cook': ['Grabbing an onion from dispenser', 'Grabbing a tomato from dispenser', 'Putting onion in pot', 'Putting tomato in pot'], 'Server': ['Grabbing dish from dispenser', 'Grabbing dish from counter', 'Placing dish closer to pot', 'Getting the soup', 'Grabbing soup from counter', 'Placing soup closer', 'Serving the soup']}, 'Division 3': {'Cook': ['Grabbing an onion from dispenser', 'Grabbing a tomato from dispenser', 'Putting onion in pot', 'Putting tomato in pot', 'Getting the soup'], 'Waiter': ['Grabbing dish from dispenser', 'Grabbing dish from counter', 'Placing dish closer to pot', 'Grabbing soup from counter', 'Placing soup closer', 'Serving the soup']}, 'Division 4': {'Food Prep': ['Grabbing an onion from dispenser', 'Grabbing a tomato from dispenser', 'Putting onion in pot', 'Putting tomato in pot', 'Getting the soup'], 'Service': ['Grabbing dish from dispenser', 'Grabbing dish from counter', 'Placing dish closer to pot', 'Grabbing soup from counter', 'Placing soup closer', 'Serving the soup']}}


# v1 of spatial role generation -- quick-n-dirty, just simple hand coded categories
# TODO: add improved spatial clustering using worker's motion planning
def generate_spatial_roles_simple(mdp, human_readable_subtasks):
    jobs_by_area = {
        "top" : set([]),
        "bottom" : set([]),
        "left" : set([]),
        "right" : set([]),
    }

    for subtask in human_readable_subtasks:
        goal_object = Subtasks.IDS_TO_GOAL_MARKERS[Subtasks.HR_SUBTASKS_TO_IDS[subtask]]
        if goal_object == "counter" or ["onion", "tomato", "cabbage", "fish", "dish", "soup"]:
            locations = mdp.get_counter_locations()
        elif goal_object == "empty_pot" or goal_object == "full_pot":
            # get any non-full pot locations
            locations = mdp.get_pot_locations()
        elif goal_object == "onion_dispenser":
            locations =  mdp.get_onion_dispenser_locations()
        elif goal_object == "tomato_dispenser":
            locations =  mdp.get_tomato_dispenser_locations()
        elif goal_object == "cabbage_dispenser":
            locations = mdp.get_cabbage_dispenser_locations()
        elif goal_object == "fish_dispenser":
            locations = mdp.get_fish_dispenser_locations()
        elif goal_object == "dish_dispenser":
            locations = mdp.get_dish_dispenser_locations()
        elif goal_object == "serving_station":
            locations = mdp.get_serving_locations()
        else:
            continue
        
        for location in locations:
            x, y = location
            if x < mdp.shape[0] // 2:
                jobs_by_area["left"].add(subtask)
            else:
                jobs_by_area["right"].add(subtask)
            if y < mdp.shape[1] // 2:
                jobs_by_area["top"].add(subtask)
            else:
                jobs_by_area["bottom"].add(subtask)


    return jobs_by_area


if __name__ == "__main__":
    """
    Sample commands
    python scripts/run_overcooked_game.py --agent human --teammate agent_models/HAHA
    """
    additional_args = [
        ('--agent', {'type': str, 'default': 'human', 'help': '"human" to used keyboard inputs or a path to a saved agent'}),
        ('--teammate', {'type': str, 'default': 'agent_models/HAHA_bcp', 'help': 'Path to saved agent to use as teammate'}),
        ('--layout', {'type': str, 'default': 'counter_circuit_o_1order', 'help': 'Layout to play on'}),
        ('--p_idx', {'type': int, 'default': 0, 'help': 'Player idx of agent (teammate will have other player idx), Can be 0 or 1.'})
    ]

    args = get_arguments(additional_args)

    t_idx = 1 - args.p_idx
    # tm = load_agent(Path(args.teammate), args)
    tm = DummyAgent()
    tm.set_idx(t_idx, args.layout, is_hrl=isinstance(tm, HierarchicalRL), tune_subtasks=True)
    if args.agent == 'human':
        agent = args.agent
    else:
        agent = load_agent(Path(args.agent), args)
        agent.set_idx(args.p_idx, args.layout, is_hrl=isinstance(agent, HierarchicalRL), tune_subtasks=False)

    # gpt = GPTRolePrompter()
    # roles = gpt.query_for_role_divisions(Subtasks.HUMAN_READABLE_ST)
    # roles 
    roles = SAMPLE_GPT_OUTPUT
    flat_role_dict = {}
    for key in roles:
        for role in roles[key]:
            flat_role_dict[role] = roles[key][role]
    # print(f'ROLES: {roles}')

    # # pick first one for testing purposes
    # for key in roles:
    #     role_set = roles[key]
    #     break
    # # agent role is the first one
    # role1, role2 = list(role_set.keys())
    # role1_tasks = role_set[role1]
    # role2_tasks = role_set[role2]

    # print(f'AGENT ROLE: {role1}')

    # mask = create_mask_from_task_list(role1_tasks)
    # print(f'MASK: {mask}')

    # print(f'HUMAN ROLE: {role2}')

    # layout = "/media/kaleb/T7/overcooked/hrc_role_assignment/overcooked_hrl/data/layouts/1p_counter_circuit_o_1order"
    # layout=args.layout
    layout = "ORA_symmetry"
    # layout = "counter_circuit_o_1order"

    dc      = App(args, agent=agent, teammate=tm, layout=layout, p_idx=args.p_idx)
    mdp     = dc.env.mdp

    # print(generate_spatial_roles_simple(mdp, Subtasks.SUBTASK
    spatial_roles =  generate_spatial_roles_simple(mdp, Subtasks.HUMAN_READABLE_ST)
    all_roles = {**flat_role_dict, **spatial_roles}
    print(all_roles)
    
    assigner = RoleAssigner(layout, None, mdp=mdp, mlam=dc.env.mlam)
    costs = assigner.evaluate_roles(all_roles)
    print(assigner.assign_roles(all_roles, costs, Subtasks.HUMAN_READABLE_ST))

    dc.on_execute()
