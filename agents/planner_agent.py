from oai_agents.common.subtasks import Subtasks
from oai_agents.agents.agent_utils import DummyPolicy
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from oai_agents.common.arguments import get_arguments
from oai_agents.agents.base_agent import OAIAgent

import numpy as np
import copy
from gym import spaces

import unittest

class PlanBasedWorkerAgent(OAIAgent):
    def __init__(self, name, mlam, p_idx, args):
        self.mlam = mlam # MediumLevelActionManager instance for the environment
        self.motion_planner = mlam.motion_planner
        # self.action_queue = []
        self.player = p_idx

        super().__init__(name, args)

    def get_distribution(self, obs, sample=True):
        plan = self.compute_subtask_plan(obs['curr_subtask'], obs['state'])
        dist = [0.0] * Action.NUM_ACTIONS
        if plan is None:
            dist[Action.ACTION_TO_INDEX[Action.STAY]] = 1.0
        else:
            action = plan[0]
            dist[Action.ACTION_TO_INDEX[action]] = 1.0
        return dist

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = False):
        # TODO implement sampling
        plan = self.compute_subtask_plan(obs['curr_subtask'], obs['state'])
        if plan is None:
            return [Action.ACTION_TO_INDEX[Action.STAY]]
        else:
            action = plan[0]
            return [Action.ACTION_TO_INDEX[action]]

    def fetch_goal_locations(self, subtask, overcooked_state):
        # stripped down version of the process_for_player function in overcooked_mdp.py
        # goal_object = Subtasks.IDS_TO_GOAL_MARKERS[Subtasks.SUBTASKS_TO_IDS[subtask]]
        goal_object = Subtasks.IDS_TO_GOAL_MARKERS[subtask]
        mdp = self.mlam.mdp
        try:
            if goal_object == "counter":
                all_empty_counters = set(mdp.get_empty_counter_locations(overcooked_state))
                # only use drop locations specified in the params - not just any counter
                # (this is a cheat but makes sure we only put ingredients in useful places)
                valid_empty_counters = [c_pos for c_pos in self.mlam.counter_drop if c_pos in all_empty_counters]
                return valid_empty_counters
            elif goal_object == "empty_pot":
                # get any non-full pot locations
                pot_states = mdp.get_pot_states(overcooked_state)
                locations = pot_states['empty'] + pot_states['1_items'] + pot_states['2_items']
                return locations
            elif goal_object == "full_pot":
                pot_states = mdp.get_pot_states(overcooked_state)
                locations = pot_states['cooking'] + pot_states['ready']
                return locations
            elif goal_object == "onion_dispenser":
                return mdp.get_onion_dispenser_locations()
            elif goal_object == "tomato_dispenser":
                return mdp.get_tomato_dispenser_locations()
            elif goal_object == "dish_dispenser":
                return mdp.get_dish_dispenser_locations()
            elif goal_object == "serving_station":
                return mdp.get_serving_locations()
            elif goal_object in ["onion", "tomato", "cabbage", "fish", "dish", "soup"]:
                # loose objects on counter for pickup
                locations = []
                for obj in overcooked_state.all_objects_list:
                    if obj.name == goal_object:
                        locations.append(obj.position)
                return locations
            else:
                return []
        except ValueError:
            print("No valid goal locations found for subtask: {}".format(subtask))
            return []

    def compute_subtask_plan(self, subtask_name, state):
        goal_locations = self.fetch_goal_locations(subtask_name, state)
        start_pos_and_or = state.players_pos_and_or[self.player]

        # determine best (closest) goal location
        try:
            min_dist_to_goal, best_goal = self.motion_planner.min_cost_to_feature(start_pos_and_or, goal_locations, with_argmin=True)
            if min_dist_to_goal == float('inf'):
                return None
            
        except ValueError:
            print("No valid goal locations found for subtask: {}".format(subtask_name))
            return None
        
        # get action plan (list of actions) to best goal
        action_plan, trajectory, cost =  self.motion_planner.get_plan(start_pos_and_or, best_goal)
        # print(action_plan, state.players_pos_and_or[self.player], best_goal)
        return action_plan
    
class HumanManager(OAIAgent):
    def __init__(self, worker, args):
        super(HumanManager, self).__init__('human_manager', args)
        self.worker = worker
        self.policy = DummyPolicy(spaces.Dict({'visual_obs': spaces.Box(0,1,(1,))}))
        # self.encoding_fn = 
        self.use_hrl_obs = False
        self.curr_subtask_id = 11
        self.prev_pcs = None

    def get_distribution(self, obs, sample=True):
        next_st = input("Enter next subtask (0-10): ")
        self.curr_subtask_id = int(next_st)
        if obs['player_completed_subtasks'] is not None:
            # Completed previous subtask, set new subtask
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {obs["player_completed_subtasks"]}')
            next_st = input("Enter next subtask (0-10): ")
            self.curr_subtask_id = int(next_st)
        obs['curr_subtask'] = self.curr_subtask_id
        return self.worker.get_distribution(obs, sample=sample)

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = False):
        print("OBSERVATION: ", obs)
        print(obs['player_completed_subtasks'])
        next_st = input("Enter next subtask (0-10): ")
        self.curr_subtask_id = int(next_st)
        if np.sum(obs['player_completed_subtasks']) == 1:
            comp_st = np.argmax(obs["player_completed_subtasks"], axis=0)
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {Subtasks.IDS_TO_SUBTASKS[comp_st]}')
            doable_st = [Subtasks.IDS_TO_SUBTASKS[idx] for idx, doable in enumerate(obs['subtask_mask']) if doable == 1]
            print('DOABLE SUBTASKS:', doable_st)
        obs['curr_subtask'] = self.curr_subtask_id
        obs.pop('player_completed_subtasks')
        obs.pop('teammate_completed_subtasks')
        return self.worker.predict(obs, state=state, episode_start=episode_start, deterministic=True)


class TestPlanBasedWorkerAgent(unittest.TestCase):
    # def test_compute_subtask_plan(self):
    #     layout = "counter_circuit"
    #     args = get_arguments() # get default arguments to make ABC happy
    #     mdp = OvercookedGridworld.from_layout_name(layout)
    #     params = {
    #         'start_orientations': False,
    #         'wait_allowed': False,
    #         'counter_goals': True, # include counters as goals in motion planner
    #         'counter_drop': [(2, 2)], # counters we can put ingredients on
    #         'counter_pickup': [], # counters we can pick things up from (ignored - can pick up anywhere)
    #         'same_motion_goals': True
    #     }
    #     mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, params, force_compute=False)
    #     start_state_fn = mdp.get_subtask_start_state_fn(mlam)
    #     agent = PlanBasedWorkerAgent("worker", mlam, args)
    #     for subtask in Subtasks.SUBTASKS:
    #         state = start_state_fn(curr_subtask=subtask, random_pos=False)
    #         print(subtask, agent.compute_subtask_plan(subtask, state))

    # def test_worker_actions(self):
    #     layout = "counter_circuit"
    #     args = get_arguments()
    #     mdp = OvercookedGridworld.from_layout_name(layout)
    #     params = {
    #         'start_orientations': False,
    #         'wait_allowed': False,
    #         'counter_goals': True,
    #         'counter_drop': [(2, 2)],
    #         'counter_pickup': [],
    #         'same_motion_goals': True
    #     }
    #     mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, params, force_compute=False)
    #     start_state_fn = mdp.get_subtask_start_state_fn(mlam)
    #     agent = PlanBasedWorkerAgent("worker", mlam, args)
    #     for subtask in Subtasks.SUBTASKS:
    #         state = start_state_fn(curr_subtask=subtask, random_pos=False)
    #         # print(subtask, agent.compute_subtask_plan(subtask, state))
    #         obs = {
    #             'curr_subtask': subtask,
    #             'state': state
    #         }
    #         print(subtask, agent.get_distribution(obs), agent.predict(obs))

    def test_human_manager(self):
        layout = "counter_circuit"
        args = get_arguments() # get default arguments to make ABC happy
        mdp = OvercookedGridworld.from_layout_name(layout)
        params = {
            'start_orientations': False,
            'wait_allowed': False,
            'counter_goals': True, # include counters as goals in motion planner
            'counter_drop': [(2, 2)], # counters we can put ingredients on
            'counter_pickup': [], # counters we can pick things up from (ignored - can pick up anywhere)
            'same_motion_goals': True
        }
        mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, params, force_compute=False)
        # start_state_fn = mdp.get_subtask_start_state_fn(mlam)
        state = mdp.get_standard_start_state()
        worker = PlanBasedWorkerAgent("worker", mlam, args)
        # for subtask in Subtasks.SUBTASKS:
        #     state = s`tart_state_fn(curr_subtask=subtask, random_pos=False)
        #     print(subtask, agent.compute_subtask_plan(subtask, state))
        manager = HumanManager(worker, args)
        obs = {'state': state,
               'player_completed_subtasks': [],
               'teammate_completed_subtasks' : []}
        
        while manager.curr_subtask_id >= 0:
            this_obs = copy.copy(obs)
            print(manager.get_distribution(this_obs))

if __name__ == "__main__":
    unittest.main()