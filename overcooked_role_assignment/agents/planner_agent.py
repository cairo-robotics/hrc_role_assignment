from overcooked_role_assignment.common.subtasks import Subtasks
from overcooked_role_assignment.agents.agent_utils import DummyPolicy
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_role_assignment.common.arguments import get_arguments
from overcooked_role_assignment.agents.base_agent import OAIAgent

import numpy as np
import copy
from gym import spaces

import unittest

# import stable_baselines3.common.distributions as sb3_distributions
# import torch as th

class PlanBasedWorkerAgent(OAIAgent):
    def __init__(self, name, mlam, p_idx, args):
        self.mlam = mlam # MediumLevelActionManager instance for the environment
        self.motion_planner = mlam.motion_planner
        # self.action_queue = []
        self.player = p_idx

        super().__init__(name, args)

    def get_distribution(self, obs: dict, sample=True):
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
        plan = self.compute_subtask_plan(obs['curr_subtask'], obs['state'])[0]
        if plan is None:
            return [Action.ACTION_TO_INDEX[Action.STAY]]
        else:
            action = plan[0]
            return [Action.ACTION_TO_INDEX[action]]

    def fetch_goal_locations(self, subtask: str, overcooked_state):
        # stripped down version of the process_for_player function in overcooked_mdp.py
        goal_object = Subtasks.IDS_TO_GOAL_MARKERS[Subtasks.SUBTASKS_TO_IDS[subtask]]
        # goal_object = Subtasks.IDS_TO_GOAL_MARKERS[Subtaskssubtask]
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
            elif goal_object == "cabbage_dispenser":
                return mdp.get_cabbage_dispenser_locations()
            elif goal_object == "fish_dispenser":
                return mdp.get_fish_dispenser_locations()
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

    def compute_subtask_plan(self, subtask_name: str, state):
        goal_locations = self.fetch_goal_locations(subtask_name, state)
        start_pos_and_or = state.players_pos_and_or[self.player]

        # determine best (closest) goal location
        try:
            min_dist_to_goal, best_goal, _ = self.motion_planner.min_cost_to_feature(start_pos_and_or, goal_locations, with_argmin=True)
            if min_dist_to_goal == float('inf'):
                print("No valid goal locations found for subtask: {}".format(subtask_name))
                return None, None
            
        except ValueError:
            print("No valid goal locations found for subtask: {}".format(subtask_name))
            return None, None
        
        # get action plan (list of actions) to best goal
        action_plan, trajectory, cost =  self.motion_planner.get_plan(start_pos_and_or, best_goal)
        # print(action_plan, state.players_pos_and_or[self.player], best_goal)
        return action_plan, cost
    
class PlanBasedManagerAgent(OAIAgent):
    def __init__(self, worker, mlam, p_idx, args):
        super(PlanBasedManagerAgent, self).__init__('manager', args)
        self.worker = worker
        self.policy = DummyPolicy(spaces.Dict({'state': spaces.Box(0,1,(1,)),
                                               'visual_obs': spaces.Box(0,1,(1,)),
                                               'player_completed_subtasks': spaces.Box(0,1,(Subtasks.NUM_SUBTASKS,)),
                                               'teammate_completed_subtasks': spaces.Box(0,1,(Subtasks.NUM_SUBTASKS,)),
                                               'current_recipes': spaces.Box(0,1,(Subtasks.NUM_SUBTASKS,))}))
        self.use_hrl_obs = False
        self.curr_subtask_id = Subtasks.SUBTASKS_TO_IDS["get_fish_from_dispenser"]
        self.prev_pcs = None
        self.mdp = mlam.mdp
        self.mlam = mlam

        self.subtask_mask = np.ones(Subtasks.NUM_SUBTASKS)
        # self.subtask_mask = np.zeros(Subtasks.NUM_SUBTASKS)

        self.INGREDIENT_DEPENDENT_SUBTASKS = Subtasks.SUBTASKS[:16]
        self.VALID_SUBTASKS_BY_INGREDIENT = {
            "onion": Subtasks.SUBTASKS[:4],
            "tomato": Subtasks.SUBTASKS[4:8],
            "cabbage": Subtasks.SUBTASKS[8:12],
            "fish": Subtasks.SUBTASKS[12:16]
        }

        # FOR ROLE TESTING
        self.subtask_mask[Subtasks.SUBTASKS_TO_IDS["get_plate_from_dish_rack"]] = 0

        self.p_idx = p_idx

    def set_subtask_mask(self, subtask_mask):
        self.subtask_mask = subtask_mask

    def get_distribution(self, obs, sample=True):
        if obs['player_completed_subtasks'] is not None:
            costs = self.get_subtask_costs(obs, self.doable_and_useful_subtasks(obs))
            next_st = np.random.choice(Subtasks.NUM_SUBTASKS, p=costs)
            self.curr_subtask_id = int(next_st)
            # Completed previous subtask, set new subtask
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {obs["player_completed_subtasks"]}')
            next_st = input("Enter next subtask (0-10): ")
            self.curr_subtask_id = int(next_st)
        obs['curr_subtask'] = Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]
        return self.worker.get_distribution(obs, sample=sample)

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = False):
        if np.sum(obs['player_completed_subtasks']) == 1:
            costs = self.get_subtask_costs(obs, self.doable_and_useful_subtasks(obs))
            next_st = np.random.choice(Subtasks.NUM_SUBTASKS, p=costs)
            self.curr_subtask_id = int(next_st)
            comp_st = np.argmax(obs["player_completed_subtasks"], axis=0)
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {Subtasks.IDS_TO_SUBTASKS[comp_st]}')
            doable_st = [Subtasks.IDS_TO_SUBTASKS[idx] for idx, doable in enumerate(obs['subtask_mask']) if doable == 1]
            print('DOABLE SUBTASKS:', doable_st)
        obs['curr_subtask'] = Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]
        obs.pop('player_completed_subtasks')
        obs.pop('teammate_completed_subtasks')
        return self.worker.predict(obs, state=state, episode_start=episode_start, deterministic=True)

    def get_subtask_costs(self, obs, subtask_mask):
        cost_by_subtask = np.zeros(Subtasks.NUM_SUBTASKS, dtype=np.float32)
        for subtask in Subtasks.SUBTASKS:
            if subtask_mask[Subtasks.SUBTASKS_TO_IDS[subtask]]:
                plan, cost = self.worker.compute_subtask_plan(subtask, obs['state'])
                if plan is None:
                    cost = 0
                else:
                    cost = 1 / (cost+1)
            else:
                cost = 0

            cost_by_subtask[Subtasks.SUBTASKS_TO_IDS[subtask]] = cost

        print(cost_by_subtask)
        cost_by_subtask /= np.sum(cost_by_subtask)
        return cost_by_subtask

    
    def get_needed_ingredients_for_recipes(self, obs):
        state = obs['state']
        current_recipes = [list(r.ingredients) for r in state.all_orders]
        pot_states_dict = self.mdp.get_pot_states(state)
        # pot_locations = self.mdp.get_pot_locations()
        # full_soups_in_pots = pot_states_dict['cooking'] + pot_states_dict['ready']
        partially_full_soups = self.mdp.get_partially_full_pots(pot_states_dict)
        empty_pots = pot_states_dict['empty']

        needed_ingredients = {loc: [] for loc in empty_pots + partially_full_soups}
        for pot in partially_full_soups:
            ingredients = state.get_object(pot).ingredients
            for recipe in current_recipes:
                recipe_copy = copy.copy(recipe)
                for i in ingredients:
                    if i in recipe:
                        recipe_copy.remove(i)
                    else:
                        break
                if len(recipe_copy) < 3:
                    needed_ingredients[pot] += recipe_copy
                    current_recipes.remove(recipe)
                    break
        
        for pot in empty_pots:
            for recipe in current_recipes:
                needed_ingredients[pot] += [i for i in recipe]
    
        print(needed_ingredients)
        # return [state.get_object(loc).ingredients for loc in partially_full_soups + full_soups_in_pots]
        needed_ingredient_sets = {
            loc: set(ingredients) for loc, ingredients in needed_ingredients.items() 
        }
        return needed_ingredient_sets
    
    def doable_and_useful_subtasks(self, obs, prev_subtask='unknown'):
        state = obs['state']
        # current_recipes = copy.copy(obs['current_recipes'])
        needed_ingredients_by_pot = self.get_needed_ingredients_for_recipes(obs)

        doable_subtasks = get_doable_subtasks(state, prev_subtask, "", self.mdp.terrain_mtx, self.p_idx, n_counters=1)
        player_pos = state.players_pos_and_or[self.p_idx]
        closest_pot_cost, closest_pot_goal, closest_pot = self.mlam.motion_planner.min_cost_to_feature(player_pos, needed_ingredients_by_pot.keys(), with_argmin=True)
        needed_ingredients = needed_ingredients_by_pot[closest_pot]
        for ingredient, subtasks in self.VALID_SUBTASKS_BY_INGREDIENT.items():
            if ingredient not in needed_ingredients:
                for subtask in subtasks:
                    doable_subtasks[Subtasks.SUBTASKS_TO_IDS[subtask]] = 0.

        for i in range(len(doable_subtasks)):
            if self.subtask_mask[i] == 0:
                doable_subtasks[i] = 0.

        return doable_subtasks

class HumanManager(OAIAgent):
    def __init__(self, worker, args):
        super(HumanManager, self).__init__('human_manager', args)
        self.worker = worker
        self.policy = DummyPolicy(spaces.Dict({'visual_obs': spaces.Box(0,1,(1,)),
                                               'state': spaces.Box(0,1,(1,)),}))
        # self.encoding_fn = 
        self.use_hrl_obs = False
        self.curr_subtask_id = 0
        self.prev_pcs = None

    def get_distribution(self, obs, sample=True):
        if obs['player_completed_subtasks'] is not None:
            next_st = input("Enter next subtask (0-10): ")
            self.curr_subtask_id = int(next_st)
            # Completed previous subtask, set new subtask
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {obs["player_completed_subtasks"]}')
            next_st = input("Enter next subtask (0-10): ")
            self.curr_subtask_id = int(next_st)
        obs['curr_subtask'] = Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]
        return self.worker.get_distribution(obs, sample=sample)

    def predict(self, obs, state=None, episode_start=None, deterministic: bool = False):
        print("OBSERVATION: ", obs)
        print(obs['player_completed_subtasks'])
        if np.sum(obs['player_completed_subtasks']) == 1:
            next_st = input("Enter next subtask (0-10): ")
            self.curr_subtask_id = int(next_st)
            comp_st = np.argmax(obs["player_completed_subtasks"], axis=0)
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {Subtasks.IDS_TO_SUBTASKS[comp_st]}')
            doable_st = [Subtasks.IDS_TO_SUBTASKS[idx] for idx, doable in enumerate(obs['subtask_mask']) if doable == 1]
            print('DOABLE SUBTASKS:', doable_st)
        obs['curr_subtask'] = Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]
        obs.pop('player_completed_subtasks')
        obs.pop('teammate_completed_subtasks')
        return self.worker.predict(obs, state=state, episode_start=episode_start, deterministic=True)


class TestPlanBasedWorkerAgent(unittest.TestCase):
    def test_get_needed_ingredients_for_recipes(self):
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
        start_state_fn = mdp.get_subtask_start_state_fn(mlam)
        start_state = start_state_fn(p_idx=1, curr_subtask="get_soup")
        print('START STATE: ', start_state)
        agent = PlanBasedManagerAgent(None, mlam, 1, args)
        obs = {'state': start_state,
                'current_recipes': start_state.all_orders}
        print(agent.get_needed_ingredients_for_recipes(obs))
        
    def test_doable_and_useful_subtasks(self):
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
        start_state_fn = mdp.get_subtask_start_state_fn(mlam)
        start_state = start_state_fn(p_idx=1, curr_subtask="get_onion_from_dispenser")
        print('START STATE: ', start_state)
        agent = PlanBasedManagerAgent(None, mlam, 1, args)
        obs = {'state': start_state,
                'current_recipes': start_state.all_orders}
        print(agent.doable_and_useful_subtasks(obs))

    def test_get_subtask_costs(self):
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
        start_state_fn = mdp.get_subtask_start_state_fn(mlam)
        worker = PlanBasedWorkerAgent("worker", mlam, 1, args)
        start_state = start_state_fn(p_idx=1, curr_subtask="get_onion_from_dispenser")
  
        agent = PlanBasedManagerAgent(worker, mlam, 1, args)
        obs = {'state': start_state,
                'current_recipes': start_state.all_orders}
        
        print(agent.get_subtask_costs(obs, agent.subtask_mask))


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

    # def test_human_manager(self):
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
    #     # start_state_fn = mdp.get_subtask_start_state_fn(mlam)
    #     state = mdp.get_standard_start_state()
    #     worker = PlanBasedWorkerAgent("worker", mlam, args)
    #     # for subtask in Subtasks.SUBTASKS:
    #     #     state = start_state_fn(curr_subtask=subtask, random_pos=False)
    #     #     print(subtask, agent.compute_subtask_plan(subtask, state))
    #     manager = HumanManager(worker, args)
    #     obs = {'state': state,
    #            'player_completed_subtasks': [],
    #            'teammate_completed_subtasks' : []}
        
    #     while manager.curr_subtask_id >= 0:
    #         this_obs = copy.copy(obs)
    #         print(manager.get_distribution(this_obs))

if __name__ == "__main__":
    unittest.main()