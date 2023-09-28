from oai_agents.common.subtasks import Subtasks
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from oai_agents.common.arguments import get_arguments
from oai_agents.agents.base_agent import OAIAgent

import unittest

class PlanBasedWorkerAgent(OAIAgent):
    def __init__(self, name, mlam, args):
        self.mlam = mlam # MediumLevelActionManager instance for the environment
        self.motion_planner = mlam.motion_planner
        self.action_queue = []
        self.player = 0
        super().__init__(name, args)

    def get_distribution(self, *args):
        pass

    def predict(self, *args):
        pass

    def fetch_goal_locations(self, subtask, overcooked_state):
        # stripped down version of the process_for_player function in overcooked_mdp.py
        goal_object = Subtasks.IDS_TO_GOAL_MARKERS[Subtasks.SUBTASKS_TO_IDS[subtask]]
        mdp = self.mlam.mdp
        try:
            if goal_object == "counters":
                return mdp.get_empty_counter_locations(overcooked_state)
            elif goal_object == "empty_pot":
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
            elif goal_object in ["onion", "tomato", "dish", "soup"]:
                locations = []
                for obj in overcooked_state.all_objects_list:
                    if obj.name == goal_object:
                        locations.append(obj)
                return locations
            else:
                return []
        except ValueError:
            print("No valid goal locations found for subtask: {}".format(subtask))
            return []

    def compute_subtask_plan(self, subtask_name, state):
        goal_locations = self.fetch_goal_locations(subtask_name, state)
        start_pos_and_or = state.players_pos_and_or[self.player]
        try:
            min_dist_to_goal, best_goal = self.motion_planner.min_cost_to_feature(start_pos_and_or, goal_locations, with_argmin=True)
            if min_dist_to_goal == float('inf'):
                return None
            
        except ValueError:
            print("No valid goal locations found for subtask: {}".format(subtask_name))
            return None
        
        # get action plan to best goal
        next_action, trajectory, cost =  self.motion_planner.get_plan(start_pos_and_or, best_goal)
        return next_action
    
class TestPlanBasedWorkerAgent(unittest.TestCase):
    def test_compute_subtask_plan(self):
        layout = "asymmetric_advantages"
        args = get_arguments()
        mdp = OvercookedGridworld.from_layout_name(layout)
        mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
        start_state_fn = mdp.get_subtask_start_state_fn(mlam)

        state = start_state_fn(curr_subtask="get_onion_from_dispenser", random_pos=False)
        agent = PlanBasedWorkerAgent("worker", mlam, args)
        print(agent.compute_subtask_plan("get_onion_from_dispenser", state))

if __name__ == "__main__":
    unittest.main()