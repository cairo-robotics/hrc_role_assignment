from oai_agents.common.subtasks import Subtasks
# from oai_agents.gym_environments.worker_env import OvercookedSubtaskGymEnv
# from oai_agents.agents.hrl import MultiAgentSubtaskWorker
from oai_agents.common.arguments import get_arguments
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState, Action, Direction
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS

from pathlib import Path
import unittest

from gurobipy import *
# import gurobipy as gp

from itertools import combinations

class RoleAssigner:
    def __init__(self, layout_name, args, n_players=2):
        """
        param env: expects oai_agents.gym_environments.base_overcooked_env.OvercookedSubtaskGymEnv
        param n_players: int, tested for 2 players only
        """
        
        self.layout_name = layout_name
        self.args = args
        self.n_players = n_players
        self.mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)
        self.start_state_fn = self.mdp.get_subtask_start_state_fn(self.mlam)

    def evaluate_single_role(self, role_subtasks):
        """
        param role: a list of subtasks in oai_agents.subtasks.SUBTASKS format
        return: heuristic eval (by way of mini TSP)
        NOTE: this version doesn't consider any ordering constraints among the subtasks
        TODO: fix the above ;)
        """
        distances = {}
        goals     = {}

        for task in role_subtasks:
            dist, goal = self.evaluate_subtask_traj(task)
            if dist is None:
                print("No connection from {} to {}, role invalid".format(task1, task2))
                
            goals[task] = goal

        for task1, task2 in combinations(role_subtasks, 2):
            dist, goal = self.evaluate_subtask_traj(task1, goals[task2])

            distances[task1, task2] = dist
            print("Distance from {} to {} is {}".format(task1, task2, dist))

        m = Model()
        vars = m.addVars(distances.keys(), obj=distances, vtype=GRB.BINARY, name="x")

        for i, j in distances:
            vars[j, i] = vars[i, j]

        cons = m.addConstrs((vars.sum(i, '*') == 2  for i in role_subtasks), name="c")

        # cons = m.addConstr(vars.sum("start", '*') == 1, name="c")

        m._vars = vars
        m.optimize()
        solution = m.getAttr('x', vars)
        selected = [(i, j) for i, j in solution.keys() if solution[i, j] > 0.5]
        selected_goals = [goals[i] for i, j in selected]

        for i in range(len(selected) - 1):
            if selected[i] == selected[i+1]:
                # selected.pop(i+1)
                selected_goals[i+1] = None

        return m.objVal, selected, selected_goals

    def fetch_goal_locations(self, subtask, overcooked_state=None):
        # stripped down version of the process_for_player function in overcooked_mdp.py
        goal_object = Subtasks.IDS_TO_GOAL_MARKERS[Subtasks.SUBTASKS_TO_IDS[subtask]]
        mdp = self.mdp
        if overcooked_state is None:
            try:
                overcooked_state = self.start_state_fn(curr_subtask=subtask, random_pos=False)
            except ValueError:
                print("No valid start state found for subtask: {}".format(subtask))
                return []
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
    
    def evaluate_subtask_traj(self, subtask, prev_location = None):
        """
        param start_state: Overcooked PlayerState (?)
        param subtask: subtask name (string)
        return: heuristic eval
        """

        try:
            start_state = self.start_state_fn(curr_subtask=subtask, random_pos=False)
        except ValueError:
            print("No valid start state found for subtask: {}".format(subtask))
            return None, None
        # print(start_state)
        goal_locations = self.fetch_goal_locations(subtask, start_state)
        start_pos_and_or = start_state.players_pos_and_or[0]
        try:
            if prev_location is not None:
                min_dist_to_goal, best_goal = self.mlam.motion_planner.min_cost_between_features([prev_location], goal_locations, with_argmin=True)
            else:
                min_dist_to_goal, best_goal = self.mlam.motion_planner.min_cost_to_feature(start_pos_and_or, goal_locations, with_argmin=True)
            return min_dist_to_goal, best_goal
        except AssertionError:
            print("No connection from player position {} to goal locations {}".format(prev_location or start_pos_and_or, goal_locations))
            return None, None

    def assign_roles(self, roles, time, tasksNeeded):
        """
        param all_roles: role suggestions in format {role_name: [subtask1, subtask2, ...]}
        param subtask_list: list of subtasks that need to be completed in format [subtask1, subtask2, ...]
        """

        # Model
        m = Model()

        # Variables
        x = {}
        for role in roles:
            x[role] = m.addVar(vtype=GRB.BINARY, name=f"x[{role}]")
                
        y = {}
        for role in roles:
            for task in roles[role]:
                y[role, task] = m.addVar(vtype=GRB.BINARY, name=f"y[{role},{task}]")

        # Each task that is needed must be covered by at least one role
        for task in tasksNeeded:
            m.addConstr(sum(y[role, task] for role in roles if task in roles[role]) >= 1, name=f"task_{task}")

        # Role-task pair can only be active if the role is assigned
        for role in roles:
            for task in roles[role]:
                m.addConstr(y[role,task] <= x[role], name=f"link_{role},{task}")

        # Objective: Minimize total time
        m.setObjective(sum(x[role] * time[role] for role in roles), GRB.MINIMIZE)

        # Solve
        m.optimize()

        # Print solution
        for v in m.getVars():
            if v.x > 0:
                print(f"{v.varName} = {v.x}")


class TestRoleAssignment(unittest.TestCase):

    def test_fetch_goal_locations(self):
        layout = "asymmetric_advantages"

        print("Testing goal location fetching for layout: {}".format(layout))
        role_assigner = RoleAssigner(layout_name=layout, args=None)
        for sbt in Subtasks.SUBTASKS:
            print("Testing goal locations for subtask: {}".format(sbt))
            print(role_assigner.fetch_goal_locations(sbt))

    def test_evaluate_subtask_traj(self):
        print("Testing start to end evaluation for layout: asymmetric_advantages")
        args = get_arguments()
        role_assigner = RoleAssigner(layout_name="asymmetric_advantages", args=args)
        player_pose = None
        for sbt in Subtasks.SUBTASKS:
            print("Testing start to end evaluation for subtask: {}".format(sbt))
            score, goal = role_assigner.evaluate_subtask_traj(sbt, player_pose)
            print(score, goal)
            if goal is not None:
                player_pose = goal

    def test_evaluate_single_role(self):
        print("Testing role evaluation")
        role_assigner = RoleAssigner(layout_name="large_room", args=None)

        role_subtasks = ['get_onion_from_dispenser', 'put_onion_in_pot', 
                'get_plate_from_dish_rack', 'get_soup',
                  'serve_soup']
        score, route, goal = role_assigner.evaluate_single_role(role_subtasks)
        print(score, route, goal)

    def test_assign_roles(self):
        print("Testing role assignment")
        role_assigner = RoleAssigner(layout_name="cramped_room", args=None)

        time = {'A': 2, 'B': 4, 'C': 1}  # Minimum time it takes for either player to perform the role
        roles = {
        'A': ['get_onion', 'cook_onion', 'serve_soup'], 
        'B': ['get_onion', 'place_onion', 'chop_onion'], 
        'C': ['place_onion', 'chop_onion', 'serve_soup']
        }
        tasksNeeded = ['get_onion', 'chop_onion', 'cook_onion', 'serve_soup']
        role_assigner.assign_roles(roles, time, tasksNeeded)

if __name__ == "__main__":
    unittest.main()