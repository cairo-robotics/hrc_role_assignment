from oai_agents.common.subtasks import Subtasks
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS

from gurobipy import *
# import gurobipy as gp

from itertools import combinations

class RoleAssigner:
    def __init__(self, layout_name, args, n_players=2, mdp=None, mlam=None):
        """
        param env: expects oai_agents.gym_environments.base_overcooked_env.OvercookedSubtaskGymEnv
        param n_players: int, tested for 2 players only
        """
        
        self.layout_name = layout_name
        self.args = args
        self.n_players = n_players
        self.mdp = mdp
        if mdp is None:
            self.mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        
        self.mlam = mlam
        if mlam is None:
            self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)
        self.start_state_fn = self.mdp.get_subtask_start_state_fn(self.mlam)

    def evaluate_single_role(self, role_subtasks, player=0):
        """
        param role: a list of subtasks in oai_agents.subtasks.SUBTASKS format
        return: heuristic eval (by way of mini TSP)
        NOTE: this version doesn't consider any ordering constraints among the subtasks OR start position for player***
        TODO: fix the above ;)
        """
        distances = {}
        goals     = {}

        doable_subtasks = []


        if len(role_subtasks) == 1:
            dist, goal = self.evaluate_subtask_traj(role_subtasks, player=player)
            return dist, goal
        
        # TODO: fix behavior for when there is an invalid task in the role -- 
        #  we probably want to evaluate the role minus that task(s)

        for task in role_subtasks:
            dist, goal = self.evaluate_subtask_traj(task, player=player)
            # if goal is None:
            #     # print("No connection to {} for player {}, role invalid".format(task, player))
            #     # return None, None, None
            #     continue
            if goal is not None:
                doable_subtasks.append(task)
                # distances[task] = dist
            goals[task] = goal

        for task1, task2 in combinations(role_subtasks, 2):
            if goals[task1] is None or goals[task2] is None:
                continue
            dist, goal = self.evaluate_subtask_traj(task1, goals[task2], player=player)

            distances[task1, task2] = dist
            print("Distance from {} to {} is {}".format(task1, task2, dist))

        if len(doable_subtasks) <= 3:
            # Invalid for TSP, so just return the sum
            return sum(distances.values()), role_subtasks, [goals[task] for task in role_subtasks if goals[task] is not None]


        m = Model()

        vars = m.addVars(distances.keys(), obj=distances, vtype=GRB.BINARY, name="x")

        for i, j in distances:
            vars[j, i] = vars[i, j]

        cons = m.addConstrs((vars.sum(i, '*') == 2  for i in doable_subtasks), name="c")

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
        goal_object = Subtasks.IDS_TO_GOAL_MARKERS[Subtasks.HR_SUBTASKS_TO_IDS[subtask]]
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
    
    def evaluate_subtask_traj(self, subtask, prev_location = None, player=0):
        """
        param subtask: subtask name (string)
        return: heuristic eval
        """

        try:
            start_state = self.start_state_fn(p_idx=player, curr_subtask=subtask, random_pos=False)
        
        except ValueError:
            print("No valid start state found for subtask: {}".format(subtask))
            return None, None
        # print(start_state)
        goal_locations = self.fetch_goal_locations(subtask, start_state)
        start_pos_and_or = start_state.players_pos_and_or[player]
        try:
            if prev_location is not None:
                # print("prev_location:", prev_location)
                min_dist_to_goal, best_goal, best_feature = self.mlam.motion_planner.min_cost_to_feature(prev_location, goal_locations, with_argmin=True)
            else:
                min_dist_to_goal, best_goal, best_feature = self.mlam.motion_planner.min_cost_to_feature(start_pos_and_or, goal_locations, with_argmin=True)
            return min_dist_to_goal, best_goal
        except AssertionError:
            print("No connection from player position {} to goal locations {}".format(prev_location or start_pos_and_or, goal_locations))
            return None, None
        
    def evaluate_roles(self, roles):
        cost_dicts = []
        for player in range(self.n_players):
            cost_dicts.append({})
            for role in roles.keys():
                print("Evaluating role {} for player {}".format(role, player))
                value, route, goal = self.evaluate_single_role(roles[role], player=player)
                if value is not None:
                    cost_dicts[player][role] = value

        return cost_dicts

    def assign_roles(self, roles, time, tasksNeeded):
        """
        param all_roles: role suggestions in format {role_name: [subtask1, subtask2, ...]}
        param subtask_list: list of subtasks that need to be completed in format [subtask1, subtask2, ...]
        param time: minimum time it takes for each player to perform the role in format [{role_name: time_p1,...}, {role_name: time_p2,...},...]
        """
        # print(roles)
        # if len(roles[0].keys()) == 0 and len(roles[1].keys()) == 0:
        #     print("No valid roles for either player")
        #     return None, None

        print("Tasks needed: {}".format(tasksNeeded))
        print("Roles: {}".format(roles))
        print("Time: {}".format(time))

        # Model
        m = Model()

        # Variables

        players = [0, 1]
        # role assignment
        x = {}
        for player in players:
            for role in roles:
                x[role, player] = m.addVar(vtype=GRB.BINARY, name=f"x[{role},{player}]")
        
        # task coverage by role assignment
        y = {}
        for role in roles:
            for task in roles[role]:
                y[role, task] = m.addVar(vtype=GRB.BINARY, name=f"y[{role},{task}]")

        # max constraint
        total_time_per_player = m.addVars(players, vtype=GRB.CONTINUOUS, name="total_time_per_player")
        total_constr = m.addConstrs((total_time_per_player[player] == sum(x[role, player] * time[player][role] for role in roles) for player in players), name="total_constr")

        max_obj = m.addVar(vtype=GRB.CONTINUOUS, name="max_obj")
        max_constr = m.addConstr(max_obj == max_(total_time_per_player[player] for player in players), name="max_constr")

        # Each task that is needed must be covered by at least one role
        for task in tasksNeeded:
            m.addConstr(sum(y[role, task] for role in roles if task in roles[role]) >= 1, name=f"task_{task}")

        # Role-task pair can only be active if the role is assigned to at least one player
        for role in roles:
            for task in roles[role]:
                m.addConstr(y[role, task] <= sum(x[role, player] for player in players) , name=f"role_{role}_task_{task}")

        # Objective: Minimize maximum time taken by any player
        m.setObjective(max_obj, GRB.MINIMIZE)


        # Solve
        m.optimize()

        # Print solution
        for v in m.getVars():
            if v.x > 0:
                print(f"{v.varName} = {v.x}")

        # Get solution:
        solution = m.getAttr('x', x)
        selected_roles = [role for role in solution.keys() if solution[role] > 0.5]
        print("Selected roles: {}".format(selected_roles))
        return solution, selected_roles