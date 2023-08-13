from oai_agents.common.subtasks import Subtasks
from oai_agents.gym_environments.worker_env import OvercookedSubtaskGymEnv
from oai_agents.agents.hrl import MultiAgentSubtaskWorker
from oai_agents.common.arguments import get_arguments

from pathlib import Path
import unittest

from gurobipy import *
# import gurobipy as gp

class RoleAssigner:
    def __init__(self, layout_name, args, n_players=2):
        """
        param env: expects oai_agents.gym_environments.base_overcooked_env.OvercookedSubtaskGymEnv
        param n_players: int, tested for 2 players only
        """
        
        self.layout_name = layout_name
        self.args = args
        self.n_players = n_players

    def evaluate_single_role(self, role_subtasks):
        """
        param role: a list of subtasks in oai_agents.subtasks.SUBTASKS format
        return: heuristic eval
        """
        
        raise NotImplementedError
    
    def evaluate_start_to_end(self, start_state, subtask):
        """
        param start_state: Overcooked PlayerState (?)
        param subtask: subtask name (string)
        return: heuristic eval, end state
        """
        subtask_id = Subtasks.SUBTASKS_TO_IDS[subtask]
        env_kwargs = {'single_subtask_id': subtask_id, 'stack_frames': False, 'full_init': True, 'args': self.args}
        env = OvercookedSubtaskGymEnv(layout_name=self.layout_name, **env_kwargs)
        worker_agent = MultiAgentSubtaskWorker.load(Path(self.args.base_dir / 'agent_models' / "HAHA" / "worker"), self.args)



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
    def test_evaluate_start_to_end(self):
        print("Testing start to end evaluation")
        args = get_arguments()
        role_assigner = RoleAssigner(layout_name="cramped_room", args=args)
        subtask = 'get_onion_from_dispenser'
        role_assigner.evaluate_start_to_end(None, subtask)

    def test_evaluate_single_role(self):
        print("Testing role evaluation")
        role_assigner = RoleAssigner(layout_name="cramped_room", args=None)

        role = ['get_onion', 'cook_onion', 'serve_soup']
        role_assigner.evaluate_single_role(role)

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