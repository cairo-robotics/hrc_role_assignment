import unittest
from oai_agents.common.arguments import get_arguments
from oai_agents.common.subtasks import Subtasks

from scripts.role_assignment import RoleAssigner

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
        score, route, goal = role_assigner.evaluate_single_role(role_subtasks, player=1)
        print(score, route, goal)

    def test_evaluate_all_roles(self):
        print("Testing bulk role evaluation in large_room")
        role_assigner = RoleAssigner(layout_name="large_room", args=None)
        roles = {
        'A': ['get_onion_from_dispenser', 'put_onion_in_pot'],
        'B': ['get_plate_from_dish_rack', 'get_soup'],
        'C': ['get_onion_from_dispenser', 'serve_soup'],
        'D': ['get_plate_from_dish_rack', 'put_soup_closer']
        }

        cost_dicts = role_assigner.evaluate_roles(roles)
        print(cost_dicts)


    def test_evaluate_all_roles_assymetric(self):
        print("Testing bulk role evaluation in asymmetric_advantages")
        role_assigner = RoleAssigner(layout_name="asymmetric_advantages", args=None)
        roles = {
        'A': ['get_onion_from_dispenser', 'put_onion_in_pot'],
        'B': ['get_plate_from_dish_rack', 'get_soup'],
        'C': ['get_onion_from_dispenser', 'serve_soup'],
        'D': ['get_plate_from_dish_rack', 'put_soup_closer']
        }

        cost_dicts = role_assigner.evaluate_roles(roles)
        print(cost_dicts)

    def test_assign_roles(self):
        print("Testing role assignment")
        role_assigner = RoleAssigner(layout_name="cramped_room", args=None)

        # time = {'A': 2, 'B': 4, 'C': 1}  # Minimum time it takes for either player to perform the role
        time = [
            {'A': 2, 'B': 4, 'C': 1},
            {'A': 2, 'B': 2, 'C': 1}
        ]
        roles = {
        'A': ['get_onion', 'cook_onion', 'serve_soup'], 
        'B': ['get_onion', 'place_onion', 'chop_onion'], 
        'C': ['place_onion', 'chop_onion', 'serve_soup']
        }
        tasksNeeded = ['get_onion', 'chop_onion', 'cook_onion', 'serve_soup']
        role_assigner.assign_roles(roles, time, tasksNeeded)

if __name__ == "__main__":
    unittest.main()