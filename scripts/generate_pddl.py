#!/usr/bin/env python
import os
from typing import Any, Callable, Optional, Union, Tuple




class GenerateProblemPDDL:
    def __init__(self,
                 subtasks: list[str],
                 task_assignments: dict[str,list[str]],
                 agents: list[str],
                 orders: list[Tuple[str, str, str]],
                 ):
        """"
        Generates a PDDL problem file for the overcooked domain.
        The problem file defines the initial state of the domain as well
        as the goal.
        """

        self._subtasks = subtasks
        self._task_assignments = task_assignments
        self._agents = agents
        self._orders = orders
        # self._state = state

        # Get path to the grandparent directory of this file
        self._path = os.path.dirname(os.path.abspath(__file__))
        self._path = os.path.join(self._path, "..", "pddl/overcooked/")
        # self._path = os.path.dirname(os.path.abspath(__file__))
        # self._path = os.path.join(self._path, "...", "pddl/overcooked/")
        print(self._path)

        objects_str = self._generate_objects()

        self._problem_template = """ (define (problem overcooked)
            (:domain overcooked)
            (:objects\n\t\t{objects})
            (:init\n{init})
            (:goal {goal})
        )"""

    def __str__(self) -> str:
        return self._problem_template.format(objects=self._generate_objects(),
                                             init=self._generate_init(),
                                             goal=self._generate_goal())
    def _generate_objects(self) -> str:
        """
        Generates the objects section of the PDDL problem file.

        Here the objects are the ingredients, plates,  and agents.
        """

        objects_str = ""
        # For each ingredient in orders need to add predicate of the form
        # ing-i - ingerdient\n
        self._ingredient_list = ["{}-{}".format(ing, i) for order in self._orders for i, ing in enumerate(order)]
        objects_str += " - ingredient\n\t\t".join(self._ingredient_list)
        objects_str += " - ingredient\n\t\t"
        # Then all the plates
        self._plate_list = ["plate-{}".format(i) for i in range(len(self._orders))]
        objects_str += " - plate\n\t\t".join(self._plate_list)
        objects_str += " - plate\n\t\t"
        # As we cant create new objects with PDDL, we need to define in advance
        # the new meals that can be created
        self._new_meal_list = ["new-{}".format(i) for i in range(len(self._orders))]
        objects_str += " - ingredient\n\t\t".join(self._new_meal_list)
        objects_str += " - ingredient\n\t\t"

        objects_str += " - subtask\n\t\t".join(self._subtasks)
        objects_str += " - subtask\n\t\t"
        # # Then all the agents
        objects_str += " - agent\n\t\t".join(self._agents)
        objects_str += " - agent\n\t\t"

        return objects_str

    def _generate_init(self) -> str:
        init_str = ""
        # Flatten orders list and turn into set to get unique ingredients
        ingredient_types = set([ing for order in self._orders for ing in order])
        print(ingredient_types)
        for ing in self._ingredient_list:
            for ing_type in ingredient_types:
                if ing_type in ing:
                    init_str += "\t\t(is{} {})\n".format(ing_type, ing)
                    init_str += "\t\t(instorage {})\n".format(ing)
        # First assign all tubtasks to the predicate 'incomplete'
        # Needs to be of the form (incomplete subtask_1) (incomplete subtask_2) ...
        for plate in self._plate_list:
            init_str += "\t\t(isplate {})\n".format(plate)
            init_str += "\t\t(empty {})\n".format(plate)
            init_str += "\t\t(instorage {})\n".format(plate)
        # for subtask in self._subtasks:
        #     init_str += "\t\t(incomplete " + subtask + ")\n"
        # Then assign each agent to a particular subtask based on task_assignments
        # Should be of the form (assing subtask_1 agent_1) (assign subtask_2 agent_2) ...
        for agent, subtasks in self._task_assignments.items():
            for subtask in subtasks:
                init_str += "\t\t(assign " + subtask + " " + agent + ")\n"

        for meal in self._new_meal_list:
            init_str += "\t\t(hypothetical {})\n".format(meal)

        return init_str

    def _generate_goal(self) -> str:
        # TODO: Figure out the correct goal ought to be.
        # For now, let's assume its the last valid subtask
        goal_str = "(and\n"

        for i, order in enumerate(self._orders):
            # Create pred str of the form ising1ing2ing3soup
            goal_str += "\t\t(is" + order[0] + order[1] + order[2] + "soup" + " new-{}".format(i) + ")\n"
            goal_str += "\t\t(isserved new-{}".format(i) + ")\n"

        goal_str += "\t)"


        return goal_str

    def save_to_file(self) -> None:
        """
        Writes pddl string to file.
        """
        # First make sure the directory exists
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        with open(os.path.join(self._path, "problem.pddl"), "w") as f:
            f.write(str(self))

        print("[save_to_file] Saved to file.")


class GenerateDomainPDDL:
    def __init__(self,
                 subtasks: list[str],
                 task_assignments: dict[str,list[str]],
                 agents: list[str],
                 orders: list[Tuple[str, str, str]],
                 ):
        """
        Generates a PDDL domain file for the overcooked domain.
        This defines the actions that can be taken by the mangager.

        TODO: Finish this!
        """
        self._subtasks = subtasks
        self._task_assignments = task_assignments
        self._agents = agents
        self._orders = orders


        domain_template = """(define (domain overcooked)
            (:requirements :typing)
            (:types subtask agent)
            (:predicates
                {predicates}
            )
            {actions}
        )""".format(predicates=self._generate_predicates(),
                    actions=self._generate_actions())


    def _generate_predicates(self) -> str:
        """
        Generates the predicates section of the PDDL domain file.

        Here the predicates refer to both object predicates as well as functions
        """
        predicates_str = ""
        ingredient_types = set([ing for order in self._orders for ing in order])
        for ing_type in ingredient_types:
            predicates_str = "\t\t(is{} ?ing - ingredient)\n".format(ing_type)

        for i, order in enumerate(self._orders):
            # Create pred str of the form ising1ing2ing3soup
            predicates_str += "\t\t(is" + order[0] + order[1] + order[2] + "soup" + " new-{}".format(i) + ")\n"

        predicates_str += "\t\t(holding ?object - object ?agent - agent)\n"
        predicates_str += "\t\t(isplate ?plate - plate)\n"
        predicates_str += "\t\t(isagent ?agent - agent)\n"
        predicates_str += "\t\t(issoup ?soup - ingredient)\n"
        predicates_str += "\t\t(assign ?subtask - subtask ?agent - agent)\n"
        predicates_str += "\t\t(isserved ?ing - ingredient)\n"
        predicates_str += "\t\t(instorage ?x - object)\n"
        predicates_str += "\t\t(oncounter ?x - object)\n"
        predicates_str += "\t\t(onplate ?ing - ingredient ?plate - plate)\n"
        predicates_str += "\t\t(inpot ?ing  - ingredient)\n"
        predicates_str += "\t\t(empty ?plate - plate)\n"
        predicates_str += "\t\t(hypothetical ?ing - ingredient)\n"
        # TODO: Add predicates for each subtask


        return predicates_str

    def _generate_actions(self) -> str:
        """
        Generates the actions section of the PDDL domain file.

        Here the actions refer to both object predicates as well as functions
        """
        pass

    def _get_ing_from_dispenser(self, ing: str) -> str:

        if ing == "plate":
            loc = "dish_rack"
        else:
            loc = "dispenser"

        action_str  = f"(:action get_{ing}_from_{loc}\n"
        action_str += f":parameters (?agent - agent ?{ing} - ingredient)\n"
        action_str += f":precondition (and (isagent ?agent) (assign get_{ing}_from_dispenser ?agent)(is{ing} ?{ing}) (instorage ?{ing}))\n"
        action_str += f":effect (and (not (instorage ?{ing})) (holding ?{ing} ?agent)))\n"


        return action_str

    def _get_ing_from_counter(self, ing: str) -> str:

        action_str  = f"(:action get_{ing}_from_counter\n"
        action_str += f":parameters (?agent - agent ?{ing} - ingredient)\n"
        action_str += f":precondition (and (isagent ?agent) (is{ing} ?{ing}) (oncounter ?{ing}))\n"
        action_str += f":effect (and (not (oncounter ?{ing})) (holding ?{ing} ?agent)))\n"

        return action_str

    def _put_ing_in_pot(self, ing: str) -> str:

        action_str  = f"(:action put_{ing}_in_pot\n"
        action_str += f":parameters (?agent - agent ?{ing} - ingredient)\n"
        action_str += f":precondition (and (isagent ?agent) (assign put_{ing}_in_pot ?agent) (is{ing} ?{ing}) (holding ?{ing} ?agent))\n"
        action_str += f":effect (and (not (holding ?{ing} ?agent)) (inpot ?{ing})))\n"

        return action_str

    def _get_soup(self) -> str:

        action_str  = "(:action get_soup\n"
        action_str = ":parameters (?agent - agent ?soup - ingredient ?plate - plate)\n"
        action_str = ":precondition (and (isagent ?agent) (assign get_soup ?agent) (issoup ?soup) (inpot ?soup) (holding ?plate))\n"
        action_str = ":effect (and (not (inpot ?soup)) (onplate ?soup ?plate)))\n"

        return action_str

    def _put_ing_closer(self, ing: str) -> str:

        action_str  = f"(:action put_{ing}_closer\n"
        action_str += f":parameters (?agent - agent ?{ing} - ingredient)\n"
        action_str += f":precondition (and (isagent ?agent) (assign put_{ing}_closer ?agent) (is{ing} ?{ing}) (holding ?{ing} ?agent))\n"
        action_str += f":effect (and (not (holding ?{ing} ?agent)) (oncounter ?{ing})))\n"

        return action_str


    def _serve_soup(self) -> str:

        action_str  = "(:action serve_soup\n"
        action_str += ":parameters (?agent - agent ?plate - plate ?soup - ingredient)\n"
        action_str += ":precondition (and (isagent ?agent) (assign serve_soup ?agent) (issoup ?soup) (onplate ?soup ?plate))\n"
        action_str += ":effect (and (not (onplate ?soup ?plate)) (isserved ?soup)))\n"

        return action_str





if __name__ == "__main__":


    subtasks  = {0: 'get_onion_from_dispenser', 1: 'get_onion_from_counter', 2: 'put_onion_in_pot', 3: 'put_onion_closer', 4: 'get_tomato_from_dispenser', 5: 'get_tomato_from_counter', 6: 'put_tomato_in_pot', 7: 'put_tomato_closer', 8: 'get_cabbage_from_dispenser', 9: 'get_cabbage_from_counter', 10: 'put_cabbage_in_pot', 11: 'put_cabbage_closer', 12: 'get_fish_from_dispenser', 13: 'get_fish_from_counter', 14: 'put_fish_in_pot', 15: 'put_fish_closer', 16: 'get_plate_from_dish_rack', 17: 'get_plate_from_counter', 18: 'put_plate_closer', 19: 'get_soup', 20: 'get_soup_from_counter', 21: 'put_soup_closer', 22: 'serve_soup'}

    subtasks = list(subtasks.values())
    agents  = ["agent_1", "agent_2"]
    orders = [('cabbage', 'onion', 'tomato'), ('fish', 'fish', 'onion')]
    assignments = {"agent_1": ["get_onion_from_dispenser", "put_onion_in_pot"],
                   "agent_2": ["get_onion_from_counter", "put_onion_closer"]}


    pddl = GenerateProblemPDDL(subtasks, assignments, agents, orders)
    print(pddl)
    pddl.save_to_file()
