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

        Here the objects are always the current subtasks, agents, and orders
        """

        objects_str = ""
        # First add all the subtasks
        # Needs to be of the from subtask_1 - subtask\n
        objects_str += " - subtask\n\t\t".join(self._subtasks) 
        objects_str += " - subtask\n\t\t"
        # Then all the agents
        objects_str += " - agent\n\t\t".join(self._agents) 
        objects_str += " - agent\n\t\t"

        return objects_str

    def _generate_init(self) -> str:
        init_str = ""
        # First assign all tubtasks to the predicate 'incomplete'
        # Needs to be of the form (incomplete subtask_1) (incomplete subtask_2) ...
        for subtask in self._subtasks:
            init_str += "\t\t(incomplete " + subtask + ")\n"
        # Then assign each agent to a particular subtask based on task_assignments
        # Should be of the form (assing subtask_1 agent_1) (assign subtask_2 agent_2) ...
        for agent, subtasks in self._task_assignments.items():
            for subtask in subtasks:
                init_str += "\t\t(assign " + subtask + " " + agent + ")\n"


        return init_str

    def _generate_goal(self) -> str:
        # TODO: Figure out the correct goal ought to be.
        # For now, let's assume its the last valid subtask

        goal_str = "(complete " + self._subtasks[-2] + ")"

        # goal_str = "(and\n"
        # for subtask in self._subtasks:
        #     goal_str += "\t\t(complete " + subtask + ")\n"
        # goal_str += "\t)"

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
    def __init__(self):
        """
        Generates a PDDL domain file for the overcooked domain.
        This defines the actions that can be taken by the mangager.

        TODO: Finish this!
        """
        domain_template = """(define (domain overcooked)
            (:requirements :typing)
            (:types subtask agent)
            (:predicates
                {predicates}
            )
            {actions}
        )"""
    


if __name__ == "__main__":


    subtasks  = {0: 'get_onion_from_dispenser', 1: 'get_onion_from_counter', 2: 'put_onion_in_pot', 3: 'put_onion_closer', 4: 'get_tomato_from_dispenser', 5: 'get_tomato_from_counter', 6: 'put_tomato_in_pot', 7: 'put_tomato_closer', 8: 'get_cabbage_from_dispenser', 9: 'get_cabbage_from_counter', 10: 'put_cabbage_in_pot', 11: 'put_cabbage_closer', 12: 'get_fish_from_dispenser', 13: 'get_fish_from_counter', 14: 'put_fish_in_pot', 15: 'put_fish_closer', 16: 'get_plate_from_dish_rack', 17: 'get_plate_from_counter', 18: 'put_plate_closer', 19: 'get_soup', 20: 'get_soup_from_counter', 21: 'put_soup_closer', 22: 'serve_soup', 23: 'unknown'}

    subtasks = list(subtasks.values())
    agents  = ["agent_1", "agent_2"]
    orders = [('cabbage', 'onion', 'tomato'), ('fish', 'fish', 'onion')]
    assignments = {"agent_1": ["get_onion_from_dispenser", "put_onion_in_pot"],
                   "agent_2": ["get_onion_from_counter", "put_onion_closer"]}


    pddl = GenerateProblemPDDL(subtasks, assignments, agents, orders)
    print(pddl)
    pddl.save_to_file()







