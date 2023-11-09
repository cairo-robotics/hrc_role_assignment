from oai_agents.common.subtasks import Subtasks, get_doable_subtasks
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from oai_agents.agents.base_agent import OAIAgent
from oai_agents.agents.agent_utils import DummyPolicy
from agents.planner_agent import PlanBasedWorkerAgent

import numpy as np
from gym import spaces

class PDDLManager: # TODO complete
    """
    class that takes the role of the manager agent, choosing next subtask
    given current state and role information
    p_idx: int, index of this player in the game (0 for p1, 1 for p2)
    """
    def __init__(self, name: str,  
                 mlam: MediumLevelActionManager, role_mask: np.ndarray,
                 p_idx: int):
        self.name = name
        self.mlam = mlam
        self.mdp = mlam.mdp
        self.role_mask = role_mask
        self.p_idx = p_idx

    def predict(self, state: OvercookedState) -> int:
        """
        Predicts the next action to take based on the current state and
        the subtasks allowed by the agent's current role.
        return: int, index of next subtask goal in Subtasks.SUBTASKS
        """
        # raise NotImplementedError
        curr_orders = state.all_orders
        return 0 # this corresponds to get_onion_from_dispenser, FYI


class PDDLAgent(OAIAgent):
    def __init__(self, name: str, mlam: MediumLevelActionManager, 
                 role_mask: np.ndarray,
                 p_idx: int, args):
        super(PDDLAgent, self).__init__(name, args)
        self.manager = PDDLManager("pddl_agent_manager", mlam, role_mask, p_idx) # agent responsible for choosing subtasks
        self.worker = PlanBasedWorkerAgent("pddl_worker", mlam, p_idx, args) # agent responsible for executing subtasks
        self.mlam = mlam
        self.curr_subtask_id = None
        self.mdp = mlam.mdp

        self.policy = DummyPolicy(spaces.Dict({'state': spaces.Box(0,1,(1,)),
                                        'visual_obs': spaces.Box(0,1,(1,)),
                                        'player_completed_subtasks': spaces.Box(0,1,(Subtasks.NUM_SUBTASKS,)),
                                        'teammate_completed_subtasks': spaces.Box(0,1,(Subtasks.NUM_SUBTASKS,)),
                                        'current_recipes': spaces.Box(0,1,(Subtasks.NUM_SUBTASKS,))}))

        self.subtask_mask = np.ones(Subtasks.NUM_SUBTASKS)
        self.p_idx = p_idx

    def predict(self, obs: dict, state=None, episode_start=None, deterministic: bool = False):
        if self.curr_subtask_id is None or np.sum(obs['player_completed_subtasks']) == 1:
            next_st = self.manager.predict(obs['state'])
            self.curr_subtask_id = int(next_st)
            comp_st = np.argmax(obs["player_completed_subtasks"], axis=0)
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {Subtasks.IDS_TO_SUBTASKS[comp_st]}')
            doable_st = [Subtasks.IDS_TO_SUBTASKS[idx] for idx, doable in enumerate(obs['subtask_mask']) if doable == 1]
            print('DOABLE SUBTASKS:', doable_st)
        obs['curr_subtask'] = Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]
        obs.pop('player_completed_subtasks')
        obs.pop('teammate_completed_subtasks')
        return self.worker.predict(obs, state=state, episode_start=episode_start, deterministic=True)

    def get_distribution(self, obs, sample=True):
        if obs['player_completed_subtasks'] is not None:
            next_st = self.smanager.predict(obs['state'])
            self.curr_subtask_id = int(next_st)
            # Completed previous subtask, set new subtask
            print(f'GOAL: {Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]}, DONE: {obs["player_completed_subtasks"]}')
            next_st = input("Enter next subtask (0-10): ")
            self.curr_subtask_id = int(next_st)
        obs['curr_subtask'] = Subtasks.IDS_TO_SUBTASKS[self.curr_subtask_id]
        return self.worker.get_distribution(obs, sample=sample)