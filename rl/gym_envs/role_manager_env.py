from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv, USEABLE_COUNTERS
from oai_agents.common.subtasks import Subtasks, get_doable_subtasks, calculate_completed_subtask
from oai_agents.gym_environments.manager_env import OvercookedManagerGymEnv
# from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction

import numpy as np

class OvercookedRoleManagerGymEnv(OvercookedManagerGymEnv):
    def __init__(self, assigned_subtasks, worker=None, **kwargs):
        # assigned_subtasks is a list of subtask indices that the role manager is responsible for
        self.role_subtasks = assigned_subtasks
        super(OvercookedRoleManagerGymEnv, self).__init__(worker=worker, **kwargs)
    
    def action_masks(self, p_idx=None):
        p_idx = p_idx or self.p_idx
        available = get_doable_subtasks(self.state, self.prev_subtask[p_idx], self.layout_name, self.terrain, p_idx, USEABLE_COUNTERS[self.layout_name]).astype(bool)
        return np.array([i in self.role_subtasks for i in range(Subtasks.NUM_SUBTASKS)]) & available