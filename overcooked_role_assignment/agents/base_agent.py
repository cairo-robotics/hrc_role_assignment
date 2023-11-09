from overcooked_role_assignment.agents.agent_utils import load_agent
from overcooked_role_assignment.common.arguments import get_args_to_save, set_args_from_load, get_arguments
from overcooked_role_assignment.common.state_encodings import ENCODING_SCHEMES
from overcooked_role_assignment.common.subtasks import calculate_completed_subtask, get_doable_subtasks, Subtasks
from overcooked_role_assignment.gym_environments.base_overcooked_env import USEABLE_COUNTERS

from overcooked_ai_py.mdp.overcooked_mdp import Action

from abc import ABC, abstractmethod
import argparse
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import numpy as np
# import torch.nn as nn
from typing import List, Tuple, Union

import wandb


class OAIAgent(ABC):
    """
    A smaller version of stable baselines Base algorithm with some small changes for my new agents
    https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm
    Ensures that all agents play nicely with the environment
    """

    def __init__(self, name, args):
        super(OAIAgent, self).__init__()
        self.name = name
        # Player index and Teammate index
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.args = args
        # Must define a policy. The policy must implement a get_distribution(obs) that returns the action distribution
        self.policy = None
        # Used in overcooked-demo code
        self.p_idx = None
        self.mdp = None
        self.horizon = None
        self.prev_subtask = Subtasks.SUBTASKS_TO_IDS['unknown']
        self.use_hrl_obs = False

    @abstractmethod
    def predict(self, obs, state=None, episode_start=None, deterministic: bool = False):
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """

    @abstractmethod
    def get_distribution(self, obs):
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """

    def set_idx(self, p_idx, layout_name, is_hrl=False, output_message=True, tune_subtasks=False):
        self.p_idx = p_idx
        self.layout_name = layout_name
        self.prev_state = None
        self.stack_frames = self.policy.observation_space['visual_obs'].shape[0] == (27 * self.args.num_stack)
        self.stackedobs = StackedObservations(1, self.args.num_stack, self.policy.observation_space['visual_obs'], 'first')
        if is_hrl:
            self.set_play_params(output_message, tune_subtasks)

    def set_encoding_params(self, mdp, horizon):
        self.mdp = mdp
        self.horizon = horizon
        self.terrain = self.mdp.terrain_mtx
        self.grid_shape = (7, 7)

    def action(self, state, deterministic=False):
        if self.p_idx is None or self.mdp is None or self.horizon is None:
            raise ValueError('Please call set_idx() and set_encoding_params() before action. '
                             'Or, call predict with agent specific obs')

        obs = self.encoding_fn(self.mdp, state, self.grid_shape, self.horizon, p_idx=self.p_idx)
        if self.stack_frames:
            obs['visual_obs'] = np.expand_dims(obs['visual_obs'], 0)
            if self.prev_state is not None:
                obs['visual_obs'] = self.stackedobs.reset(obs['visual_obs'])
            else:
                obs['visual_obs'], _ = self.stackedobs.update(obs['visual_obs'], np.array([False]), [{}])
            obs['visual_obs'] = obs['visual_obs'].squeeze()
        if 'player_completed_subtasks' in self.policy.observation_space.keys():
            # If this isn't the first step of the game, see if a subtask has been completed
            comp_st = [None, None]
            if self.prev_state is not None:
                for i in range(2):
                    try:
                        cst = calculate_completed_subtask(self.terrain, self.prev_state, state, i)
                    except ValueError as e:
                        cst = None
                    comp_st[i] = cst
                # If a subtask has been completed, update counts
                if comp_st[self.p_idx] is not None:
                    player_completed_tasks = np.eye(Subtasks.NUM_SUBTASKS)[comp_st[self.p_idx]]
                    self.prev_subtask = comp_st[self.p_idx]
                else:
                    player_completed_tasks = np.zeros(Subtasks.NUM_SUBTASKS)
                if comp_st[1 - self.p_idx] is not None:
                    tm_completed_tasks = np.eye(Subtasks.NUM_SUBTASKS)[comp_st[1 - self.p_idx]]
                else:
                    tm_completed_tasks = np.zeros(Subtasks.NUM_SUBTASKS)
                # If this is the first step of the game, reset subtask counts to 0
            else:
                player_completed_tasks = np.zeros(Subtasks.NUM_SUBTASKS)
                tm_completed_tasks = np.zeros(Subtasks.NUM_SUBTASKS)
            obs['player_completed_subtasks'] = player_completed_tasks
            obs['teammate_completed_subtasks'] = tm_completed_tasks
        if 'subtask_mask' in self.policy.observation_space.keys():
            obs['subtask_mask'] = get_doable_subtasks(state, self.prev_subtask, self.layout_name, self.terrain, self.p_idx, USEABLE_COUNTERS[self.layout_name]).astype(bool)

        self.prev_state = deepcopy(state)
        obs = {k: v for k, v in obs.items() if k in self.policy.observation_space.keys()}

        try:
            agent_msg = self.get_agent_output()
        except AttributeError as e:
            agent_msg = ' '

        action, _ = self.predict(obs, deterministic=deterministic)
        return Action.INDEX_TO_ACTION[int(action)], agent_msg

    def _get_constructor_parameters(self):
        return dict(name=self.name, args=self.args)

    def step(self):
        pass

    def reset(self):
        pass

    def save(self, path: Path) -> None:
        """
        Save model to a given location.
        :param path:
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        save_path = path / 'agent_file'
        args = get_args_to_save(self.args)
        # th.save({'agent_type': type(self), 'state_dict': self.state_dict(),
        #          'const_params': self._get_constructor_parameters(), 'args': args}, save_path)

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> 'OAIAgent':
        """
        Load model from path.
        :param path: path to save to
        :param device: Device on which the policy should be loaded.
        :return:
        """
        # device = args.device
        # load_path = path / 'agent_file'
        # saved_variables = th.load(load_path, map_location=device)
        # set_args_from_load(saved_variables['args'], args)
        # saved_variables['const_params']['args'] = args
        # # Create agent object
        # model = cls(**saved_variables['const_params'])  # pytype: disable=not-instantiable
        # # Load weights
        # model.load_state_dict(saved_variables['state_dict'])
        # model.to(device)
        # return model
        pass
