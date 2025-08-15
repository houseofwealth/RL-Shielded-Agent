from stable_baselines3.her import HerReplayBuffer

from ipdb import set_trace

from stable_baselines3.common.type_aliases import DictReplayBufferSamples
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy


class CustomHerReplayBuffer(HerReplayBuffer):
        '''
            Overwritten version of SB3.HerReplayBuffer that enables pretraining
            Refer to SB3.HerReplayBuffer for additional documentation.
        '''
        def __init__(
            self,
            env,
            buffer_size,
            device="auto",
            replay_buffer=None,
            max_episode_length=None,
            n_sampled_goal=4,
            goal_selection_strategy="future",
            online_sampling=True,
            handle_timeout_termination=True,
        ):

            super().__init__(
                env, buffer_size, device,
                replay_buffer, max_episode_length, n_sampled_goal,
                goal_selection_strategy, online_sampling, handle_timeout_termination
            )

            # Split achieved and desired goals
            self.achieved_goal_shape = get_obs_shape(self.env.observation_space.spaces["achieved_goal"])
            self.desired_goal_shape = get_obs_shape(self.env.observation_space.spaces["desired_goal"])

            # input dimensions for buffer initialization
            input_shape = {
                "observation": (self.env.num_envs,) + self.obs_shape,
                "achieved_goal": (self.env.num_envs,) + self.achieved_goal_shape,
                "desired_goal": (self.env.num_envs,) + self.desired_goal_shape,
                "action": (self.action_dim,),
                "reward": (1,),
                "next_obs": (self.env.num_envs,) + self.obs_shape,
                "next_achieved_goal": (self.env.num_envs,) + self.achieved_goal_shape,
                "next_desired_goal": (self.env.num_envs,) + self.desired_goal_shape,
                "done": (1,),
            }
            self._observation_keys = ["observation", "achieved_goal", "desired_goal"]
            self._buffer = {
                key: np.zeros((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
                for key, dim in input_shape.items()
            }
            # Store info dicts are it can be used to compute the reward (e.g. continuity cost)
            self.info_buffer = [deque(maxlen=self.max_episode_length) for _ in range(self.max_episode_stored)]
            # episode length storage, needed for episodes which has less steps than the maximum length
            self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)
        
        def _sample_transitions(
            self,
            batch_size,
            maybe_vec_env,
            online_sampling,
            n_sampled_goal=None,
        ):
            """
            :param batch_size: Number of element to sample (only used for online sampling)
            :param env: associated gym VecEnv to normalize the observations/rewards
                Only valid when using online sampling
            :param online_sampling: Using online_sampling for HER or not.
            :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
            :return: Samples.
            """
            # Select which episodes to use
            if online_sampling:
                assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
                # Do not sample the episode with index `self.pos` as the episode is invalid
                if self.full:
                    episode_indices = (
                        np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos
                    ) % self.n_episodes_stored
                else:
                    episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
                # A subset of the transitions will be relabeled using HER algorithm
                her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
            else:
                assert maybe_vec_env is None, "Transitions must be stored unnormalized in the replay buffer"
                assert n_sampled_goal is not None, "No n_sampled_goal specified for offline sampling of HER transitions"
                # Offline sampling: there is only one episode stored
                episode_length = self.episode_lengths[0]
                # we sample n_sampled_goal per timestep in the episode (only one is stored).
                episode_indices = np.tile(0, (episode_length * n_sampled_goal))
                # we only sample virtual transitions
                # as real transitions are already stored in the replay buffer
                her_indices = np.arange(len(episode_indices))

            ep_lengths = self.episode_lengths[episode_indices]

            if online_sampling:
                # Select which transitions to use
                transitions_indices = np.random.randint(ep_lengths)
            else:
                if her_indices.size == 0:
                    # Episode of one timestep, not enough for using the "future" strategy
                    # no virtual transitions are created in that case
                    return {}, {}, np.zeros(0), np.zeros(0)
                else:
                    # Repeat every transition index n_sampled_goals times
                    # to sample n_sampled_goal per timestep in the episode (only one is stored).
                    # Now with the corrected episode length when using "future" strategy
                    transitions_indices = np.tile(np.arange(ep_lengths[0]), n_sampled_goal)
                    episode_indices = episode_indices[transitions_indices]
                    her_indices = np.arange(len(episode_indices))

            # get selected transitions
            transitions = {key: self._buffer[key][episode_indices, transitions_indices].copy() for key in self._buffer.keys()}

            # sample new desired goals and relabel the transitions
            new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
            num_obs = new_goals.shape[2]
            num_dims = transitions["desired_goal"].shape[2]
            num_preds = num_obs / num_dims

            # We will randomly choose a predator trajectory to 
            # reassign the goal to
            # if num_preds > 1:                
            num_her_indices = len(her_indices)
            chosen_pred = np.random.choice(int(num_preds), size=num_her_indices)
            start_idx = chosen_pred * num_dims
            end_idx = (chosen_pred + 1) * num_dims
            
            for num, hi in enumerate(her_indices):
                transitions["desired_goal"][hi, :, :] = new_goals[num, :, start_idx[num]:end_idx[num]]

            # Convert info buffer to numpy array
            transitions["info"] = np.array(
                [
                    self.info_buffer[episode_idx][transition_idx]
                    for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
                ]
            )

            # Edge case: episode of one timesteps with the future strategy
            # no virtual transition can be created
            if len(her_indices) > 0:
                try:
                    infos = transitions["info"][her_indices, 0]
                except:
                    infos = transitions["info"][her_indices]
                    
                # Vectorized computation of the new reward
                transitions["reward"][her_indices, 0] = self.env.env_method(
                    "compute_reward",
                    # the new state depends on the previous state and action
                    # s_{t+1} = f(s_t, a_t)
                    # so the next_achieved_goal depends also on the previous state and action
                    # because we are in a GoalEnv:
                    # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                    # therefore we have to use "next_achieved_goal" and not "achieved_goal"
                    transitions["next_achieved_goal"][her_indices, 0],
                    # here we use the new desired goal
                    transitions["desired_goal"][her_indices, 0],
                    infos,
                )

            # concatenate observation with (desired) goal
            observations = self._normalize_obs(transitions, maybe_vec_env)

            # HACK to make normalize obs and `add()` work with the next observation
            next_observations = {
                "observation": transitions["next_obs"],
                "achieved_goal": transitions["next_achieved_goal"],
                # The desired goal for the next observation must be the same as the previous one
                "desired_goal": transitions["desired_goal"],
            }
            next_observations = self._normalize_obs(next_observations, maybe_vec_env)

            if online_sampling:
                next_obs = {key: self.to_torch(next_observations[key][:, 0, :]) for key in self._observation_keys}

                normalized_obs = {key: self.to_torch(observations[key][:, 0, :]) for key in self._observation_keys}

                return DictReplayBufferSamples(
                    observations=normalized_obs,
                    actions=self.to_torch(transitions["action"]),
                    next_observations=next_obs,
                    dones=self.to_torch(transitions["done"]),
                    rewards=self.to_torch(self._normalize_reward(transitions["reward"], maybe_vec_env)),
                )
            else:
                return observations, next_observations, transitions["action"], transitions["reward"]