from custom_mlp_extractor import CustomActorMlpExtractor, CustomCriticMlpExtractor, CustomBothMlpExtractor
from stable_baselines3.ppo.policies import MultiInputPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from ipdb import set_trace

import sys
import time

from typing import Callable, Dict, List, Optional, Tuple, Type, Union, TypeVar
import numpy as np
import torch as th
from torch import nn
import gymnasium as gym
from gymnasium import spaces


from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


from stable_baselines3.common.distributions import (
    make_proba_distribution,
)

# SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")




class CustomActorCriticPolicy_Actor(MultiInputPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        num_fake_preds: int  = 0,
        *args,
        **kwargs,
    ):

        # build mlp in super:
        super(CustomActorCriticPolicy_Actor, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.num_dims = observation_space['desired_goal'].shape[0]
        assert self.num_dims <= 3, 'Too man physical dimensions'   
        # self.num_preds = int((observation_space['achieved_goal'].shape[0] / self.num_dims) - num_fake_preds) 
        self.num_preds = int(observation_space['achieved_goal'].shape[0] / self.num_dims) 
        # self.features_dim = self.features_extractor.features_dim
        orig_action_space_size = action_space.shape[0]
        self.pred_action_space = spaces.Box(-1, 1, shape=(int(orig_action_space_size/(self.num_preds )),))
        # Action distribution
        self.action_dist = make_proba_distribution(self.pred_action_space, use_sde=kwargs['use_sde'], dist_kwargs=self.dist_kwargs)
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = CustomActorMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            observation_space = self.observation_space,
            # num_preds = self.num_preds,
        )
 
    def forward_components(self, latent, deterministic):
        distribution = self._get_action_dist_from_latent(latent)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.pred_action_space.shape)
        return actions, log_prob
    
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        # set_trace()
        features = self.extract_features(obs)
        # if self.share_features_extractor:
        #     latent_pi, latent_vf = self.mlp_extractor(features)
        # else:
        #     pi_features, vf_features = features
        latent_pi = self.mlp_extractor.forward_actor(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        # set_trace()
        chunks = th.chunk(latent_pi, self.num_preds, dim=1)
        act_list = []
        prob_list = []
        for chunk in chunks:
            act, prob = self.forward_components(chunk, deterministic)
            act_list.append(act)
            prob_list.append(prob)
        actions = th.cat(tuple(act_list), dim=1)
        log_prob = th.mean(th.cat(tuple(prob_list)))
        # distribution = self._get_action_dist_from_latent(latent_pi)
        # actions = distribution.get_actions(deterministic=deterministic)
        # log_prob = distribution.log_prob(actions)
        # actions = actions.reshape((-1,) + self.action_space.shape)
        # actions, log_prob = self.forward_components(latent_pi, deterministic)
        # print(f'Values: {values.shape}')
        # set_trace()
        return actions, values, log_prob
    
    def eval_components(self, latent, actions):
        distribution = self._get_action_dist_from_latent(latent)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy
        
        
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        # set_trace()
        features = self.extract_features(obs)
        # if self.share_features_extractor:
        #     latent_pi, latent_vf = self.mlp_extractor(features)
        # else:
        #     pi_features, vf_features = features
        latent_pi = self.mlp_extractor.forward_actor(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        act_chunks = th.chunk(actions, self.num_preds, dim=1)
        laten_chunks = th.chunk(latent_pi, self.num_preds, dim=1)
        prob_list = []
        entr_list = []
        for a, l in zip(act_chunks, laten_chunks):
            prob, entr = self.eval_components(l, a)
            prob_list.append(prob)
            entr_list.append(entr)
        log_prob = th.mean(th.stack(tuple(prob_list)), dim=0)
        entropy = th.mean(th.stack(tuple(entr_list)), dim=0)
        # _, log_prob, entropy = self.actor_process(latent_pi, deterministic)
        # distribution = self._get_action_dist_from_latent(latent_pi)
        # log_prob = distribution.log_prob(actions)
        # entropy = distribution.entropy()
        # print(f'Values: {values.shape}')
        return values, log_prob, entropy
    
    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

            # Switch to eval mode (this affects batch norm / dropout)
            self.set_training_mode(False)

            # observation, vectorized_env = self.obs_to_tensor(observation)

            with th.no_grad():
                actions = self._predict(observation, deterministic=deterministic)
            # Convert to numpy, and reshape to the original action shape
            # set_trace()
            actions = actions.cpu().numpy().reshape(self.action_space.shape)

            if isinstance(self.action_space, spaces.Box):
                if self.squash_output:
                    # Rescale to proper domain when using squashing
                    actions = self.unscale_action(actions)
                else:
                    # Actions could be on arbitrary scale, so clip the actions to avoid
                    # out of bound error (e.g. if sampling from a Gaussian distribution)
                    actions = np.clip(actions, self.action_space.low, self.action_space.high)
            
            # # Remove batch dimension if needed
            # if not vectorized_env:
            #     actions = actions.squeeze(axis=0)

            return actions, state
    
    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        distributions = self.get_distribution(observation)
        actions = []
        for dist in distributions:
            actions.append(dist.get_actions(deterministic=deterministic))
        joined_actions = th.cat(tuple(actions), dim=1)
        return joined_actions
    
    def get_distribution(self, obs):
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        # features = super().extract_features(obs, self.pi_features_extractor)
        distribution_list = []
        latent_pi = self.mlp_extractor.forward_actor(obs)
        chunks = th.chunk(latent_pi, self.num_preds, dim=1)
        for chunk in chunks:
            distribution_list.append(self._get_action_dist_from_latent(chunk))
        # set_trace()
        return distribution_list


class CustomActorCriticPolicy_Critic(MultiInputPolicy): 
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy_Critic, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

        self.num_dims = observation_space['desired_goal'].shape[0]
        assert self.num_dims <= 3, 'Too man physical dimensions'   
        self.num_preds = int(observation_space['achieved_goal'].shape[0] / self.num_dims)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = CustomCriticMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            observation_space = self.observation_space,
        )    
        
        
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        # set_trace()
        features = self.extract_features(obs)
        # if self.share_features_extractor:
        #     latent_pi, latent_vf = self.mlp_extractor(features)
        # else:
        #     pi_features, vf_features = features
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(obs)
        # Evaluate the values for the given observations
        vals = []
        for chunk in th.chunk(latent_vf, self.num_preds, dim=1):
            val = self.value_net(chunk)
            vals.append(val)
        values = th.mean(th.cat(tuple(vals), dim=1), dim=1)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)

        return actions, values, log_prob
    
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        # set_trace()
        features = self.extract_features(obs)
        # if self.share_features_extractor:
        #     latent_pi, latent_vf = self.mlp_extractor(features)
        # else:
        #     pi_features, vf_features = features
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(obs)
        
        vals = []
        for chunk in th.chunk(latent_vf, self.num_preds, dim=1):
            val = self.value_net(chunk)
            vals.append(val)
        # print(f'vals: {vals}')
        values = th.mean(th.cat(tuple(vals), dim=1), dim=1)
        # print(f' values: {values.shape}')
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        # values = self.value_net(latent_vf)
        # print(f'log probs: {log_prob.shape}')
        # print(f'values: {values.shape}')
        entropy = distribution.entropy()
        return values, log_prob, entropy
    
    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        # features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(obs)
        vals = []
        for chunk in th.chunk(latent_vf, self.num_preds, dim=1):
            val = self.value_net(chunk)
            vals.append(val)
        values = th.mean(th.cat(tuple(vals), dim=1), dim=1)
        return values



class CustomActorCriticPolicy_Both(MultiInputPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
        ):
     
        super(CustomActorCriticPolicy_Both, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


        # Disable orthogonal initialization
        self.ortho_init = False
        self.num_dims = observation_space['desired_goal'].shape[0]
        assert self.num_dims <= 3, 'Too man physical dimensions'   
        self.num_preds = int(observation_space['achieved_goal'].shape[0] / self.num_dims)
        # self.features_dim = self.features_extractor.features_dim
        orig_action_space_size = action_space.shape[0]
        self.pred_action_space = spaces.Box(-1, 1, shape=(int(orig_action_space_size/self.num_preds),))
        # Action distribution
        self.action_dist = make_proba_distribution(self.pred_action_space, use_sde=kwargs['use_sde'], dist_kwargs=self.dist_kwargs)
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = CustomBothMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            observation_space = self.observation_space,
        )
        
 
    def forward_components(self, latent, deterministic):
        distribution = self._get_action_dist_from_latent(latent)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.pred_action_space.shape)
        return actions, log_prob
 
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        # set_trace()
        features = self.extract_features(obs)
        # if self.share_features_extractor:
        #     latent_pi, latent_vf = self.mlp_extractor(features)
        # else:
        #     pi_features, vf_features = features
        latent_pi = self.mlp_extractor.forward_actor(obs)
        latent_vf = self.mlp_extractor.forward_critic(obs)
        # Evaluate the values for the given observations
        # values = self.value_net(latent_vf)
        # set_trace()
        chunks = th.chunk(latent_pi, self.num_preds, dim=1)
        act_list = []
        prob_list = []
        for chunk in chunks:
            act, prob = self.forward_components(chunk, deterministic)
            act_list.append(act)
            prob_list.append(prob)
        actions = th.cat(tuple(act_list), dim=1)
        log_prob = th.mean(th.cat(tuple(prob_list)))
        
        vals = []
        for chunk in th.chunk(latent_vf, self.num_preds, dim=1):
            val = self.value_net(chunk)
            vals.append(val)
        values = th.mean(th.cat(tuple(vals), dim=1), dim=1)
        # distribution = self._get_action_dist_from_latent(latent_pi)
        # actions = distribution.get_actions(deterministic=deterministic)
        # log_prob = distribution.log_prob(actions)
        # actions = actions.reshape((-1,) + self.action_space.shape)
        # actions, log_prob = self.forward_components(latent_pi, deterministic)

        return actions, values, log_prob
    
    def eval_components(self, latent, actions):
        distribution = self._get_action_dist_from_latent(latent)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy
        
        
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
      
        features = self.extract_features(obs)
        # if self.share_features_extractor:
        #     latent_pi, latent_vf = self.mlp_extractor(features)
        # else:
        #     pi_features, vf_features = features
        latent_pi = self.mlp_extractor.forward_actor(obs)
        latent_vf = self.mlp_extractor.forward_critic(obs)
        # values = self.value_net(latent_vf)
        act_chunks = th.chunk(actions, self.num_preds, dim=1)
        laten_chunks = th.chunk(latent_pi, self.num_preds, dim=1)
        prob_list = []
        entr_list = []
        for a, l in zip(act_chunks, laten_chunks):
            prob, entr = self.eval_components(l, a)
            prob_list.append(prob)
            entr_list.append(entr)
        log_prob = th.mean(th.stack(tuple(prob_list)), dim=0)
        entropy = th.mean(th.stack(tuple(entr_list)), dim=0)
        
        vals = []
        for chunk in th.chunk(latent_vf, self.num_preds, dim=1):
            val = self.value_net(chunk)
            vals.append(val)
        values = th.mean(th.cat(tuple(vals), dim=1), dim=1)
        # _, log_prob, entropy = self.actor_process(latent_pi, deterministic)
        # distribution = self._get_action_dist_from_latent(latent_pi)
        # log_prob = distribution.log_prob(actions)
        # entropy = distribution.entropy()
        return values, log_prob, entropy
    
    def predict(self, *args, **kwargs):
        raise 'Predict Broken'
    
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)
    
    def get_distribution(self, obs: th.Tensor):
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        # features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(obs)
        return self._get_action_dist_from_latent(latent_pi)

    
    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        # features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(obs)
        vals = []
        for chunk in th.chunk(latent_vf, self.num_preds, dim=1):
            val = self.value_net(chunk)
            vals.append(val)
        values = th.mean(th.cat(tuple(vals), dim=1), dim=1)
        return values




class CustomPPOPolicy(MultiInputPolicy):

    def make_actor(self, features_extractor=None):
        print('------Making Actor--------')
        set_trace()
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        # set_trace()
        return CustomActor(**actor_kwargs).to(self.device)
    
    
 