
from typing import Dict, List, Tuple, Type, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device

from ipdb import set_trace


class CustomActorMlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        observation_space = None,
        # num_preds = 0,
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim
        self.device = device

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.num_dims = observation_space['desired_goal'].shape[0]
        assert self.num_dims <= 3, 'Too man physical dimensions'   
        # self.num_preds = num_preds
        self.num_preds = int(observation_space['achieved_goal'].shape[0] / self.num_dims)
        
        # add layer to take policy_net from 2*feature_dim to feature_dim
        # policy_net.append(nn.Linear(last_layer_dim_pi, int(last_layer_dim_pi/self.num_preds)))
        # policy_net.append(activation_fn())
        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        
        assert observation_space is not None, 'Observatin space failed to pass through'
        self.observation_space = observation_space
        # policy distributions require a latent_dim * num_preds 
        # self.latent_dim_pi = self.latent_dim_pi*self.num_preds
        
    def get_heads(self, observations):
        # set_trace()
        # # print(observations)
        # self.num_dims = observations['desired_goal'].shape[1]
        # assert self.num_dims <= 3, 'Too man physical dimensions'   
        # self.num_preds = int(observations['achieved_goal'].shape[1] / self.num_dims)
        
        # Create heads
        heads = []
        pred_chunks = th.chunk(observations['achieved_goal'], self.num_preds, dim=1)
        for pred_num in range(self.num_preds):
            current_pred = pred_chunks[pred_num]
            try:
                other_preds = th.cat(
                    [pred_chunks[i] for i in range(self.num_preds) if i != pred_num],
                    axis=1,
                )
            except:
                other_preds = th.tensor([])
            current_obs = th.cat(
                (
                current_pred,
                other_preds.to(pred_chunks[0].device),
                observations['desired_goal'],
                observations['observation'],
                ),
                dim=1
            )
            heads.append(current_obs)
        heads = tuple(heads)
        return heads

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        features = self.get_heads(features)
        output_heads = [self.policy_net(feature.float()) for feature in features]
        output = th.cat(tuple(output_heads), dim=1)
        return output


    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
    
    
    
    
    
class CustomCriticMlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        observation_space = None,
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        self.num_dims = observation_space['desired_goal'].shape[0]
        assert self.num_dims <= 3, 'Too man physical dimensions'   
        self.num_preds = int(observation_space['achieved_goal'].shape[0] / self.num_dims)
        
        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi 
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        
    def get_heads(self, observations):
        # set_trace()
        # # print(observations)
        # self.num_dims = observations['desired_goal'].shape[1]
        # assert self.num_dims <= 3, 'Too man physical dimensions'   
        # self.num_preds = int(observations['achieved_goal'].shape[1] / self.num_dims)
        
        # Create heads
        heads = []
        pred_chunks = th.chunk(observations['achieved_goal'], self.num_preds, dim=1)
        for pred_num in range(self.num_preds):
            current_pred = pred_chunks[pred_num]
            try:
                other_preds = th.cat(
                    [pred_chunks[i] for i in range(self.num_preds) if i != pred_num],
                    axis=1,
                )
            except:
                other_preds = th.tensor([])
            current_obs = th.cat(
                (
                current_pred,
                other_preds.to(pred_chunks[0].device),
                observations['desired_goal'],
                observations['observation'],
                ),
                dim=1
            )
            heads.append(current_obs)
        heads = tuple(heads)
        return heads

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features = self.get_heads(features)
        output_heads = [self.value_net(feature.float()) for feature in features]
        output = th.cat(tuple(output_heads), dim=1)
        return output
    
    
    
class CustomBothMlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        observation_space = None,
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        self.num_dims = observation_space['desired_goal'].shape[0]
        assert self.num_dims <= 3, 'Too man physical dimensions'   
        self.num_preds = int(observation_space['achieved_goal'].shape[0] / self.num_dims)
        
        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi 
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        
    def get_heads(self, observations):
        # set_trace()
        # # print(observations)
        # self.num_dims = observations['desired_goal'].shape[1]
        # assert self.num_dims <= 3, 'Too man physical dimensions'   
        # self.num_preds = int(observations['achieved_goal'].shape[1] / self.num_dims)
        
        # Create heads
        heads = []
        pred_chunks = th.chunk(observations['achieved_goal'], self.num_preds, dim=1)
        for pred_num in range(self.num_preds):
            current_pred = pred_chunks[pred_num]
            try:
                other_preds = th.cat(
                    [pred_chunks[i] for i in range(self.num_preds) if i != pred_num],
                    axis=1,
                )
            except:
                other_preds = th.tensor([])
            current_obs = th.cat(
                (
                current_pred,
                other_preds.to(pred_chunks[0].device),
                observations['desired_goal'],
                observations['observation'],
                ),
                dim=1
            )
            heads.append(current_obs)
        heads = tuple(heads)
        return heads

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        features = self.get_heads(features)
        output_heads = [self.policy_net(feature.float()) for feature in features]
        output = th.cat(tuple(output_heads), dim=1)
        return output

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features = self.get_heads(features)
        output_heads = [self.value_net(feature.float()) for feature in features]
        output = th.cat(tuple(output_heads), dim=1)
        return output
    
    
    
    