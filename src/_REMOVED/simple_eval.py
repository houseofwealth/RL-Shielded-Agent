from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, PPO
from meher.custom_td3 import CustomTD3
from meher.custom_ppo_nn import CustomPPOPolicy
from meher.env import meherEnv
import numpy as np
from meher.render import Render_ct_3d
from os.path import join, exists
from ipdb import set_trace
from meher.config import DEFAULT_CONFIG
import json
import torch

from warnings import warn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def obs_vals_to_tensor(obs):
    # set_trace()
    new_dict = dict()
    for k,v in obs.items():
        new_dict[k] = torch.from_numpy(obs[k]).unsqueeze(dim=-2).float().to(device)
    return new_dict

def run_eval(model, config):
    env = meherEnv(config['env_config'])
    env.AT_TARGET_RADIUS = 1.0
    

    # Evaluation
    obs = env.reset()
    rewards = []
    successes = []
    new_render = True
    while new_render:
        observations = []
        observations.append(obs)
        done = False
        while not done:
            action, _ = model.predict(obs_vals_to_tensor(obs), deterministic=True)
            obs, reward, done, info = env.step(action)
            observations.append(obs)

        rewards.append(reward)
        if reward > 0:
            successes.append(1)
        else:
            successes.append(0)

        
        obs = env.reset()
        return np.mean(rewards)

