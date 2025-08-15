
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
from jsonize_configs import generate_config
from warnings import warn
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
from copy import deepcopy

def dict_to_tensor(dict_in, device):
    for key, value in dict_in.items():
        dict_in[key] = th.tensor(value).reshape(1, -1).to(device)
    return dict_in

def dict_to_np(dict_in):
    for key, value in dict_in.items():
        if isinstance(value, th.Tensor):
            value = value.cpu.detach()
        dict_in[key] = np.asarray(value)
    return dict_in

'''
    Evaluate a saved model for multiple different target sizes
    It will be necessary to change experiment_name to the desired experiment
        and also change model_class to the model class that was used during
        training (e.g., PPO, DDPG)
'''
if __name__ == '__main__':

    # Experiment Name
    experiment_name = '8141951639' 
    # experiment_name = '6305875048'

    # Get the config
    folder_name = '/storage/sn/meherproj/meher/experiments/z3_shield/'
    # folder_name = '/home/dccrowd/projects/meher/meher/experiments/z3_shield/'
    # folder_name = '/home/dccrowd/projects/meher/test/'
    # config_folder_name = join(folder_name, 'configs') 
    config_folder_name = './configs/'
    config_name = folder_name + 'configs/' + experiment_name + '.json'
    default_config = DEFAULT_CONFIG

    if not exists(config_name):
        if '_' in experiment_name:
            experiment_name_2 = experiment_name.split('_')[1]
        else:
            experiment_name_2 = experiment_name
        config_name = join(config_folder_name, experiment_name_2 + '.json')
        if not exists(config_name):
            config_name = None
            config = default_config
            warn('Could not load config.  Using default instead.')
    
    if config_name is not None:
        with open(config_name, 'r') as file:
            config = json.load(file)
            config = generate_config(default_config, config)

    # Define env
    env = meherEnv(config['env_config'])
    env.AT_TARGET_RADIUS = 1.0              #SN: why is this redefining it?

    # Load the model
    print('Loading Model...')
    model_class = PPO
    base_folder = './'
    file_name = folder_name + 'models/' + experiment_name
    # file_name = './models/3018389892'
    combined_name = join(base_folder, file_name)
    print('Loading: ' + combined_name)
    model = model_class.load(combined_name, env, policy=CustomPPOPolicy)
    print('Done!')

    

    # Evaluation
    obs, _ = env.reset()
    rewards = []
    successes = []
    new_render = True
    while new_render:
        observations = []
        observations.append(dict_to_np(deepcopy(obs)))
        done = False
        while not done:
            action, _ = model.predict(dict_to_tensor(obs, model.device))
            obs, reward, done, truncate, info = env.step(action)
            observations.append(dict_to_np(deepcopy(obs)))

            # for key, value in observations[-1].items():
            #     if isinstance(value, th.Tensor):
            #         set_trace()

        rewards.append(reward)
        if reward > 0:
            successes.append(1)
        else:
            successes.append(0)
        # set_trace()
        
        obstacle_bounds = None
        if config['use_obstacle']:
            obstacle_bounds = config['obstacle_bounds']

        # Get the renderer
        prey_size = env.AT_TARGET_RADIUS / 2
        base_size = prey_size
        if True:
            renderer = Render_ct_3d(observations, num_dims=3, grid_size=10,
                use_set_function=False, gif_name=None, prey_radius=prey_size,
                base_radius=base_size, reward=reward,
                num_preds=config['env_config']['num_preds'],
                obstacle_bounds=obstacle_bounds,
            )

            cont = input('Continue (Y/n)?')
            if cont.lower() == 'n':
                new_render = False

        
        obs, _ = env.reset()

    print('-----------------------------')
    print(np.mean(rewards))
    print(np.mean(successes))
    print('-----------------------------')