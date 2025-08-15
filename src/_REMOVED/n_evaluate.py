
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from meher.custom_td3 import CustomTD3
from meher.env import meherEnv
import numpy as np
from meher.render import Render_ct_3d
from os.path import join, exists
from ipdb import set_trace
from meher.config import DEFAULT_CONFIG
from jsonize_configs import generate_config
from tqdm import tqdm
import json
from warnings import warn

import matplotlib.pyplot as plt

'''
    Evaluate a saved model for multiple different target sizes
    It will be necessary to change experiment_name to the desired experiment
        and also change model_class to the model class that was used during
        training (e.g., PPO, DDPG)
'''
if __name__ == '__main__':


    # Experiment Name
    at_target_radii = np.power(0.95, np.arange(20))
    experiment_name = 'max_1224448900'

    # Get the config
    config_name = './configs/' + experiment_name + '.json'
    default_config = DEFAULT_CONFIG

    if not exists(config_name):
        experiment_name_2 = experiment_name.split('_')[1]
        config_name = './configs/' + experiment_name_2 + '.json'
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
    env.AT_TARGET_RADIUS = 0.65

    # Load the model
    print('Loading Model...')
    model_class = CustomTD3
    base_folder = './'
    file_name = './models/' + experiment_name
    combined_name = join(base_folder, file_name)
    print('Loading: ' + combined_name)
    model = model_class.load(model_class, combined_name, env=env)
    print('Done!')

    performances = []
    for radius in tqdm(at_target_radii):
        env.AT_TARGET_RADIUS = radius    

        # Evaluation
        obs = env.reset()
        rewards = []
        successes = []
        for episode in range(100):
            observations = []
            observations.append(obs)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                observations.append(obs)

            rewards.append(reward)
            if reward > 0:
                successes.append(1)
            else:
                successes.append(0)

            # Get the renderer
            prey_size = env.AT_TARGET_RADIUS / 2
            base_size = prey_size
            
            obs = env.reset()

        performances.append(np.mean(rewards))

    successes = (np.asarray(performances) + 1) / 2
    print('-----------------------------')
    print(at_target_radii)
    print(performances)
    print(successes)

    workspace_volume = 2 * (2 * env.workspace_size) + 1 * (env.workspace_size)
    volumes = (4 / 3) * np.pi * (at_target_radii) ** 3
    relative_volumes = volumes / workspace_volume

    
    plt.figure()
    plt.plot(relative_volumes, successes * 100)
    plt.xlabel('Relative Target Volume (a.u.)')
    plt.ylabel('% Success')
    plt.ylim([0, 100])
    plt.savefig('successes.png')
