# from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

import torch as th
import numpy as np
import os
from ipdb import set_trace

import json
from warnings import warn

import sys
sys.path.append('..')
from jsonize_configs import format_dict
# from shielded.pretrain import spring_pretrain, load_from_model_file
# from callbacks import CurriculumCallback, CheckpointCallback
from callbacks import CheckpointCallback
from config import DEFAULT_CONFIG
# from custom_ppo_nn import CustomPPOPolicy, CustomActorCriticPolicy_Actor, CustomActorCriticPolicy_Critic, CustomActorCriticPolicy_Both
from custom_ppo_nn import CustomActorCriticPolicy_Actor
# from shielded.simple_eval import run_eval
from custom_ppo import CustomPPO
from copy import deepcopy


'''
    Create a model from the configs
    This is provided as an interface to calculate many parameters from provided parameters
    Inputs:
        config (dict) - the configuation dict - an examples is in shielded.config
        env (gym.Env) - the RL environment - an example is in shielded.env
        save_location (str) - where the experiment results should be saved
    Results:
        model (stable-baselines3.model) - the RL model
'''
def create_model(config, env, save_location):
    # print('> create_model')
    # breakpoint()

    if config['lr_schedule'] is not None:
        lr = config['lr_schedule']
    else:
        lr = config['lr']

    # action_noise = create_action_noise(config)
    # policy_kwargs = dict(
    #     optimizer_class=th.optim.Adam,
    #     optimizer_kwargs=dict(weight_decay=config['L2_param'], eps=config['eps']),
    #     num_fake_preds = config['env_config']['num_fake_preds'],
    #     )

    model = config['model_class'](
        CustomActorCriticPolicy_Actor,
        env,
        use_shield=config['use_shield'],
        shield_type=config['shield_type'],
        shield_angle=config['shield_angle'],
        num_shield_chances=config['num_shield_chances'],
        verbose=0,
        tensorboard_log=save_location,
        gamma=config['gamma'],
        learning_rate=lr,
        seed=config['seed'],
        device=config['device'],
        stop_early=config['stop_early'],
        ent_coef=config['ent_coef'], #0, #0.01, 0.05, and 0.1 - lower works better
        target_kl=0.05              #SN: suggeted by Cale
    )
    return model

'''
    Create action noise from parameters
    Inputs:
        config (dict) - the configuration file
    Returns:
        noise (stable_baselines3.OrnsteinUhlenbeckActionNoise) - the noise object
def create_action_noise(config):
    noise_config = config['action_noise']
    env_config = config['env_config']
    num_actions = env_config['num_preds'] * env_config['num_dims']
    if not noise_config['use_action_noise']:
        noise = None
    else:
        noise = OrnsteinUhlenbeckActionNoise(
           mean=(noise_config['mu'] * np.ones(num_actions)),
           sigma=(noise_config['exploration'] * np.ones(num_actions)),
        )

    return noise
'''

'''
    Create the neural network kwargs dynamically from the config
    Inputs:
        config (dict) - the configuration file
    Returns:
        policy_kwargs (dict) - a dict, defined by SB3, that describes the neural network
def create_neural_network_args(config):
    net_config = config['neural_network']
    net_arch = [net_config['nodes_per_layer'] for layer in range(net_config['num_layers'])]
    policy_kwargs = dict(
        net_arch=dict(
            pi=net_arch,
            qf=net_arch,
        )
    )
    return policy_kwargs
'''

'''
    Train the RL agent
    Note that this function is called by optimize.py for parallel experiments
    Inputs:
        config (dict) - the configuration file
    Returns:
        score (dict) - the outcome of training, which is used by optimize.py
'''
def train_rl(config):
    # print('>train_rl')
    # Determine the learning rate
    if config['use_lr_schedule']:
        def learning_rate_schedule(percent):
            COEFF = config['lr_coeff']
            MAX = config['lr_max_power']
            MIN = config['lr_min_power']
            power = -1 * (MIN - ((MIN - MAX) * percent))
            learning_rate = COEFF * (10 ** power)
            return learning_rate

        config['lr_schedule'] = learning_rate_schedule
        if config['lr_min_power'] < config['lr_max_power']:
            config['lr_min_power'] = config['lr_max_power']
            warn('Min power < max power - using max_power')

    # Each experiment gets and id and a save location
    experiment_id = np.random.randint(9E9)
    #SN: saving training data so tensorboard can display
    save_location = '../experiments/results/' + str(experiment_id)
    config['experiment_id'] = experiment_id

    save_location = os.path.abspath(save_location)
    print('Saving to: ' + save_location)

    # Create the env and model
    env = config['env'](config['env_config'])
    model = create_model(config, env, save_location)

    # Save Configs
    save_path = os.path.abspath('../experiments/configs')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = save_path + '/' + str(experiment_id) + '.json'
    save_name = os.path.abspath(save_name)
    print('Saving config to: ' + save_name)
    config2 = format_dict(config)
    with open(save_name, 'w') as file:
        json.dump(format_dict(config2), file)

    # Pretrain
    if config['pretrain']['use_pretrain']:
        if config['pretrain']['pretrain_type'] == 'spring':
            model._setup_learn(config['learning_steps'])
            print('Pretraining ...')
            model = spring_pretrain(model, env, config)

            # Get a new model with no buffer values
            new_model = create_model(config, env, save_location)
            new_model.set_parameters(model.get_parameters())
            model = new_model
        elif config['pretrain']['pretrain_type'] == 'from_file':
            model = load_from_model_file(model, env, config)
        else:
            raise ValueError('Unknown pretraining type: ' + config['pretrain']['pretrain_type'] )
            
    print('Training ...')   
    
    # Train the model
    
    model.learn(
        config['learning_steps'],
        callback=[
            # CurriculumCallback(config['callbacks_config']),
            CheckpointCallback(config)
            ],
    )
    

    # Save
    save_name = '../experiments/models/' + str(experiment_id)
    print('Saving model to: ' + os.path.abspath(save_name))
    model.save(save_name)
    print('Done Training')

    # print('n_agent_fails', model.shield.n_agent_fails)
    # Create score for optimization
    last_rewards = model.rollout_buffer.rewards
    last_rewards = last_rewards[last_rewards != 0]
    if len(last_rewards) > 100:
        score = np.mean(last_rewards[-100:])
    else:
        score = np.mean(last_rewards)

    return {'episode_reward_mean': score}



''' Test '''
if __name__ == '__main__':
    
    config = deepcopy(DEFAULT_CONFIG)
    config['env_config']['num_dims'] = 2
    config['model_class'] = CustomPPO
    config['shield_angle'] = 10
    config['use_shield'] = True
    config['shield_type'] = 'z3_shield' #'angle'
    config['env_config']['use_shield_chance'] = 0.5
    config['env_config']['num_preds'] = 1
    config['learning_steps'] = int(500E3)
    config['env_config']['MAX_DISTANCE_FROM_BASE'] = 2
    config['env_config']['mode'] = 'offense' #offense, defense
    config['env_config']['prey_spawn'] = 'above' # above, random'
    config['env_config']['workspace_size'] = 200
    config['env_config']['MAX_EPISODE_STEPS'] = 100
    config['env_config']['collide_with_herd'] = False    
    config['num_shield_chances'] = 1000
    config['lr'] = 1E-4
    train_rl(config)
