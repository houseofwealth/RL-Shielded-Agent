# from custom_td3 import CustomTD3
from env import myEnv
# from custom_her import CustomHerReplayBuffer
# from stable_baselines3 import TD3, HerReplayBuffer, PPO
from stable_baselines3 import PPO
# from ray import tune

import numpy as np

USE_PARALLEL = False
AGENT_SIZE = 0.1                                    #SN:lower it to 0.1 to reduce slop so siheld calcs are accurate when shielding on, 1 otherwise
NUM_SAMPLES = 16                                
print('AGENT_SIZE', AGENT_SIZE)
STEP_SIZE = 1                                     #SN: was 0.01
""" 20% of worksapce size"""
LObs = -4
RObs = 4
CObs = -4
FObs = 4
BObs = 3
TObs = 8

def setFlagUSC(val):
    if USE_PARALLEL:
      return tune.grid_search(np.round(np.arange(0, 1.01, 0.1), 2))
    else:
      return val

def setFlag(val):
    if USE_PARALLEL:
      return tune.grid_search([True, False])
    else:
      return val

def setFlagExpNum(val):
    if USE_PARALLEL:
      return tune.grid_search([ns for ns in range(NUM_SAMPLES)])
    else:
      return val

DEFAULT_CONFIG = {
    'model_class': PPO,
    # 'model_class': CustomPPO,
    'env':myEnv,
    'env_config': {        
        'max_velocity': 10 / STEP_SIZE,
        'num_preds': 1,
        'predators_max_speeds': [1 / STEP_SIZE, 1 / STEP_SIZE, 1 / STEP_SIZE,], # min length is >= num_preds
        'num_fake_preds': 0,
        'correlate_fake_preds': 'none', # none, predator, prey
        'preds_collide_with_fake': False,
        'num_dims': 2,
        'AT_TARGET_RADIUS': 0, # 0.1 for shield, 1 for no shield
        'MAX_EPISODE_STEPS': 20,
        'MIN_DISTANCE_FROM_BASE': AGENT_SIZE,
        'MAX_DISTANCE_FROM_BASE': 2, #SN: in case agent size is 0
        'preds_collide': True,
        'collide_with_ground': False,
        'collide_with_herd': False,
        'flat_herd': True,
        'use_shield_chance': setFlagUSC(1), #tune.grid_search(np.round(np.arange(0, 1.01, 0.1), 2)),
        'mode': 'defense', # offense, defense
        'prey_spawn': 'above', # above, random

        #Dont overwrite these in experiment.py they won't be read from there!
        'STEP_SIZE':            STEP_SIZE,
        'max_acceleration':     10,                           #SN: was int(10E3),
        'workspace_size':       10,
        'GEOFENCING' :          setFlag(False), #was tune.grid_search([True, False]),
        'DOING_OBSTACLES':      setFlag(True), #tune.grid_search([True, False]),
        'DOING_BOUNDED':        setFlag(False), #tune.grid_search([True, False]),
        'STEPS_BOUND':          3,
        'LObs' :                LObs,
        'RObs' :                RObs,
        'CObs' :                CObs,
        'FObs' :                FObs,
        'BObs' :                BObs,
        'TObs' :                TObs
    },
    'callbacks_config': {
        'INITIAL_TIMEOUT': 100,
        'WINDOW_SIZE': 100,
        'MIN_TARGET_RADIUS': 0.1,
        'MIN_MEAN_REWARD': 1.1,
        'TARGET_RADIUS_COEFF': 0.95,
        'STEP_SIZE_COEFF': 1.0,
        'min_difference': 0.05,
        'min_performance': -1,
        'max_performance': 1,
    },
    'n_sampled_goal': 0,
    'max_episode_length': 100,
    'learning_steps': int(500E3),
    # 'replay_buffer_class': HerReplayBuffer, #HerReplayBuffer
    'lr': 1E-4,
    'use_lr_schedule': True,
    'lr_coeff': 1.0,
    'lr_min_power': 3.0,
    'lr_max_power': 3.0,
    'lr_schedule': None,
    'pretrain': {
        'use_pretrain': False,
        'pretrain_type': 'from_file', # 'spring', 'from_file'
        'pretrain_steps': 1000,
        'pretrain_model': 'max_4886045205',
    },
    'action_noise':{
        'use_action_noise': False,
        'mu': 0.0,
        'exploration': 1.0,
    },
    'neural_network': {
        'num_layers': 2,
        'nodes_per_layer': 64,
    },
    'learning_starts': int(10E3),
    'buffer_size': int(30E3),
    'batch_size': 100,
    'gamma': 0.99,
    'train_freq': (1, 'episode'),
    'gradient_steps': -1,
    'policy_delay': 10,
    'target_policy_noise': 0.2,
    'target_noise_clip': 0.5,
    'seed': None,
    'tau': 0.001,
    'device': 'auto',
    'L2_param': 0.0005,
    'eps':1e-5,
    'use_shield': True,
    'shield_type': 'z3_shield',
    'num_shield_chances': 100,
    'shield_angle': 10,
    'experiment_number': setFlagExpNum(0), #tune.grid_search([ns for ns in range(NUM_SAMPLES)]),
    'ent_coef': 0.00,
    'stop_early': False,
    'use_obstacle': True,
    'obstacle_bounds': (LObs, RObs, BObs, TObs) #was (-4, 4, 3, 8),
}
