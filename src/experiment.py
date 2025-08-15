from rl import train_rl
from config import DEFAULT_CONFIG, USE_PARALLEL
# from ray import tune
from copy import deepcopy
from custom_ppo import CustomPPO
import numpy as np
import pdb

'''
TBD: agent size and min distance from base and agent size  confusion should prolly just recalc the BL shield to account for agentsize '''
'''NOTe: DONT CHANGE AGENT_SIZE, STEP_SIZE, Geofencing, ETC FLAGS IN HERE, ONLY IN CONFIG.PY'''
if __name__ == '__main__':

    config = deepcopy(DEFAULT_CONFIG)
    config['model_class'] = CustomPPO
    print('use_shield', config['use_shield'])                            #SN:


    # moved -> config: USE_PARALLEL = False
    if USE_PARALLEL:
        run(
            train_rl,
            num_samples=1,
            config=config,
            num_concurrent=None,
            num_nodes=None,
            experiment_name='shields_test')
    else:
        # moved: config['env_config']['use_shield_chance'] = 0.5
        # moved: config['experiment_number'] = 0
        train_rl(config)  
