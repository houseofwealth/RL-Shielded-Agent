# from meher.run import run
from meher.rl import train_rl
from meher.run import run
from meher.config import DEFAULT_CONFIG
from ray import tune
from copy import deepcopy
from meher.custom_ppo import CustomPPO
import numpy as np
import pdb

if __name__ == '__main__':

    config = deepcopy(DEFAULT_CONFIG)
    config['model_class'] = CustomPPO   

    run(
        train_rl,
        num_samples=1,
        config=config,
        num_concurrent=None,
        num_nodes=None,
        experiment_name='shields_fixed_end_early_policy_on_fail')

