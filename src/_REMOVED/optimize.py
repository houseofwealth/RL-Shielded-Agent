from meher.custom_td3 import CustomTD3
from meher.env import meherEnv
from meher.custom_her import CustomHerReplayBuffer
import numpy as np
import ray
from ray import tune
from ray import air
from ray.tune.search.hyperopt import HyperOptSearch
from meher.rl import train_rl
from ipdb import set_trace
from meher.config import DEFAULT_CONFIG

# AGENT_SIZE = 1.0
DEFAULT_CONFIG['learning_steps'] = int(500E3)
DEFAULT_CONFIG['env_config']['preds_collide'] = True #tune.grid_search([True, False])
DEFAULT_CONFIG['env_config']['num_preds'] = 3 # tune.grid_search([1, 2, 3])
DEFAULT_CONFIG['env_config']['num_fake_preds'] = 0 #tune.grid_search([0, 1, 2])
DEFAULT_CONFIG['env_config']['correlate_fake_preds'] = 'none' # tune.grid_search(['none', 'prey', 'predator'])
DEFAULT_CONFIG['env_config']['preds_collide_with_fake'] = False #tune.grid_search([True, False])
# DEFAULT_CONFIG['L2_param'] = tune.grid_search([0.0005, 0])
# DEFAULT_CONFIG['eps'] = tune.grid_search([1e-5, 1e-8])



'''
    Optimize parameters via Bayesian hyperparameter optimization
'''
if __name__ == '__main__':
    ray.init(num_gpus=0)
    config = DEFAULT_CONFIG
    NUM_SAMPLES = 16
    # Hyperopt
    max_concurrent = 50
    algo = None
    # algo = HyperOptSearch()
    # algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=max_concurrent)

    tuner = tune.Tuner(
        train_rl,
        param_space=config,
        run_config=air.RunConfig(
            name='actor_3_preds_master_branch_edit_init_order',
            local_dir='./tune/',
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=False
            ),
        ),
        tune_config=tune.TuneConfig(
            num_samples=NUM_SAMPLES,
            search_alg=algo,
            metric='score',
            mode='max'
        ),
    )

    results = tuner.fit()
    print('Best result:')
    print(results.get_best_result("score", mode='max'))

    # set_trace()