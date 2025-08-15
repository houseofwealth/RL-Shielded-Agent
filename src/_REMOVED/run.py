# from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray import air
from os.path import join

def run(train, num_samples=2, config=None, num_concurrent=None, num_nodes=None, experiment_name=None):
    if config is None:
        config = {}

    if num_nodes is None:
        num_nodes = 1

    if num_concurrent is None:
        num_concurrent = num_samples
    
    # trainable_with_gpu = tune.with_resources(train, {"gpu": (num_nodes/num_concurrent)})
    trainable_with_cpu = tune.with_resources(train, {"cpu": 1})
    # if experiment_name is None:
    #     experiment_name = get_name(config)
    # else:
    experiment_name = str(experiment_name)
    
    local_dir = './' + experiment_name

    tuner = tune.Tuner(
        #trainable_with_gpu,
        trainable_with_cpu,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric='episode_reward_mean',
            mode='max',
        ),
        run_config=air.RunConfig(
            name=experiment_name,
            local_dir=local_dir,
        ),
    )

    print('Saving results to: ' + local_dir +  ' | ' + experiment_name)
    results = tuner.fit()
    results_df = results.get_dataframe()
    results_loc = join(local_dir, experiment_name, 'results.csv')
    results_df.to_csv(results_loc)
