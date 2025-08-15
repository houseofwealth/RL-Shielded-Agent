import numpy as np
from scipy import interpolate
from scipy.stats import iqr
from tbparse import SummaryReader


def parse_tb_data(path_to_folder):
    reader = SummaryReader(path_to_folder, pivot=True, extra_columns={'dir_name'})
    df = reader.scalars
    exp_names = list(set(df['dir_name']))
    df_dict = {}
    for e in exp_names:
        df_dict[e]=df[df['dir_name'] == e]
    return df_dict


def extend_array(experiment_df, num_steps):
    exp_arr = experiment_df.to_numpy()
    a = np.array([[0, exp_arr[0,1]]])
    b = np.array([[num_steps, exp_arr[-1, 1]]])
    return np.concatenate((a, exp_arr, b))

def extrapolate_rewards(experiment_df, num_steps):
    extended_arr = extend_array(experiment_df, num_steps)
    global_steps = np.arange(num_steps)
    func = interpolate.interp1d(extended_arr[:,0], extended_arr[:,1])
    return func(global_steps)

def extrapolate_all_experiments(experiment_dict, num_steps):
    all_experiment_rewards = np.zeros((len(experiment_dict), num_steps))
    # exper_dict[experiments[0]][['step','rollout/ep_rew_mean']]
    for i, df in enumerate(experiment_dict.values()):
        all_experiment_rewards[i, :] = extrapolate_rewards(df[['step','rollout/ep_rew_mean']], num_steps)
    
    return all_experiment_rewards
   
   
def tb_to_stats(path_to_tb_repeats, num_steps):
    df_dict = parse_tb_data(path_to_tb_repeats)
    all_repeat_rew = extrapolate_all_experiments(df_dict, num_steps)
    top, bottom = iqr(all_repeat_rew, axis=0, rng=(50,75)), iqr(all_repeat_rew, axis=0, rng=(25,50))
    mean, std = np.median(all_repeat_rew, axis=0), np.std(all_repeat_rew, axis=0)
    return (mean, std, top, bottom, all_repeat_rew)
