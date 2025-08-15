from tbparse import SummaryReader
from os.path import join
from ipdb import set_trace
import matplotlib.pyplot as plt
import numpy as np

class Processor():

    def __init__(self):
        self.figure_name_1 = join('.', 'pred_1_2.png')
        self.figure_name_2 = join('.', 'pred_1_2.pdf')

        self.base_folder = join('..', 'tune')
        self.pred_1_folder = join(self.base_folder, 'ppo')
        self.pred_2_folder = join(self.base_folder, 'ppo-2-preds')
        self.load_data()
        self.plot_data()

    def load_data(self):

        # Get the part of the data that we care about
        def get_sub_data(data_in):
            data_out = data_in.loc[data_in['tag'] == 'rollout/ep_rew_mean', :]
            _, dir_idxs = np.unique(data_out['dir_name'], return_inverse=True)
            data_out['dir_name'] = dir_idxs
            return data_out
        
        # Calculate the summary statistics
        def get_stats(data_in):
            timesteps = np.unique(data_in.step)
            num_timesteps = len(timesteps)

            medians = np.zeros(num_timesteps)
            lows = np.zeros(num_timesteps)
            highs = np.zeros(num_timesteps)

            for num, t in enumerate(timesteps):
                sub_data = data_in.loc[data_in['step'] == t]
                # medians[num] = np.median(sub_data.value)
                lows[num], medians[num], highs[num] = np.percentile(sub_data.value, [25, 50, 75])
            
            summary_data = {
                't': timesteps,
                'low': lows,
                'median': medians,
                'high': highs,
            }

            return summary_data


        self.p1_data = SummaryReader(self.pred_1_folder, extra_columns={'dir_name'}).scalars
        self.p1_data = get_sub_data(self.p1_data)
        self.p1_summary_data = get_stats(self.p1_data)
        self.p2_data = SummaryReader(self.pred_2_folder, extra_columns={'dir_name'}).scalars
        self.p2_data = get_sub_data(self.p2_data)
        self.p2_summary_data = get_stats(self.p2_data)

    def plot_data(self):
        plt.figure()

        def fill_between(data):
            plt.fill_between(data['t'], data['low'], data['high'], alpha=0.5)

        def plot(data):
            plt.plot(data['t'], data['median'], linewidth=2)

        for data in [self.p1_summary_data, self.p2_summary_data]:
            fill_between(data)

        for data in [self.p1_summary_data, self.p2_summary_data]:
            plot(data)

        plt.ylabel('Rewards')
        plt.xlabel('Training Steps')
        plt.title('Mutiple Predators')
        plt.legend(['1 Predator', '2 Predators'], loc='lower right')
        plt.savefig(self.figure_name_1)
        plt.savefig(self.figure_name_2)
        # set_trace()



if __name__ == '__main__':
    p = Processor()
