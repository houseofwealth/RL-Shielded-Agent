from tbparse import SummaryReader
from os.path import join
from ipdb import set_trace
import matplotlib.pyplot as plt
import numpy as np

class Processor():

    def __init__(self):
        self.figure_name_1 = join('.', 'td3_1_2.png')
        self.figure_name_2 = join('.', 'td3_1_2.pdf')

        self.base_folder = join('..', 'tune')
        # self.folder_1 = join(self.base_folder, 'ppo')
        # self.folder_2 = join(self.base_folder, 'td3')
        # self.folder_3 = join(self.base_folder, 'td3-her')
        self.folders = ['ppo', 'td3', 'td3-her']

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
            timesteps = sorted(np.unique(data_in.step))
            num_timesteps = len(timesteps)

            MAX_NUM_TIMESTEPS = 101
            if num_timesteps > MAX_NUM_TIMESTEPS:
                timesteps = np.linspace(timesteps[0], timesteps[-1], MAX_NUM_TIMESTEPS)
                num_timesteps = MAX_NUM_TIMESTEPS

                new_timesteps = timesteps[np.digitize(data_in.step, timesteps) - 1]
                data_in['step'] = new_timesteps


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
        

        self.data = []
        self.summary_data = []

        for folder in self.folders:
            full_name = join(self.base_folder, folder)
            data = SummaryReader(full_name, extra_columns={'dir_name'}).scalars
            data = get_sub_data(data)
            summary_data = get_stats(data)
            self.data.append(data)
            self.summary_data.append(summary_data)

        # set_trace()

    def plot_data(self):
        # set_trace()
        plt.figure()

        def fill_between(data):
            plt.fill_between(data['t'], data['low'], data['high'], alpha=0.5)

        def plot(data):
            plt.plot(data['t'], data['median'], linewidth=2)

        for data in self.summary_data:
            fill_between(data)

        for data in self.summary_data:
            plot(data)

        plt.ylabel('Rewards')
        plt.xlabel('Training Steps')
        plt.title('On-Policy vs. Off-Policy')
        plt.legend(self.folders, loc='lower right')
        plt.savefig(self.figure_name_1)
        plt.savefig(self.figure_name_2)
        # set_trace()



if __name__ == '__main__':
    p = Processor()
