from tbparse import SummaryReader
from ipdb import set_trace
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import pandas as pd

'''
    Constants
'''
PARSE_NAMES = True
YEAR = 2023
# -----------------------------------
# NEEDED_VARS = ['preds_collide_with_fake']
# FILE_NAME = 'prey_collide'
# FOLDER = '/home/dccrowd/projects/meher3/tune/reward_complexity_prey_corr'
# LABELS = {
#     ('False'): 'No Collisions',
#     ('True'): 'Collisions',
# }
# TITLE = 'Complicated Solution Space'
# -----------------------------------
# NEEDED_VARS = ['correlate_fake_preds']
# FILE_NAME = 'noise'
# FOLDER = '/share/mltrapp/correlation'
# LABELS = {
#     ('none'): 'No Correlation',
#     ('prey'): 'Correlated w/ Prey',
#     ('predator'): 'Correlated w/ Predator',
# }
# TITLE = 'Noisy Observations'
# CONDITIONS_TO_EXCLUDE = []
# -----------------------------------
NEEDED_VARS = ['num_fake_preds']
FILE_NAME = 'uncorr'
FOLDER = '/home/dccrowd/projects/meher3/tune/state_size'
LABELS = {
    ('0/train'): '0 Fake Predators',
    ('1/train'): '1 Fake Predators',
    ('2/train'): '2 Fake Predators',
}
TITLE = 'Large State Space'
CONDITIONS_TO_EXCLUDE = []
# -----------------------------------

YLABEL = 'Episode Return'
XLABEL = '10^3 Steps'


class Processor():
    '''
        Process the data in the folder specified by FOLDER and FILE_NAME
        Automatically parse into different experiments based on different names.
            Experiment names should be stored in NEEDED_VARS
        Stores pngs and pdfs in the local directory and in the data directory
        Replabels conditions for plotting according to LABELS
        Excludes conditions in CONDITIONS_TO_EXCLUDE
        Labels plot according to TITLE, XLABEL, and YLABEL
    '''
    def __init__(
            self,
            folder=FOLDER,
            year=YEAR,
            needed_vars=NEEDED_VARS,
            file_name=FILE_NAME,
            labels=LABELS,
            conditions_to_exclude=CONDITIONS_TO_EXCLUDE,
            title=TITLE,
            ylabel=YLABEL,
            xlabel=XLABEL,
    ):
        folders = next(os.walk(folder))[1]

        # Get the data
        
        conditions = []
        if PARSE_NAMES:
            data = SummaryReader(folder, extra_columns={'dir_name'}).scalars
            dir_names = data['dir_name'].values

            for var in needed_vars:
                val = []
                for dn in dir_names:
                    dn = dn.split(var + '=')[1]
                    if ',' in dn:
                        dn = dn.split(',')[0]
                    if '_' in dn:
                        dn = dn.split('_')[0]
                    val.append(dn)
                data[var] = val

            data = data.loc[data.tag == 'rollout/ep_rew_mean']
            all_vars = copy(needed_vars)
            all_vars.append('step')
            result = data.groupby(all_vars).quantile([0.25, 0.5, 0.75])

            result = result.reset_index()

            # Get the stats
            med = result.loc[result['level_' + str(len(needed_vars) + 1)] == 0.5]
            low = result.loc[result['level_' + str(len(needed_vars) + 1)] == 0.25]
            high = result.loc[result['level_' + str(len(needed_vars) + 1)] == 0.75]
            all_vals = med.copy().rename({'value': 'med'}, axis=1)
            all_vals.loc[:, 'low'] = low.value.values
            all_vals.loc[:, 'high'] = high.value.values
            # set_trace()

            # Summarize the max reward and learning time

            grouped_med = med.groupby(needed_vars)
            maxes = grouped_med.value.max()
            
            steps = deepcopy(maxes)
            for group_name, group in grouped_med:
                steps[group_name] = group.loc[group.value == maxes[group_name], 'step'].min()

            maxes = maxes.to_frame()
            steps = steps.to_frame()
            maxes = maxes.rename({'value': 'max_reward'}, axis=1)
            steps = steps.rename({'value': 'time_to_learn'}, axis=1)
            maxes = maxes.assign(time_to_learn=steps.time_to_learn)
            for folder in ['./', FOLDER]:
                maxes.to_csv(os.path.join(folder, file_name + '.csv'))


            # Create the plot
            fig = plt.figure(figsize=(4, 4))
            ax = fig.subplots(1, 1)
            
            # Plot the data
            groupd_data = all_vals.groupby(needed_vars)            
            for key in list(labels.keys()):
                if key in conditions_to_exclude:
                    continue
                group = groupd_data.get_group(key)
                plt.fill_between(group.step / 1000, group.low, group.high, alpha=0.3)

            for key in list(labels.keys()):
                if key in conditions_to_exclude:
                    continue
                group = groupd_data.get_group(key)
                label = labels[key]
                plt.plot(group.step / 1000, group.med, linewidth=1, label=label)

            # plt.legend(loc='best', bbox_to_anchor=(0.35, 0.15))
            plt.legend()
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tight_layout()

        else:
            for folder in folders:
                # Parse the condition
                full_folder = os.path.join(self.FOLDER, folder)
                num_preds = int(folder.split('=')[1].split(',')[0])
                collision = (folder.split('=')[-1] == 'True')

                # Load the data
                data = SummaryReader(full_folder).scalars
                data = data.loc[data.tag == 'rollout/ep_rew_mean']
                result = data.value.groupby(data.step).quantile([0.25, 0.5, 0.75])
                result = result.reset_index()

                med = result.loc[result.level_1 == 0.5]
                low = result.loc[result.level_1 == 0.25]
                high = result.loc[result.level_1 == 0.75]

                condition = {
                    'num_preds': num_preds,
                    'collision': collision,
                    'low': low,
                    'med': med,
                    'high': high,
                }

                conditions.append(condition)

            # Plot the data
            plt.figure()
            for condition in conditions:
                if (condition['num_preds'] == 1) and (condition['collision']):
                    continue
                plt.fill_between(condition['low'].step, condition['low'].value, condition['high'].value, alpha=0.3, label='_nolegend_')

            # legend = []
            for condition in conditions:
                if (condition['num_preds'] == 1) and (condition['collision']):
                    continue
                label = str(condition['num_preds']) + ' predators, '
                # legend.append(label)
                if condition['num_preds'] > 1:
                    if condition['collision']:
                        label += 'collision'
                    else:
                        label += 'no collision'
                plt.plot(condition['med'].step, condition['med'].value, label=label)

        # plt.legend()

        png_name = os.path.join(folder, file_name + '.png')
        pdf_name = os.path.join(folder, file_name + '.pdf')
        png_name_2 = os.path.join('./', file_name + '.png')
        pdf_name_2 = os.path.join('./', file_name + '.pdf')
        print('Saving to: ' + png_name)
        for file_name in [png_name, png_name_2, pdf_name, pdf_name_2]:
            plt.savefig(file_name)
        # plt.savefig(png_name)
        # plt.savefig(pdf_name)


if __name__ == '__main__':
    p = Processor()