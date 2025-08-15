import pandas as pd
from pandas.io.formats.style import Styler
import os
from ipdb import set_trace
from copy import copy
import numpy as np


''''
    Combine the CSV files produced by processor.py into a single Latex table
    The Latex table will be saved as ./latex_data.tex
'''

if __name__ == '__main__':

    NEEDED_VARS = ['num_preds', 'preds_collide']

    files = os.listdir('./')
    files = [file for file in files if '.csv' in file]
    condition_names = [' '.join(file.split('.')[0].split('_')).title() for file in files]
    condition_names[condition_names == 'Custom None'] = 'Centralized'

    for num, (file, cond_name) in enumerate(zip(files, condition_names)):
        if num == 0:
            data = pd.read_csv(file)
            data = data.assign(condition=cond_name)
        else:
            new_data = pd.read_csv(file)
            new_data = new_data.assign(condition=cond_name)
            data = pd.concat((data, new_data))
    data = data.reset_index()

    # Drop the preds==1, collide==true condition
    data = data.drop(data.loc[(data.num_preds == 1) & (data.preds_collide == True)].index)

    # Drop all but one of the preds==1, collide=False condition
    redundant_conditions = data.loc[(data.num_preds == 1) & (data.preds_collide == False)].index
    data = data.drop(redundant_conditions.values[1:])
    data = data.drop('index', axis=1)

    new_names = {
        'num_preds': '\# Predators',
        'preds_collide': 'Predators Collide',
        'max_reward': 'Max Reward',
        'time_to_learn': 'Steps to Learn',
        'condition': 'Condition',
    }
    data = data.rename(new_names, axis=1)

    needed_vars = [new_names[var] for var in NEEDED_VARS]
    needed_vars.append('Condition')
    data['Steps to Learn'] = data['Steps to Learn'].astype(int)
    
    data = data[['\# Predators', 'Predators Collide', 'Condition', 'Max Reward', 'Steps to Learn']]
    data = data.set_index(['\# Predators', 'Predators Collide', 'Condition'])
    data = data.sort_index(axis=0)
    
    styler = Styler(data).format({
        'Max Reward': '{:.2f}',
        'Max Reward': '{:.2f}',
    })
    latex = styler.to_latex(sparse_index=True, clines='all;data')

    with open('latex_data.tex', 'w') as latex_file:
        latex_file.write(latex)
    # set_trace()
