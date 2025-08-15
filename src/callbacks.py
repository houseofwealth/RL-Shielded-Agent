from stable_baselines3.common.callbacks import BaseCallback
from ipdb import set_trace
import numpy as np

class CurriculumCallback(BaseCallback):
    '''
        Decrease the size of the agents when success is high
        Inputs:
            config (dict) - the meher.config dictionary
            verbose(int) - how verbose the printing should be
    '''

    def __init__(self, config, verbose=0):
        super().__init__(verbose=verbose)
        self.total_num_steps = 0
        self.rewards = []
        
        self.INITIAL_TIMEOUT = config['INITIAL_TIMEOUT']
        self.WINDOW_SIZE = config['WINDOW_SIZE']
        self.MIN_TARGET_RADIUS = config['MIN_TARGET_RADIUS']
        self.MIN_MEAN_REWARD = config['MIN_MEAN_REWARD']
        self.TARGET_RADIUS_COEFF = config['TARGET_RADIUS_COEFF']
        self.STEP_SIZE_COEFF = config['STEP_SIZE_COEFF']


        self.timeout = self.INITIAL_TIMEOUT
    def _on_step(self):
        self.total_num_steps += 1
        if self.locals['dones']:
            self.timeout -= 1
            self.rewards.append(np.sign(self.locals['rewards'][0]))
        
            if len(self.rewards) < self.WINDOW_SIZE:
                rewards = self.rewards
            else:
                rewards = self.rewards[-self.WINDOW_SIZE:]
            mean_reward = np.mean(rewards)

            if mean_reward > self.MIN_MEAN_REWARD and self.timeout <= 0.0:
                self.locals['env'].envs[0].env.AT_TARGET_RADIUS *= self.TARGET_RADIUS_COEFF
                self.locals['env'].envs[0].env.STEP_SIZE *= self.STEP_SIZE_COEFF
                if self.locals['env'].envs[0].env.AT_TARGET_RADIUS < self.MIN_TARGET_RADIUS:
                    self.locals['env'].envs[0].env.AT_TARGET_RADIUS = self.MIN_TARGET_RADIUS
                else:
                    print(self.locals['env'].envs[0].env.AT_TARGET_RADIUS)
                self.timeout = self.INITIAL_TIMEOUT

                self.locals['self'].replay_buffer.reset()

        return True
    

class CheckpointCallback(BaseCallback):
    '''
        Save a checkpoint of the model when the performance is high
        Inputs:
            config (dict) - the meher.config dictionary
            verbose(int) - how verbose the printing should be
    '''

    def __init__(self, config, verbose=0):
        super().__init__(verbose=verbose)
        self.total_num_steps = 0
        self.rewards = []    


        self.WINDOW_SIZE = config['callbacks_config']['WINDOW_SIZE']
        self.MIN_DIFFERENCE = config['callbacks_config']['min_difference']
        self.min_performance = config['callbacks_config']['min_performance']
        self.max_performance = config['callbacks_config']['max_performance']
        self.save_name_fragment = './models/max_' + str(config['experiment_id'])

    def create_checkpoint(self):
        self.locals['self'].save(self.save_name_fragment)

    def _on_step(self):
        self.total_num_steps += 1

        # Only save rewards on last step
        if self.locals['dones']:
            self.rewards.append(np.sign(self.locals['rewards'][0]))
        
            # Determine if we should create a checkpoint
            if len(self.rewards) > self.WINDOW_SIZE:
                rewards = self.rewards[-self.WINDOW_SIZE:]
                mean_reward = np.mean(rewards)

                # See if the mean reward is higher than the threshold
                above_threshold = (mean_reward >= self.min_performance + self.MIN_DIFFERENCE)

                # See if the reward is higher (when we are close to the max reward)
                better_reward = (
                        ((self.max_performance - self.MIN_DIFFERENCE) < mean_reward)
                        and (mean_reward > self.min_performance)
                    )

                if above_threshold or better_reward:
                    self.create_checkpoint()
                    self.min_performance = mean_reward
                    print(
                        'Mean reward = ' + str(mean_reward)
                        + ' at ' + str(self.total_num_steps)
                        + ' steps'
                    )

        return True
