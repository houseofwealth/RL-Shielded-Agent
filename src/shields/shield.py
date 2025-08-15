from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import torch as th
from ipdb import set_trace


class Shield(ABC):
    '''
        Base class for shields
        Inputs:
            env (gym.Env) - the environment that we're using
            policy(sb3.policy) - the policy from which we derive the action
            num_chances (int) - the number of times to draw from the policy
    '''

    def __init__(self, env, policy, num_chances):
        self.env = env
        self.policy = policy
        self.num_chances = num_chances

        if self.env.num_preds > 1:
            raise NotImplementedError('We have only implemented a shield for 1 wolf!')
    '''
        Inputs:
            obs (th.tensor) - the observation
    '''
    # @abstractmethod
    # def get_action(self, obs):
    #     pass
    #     actions, values, log_probs = (None, None, None)
    #     return actions, values, log_probs
    
    @abstractmethod
    def evaluate_actions(self, actions, obs, single_obs):
        pass
        actions, valid_actions = (None, None)
        return actions, valid_actions
    
    def get_uniform_actions(self):
        actions = self.env.action_space.sample()
        return actions
    
    def repeat_obs(self, obs):
        if isinstance(obs, dict):
            for key, value in obs.items():
                obs[key] = value.repeat(self.num_chances, 1)
        else:
            raise NotImplementedError('This has only been implemented for dict observation spaces!')
        return obs  
    
    def get_positions(self, actions, num_actions=1, return_velocity=False):
        step_size = self.env.STEP_SIZE
        accelerations = self.env.action_to_acceleration(actions)
        temp_pred = deepcopy(self.env.predators[0])
        for _ in range(num_actions):
            positions, velocities = temp_pred.try_actions(accelerations, step_size)

        if return_velocity:
            return positions, velocities
        return positions
    
    def sample_policy(self, obs, single_obs):
        actions, values, log_probs = self.policy(obs)
        actions = actions.cpu().numpy()
        valid_actions = self.evaluate_actions(actions, obs, single_obs)
        return actions, valid_actions
    
    def get_policy_action(self, obs, single_obs):

        # Draw actions
        actions, valid_actions = self.sample_policy(obs, single_obs)               

        # Choose an action
        if len(valid_actions) > 0:
            chosen_action = valid_actions[0]
            actions = actions[chosen_action, :]
            need_random = False

            values, log_probs, _ = self.policy.evaluate_actions(
                single_obs,
                th.tensor(actions.reshape(1, self.env.num_dims)).to(self.device)
            )
            # print('values', values)
        else:
            chosen_action = 0
            need_random = True
            #values = None
            #log_probs = None
            actions = actions[chosen_action, :]
            values, log_probs, _ = self.policy.evaluate_actions(
                single_obs,
                th.tensor(actions.reshape(1, self.env.num_dims)).to(self.device)
            )


        # Determine the typf of shielding that we are applying
        if chosen_action == 0:
            policy_shield = False
        else:
            policy_shield = True

        return actions, values, log_probs, need_random, policy_shield
    
    def sample_uniformly(self, obs, single_obs):

        # Draw random action
        low = self.env.action_space.low
        high = self.env.action_space.high
        actions = np.random.uniform(
            low, high, size=(self.num_chances, self.env.num_dims))
        valid_actions = self.evaluate_actions(actions, obs, single_obs)  

        return actions, valid_actions
    
    def get_random_actions(self, obs, single_obs):

        # Get the actions
        actions, valid_actions = self.sample_uniformly(obs, single_obs)
           
        # Choose an action
        if len(valid_actions) > 0:
            chosen_action = valid_actions[0]
            actions = actions[chosen_action, :]
            random_shield = True
        else:
            actions = actions[0, :]
            random_shield = False

        values, log_probs, _ = self.policy.evaluate_actions(
                single_obs,
                th.tensor(actions.reshape(1, self.env.num_dims)).to(self.device)
            )

        return actions, values, log_probs, random_shield 
    
    def get_action(self, obs):

        # Copy action
        single_obs = deepcopy(obs)
        obs = self.repeat_obs(obs)

        (
            policy_actions,
            policy_values,
            policy_log_probs,
            need_random,
            policy_shield
        ) = self.get_policy_action(
            deepcopy(obs), deepcopy(single_obs)
        )
        #old_actions = deepcopy(actions)
        #old_values = deepcopy(values.detach().numpy())
        #old_log_probs = deepcopy(log_probs.detach().numpy())
        if need_random:
            (random_actions,
             random_values,
             random_log_probs,
             random_shield) = self.get_random_actions(deepcopy(obs), deepcopy(single_obs))


        if policy_shield:
            shield_type = 'policy'
        elif (not policy_shield) and (not need_random):
            shield_type = 'none'
        elif random_shield:
            shield_type = 'random'
        elif not random_shield:
            shield_type = 'fail'
           
        if shield_type != 'random':
            actions = policy_actions
            values = policy_values
            log_probs = policy_log_probs
        else:
            actions = random_actions
            values = random_values
            log_probs = random_log_probs
       
        actions = np.asarray([actions])
        assert actions.size == self.env.num_dims, 'action is wrong size!'

        '''
            print(deepcopy(values))
            print(deepcopy(old_values))
            set_trace()
            values = old_values
            actions = old_actions
            log_probs = old_log_probs
            print(deepcopy(values))
            set_trace()
        '''
        return actions, values, log_probs, shield_type
