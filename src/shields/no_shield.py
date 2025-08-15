from .shield import Shield
import numpy as np

class NoShield(Shield):
    '''
        Take first action (no shield)
    '''

    def get_action(self, obs):
        actions, values, log_probs = self.policy(obs)
        actions = actions.cpu().numpy()
        shield_type = 'none'
        return actions, values, log_probs, shield_type
    
    def evaluate_actions(self, actions, obs, single_obs):
        pass