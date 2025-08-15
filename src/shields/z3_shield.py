# from .shield import Shield
from .shield import *
from shields.builder.buildShield import *
# from shield import Shield
import numpy as np
# print(__package__)  #returns shields
from ipdb import set_trace
from tqdm import tqdm

MAX_ACTION_TRIES = 99

class Z3Shield(Shield):
    '''
        Do not take any step from which we cannot reach the prey.  These calculations
            do not account for inertia.
        Inputs:
            env (gym.Env) - the environment that we're using
            policy(sb3.policy) - the policy from which we derive the action
            num_chances (int) - the number of times that we should try to
                draw an action from the policy
            device - the device for torch
    '''
    def __init__(self, env, policy, num_chances, device):
        super().__init__(env, policy, num_chances)
        self.DOING_BOUNDED = env.DOING_BOUNDED
        self.DOING_OBSTACLES = env.DOING_OBSTACLES
        self.GEOFENCING = env.GEOFENCING
        self.device = device
        self.n_inv_not_satisfied = 0
        self.n_agent_fails = 0
        # buildShield()
        print('**WARNING, not building shield')
        # self.bound = 3

    def evaluate_actions(self, actions, obs, single_obs):
        # Get the positions that would result from the actions
        num_dims = self.env.num_dims
        current_position = single_obs['achieved_goal'].cpu().numpy()
        current_velocity = single_obs['observation'][:, :num_dims].cpu().numpy()
        # print('current_position', current_position,'current_velocity', current_velocity)
        current_state = np.concatenate((current_position, current_velocity), axis=1).squeeze().tolist()
        current_state_z3 = list(map(toZ3Type, current_state))

        # positions, velocities = self.get_positions(actions)
        valid_actions = []
        #SN:
        prey_pos = self.env.prey.position
        prey_pos = prey_pos.tolist()
        prey_vel = self.env.prey.velocity.tolist()
        prey_st = prey_pos + prey_vel

        if self.env.start_of_episode:
          self.env.start_of_episode = False
          se = solnExists(current_state, prey_st, self.env.STEPS_BOUND)
          if se: print('solnExists in', current_state, prey_st, self.env.STEPS_BOUND)
          else: print('***WARNING: no solution from', current_state, prey_st, self.env.STEPS_BOUND)

        if True: #moved -> env.py :solnExistsPy(current_state, prey_st, self.env.STEPS_BOUND - self.env.n_steps_to_bound):
            # print('current_state', current_state)
            for num, action in enumerate(actions): #SN: doesn't compile: enumerate(tqdm(actions)):
                agent_action = action.tolist()
                # print('agent_action', agent_action)
                # agent_action = list(map(toZ3Type, agent_action))
                #this simply tests if the action leads to an OK next state, not necc beyond that
                agent_acceleration = self.env.action_to_acceleration(agent_action)
                #SN: why does accerlation not requred to be converted to a Z3 type
                # print('agent_acceleration', agent_acceleration)
                # if quickOK(agent_acceleration, current_state, prey_st, self.env.STEPS_BOUND - self.env.n_steps_to_bound):
                
                res = OK(
                    agent_acceleration,
                    current_state,
                    prey_st,
                    self.env.STEPS_BOUND - self.env.n_steps_to_bound,
                    doing_bounded=self.DOING_BOUNDED,
                    doing_obstacles=self.DOING_OBSTACLES,
                    geofencing=self.GEOFENCING
                    )
                # if not res: print('quickOK failed on', agent_acceleration, current_state, prey_st, self.env.STEPS_BOUND - self.env.n_steps_to_bound)
                if res:
                    # if num>0 and res == True: print('accepted', agent_acceleration, current_state)
                    # if num>75: print('tried', num, ' actions before finding OK one')
                    if res != True and len(res)==self.env.num_dims: #this will only happen if DOING_BOUNDS
                        # print('res acc', res)
                        res = self.env.acceleration_to_action(res) #np.array(res)).tolist()
                        # print('res action', res)
                        actions[0] = res
                        valid_actions = np.asarray([0])   #SN: return index of succesful action
                    else:
                        # print('action', num, 'is ok')
                        valid_actions = np.asarray([num])
                    break
                # else: 
                    # print('agent_acceleration', agent_acceleration, "failed in", current_state)
                # Break if the shield took too long
                if (num == MAX_ACTION_TRIES):
                    # print("**Agent failed to find ok action") #set_trace()
                    self.n_agent_fails += 1
                    if self.n_agent_fails%100 == 0:
                        print('n_agent_fails', self.n_agent_fails)
        else: 
            self.n_inv_not_satisfied += 1
            breakpoint()
            if self.n_inv_not_satisfied%100 == 0: 
                print('variant/invariant not satisfied', self.n_inv_not_satisfied)
            # print('no soln exists in state', current_state)
        # print('valid_actions', valid_actions)
        return valid_actions    
    


