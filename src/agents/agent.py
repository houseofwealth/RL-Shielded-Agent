from abc import ABC
import numpy as np
from ipdb import set_trace

# DOING_OBSTACLES = True
# #Obstacle is a N dim block with LObs,RObs in all dims but the last where its BObs and TObs
# # LObs = -8; RObs=8; BObs=5; TObs=15  #TBD: make it config setting and import into MR
# LObs = -4; RObs=4; BObs=3; TObs=8            #is 16x10
# print('DOING_OBSTACLES', DOING_OBSTACLES)
# if DOING_OBSTACLES: print('LObs RObs BObs TObs', LObs, RObs, BObs, TObs)

class BaseAgent(ABC):
    def __init__(self, num_dims, max_velocity, workspace_size, doing_obstacles=False, doing_geofence=False,
                 LObs=0, RObs=0, TObs=0, BObs=0):
        self._max_velocity = max_velocity
        self._workspace_size = workspace_size
        self.num_dims = num_dims
        self._position = np.zeros(num_dims)
        self._velocity = np.zeros(num_dims)
        self._is_live = True
        self.hit_geofence = False
        # self.n_hit_geofence_or_obs = 0
        self.doing_obstacles = doing_obstacles
        self.doing_geofence = doing_geofence
        self.LObs = LObs
        self.RObs = RObs
        self.TObs = TObs
        self.BObs = BObs


    @property
    def max_velocity(self):
        return self._max_velocity

    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, new_position):
        self._position = new_position
    
    @property
    def velocity(self):
        return self._velocity
    
    @velocity.setter
    def velocity(self, new_velocity):
        self._velocity = new_velocity
    
    @property
    def is_live(self):
        return self._is_live
    
    @is_live.setter
    def is_live(self, new_value:bool):
        self._is_live = new_value

    """     #SN: why all this trouble for a simple 1 col vector?
    def clip_entity_position(self):
        position = self._position.reshape(-1, self.num_dims)
        #Dim N values have to stay >= 0 (above ground) along z axis
        z_clip = np.clip(position[:, -1], a_min=0, a_max=self._workspace_size).reshape(-1, 1)
        dims_clipped = np.clip(position[:, :-1], a_min=-self._workspace_size, a_max=self._workspace_size,
        )
        self._position = np.append(dims_clipped, z_clip, axis=1).squeeze() 
    """
    '''clips to the workspace, in addition sets a flag if the workspace boundary was hit (needed for geofencing)'''
    def clip_entity_position(self, isreset=False):
        position = self._position.reshape(-1, self.num_dims)

        # Dim N values have to stay >= 0 (above ground) along z axis
        z_vals = position[:, -1]
        z_violation = (z_vals < 0) | (z_vals > self._workspace_size)
        z_clip = np.clip(z_vals, a_min=0, a_max=self._workspace_size).reshape(-1, 1)

        dims_vals = position[:, :-1]
        dims_violation = (dims_vals < -self._workspace_size) | (dims_vals > self._workspace_size)
        dims_clipped = np.clip(dims_vals, a_min=-self._workspace_size, a_max=self._workspace_size)

        # Increment counter if any violations occurred
        if np.any(z_violation) or np.any(dims_violation):
            # if not isreset: breakpoint()
            # self.n_hit_geofence_or_obs += 1
            self.hit_geofence = True

        self._position = np.append(dims_clipped, z_clip, axis=1).squeeze()


    #SN: true if it hit geofence or an obstacle. TBD: fix for 3D!!
    def hitGeoFenceOrObs(self): 
        last_element = self._position[self.num_dims - 1]
        #checks if any of the elements of the position falls in the obstacle area
        if self.doing_obstacles: 
            hit_obstacle = \
              any(self.LObs <= i and i <= self.RObs for i in self._position[0:(self.num_dims - 1)]) and \
              self.BObs <= last_element and\
              last_element <= self.TObs
        else: 
            hit_obstacle = False
        # if hit_obstacle: 
        #     self.n_hit_geofence_or_obs += 1 
        #     # breakpoint()
        #     if self.n_hit_geofence_or_obs%100 == 0: print('n_hit_geofence_or_obs =', self.n_hit_geofence_or_obs)

        #do the first N-1 dims then the final Nth dim which has different range starting @ 0
        # hit_geofence =  any( i < -self._workspace_size or i > self._workspace_size \
        #                      for i in self._position[0:(self.num_dims - 1)]) or \
        #                 last_element < 0 or \
        #                 last_element > self._workspace_size
        if (self.doing_geofence and self.hit_geofence) or\
           (self.doing_obstacles and hit_obstacle):
        # if (self.doing_obstacles and hit_obstacle):            #print('***new pos out of bounds', self._position)
            # print('hit geofence at', self._position)
            return True               

    def apply_acceleration(self, acceleration, step_size):
        # Don't update dead agents
        if not self.is_live:
            # breakpoint()
            return
        
        # Cap max_velocity #SN: what's this for? A: representes terminal velocity TBD: when reinstated ensure shield calc does the same thing
        # speed = np.linalg.norm(self._velocity)
        # if speed > self.max_velocity:
        #     self._velocity = (
        #         self.max_velocity
        #         * self._velocity
        #         / speed)
        old_position = self._position.copy()
        # print('apply_acceleration: old_position', old_position, 'acceleration', acceleration)
        # if old_position.shape[0] > 1:
        #     set_trace()
        #SN: THIS IS INCORRECT!!
        # self._position = self._position + (self._velocity * step_size)  
        delta_v = acceleration * step_size
        self._position = self._position + (self._velocity * step_size) + (delta_v/2 * step_size)
        self._velocity = self._velocity + (acceleration * step_size)

        # compute the true velocity after clipping positions 
        # self._velocity = self._position - old_position  #SN: THIS IS INCORRECT TOO

    def try_actions(self, accelerations, step_size):
        num_trials = accelerations.shape[0]
        if self._velocity.ndim == 1:
            self._velocity = np.tile(self._velocity, (num_trials, 1))
            self._position = np.tile(self._position, (num_trials, 1))
        self.apply_acceleration(accelerations, step_size)
        return self._position, self._velocity