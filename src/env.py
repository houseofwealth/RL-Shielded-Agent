from gymnasium import spaces, Env
import numpy as np

from agents.predator import Predator
from agents.prey import Prey
from agents.agent import BaseAgent
from warnings import warn


class myEnv(Env):
    def __init__(self, config):

        # Load config - i think this allows entries in the config to be accessed as if they were fields of env
        assert isinstance(config, dict), 'Config should be a dict!'
        for key, value in config.items():
            self.__dict__[key] = value

        # Add constants
        self.MAX_PREY_VELOCITY = 0.5  / self.STEP_SIZE  #SN: was 1/STEP_SIZE...
        if self.max_velocity is None:
            self.max_velocity = self.MAX_PREY_VELOCITY
        self.GROUND_COLLISION_VELOCITY = -1.0        

        # Create agents
        if 'predators_max_speeds' in config:
            self.predators = []
            for pred_index in range(self.num_preds):
                self.predators.append(Predator(num_dims=self.num_dims, 
                                               max_velocity=config['predators_max_speeds'][pred_index],
                                               workspace_size=self.workspace_size,
                                               doing_obstacles=self.DOING_OBSTACLES,
                                               doing_geofence = self.GEOFENCING,
                                               LObs=self.LObs, RObs=self.RObs, TObs=self.TObs, BObs=self.BObs))
        else:
            self.predators = [Predator(num_dims=self.num_dims, 
                                       max_velocity=self.max_velocity, 
                                       workspace_size=self.workspace_size,
                                       doing_obstacles=config.DOING_OBSTACLES,
                                       LObs=config.LObs, RObs=config.RObs, TObs=config.TObs, BObs=config.BObs)
                                                                    for i in range(self.num_preds)]
            
        self.fake_predators = [FakePredator(num_dims=self.num_dims, workspace_size=self.workspace_size) #SN: ToFix
                               for i in range(self.num_fake_preds)]
        self.prey = Prey(num_dims=self.num_dims, max_velocity=self.MAX_PREY_VELOCITY, workspace_size=self.workspace_size)
       
        self.total_timesteps = 0
        # Initialize
        self.define_spaces()
        self.reset()
        self.hitGeoFenceOrObs = False
        self.n_hit_geofence = 0
        self.n_prey_caught = 0
        self.n_tot_rew = 0
        self.n_bound_exceeded = 0
        # print('**WARNING, prey may have 0 velocity')

    def define_spaces(self):
        total_num_preds = self.num_preds + self.num_fake_preds
        self.action_space = spaces.Box(-1, 1, shape=(self.num_dims * self.num_preds,))

        # Observation: pred velocities + prey velocity
        observation = spaces.Box(
            -self.workspace_size , self.workspace_size , shape=(total_num_preds * self.num_dims + self.num_dims,)
        )

        # Achieved_goal: pred positions
        achieved_goal = spaces.Box(
            -self.workspace_size , self.workspace_size , shape=(total_num_preds * self.num_dims,))
        
        # Desired_goal: prey position
        desired_goal = spaces.Box(
            -self.workspace_size , self.workspace_size , shape=(self.num_dims,)
        )
        
        self.observation_space = spaces.Dict({
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
        })

    def compute_reward(self, achieved_goal, desired_goal, info):    #SN: unused?
        rewards = []
        for ag, dg, i in zip(achieved_goal, desired_goal, info):
            reward = self._compute_reward(ag, dg, i)
            rewards.append(reward)
        return rewards

    def _compute_reward(self, achieved_goal, desired_goal, info):
        if info['done']:
            if self.at_target(achieved_goal, desired_goal):
                # print('achieved_goal, desired_goal', achieved_goal, desired_goal)
                reward = 1
            # elif info['task_failed']:   #SN: typically means its collided w/ sth
            #     reward = -0.5
            else:
                ##SN: was -1, -0.1 converges faster but doesn't help with learning rate 
                # print('reward -1')
                reward = -1 
                # breakpoint()  
        else: #episode not finished, no reward yet
            reward = 0
        return reward

    def reset(self, options=None, seed=0):
        # TODO  seeding doesn't currently actually seed anything. 
        
        self.end_now = False
        
        self.num_episode_steps = 0
        self.use_shield = np.random.choice([True, False], p=[self.use_shield_chance, 1 - self.use_shield_chance])

        self.n_steps_to_bound = 0   #SN:
        # Base position
        self.base_position = np.zeros(self.num_dims)

        # Prey position
        self.prey.is_live = True
        if self.prey_spawn == 'above':
            if self.num_dims == 3:
                self.prey.position = np.asarray([0.0, 0.0, 10.0])
            elif self.num_dims == 2:
                self.prey.position = np.asarray([0.0, 10.0])
        elif self.prey_spawn == 'random':   #SN: couldnt this put prey outside geofence?
            self.prey.position = np.random.uniform(
                -self.workspace_size,
                self.workspace_size,
                size=(self.num_dims))
            self.prey.clip_entity_position(True)                         #SN: TBD needs better soln
            # print('reset: prey.position', self.prey.position)
            if self.num_dims == 3:
                if self.workspace_size >= 10:
                    self.prey.position[-1] = 10
                else:
                    self.prey.position[-1] = self.workspace_size
                    warn('setting prey position at workspace limit')
        else:
            raise ValueError('Unknown Prey spawner: ' + str(self.prey_spawn))
        
        # Create offsets for fake predators.
        # We will allow these to overlap for now
        # We will also allow for (z > workspace size) so that the solution is not trivial
        if self.num_fake_preds > 0:
            for fake_predator in self.fake_predators:
                direction = np.random.uniform(
                    -1, 1, size=self.num_dims
                )
                if self.num_dims == 3:
                    direction[self.num_dims-1] = 0
                    
                direction = direction / np.linalg.norm(direction)
                distance = np.random.uniform(
                    self.MIN_DISTANCE_FROM_BASE, self.MAX_DISTANCE_FROM_BASE, size=1,
                ) # self.MAX_DISTANCE_FROM_BASE 
                fake_predator.offset = distance * direction

        # Predator Locations & Velocities
        has_collision = True
        while has_collision:

            distance_from_base = np.random.uniform(
                self.MIN_DISTANCE_FROM_BASE,
                self.MAX_DISTANCE_FROM_BASE,
                size=self.num_preds,
            )
            direction_to_base = np.random.uniform(
                -1, 1, size=(self.num_dims * self.num_preds)
            )        
            
            # Always start predators on bottom
            if self.num_dims == 3:
                z_dims = np.arange(2, (self.num_dims * self.num_preds + 1), 3)
                direction_to_base[z_dims] = 0

            # Create all of the position
            for pred_num, predator in enumerate(self.predators):
                start_idx = pred_num * self.num_dims
                end_idx = (pred_num + 1) * self.num_dims
                current_pred_position = direction_to_base[start_idx:end_idx]
                current_pred_distance = distance_from_base[pred_num]
                # print('current_pred_distance, current_pred_position', current_pred_distance, current_pred_position)
                predator.position = (
                    current_pred_distance
                    * current_pred_position
                    / np.linalg.norm(current_pred_position)
                )
                predator.clip_entity_position(True)             #SN: to ensure starting point isn't outside geofence
                predator.is_live = True
                predator.velocity = np.zeros(self.num_dims)
                predator.hit_geofence = False
                # predator.n_hit_geofence_or_obs = 0
                # print('reset: pred.pos', predator.position)

            # if the generation spawned predators on top of each other regenerate
            obs = self.get_observation()[0]
            collision, _, _ = self.pred_collided()
            if not np.any(collision):
                has_collision = False
            else:
                print('**WARNING: preds landed on top of each other, not curently handled')

        # Prey velocity towards [0] * Dims (herd location)
        if self.mode == 'defense':
            target_velocity = (self.base_position - self.prey.position) #SN: target direction?
            max_prey_velocity = self.prey.max_velocity
        elif self.mode == 'offense':
            position = np.zeros(self.num_dims)
            for predator in self.predators:
                position += predator.position
            average_pred_position = position / self.num_preds
            target_velocity = (self.prey.position - average_pred_position)
            max_prey_velocity = self.max_velocity / 2
        else:
            raise ValueError('Unknown Prey policy: ' + str(self.mode))
        # self.prey.velocity = np.asarray([0.0, 0.0])
        self.prey.velocity = (
            max_prey_velocity
            * target_velocity
            / np.linalg.norm(target_velocity)
            * self.STEP_SIZE)

        obs = self.get_observation()[0]
        """ This is best place for this but creates cyclic dependency b/c shield <- model_1pt <- config <- env.     
        se = solnExistsPy(self.predators[0].position + self.predators[0].velocity, 
                          self.prey.position + self.prey.velocity, 
                          self.env.STEPS_BOUND)
        print('solnExists', se) """
        self.start_of_episode = True
        return obs, []

    '''returns new system state and whether preds collided with each other, geofence, etc'''
    def get_observation(self):

        # Get the total number of predators
        total_num_preds = self.num_preds + self.num_fake_preds    

        # Define a function to replace real predator positions
        def overwrite_real(random_draw):           
            # Get the random samples
            pred_velocity = random_draw['observation'][:-self.num_dims]
            pred_position = np.ravel(random_draw['achieved_goal'])

            # Replace the random samples with real values from actual predators
            pred_idxs = self.num_preds * self.num_dims
            pred_velocity[:pred_idxs] = [predator.velocity for predator in self.predators]
            pred_position[:pred_idxs] = [predator.position for predator in self.predators]

            return pred_position, pred_velocity

        # We just copy predator kinematics if there are no fake predators
        if self.num_fake_preds == 0:
            pred_velocity = np.concatenate([predator.velocity for predator in self.predators])
            achieved_goal = np.concatenate([predator.position for predator in self.predators])
        # If there are fake predators, we need to sample and then overwrite
        else:
            # For time-varying distribution, we draw relative to first predator
            # position so that mean will vary with time
            if self.correlate_fake_preds == 'predator':
                for fake_predator in self.fake_predators:
                    # The fake pred positions will be calculated using the predator position
                    fake_predator.position = self.predators[0].position + fake_predator.offset
                    fake_predator.velocity = self.predators[0].velocity
                    self.clip_entity_position(fake_predator)

                # Join the real and fake predator positions
                achieved_goal = np.concatenate(([predator.position for predator in self.predators],
                                                 [fake_predator.position for fake_predator in self.fake_predators]))

                # Join the real and fake predator velocities
                pred_velocity = np.concatenate(([predator.velocity for predator in self.predators],
                                                 [fake_predator.velocity for fake_predator in self.fake_predators]))
                
            elif self.correlate_fake_preds == 'prey':
                # raise NotImplementedError('Must implement resetter & collider as well')
            
                for fake_predator in self.fake_predators:
                    # The fake pred positions will be calculated using the predator position
                    fake_predator.position = self.prey.position + fake_predator.offset
                    fake_predator.velocity = self.prey.velocity
                    self.clip_entity_position(fake_predator)


                # Join the real and fake predator positions
                achieved_goal = np.concatenate(([predator.position for predator in self.predators],
                                                 [fake_predator.position for fake_predator in self.fake_predators]))

                # Join the real and fake predator velocities
                pred_velocity = np.concatenate(([predator.velocity for predator in self.predators],
                                                 [fake_predator.velocity for fake_predator in self.fake_predators]))

                

            # For time-invariant distributions, we draw uniformly across the
            # state space
            elif self.correlate_fake_preds == 'none':

                # Random draw from distribution
                random_draw = self.observation_space.sample()
                achieved_goal, pred_velocity = overwrite_real(random_draw)
                # since we are randomly drawing make sure to clip all positions
                achieved_goal = self.clip_positions(achieved_goal)
            
            else:
                raise ValueError('Value for correlate_fake_preds not recognized: ' + str(self.correlated_fake_preds))

        # If the predators collide, set the position to all -1
        task_failed = False
        _, achieved_goal, task_failed = self.pred_collided()

        if self.num_fake_preds == 0:
            self.position = np.copy(achieved_goal)
        else:
            self.position = np.copy(achieved_goal[:(self.num_preds * self.num_dims)])

        obs = {
            'observation': np.concatenate((pred_velocity, self.prey.velocity)),
            'achieved_goal': achieved_goal,
            'desired_goal': self.prey.position,
        }

        return obs, task_failed
            
    '''check if predators collided with each other, fake preds, base, ground, geofence or obstacles'''
    def pred_collided(self):
        collided = np.zeros(self.num_preds, dtype=bool)
        hits = np.zeros(self.num_preds)           #SN:
        task_failed = False

        # Check if predators collide with each other
        if self.preds_collide and self.num_preds > 1:
            for pred_1_idx, pred_1 in enumerate(self.predators):
                if not pred_1.is_live:
                    continue
                for pred_2_idx, pred_2 in enumerate(self.predators[pred_1_idx+1:]):

                    if np.linalg.norm(pred_1.position - pred_2.position) < self.AT_TARGET_RADIUS:
                        collided[pred_1_idx] = True
                        collided[pred_2_idx+pred_1_idx + 1] = True

        # Check if predators collide with fake preds
        if self.preds_collide_with_fake:
            for pred_index, predator in enumerate(self.predators):
                for fake_predator in self.fake_predators:
                    if np.linalg.norm(predator.position - fake_predator.position) < self.AT_TARGET_RADIUS:
                        # For now, fake preds are not destroyed
                        collided[pred_index] = True
                        # collided[fake_pred] = True

        # Check if predators collide with base
        if self.collide_with_herd:
            for pred_index, predator in enumerate(self.predators):
                collided_with_herd = False

                if self.flat_herd:
                    if (
                        (np.linalg.norm(
                            predator.position[:2]
                            - self.base_position[0:2]
                            ) < self.AT_TARGET_RADIUS 
                        )
                        and ((predator.position[-1] - self.base_position[-1]) < (self.AT_TARGET_RADIUS / 2))
                    ):
                        collided_with_herd = True
                else:
                    if np.linalg.norm(predator.position - self.base_position) < self.AT_TARGET_RADIUS:
                        collided_with_herd = True

                if collided_with_herd:                    
                    collided[pred_index] = True
                    task_failed = True

        # Check if predators collide with ground with large velocity
        if self.collide_with_ground:
            for pred_idx, predator in enumerate(self.predators):
                if ((predator.position[-1] < self.AT_TARGET_RADIUS)
                    and (predator.velocity[-1] <= self.GROUND_COLLISION_VELOCITY)
                    and predator.is_live):
                    collided[pred_idx] = True

        #SN: Check if predators collided with obstacle or geofence
        for pred_idx, predator in enumerate(self.predators):
          if predator.hitGeoFenceOrObs():
            # breakpoint()
            collided[pred_idx] = True
            task_failed = True      #SN: this flag causes step to return done
            self.n_hit_geofence += 1
            if self.n_hit_geofence % 10 == 0: print('n_hit_geofence or obstacle', self.n_hit_geofence)
            # print('**predator', pred_idx, ' collided with geofence at', predator.position)  #SN:

        # set their is_live to false, position to [-1] * num_dims, velocity to [0] * num_dims
        for predator_index, predator in enumerate(self.predators):
            if collided[predator_index]:
                predator.is_live = False
                # breakpoint()                #SN:
                #reset predator position after collision
                predator.position = np.array([-1]*self.num_dims)
                predator.velocity = np.zeros(self.num_dims)

        updated_positions = np.concatenate([predator.position for predator in self.predators])
        return collided, updated_positions, task_failed
    
    def at_target(self, achieved_goal, desired_goal):
        is_at_target = False
        for pred_num in range(self.num_preds):
            start_idx = pred_num * self.num_dims
            end_idx = (pred_num + 1) * self.num_dims
            if np.linalg.norm(
                achieved_goal[start_idx:end_idx]
                - desired_goal
            ) < self.MIN_DISTANCE_FROM_BASE: #was self.AT_TARGET_RADIUS:
                is_at_target = True
        return is_at_target
    
    def predator_at_prey(self):
        for predator in self.predators:
            if self.entityCollidedAt(predator, self.prey.position):
                self.prey.is_live = False
                self.n_prey_caught += 1
                if self.n_prey_caught%10 == 0: print('n_prey_caught',self.n_prey_caught)
                return True
        return False
    
    def entityCollidedAt(self, entity: BaseAgent, position):
        return np.linalg.norm(entity.position - position) < self.MIN_DISTANCE_FROM_BASE #was self.AT_TARGET_RADIUS
       
    '''only used for fake preds'''
    def clip_positions(self):
        for predator in self.predators:
           self.clip_entity_position(predator)
    
        for fake_predator in self.fake_predators:
           self.clip_entity_position(fake_predator)

        self.clip_entity_position(self.prey)

    def action_to_acceleration(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action * self.max_acceleration

    #SN: added so evaluate_action in z3_shield can convert from acc back to action
    def acceleration_to_action(self, acc):
        action = [i / self.max_acceleration for i in acc]
        # maybe assert (self.action_space.low <= elt for elt in action).all() and .. 
        # SN: tried assert self.action_space.log <= action, etc but didnt like that either
        # assert all(self.action_space.low <= elt and elt <= self.action_space.high for elt in action), "provided acceleration value outside action space limits"
        return action
      
    def step(self, action):
        # print('env_action', action)
        self.num_episode_steps += 1
        self.total_timesteps += 1
        acceleration = self.action_to_acceleration(action)

        # Update predators
        for pred_num, predator in enumerate(self.predators):
            # Don't update dead predators
            if not predator.is_live:
                print('predator dead', pred_num, ' at ', predator.position) #SN: never gets here
                continue

            start_idx = pred_num * self.num_dims
            end_idx = (pred_num + 1) * self.num_dims
            # print('predator at ', predator.position, 'moving', predator.velocity, ' applying', acceleration)
            predator.apply_acceleration(acceleration[start_idx:end_idx], self.STEP_SIZE)
            # compute the true velocity after clipping positions    
            # print('clipping predator posn from', predator.position)
            predator.clip_entity_position()    #SN: added
            # print('new pos (after any clipping)', predator.position)
        
        # Check if predators are all dead   SN: THIS IS IN THE WRONG PLACE! Moved to after get_obs
        # predators_dead = np.all([not predator.is_live for predator in self.predators])
        # Update Prey - move a constant velocity towards herd
        self.prey.position += self.prey.velocity
        self.prey.clip_entity_position()

        obs, task_failed = self.get_observation()
        #SN:
        # breakpoint()
        if self.DOING_BOUNDED: 
            bound_exceeded = self.n_steps_to_bound > (self.STEPS_BOUND - 1) #wy was this 2??
            if bound_exceeded: 
                self.n_bound_exceeded += 1
                if self.n_bound_exceeded%100 == 0: print('n_bound_exceeded', self.n_bound_exceeded)
            task_failed = task_failed or bound_exceeded
        
        predators_dead = np.all([not predator.is_live for predator in self.predators])  #SN: moved from earlier
        # truncated = False
        truncated = self.num_episode_steps >= self.MAX_EPISODE_STEPS
        # if self.predator_at_prey(): breakpoint() #print('shield would have rejected', action)

        '''done just means end the episode here, task_failed if preds collided (w/ each other, geofence, bound exceeded, etc), it got truncated if it took too long'''
        done =  task_failed or \
                self.predator_at_prey() or \
                truncated or \
                self.entityCollidedAt(self.prey, self.base_position) 

        '''
        print(str(obs['achieved_goal']) + ' | ' + str(obs['desired_goal']))
        if done:
            set_trace()
        '''
        #SN: if done is true the episode ends, with the reward below
        #SN:
        # if task_failed:
        #     print('task failed')  #SN:
        #     # done = True
        # elif self.predator_at_prey():
        #     print('caught prey at', self.prey.position)
        #     # done = True
        # elif self.num_episode_steps >= self.MAX_EPISODE_STEPS:
        #     # done = True
        #     print('truncated')
        # elif self.entityCollidedAt(self.prey, self.base_position):
        #     # done = True
        #     print('prey reached base')
        # else:
        #     done = False
        # breakpoint()
        info = {}
        info['done'] = done
        info['truncated'] = truncated
        info['task_failed'] = task_failed
        info['target_size'] = self.AT_TARGET_RADIUS
        info['pred_is_dead'] = predators_dead
            
        rew = self._compute_reward(
            obs['achieved_goal'],
            obs['desired_goal'],
            info
        )

        ''' 
        if done and (rew < 0) and (self.total_timesteps > int(50E3)):
            set_trace()
        '''

        if rew != 0: 
            self.n_tot_rew += rew
            # print('step: n_tot_rew', self.n_tot_rew, 'n_hit_geofence', self.n_hit_geofence, 'n_prey_caught', self.n_prey_caught)
        # if task_failed: 
            # breakpoint()
            # print('step: rew', rew)
        # truncated flag only true when max_step_count exceeded
        self.n_steps_to_bound += 1
        
        if self.end_now:
            rew = -1
            done = True
        
        return obs, rew, done, truncated, info
