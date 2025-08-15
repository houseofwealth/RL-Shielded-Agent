import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import imageio
from os import remove, listdir
from ipdb import set_trace
from tqdm import tqdm
from meher.sphere import draw_sphere # SN:
from meher.rectangle import draw_rectangle


'''
    Evaluate for another epoch and render the result
    This is useful for repeatedly making gifs of agent behavior, especially
        if previous gifs show trivial agent behavior (e.g., the prey is caught
        in one timestep)
    
    Inputs:
        ev2render (rllib_wolfpack.cont_predator_prey.evaluator.Evaluator()) - the
            evaluator that should be used to evaluate the agent performance.

'''
def rerender(ev2render, gif_name=None):
        ev2render.reset()
        ev2render.evaluate()
        Render(
            ev2render.obs_list, grid_size=ev2render.grid_size,
            use_set_function=ev2render.use_set_function, gif_name=gif_name,
            prey_radius=ev2render.prey_radius, base_radius=ev2render.base_radius,
        )


class Render():

    '''
        Render an epoch as a gif

        Inputs:
            observations (list of dicts) - the observations of all agents at each timestep
            num_dims (int) - the number of physical dimensions (2D, 3D, etc...)
            grid_size (int) - the size of the workspace in arbitrary units.  The
                workspace is assumed to be square (or, at least, hypercube)
    '''

    def __init__(
        self, observations, num_dims=2, grid_size=10,
        use_set_function=False, gif_name=None, prey_radius=1, base_radius=1
        ):
        
        # Input handling
        self.observations = self.get_observations(observations)
        self.num_dims = num_dims
        self.grid_size = grid_size
        self.use_set_function = use_set_function
        # Get the names
        self.names = list(self.observations.keys())

        if gif_name is None:
            self.gif_name='./episode.gif'
        else:
            self.gif_name = str(gif_name)

        # Initialize the plot 
        plt.figure()
        plt.ylim([0 - 1, self.grid_size + 1])
        plt.xlim([0 - 1, self.grid_size + 1]) 
        ax = plt.gca()
        ax.set_aspect('equal')
        # set_trace()
                
        # Initialize the agents
        self.agents = {}
        for num, agent in enumerate(self.names):
            if 'prey' in agent:
                color='b'
                radius = prey_radius
            elif 'base' in agent:
                color = 'k'
                radius = base_radius
            else:
                color='r'
                radius = prey_radius

            pos = self.observations[agent][0, :]
            circle  = Circle(tuple(pos), radius=radius, alpha=0.5, color=color)
            self.agents[agent] = circle
            ax.add_artist(circle)
            plt.title('Step #' + str(0))
            plt.savefig(str(0) + '.png')

            count = self.create_pngs()
            self.create_gif(count)

    def create_pngs(self):

        # Update
        count = 1
        print('Creating *.pngs:')
        for num in range(np.shape(self.observations['prey'])[0]):

            for key, val in self.observations.items():
                pos = val[num, :]
                try:
                    self.agents[key].set(center=pos)
                except:
                    set_trace()
            plt.title('Step #' + str(num))
            plt.savefig(str(count) + '.png')
            count += 1
        return count - 1

    def create_gif(self, count):
        # Create gif
        file_names = ['_' + str(i) + '.png' for i in range(count)]

        if len(file_names) == 0:
            print('No movements - predator spawned on prey')
            return

        with imageio.get_writer(self.gif_name, mode='I', duration=0.01) as gif_writer:

            print('Creating *.gif:')
            # Every movement
            for file_name in tqdm(file_names):
                image = imageio.imread(file_name)
                gif_writer.append_data(image)

            # Hold at end for a while
            for i in range(10):
                image = imageio.imread(file_name)
                gif_writer.append_data(image)


        # Remove png
        while len(file_names) > 0:
            for file in file_names:
                remove(file)
            file_names = listdir('./')
            file_names = [f for f in file_names if '_*.png' in f]

        # Notify user
        print('GIF saved!')

    def get_observations(self, observations):
        obs_only = np.asarray([obs['observation'] for obs in observations])
        prey_positions = obs_only[:, :2]
        pred_positions = obs_only[:, 4:6]
        base_positions = np.asarray([obs['desired_goal'] for obs in observations])

        observations = {
            'prey': prey_positions,
            'predator': pred_positions,
            'base': base_positions
        }

        return observations


class Render_ct(Render):

    def get_observations(self, observations):
        prey_positions = np.asarray([obs['achieved_goal'] for obs in observations])
        pred_positions = np.asarray([obs['undesired_goal'] for obs in observations])
        base_positions = np.asarray([obs['desired_goal'] for obs in observations])
        
        observations = {
            'prey': prey_positions,
            'base': base_positions
        }

        num_dims = np.shape(prey_positions)[1]
        num_preds = int(np.shape(pred_positions)[1] / num_dims)
        for pred_num in range(num_preds):
            observations['predator' + str(pred_num)] = pred_positions[
                :, (num_dims * pred_num):(num_dims * (pred_num + 1))
            ]

        return observations


class Render3d(Render):

    '''
        Render an epoch as a gif in 3d

        Inputs:
            observations (list of dicts) - the observations of all agents at each timestep
            num_dims (int) - the number of physical dimensions (2D, 3D, etc...)
            grid_size (int) - the size of the workspace in arbitrary units.  The
                workspace is assumed to be square (or, at least, hypercube)
    '''

    def __init__(
        self, observations, num_dims=3, grid_size=10,
        use_set_function=False, gif_name=None, prey_radius=1,
        base_radius=1, reward=None, num_preds=np.inf, obstacle_bounds=None,
        ):
        
        # Input handling
        self.observations = self.get_observations(observations)
        self.num_dims = num_dims
        self.grid_size = grid_size
        self.use_set_function = use_set_function
        self.obstacle_bounds = obstacle_bounds

        # Get the names
        self.names = list(self.observations.keys())
        self.color_names = np.copy(self.names)
        if num_preds < np.inf:
            is_pred_name = ['predator' in name for name in self.names]
            pred_names = np.asarray(self.names)[is_pred_name]
            pred_numbers = np.asarray([int(name.split('predator')[1]) for name in pred_names])
            is_fake_pred = [pd >= num_preds for pd in pred_numbers]
            fake_pred_names_old = pred_names[is_fake_pred]
            fake_pred_names_new = ['fake_pred' + str(num) for num in pred_numbers[is_fake_pred]]
            for num, name in enumerate(self.names):
                if name in fake_pred_names_old:
                    name_loc = np.where(fake_pred_names_old == name)[0][0]
                    new_name = fake_pred_names_new[name_loc]
                    self.color_names[num] = new_name

        if gif_name is None:
            self.gif_name='./episode.gif'
        else:
            self.gif_name = str(gif_name)

        # Initialize the plot 
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        def get_agent_color(agent):
            if 'prey' in agent:
                color='b'
                radius = prey_radius
            elif 'base' in agent:
                color = 'k'
                radius = base_radius
            elif 'fake_pred' in agent:
                color = 'g'
                radius = prey_radius
            else:
                color='r'
                radius = prey_radius

            return color, radius

        count = 0
        print('Creating *.pngs:')
        for obs_num in range(np.shape(self.observations['prey'])[0]):

            # Clear the figure
            ax.clear()

            # Plot hidden spheres in corners to show whole workspace
            zero_z = np.ones(3)
            zero_z[-1] = 0
            #SN: 3 draw_sphere comments out b/c library not found
            draw_sphere(ax, 1E-6, -11 * zero_z, 'b')
            draw_sphere(ax, 1E-6, 11 * np.ones(3), 'b')

            # set_trace()
            if self.obstacle_bounds is not None:
                if len(self.obstacle_bounds) == 4:
                    obstacle_bounds = [ob for ob in self.obstacle_bounds]
                    obstacle_bounds.extend([0, 0.01])
                else:
                    obstacle_bounds = self.obstacle_bounds

                draw_rectangle(
                    obstacle_bounds[0],
                    obstacle_bounds[1],
                    obstacle_bounds[2],
                    obstacle_bounds[3],
                    obstacle_bounds[4],
                    obstacle_bounds[5],
                    ax,
                )

                # draw_rectangle(
                #     -10,
                #     10,
                #     -10,
                #     10,
                #     -10,
                #     10,
                # )
            # rectangle = 

            

            # Plot each agent
            for agent_num, (color_name, agent) in enumerate(zip(self.color_names, self.names)):
                color, radius = get_agent_color(color_name)

                # Get the position
                pos = self.observations[agent][obs_num, :]
                if len(pos) == 2:
                    # Make 3D, if necessary
                    pos = np.append(pos, [0])

                draw_sphere(ax, radius, pos, color) #, added next 3 lines -- throws error
                # posT = self.observations[agent][0, :]
                # circle  = Circle(tuple(posT), radius=radius, alpha=0.5, color='green')
                # ax.add_artist(circle)

                # Plot lines
                for dim in range(self.num_dims):
                    pos2 = np.copy(pos)
                    if dim == 0:
                        pos2[dim] = -10
                    elif dim == 1:
                        pos2[dim] = 10
                    else:
                        pos2[dim] = 0
                    
                    ax.plot(
                        [pos[0], pos2[0]],
                        [pos[1], pos2[1]],
                        [pos[2], pos2[2]],
                        linewidth=0.5,
                        color=color,
                    )
                    ax.set_ylim([-10, 10])
            
            result_string = ''
            if obs_num == (np.shape(self.observations['prey'])[0] - 1):
                if reward is not None:
                    if reward > 0:
                        result_string = ' - Success'
                    elif reward < 0:
                        result_string = ' - Failure'

            # Title and save
            plt.title('Step #' + str(obs_num) + result_string)
            plt.savefig('_' + str(obs_num) + '.png')
            count += 1      
        
        self.create_gif(count)


class Render_ct_3d(Render3d):

    def get_observations(self, observations):
        pred_positions = np.asarray([obs['achieved_goal'] for obs in observations])
        prey_positions = np.asarray([obs['desired_goal'] for obs in observations])
        num_dims = np.shape(prey_positions)[1]

        base_positions = np.zeros((prey_positions.shape[0], num_dims))
        
        observations = {
            'prey': prey_positions,
            'base': base_positions
        }

        
        num_preds = int(np.shape(pred_positions)[1] / num_dims)
        for pred_num in range(num_preds):
            observations['predator' + str(pred_num)] = pred_positions[
                :, (num_dims * pred_num):(num_dims * (pred_num + 1))
            ]

        return observations


if __name__ == '__main__':
    NUM_OBS = 10
    NUM_FEATURES = 6
    observations = []
    for _ in range(NUM_OBS):
        obs = {
            'achieved_goal': np.random.random(size=(2,)),
            'desired_goal': np.random.random(size=(2,)),
            'observation': np.random.random(size=(NUM_FEATURES,)),
        }
        observations.append(obs)

    renderer = Render3d(
        observations, num_dims=2, grid_size=10,
        use_set_function=False, gif_name=None, prey_radius=1, base_radius=1
    )