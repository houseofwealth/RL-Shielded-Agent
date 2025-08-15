# Import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ipdb import set_trace


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Inspired by:
# https://stackoverflow.com/questions/76002425/plot-surfaces-like-a-box-in-3d
def draw_rectangle(xmin, xmax, ymin, ymax, zmin, zmax, ax=None):
    # x_data = [xmin, xmax]
    # y_data = [ymin, ymax]

    GRID_POINTS = 10

    xModel = np.linspace(xmin, xmax, GRID_POINTS)
    yModel = np.linspace(ymin, ymax, GRID_POINTS)
    zModel = np.linspace(zmin, zmax, GRID_POINTS)
    X, Y = np.meshgrid(xModel, yModel)
    _, Z = np.meshgrid(xModel, zModel)
    Z2, _ = np.meshgrid(zModel, xModel)

    def func2(data, m):
        return m*np.ones([len(data[0]), len(data[1])])

    points = np.array([[xmin, ymin, zmin], 
            [xmin, ymax, zmin],
            [xmax, ymin, zmin], 
            [xmax, ymax, zmin], 
            [xmin, ymin, zmax], 
            [xmin, ymax, zmax],
            [xmax, ymin, zmax], 
            [xmax, ymax, zmax]])
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    alpha = 0.35
    # ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])

    zero_plane = np.zeros((GRID_POINTS, GRID_POINTS))

    # There are 6 sides of the cube to plot
    # Bottom
    ax.plot_surface(X, Y, zero_plane + zmin, rstride=1, cstride=1,
                    linewidth=1, antialiased=True, alpha=alpha, facecolor='r')

    # Top. Not plotted intentionally.
    ax.plot_surface(X, Y, zero_plane + zmax, rstride=1, cstride=1,
                   linewidth=1, antialiased=True, alpha=alpha, facecolor='r')

    # Sides
    ax.plot_surface(zero_plane + xmin, Y, Z2, rstride=1, cstride=1,
                    linewidth=1, antialiased=True, alpha=alpha, facecolor='r')
    
    ax.plot_surface(zero_plane + xmax, Y, Z2, rstride=1, cstride=1, 
                    linewidth=1, antialiased=True, alpha=alpha, facecolor='r')

    ax.plot_surface(X, zero_plane + ymin, Z, rstride=1, cstride=1, 
                    linewidth=1, antialiased=True, alpha=alpha, facecolor='r')
    
    ax.plot_surface(X, zero_plane + ymax, Z, rstride=1, cstride=1, 
                    linewidth=1, antialiased=True, alpha=alpha, facecolor='r')
    
    for num, point in enumerate(points):
        for point2 in points[num + 1:]:
            point_diff = point2 - point
            if np.sum(point_diff != 0) == 1:
                ax.plot(
                    [point[0], point2[0]],
                    [point[1], point2[1]],
                    [point[2], point2[2]],
                    'k',
                )
            # set_trace()

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    # set_trace()


if __name__ == '__main__':
    xmin = -4
    xmax = 4
    ymin = 3
    ymax = 8
    zmin = 0
    zmax = 10
    draw_rectangle(xmin, xmax, ymin, ymax, zmin, zmax)
    plt.show()



'''
# Create axis
axes = [5, 5, 5]

# Create Data
data = np.ones(axes, dtype=bool)
set_trace()

# Control Transparency
alpha = 0.9

# Control colour
colors = np.empty(axes + [4], dtype=np.float32)

colors[:] = [1, 0, 0, alpha] # red

# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Voxels is used to customizations of the
# sizes, positions and colors.
ax.voxels(data, facecolors=colors)
'''

# plt.show()