'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from cutter import RainbowCutter
from matplotlib import colors as mcolors

cut = RainbowCutter()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n_points = 35
r, b, g = np.meshgrid(
            np.linspace(0, 1, n_points), 
            np.linspace(0, 1, n_points), 
            np.linspace(0, 1, n_points)
        )

rgb = np.array([r.flatten(), g.flatten(), b.flatten()]).T

mask = cut.cut_function(r,g,b)
ax.scatter(255*r[mask], 255*b[mask], 255*g[mask], c=rgb[mask.flatten()], marker='o')

ax.set_xlabel('Rot')
ax.set_ylabel('Blau')
ax.set_zlabel('Grün')

plt.savefig('test.png')

