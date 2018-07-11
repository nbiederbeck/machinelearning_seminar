import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mls.rainbow.cutter import RainbowCutter
from mls.rainbow.painter import RainbowPainter
from mls.rainbow.plotter import RainbowPlotter

from cutimage import *


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

cutter = RainbowCutter(256)
mask_cube = cutter.mask_cube()
plotter = plot_cube(mask_cube)


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes()

r, c = plotter.plot_plane(0, 0)

quadmesh = ax.pcolormesh(r, color=c, animated=True)

# initialization function: plot the background of each frame
def init():
    quadmesh.set_array(r.ravel())
    return (quadmesh,)


# animation function.  This is called sequentially
def animate(i):
    print(i)
    r, _ = plotter.plot_plane(0, i)
    quadmesh.set_array(r.ravel())
    return (quadmesh,)


anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=256, interval=20, blit=False
)

plt.show()


# def main():
#     r = 0
#     g = 1
#     b = 2

#     cutter = RainbowCutter()
#     painter = RainbowPainter()
#     plotter = RainbowPlotter()

#     N = painter.N
#     figsize = plotter.figsize_inches

#     rg = (  # A generator for the red green side of the color cube
#         plotter.plot_colormesh(
#             cutter.cut_mesh(
#                 painter.paint_mesh(r, g, b, i),
#                 r,
#                 g,
#                 b,
#                 np.pi / 3,
#                 scale=-4,
#                 offset_x=i / N,
#                 offset_y=i / N,
#             ),
#             return_ax=False,
#         )
#         for i in range(N)
#     )

#     # Plotting and saving each generated layer
#     for i, fig in enumerate(tqdm(rg, desc="GIF", total=N)):
#         fig.savefig("build/{}{:03d}.png".format("rg_", i), dpi=N / figsize)
#         plt.close(fig)


# if __name__ == "__main__":
#     main()
