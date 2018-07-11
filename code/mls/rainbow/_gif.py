from cutimage import *

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def main(color=0):
    N = 256
    cutter = RainbowCutter(N)
    mask_cube = cutter.mask_cube()
    plotter = plot_cube(mask_cube)

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(4, 4))
    ax = plt.axes()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=-0.1)

    r, c = plotter.plot_plane(0, 0)

    quadmesh = ax.pcolormesh([[], []])

    # initialization function: plot the background of each frame
    def init():
        quadmesh.set_array(r.ravel())
        return (quadmesh,)

    # animation function.  This is called sequentially
    def animate(i):
        print("\r{:3.2f}% erstellt.".format(100 * i / N), end="")
        r, c = plotter.plot_plane(color, N - 1 - i)
        quadmesh.set_array(r.ravel())
        # plt.draw()
        return (quadmesh,)
        # plt.pcolormesh(r, color=c)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=10, interval=20, blit=False
    )

    anim.save("animation_{}.mp4".format(color), fps=30)


if __name__ == "__main__":
    main(0)
    # for c in [0, 1, 2]:
    #     main(c)
