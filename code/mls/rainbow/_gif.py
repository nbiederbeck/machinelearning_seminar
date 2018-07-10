import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mls.rainbow.cutter import RainbowCutter
from mls.rainbow.painter import RainbowPainter
from mls.rainbow.plotter import RainbowPlotter


def main():
    r = 0
    g = 1
    b = 2

    cutter = RainbowCutter()
    painter = RainbowPainter()
    plotter = RainbowPlotter()

    N = painter.N
    figsize = plotter.figsize_inches

    rg_ = (  # A generator for the red green side of the color cube
        plotter.plot_colormesh(
            cutter.cut_mesh(painter.paint_mesh(r, g, b, i), r, g, b, np.pi / 4),
            return_ax=False,
        )
        for i in range(N)
    )

    gb_ = (  # A generator for the green blue side of the color cube
        plotter.plot_colormesh(
            cutter.cut_mesh(painter.paint_mesh(g, b, r, i), r, g, b, np.pi / 4),
            return_ax=False,
        )
        for i in range(N)
    )
    br_ = (  # A generator for the blue red side of the color cube
        plotter.plot_colormesh(
            cutter.cut_mesh(painter.paint_mesh(b, r, g, i), r, g, b, np.pi / 4),
            return_ax=False,
        )
        for i in range(N)
    )

    sides = [rg_, gb_, br_]

    # Plotting and saving each generated layer
    for l, layers in zip(["rg_", "gb_", "br_"], tqdm(sides, desc="Sides", total=3)):
        for i, fig in enumerate(tqdm(layers, desc="GIF", total=N)):
            fig.savefig("build/{}{:03d}.png".format(l, i), dpi=N / figsize)
            plt.close(fig)


if __name__ == "__main__":
    main()
