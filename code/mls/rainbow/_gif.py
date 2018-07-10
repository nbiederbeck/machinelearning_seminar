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

    rg = (  # A generator for the red green side of the color cube
        plotter.plot_colormesh(
            cutter.cut_mesh(
                painter.paint_mesh(r, g, b, i),
                r,
                g,
                b,
                np.pi / 3,
                scale=-4,
                offset_x=i / N,
                offset_y=i / N,
            ),
            return_ax=False,
        )
        for i in range(N)
    )

    # Plotting and saving each generated layer
    for i, fig in enumerate(tqdm(rg, desc="GIF", total=N)):
        fig.savefig("build/{}{:03d}.png".format("rg_", i), dpi=N / figsize)
        plt.close(fig)


if __name__ == "__main__":
    main()
