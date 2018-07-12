import numpy as np
import matplotlib.pyplot as plt
from mls.rainbow.cutter import RainbowCutter


class CubePlotter:
    """Plot the 3d RGB spectrum as a cube."""

    def __init__(self):
        """Initialize 3d RGB meshgrid."""
        self.r, self.b, self.g = np.meshgrid(
            np.linspace(0, 1, 256), np.linspace(0, 1, 256), np.linspace(0, 1, 256)
        )

    def apply_mask(self, mask):
        """Apply calculated mask on RGB colors.
        Every value catched by mask will be set to zero.

        Parameters:
            mask: np.ndarray
        """
        self.r[~mask] = 0
        self.b[~mask] = 0
        self.g[~mask] = 0

    def plot_plane(self, plane_color, plane):
        """Plot single plane in 3d RGB cube.

        Parameters:
            plane_color: int 0..2
                Choice of color (r, g, b)
            plane: int 0..255
                RGB value of chosen plane_color
        """
        if plane_color == 0:
            r, g, b = self.r[plane, :, :], self.g[plane, :, :], self.b[plane, :, :]
        elif plane_color == 1:
            r, g, b = self.r[:, plane, :], self.g[:, plane, :], self.b[:, plane, :]
        elif plane_color == 2:
            r, g, b = self.r[:, :, plane], self.g[:, :, plane], self.b[:, :, plane]

        rgb = np.array([r, g, b]).T

        color_tuple = rgb.transpose((1, 0, 2)).reshape(
            (rgb.shape[0] * rgb.shape[1], rgb.shape[2])
        )

        m = plt.pcolormesh(r, color=color_tuple, linewidth=0)
        m.set_array(None)

    def plot_cube(self):
        """Plot full 3d RGB cube.
        Paints a figure with 3 subplots.
        """
        # for i in range(256):
        #     print("\r{:3.2f}% plotted.".format(100 * i / 256), end="")

        #     fig = plt.figure(figsize=(5, 15))

        #     for r in range(3):
        #         plt.subplot("31{}".format(r))
        #         self.plot_plane(r, i)

        #     fig.tight_layout(pad=0)
        #     plt.savefig("build/{}.png".format(i))
        #     plt.close()
        for i in range(25):
            print("\r{:3.2f}% plotted.".format(10 * i / 256), end="")

            fig = plt.figure(figsize=(15, 5))

            for r in range(3):
                plt.subplot("13{}".format(r))
                self.plot_plane(r, i*10)

            fig.tight_layout(pad=0)
            plt.savefig("build/{}.png".format(i*10))
            plt.close()


def main():
    """Plot 3d RGB cube."""
    cutter = RainbowCutter(
        N=256,
        theta=-np.pi / 3.5,
        scale=-10.0,
        xshift=1.6,
        yshift=1.6,
        xbias=0.0,
        ybias=0.1,
    )

    plotter = CubePlotter()

    mask_cube = cutter.cut_function(plotter.r, plotter.g, plotter.b)

    plotter.apply_mask(mask_cube)

    plotter.plot_cube()


if __name__ == "__main__":
    main()
