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

        print(r.shape)
        rgb = np.array([r, g, b]).T

        color_tuple = rgb.transpose((1, 0, 2)).reshape(
            (rgb.shape[0] * rgb.shape[1], rgb.shape[2])
        )

        m = plt.pcolormesh(r, color=color_tuple, linewidth=0)
        m.set_array(None)

    def plot_cube(self, planes):
        """Plot full 3d RGB cube.
        Paints a figure with 3 subplots.
        """
        n = len(planes)
        for i in range(n):
            fig = plt.figure(figsize=(15, 5*n))
            print(n)
            for r in range(3):
                print("{}{}{}".format(n, 3, 3*i+r+1))
                plt.subplot("{}{}{}".format(n, 3, 3*i+r+1))
                self.plot_plane(r, planes[i])
        fig.tight_layout(pad=0)
        plt.savefig("build/cube.png")
        plt.close()


def main():
    """Plot 3d RGB cube."""
    cutter = RainbowCutter()
    plotter = CubePlotter()

    mask_cube = cutter.cut_function(plotter.r, plotter.g, plotter.b)
    plotter.apply_mask(mask_cube)
    plotter.plot_cube([0, 150])


if __name__ == "__main__":
    main()
