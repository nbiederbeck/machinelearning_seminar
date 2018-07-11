import numpy as np
import matplotlib.pyplot as plt


class RainbowCutter:
    def __init__(self, N):
        self.r, self.b, self.g = np.meshgrid(
            np.linspace(0, 1, N), np.linspace(0, 1, N), np.linspace(0, 1, N)
        )

    def _rotate(self, x, y, theta):
        xr = np.cos(theta) * x + np.sin(theta) * y
        yr = -np.sin(theta) * x + np.cos(theta) * y
        return xr, yr

    def cut_function(self, x, y, x0, y0, theta, scale):

        x_rot, y_rot = self._rotate(x, y, theta)
        x0_rot, y0_rot = self._rotate(x0, y0, theta)

        x = x_rot - x0_rot
        y = y_rot - y0_rot

        mask = y < scale * (x * x)

        return mask

    def mask_cube(self):
        return self.cut_function(
            self.r, self.g, 1.1 * self.b, 1.1 * self.b, -1 * np.pi / 3.5, -30
        )


class plot_cube:
    def __init__(self, mask, apply_mask=True):
        self.r, self.b, self.g = np.meshgrid(
            np.linspace(0, 1, 256), np.linspace(0, 1, 256), np.linspace(0, 1, 256)
        )
        if apply_mask:
            self.r[~mask] = 0
            self.b[~mask] = 0
            self.g[~mask] = 0

    def plot_plane(self, plane_color, plane):
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
        return r, color_tuple


def main():
    N = 256

    cutter = RainbowCutter(N)
    mask_cube = cutter.mask_cube()

    pltter = plot_cube(mask_cube, True)

    pltter.plot_plane(0, 150)
    pltter.plot_plane(1, 150)
    pltter.plot_plane(2, 150)


if __name__ == "__main__":
    main()
