import numpy as np
import matplotlib.pyplot as plt


class RainbowCutter:
    def __init__(self, N, theta, scale, xshift=1, yshift=1, xbias=0, ybias=0):
        """ initalize a 3dim Mesgrid with N points per axis
        for calculation of the cuts
        Parameters:
            N: int
            discretisation of axis
        """
        self.r, self.b, self.g = np.meshgrid(
            np.linspace(0, 1, N), np.linspace(0, 1, N), np.linspace(0, 1, N)
        )
        self.theta = theta
        self.scale = scale
        self.xshift = xshift
        self.yshift = yshift
        self.xbias = xbias
        self.ybias = ybias

    def _rotate(self, x, y, theta):
        """ Rotate given tuple of axis with theta at
        coordinate origin
        Parameters:
            x, y: ndarray
                to rotate arrays
            theta: double
                angle to rotate
        """
        xr = np.cos(theta) * x + np.sin(theta) * y
        yr = -np.sin(theta) * x + np.cos(theta) * y
        return xr, yr

    def cut_function(self, x, y, z):
        """ cut function to seperate background from signal
        Parameters:
            x, y: ndarray(N,N,N)
                colors to check
            x0, y0: ndarray(N,N,N)
                bias of color
            theta: double
                angle to rotate
            scale: double
                scale of the parabular
        Returns:
            mask: ndarray
                mask of true and false for given
                points
        """
        x0 = z * self.xshift + self.xbias
        y0 = z * self.yshift + self.ybias
        x_rot, y_rot = self._rotate(x, y, self.theta)
        x0_rot, y0_rot = self._rotate(x0, y0, self.theta)

        x = x_rot - x0_rot
        y = y_rot - y0_rot

        mask = y < self.scale * (x * x)

        return mask

    def cut_image(self, im):
        r = im[:, :, 0]
        g = im[:, :, 1]
        b = im[:, :, 2]
        mask = self.cut_function(r, g, b)
        im[~mask] = 0
        return im

    def mask_cube(self):
        return self.cut_function(self.r, self.g, self.b)


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

    def ful_plane(self):
        for x in range(51):
            print(x * 5)
            fig = plt.figure(figsize=(5, 15))
            for r in range(3):
                plt.subplot("31{}".format(r))
                self.plot_plane(r, x * 5)
            plt.savefig("build/{}.png".format(x * 5))
            plt.close()
        m = plt.pcolormesh(r, color=color_tuple, linewidth=0)
        m.set_array(None)
        return r, color_tuple


def main():
    N = 256

    cutter = RainbowCutter(
        N, -np.pi / 3, -20, xshift=2.0, yshift=2.0, xbias=0.2, ybias=-0.3
    )
    mask_cube = cutter.mask_cube()

    pltter = plot_cube(mask_cube, True)
    pltter.ful_plane()


if __name__ == "__main__":
    main()
