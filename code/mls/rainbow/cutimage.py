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
        return self.cut_function(self.r, self.g, 1.1*self.b, 1.1*self.b,
                -1*np.pi/3.5
                , -30)


def main():
    N = 255
    cutter = RainbowCutter(N)
    mask_cube = cutter.mask_cube()
    print(mask_cube[:, 0, :])
    print(mask_cube[:, -1, :])


    r, g= np.meshgrid(
            np.linspace(0, 1, N), np.linspace(0, 1, N)    
            )

    point_of_interest= 15
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.scatter(
            r[mask_cube[:, :, point_of_interest]], 
            g[mask_cube[:, :,point_of_interest]]
            )
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax = fig.add_subplot(312)
    ax.scatter(
            r[mask_cube[:, point_of_interest, :]], 
            g[mask_cube[:, point_of_interest, :]]
            )
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax = fig.add_subplot(313)
    ax.scatter(
            r[mask_cube[point_of_interest, :, :]],
            g[mask_cube[point_of_interest, :, :]]
            )
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.show()


if __name__ == "__main__":
    main()
