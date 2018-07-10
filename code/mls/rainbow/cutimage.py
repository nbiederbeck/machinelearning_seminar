import numpy as np


class RainbowCutter:
    def __init__(self, N):
        self.r, self.b, self.g = np.meshgrid(
            np.linspace(0, 1, N), np.linspace(0, 1, N), np.linspace(0, 1, N)
        )

    def cut_function(self, x, y, x0, y0, theta, scale):
        def _rotate(x, y, theta):
            xr = np.cos(theta) * x + np.sin(theta) * y
            yr = -np.sin(theta) * x + np.cos(theta) * y
            return xr, yr

        x_rot, y_rot = _rotate(x, y, -theta)
        x0_rot, y0_rot = _rotate(x0, y0, -theta)
        x_rot -= x0_rot
        y_rot -= y0_rot

        mask = y_rot < scale * (x_rot * x_rot)

        return mask

    def mask_cube(self):
        return self.cut_function(self.r, self.g, self.b, self.b, 1 * np.pi / 4, 10)


def main():
    cutter = RainbowCutter(47)
    mask_cube = cutter.mask_cube()
    print(mask_cube[:, 0, :])
    print()
    print(mask_cube[:, -1, :])


if __name__ == "__main__":
    main()
