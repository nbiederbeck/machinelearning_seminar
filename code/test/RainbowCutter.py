import numpy as np
import matplotlib.pyplot as plt


class RainbowCutter:
    """Cut colormeshes."""

    def __init__(self):
        pass

    def cut_mesh(self, mesh, color_x, color_y, color_z=2, theta=0):
        cut = mesh.copy()

        RGB = [mesh[:, :, 0], mesh[:, :, 1], mesh[:, :, 2]]

        x, y = self._rotate(RGB[color_x], RGB[color_y], theta)

        parabel = -20 * y * y + (2.0 * (RGB[color_z] - 0.3)) / np.cos(theta)

        mask = x < parabel

        cut[~mask] = 0

        return cut

    def _rotate(self, x, y, theta):
        """Rotate x and y by theta.

        Parameters:
            x, y: np.array
            theta: float in [0 ... 2 * pi]

        Returns:
            xr, xy: np.array
                x, y rotated by theta
        """
        xr = np.cos(theta) * x + np.sin(theta) * y
        yr = -np.sin(theta) * x + np.cos(theta) * y
        return xr, yr


def main():
    cutter = RainbowCutter()
    cut = cutter._cut_mesh_(np.zeros((100, 100, 3)), 0, 1, 2, np.pi / 4)
    print(cut)


if __name__ == "__main__":
    main()
