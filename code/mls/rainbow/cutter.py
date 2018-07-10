import numpy as np
import matplotlib.pyplot as plt


class RainbowCutter:
    """Cut colormeshes."""

    def __init__(self):
        pass

    def cut_mesh(self, mesh, color_x, color_y, color_z=2, theta=0):
        """Cut colormesh.

        Parameters:
            mesh: np.ndarray(256, 256, 3)
            color_x, color_y, color_z: int 0..2
                RGB tuple
            theta: float
                rotation angle
        Returns:
            cut: np.ndarray(256, 256, 3)
                mesh with cuts applied.
        """
        RGB = [mesh[:, :, 0], mesh[:, :, 1], mesh[:, :, 2]]

        x_arr, y_arr = self._rotate(RGB[color_x], RGB[color_y], -theta)
        x_arr = x_arr.flatten()
        y_arr = y_arr.flatten()
        z = RGB[color_z][0, 0]

        mask = []

        # for x, y in zip(x_arr, y_arr):
        #     mask.append(y > 4 * (x ** 2) + z)
        mask = y_arr > 4 * (x_arr ** 2) + z

        mask = np.array(mask).reshape(255, 255)

        cut = mesh.copy()

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
    cut = cutter.cut_mesh(np.zeros((100, 100, 3)), 0, 1, 2, np.pi / 4)
    print(cut)


if __name__ == "__main__":
    main()
