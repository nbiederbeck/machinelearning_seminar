import numpy as np
import matplotlib.pyplot as plt


class RainbowCutter:
    """Cut colormeshes."""

    def __init__(self):
        pass

    def cut_mesh(
        self,
        mesh,
        color_x,
        color_y,
        color_z=2,
        theta=0,
        offset_x=0,
        offset_y=0,
        scale=1,
    ):
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
        x0, y0 = self._rotate(offset_x, offset_y, -theta)

        x = x_arr - x0
        y = y_arr - y0

        x = x.flatten()
        y = y.flatten()

        mask = y < scale * (x * x)

        mask = mask.reshape(mesh.shape[0], mesh.shape[1])

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
