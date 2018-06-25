import numpy as np


class RainbowPainter:
    """Paint colormeshes."""

    def __init__(self, N=255):
        self.N = N
        self.colors = np.linspace(0, 1, self.N)

    def paint_mesh(self, color_x, color_y, color_z, z_value):
        """Paint colormesh.

        Parameters:
            color_x, color_y: int \in [0, 1, 2]
                Desribes which color of R, G, B on x-axis and y-axis
            color_z: int \in [0, 1, 2]
                Desribes which color of R, G, B on z-axis
            z_value: int \in [0 ... 255]
                RGB value of z-axis color

        Returns:
            mesh: np.ndarray((self.N, self.N, 3))
                Painted colormesh ready for matplotlib.pyplot.imshow,
                meaning values are 0 < x < 1.
        """
        mesh = np.zeros((self.N, self.N, 3))
        for i in range(self.N):
            mesh[i, :, color_x] = self.colors.copy()
            mesh[:, i, color_y] = self.colors.copy()
        mesh[:, :, color_z] = z_value / self.N
        return mesh


def main():
    r, g, b = 0, 1, 2
    painter = RainbowPainter()
    mesh = painter.paint_mesh(r, g, b, 255)
    print(mesh)


if __name__ == "__main__":
    main()
