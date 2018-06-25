import matplotlib.pyplot as plt


class RainbowPlotter:
    """Plot colormeshes."""

    def __init__(self, figsize_inches=4):
        self.fig = None
        self.ax = None
        self.figsize_inches = figsize_inches

    def plot_colormesh(self, mesh):
        """Plot colormesh.

        Parameters:
            mesh: numpy.ndarray((x, y, 3))
                Colormesh to plot.
        Returns:
            fig, ax: matplotlib.pyplot.(figure,axis)
        """
        fig, ax = plt.subplots()
        fig.set_size_inches(self.figsize_inches, self.figsize_inches)
        ax.imshow(mesh, origin="lower", interpolation=None)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0)
        return fig, ax


def main():
    from numpy import zeros

    plotter = RainbowPlotter()
    fig, ax = plotter.plot_colormesh(zeros((100, 100, 3)))
    fig.show()


if __name__ == "__main__":
    main()
