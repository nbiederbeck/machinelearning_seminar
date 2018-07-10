from RainbowPainter import RainbowPainter
from RainbowCutter import RainbowCutter
from RainbowPlotter import RainbowPlotter


def main():
    pi = 3.1415962

    painter = RainbowPainter()
    cutter = RainbowCutter()
    plotter = RainbowPlotter()

    r, g, b = 0, 1, 2

    mesh = painter.paint_mesh(r, g, b, 0)
    cut = cutter.cut_mesh(mesh, r, g, b, pi / 4)

    fig, ax = plotter.plot_colormesh(cut)
    fig.savefig("build/test_rainbow.png", dpi=256 / 4)


if __name__ == "__main__":
    main()
