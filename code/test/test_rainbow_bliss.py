from RainbowPainter import RainbowPainter
from RainbowCutter import RainbowCutter
from RainbowPlotter import RainbowPlotter
from PIL import Image as pil
from numpy import asarray as array
import matplotlib.pyplot as plt


def main():
    pi = 3.1415962

    im = array(pil.open("../../../../Pictures/bliss.png"))

    painter = RainbowPainter()
    cutter = RainbowCutter()
    plotter = RainbowPlotter()

    r, g, b = 0, 1, 2

    cut = cutter.cut_mesh(im, r, g, b, pi / 4)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(im)
    ax2.imshow(cut)
    fig.tight_layout()
    fig.savefig("build/bliss.png", dpi=256)


if __name__ == "__main__":
    main()
