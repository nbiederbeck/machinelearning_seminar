from mls.rainbow.cutter import RainbowCutter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as pil

import fire


def show_pictures(cutter, path):
    im = np.array(pil.open(path), dtype=np.uint16)
    im = im / 255

    fig, axes = plt.subplots(nrows=2)
    axes[0].imshow(im)

    cut = cutter.cut_image(im)
    axes[1].imshow(cut)

    fig.tight_layout(pad=0)

    plt.show()


def main(path):
    cutter = RainbowCutter(
        N=256,
        theta=-np.pi / 2.5,
        scale=-2.8,
        xshift=1.2,
        yshift=1.2,
        xbias=0.0,
        ybias=0.2,
    )

    show_pictures(cutter, path)


if __name__ == "__main__":
    fire.Fire(main)
