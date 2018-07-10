import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import imageio
from tqdm import tqdm
from mls.rainbow.cutter import RainbowCutter
from mls.rainbow.painter import RainbowPainter

N = 256
colors = np.linspace(0, 1, N)
rg = np.zeros((N, N, 3))
gb = np.zeros((N, N, 3))
br = np.zeros((N, N, 3))
figsize_inches = 4

r = 0
g = 1
b = 2

cutter = RainbowCutter()
painter = RainbowPainter()
N = painter.N


def layer(mesh):
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize_inches, figsize_inches)
    ax.imshow(mesh, origin="lower", interpolation=None)
    ax.set_xticks([])
    ax.set_yticks(ax.get_xticks())
    fig.tight_layout(pad=0)
    return fig


rg_ = (
    layer(cutter.cut_mesh(painter.paint_mesh(r, g, b, i), r, g, b, np.pi / 4))
    for i in range(N)
)  # A generator for the red green side of the color cube

gb_ = (
    layer(cutter.cut_mesh(painter.paint_mesh(g, b, r, i), r, g, b, np.pi / 4))
    for i in range(N)
)  # A generator for the green blue side of the color cube
br_ = (
    layer(cutter.cut_mesh(painter.paint_mesh(b, r, g, i), r, g, b, np.pi / 4))
    for i in range(N)
)  # A generator for the blue red side of the color cube

sites = [rg_, gb_, br_]

# Plotting and saving each generated layer
for l, layers in zip(
    ["rg_", "gb_", "br_"], tqdm(sites, ascii=True, desc="Sites", total=3)
):
    for i, fig in enumerate(tqdm(layers, ascii=True, desc="GIF", total=N)):
        output = BytesIO()
        fig.savefig("build/{0}{1:03d}.png".format(l, i), dpi=N / figsize_inches)
        plt.close(fig)

print("DONE")
