import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import imageio
from tqdm import tqdm
from RainbowCutter import RainbowCutter
from RainbowPainter import RainbowPainter

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

# for i in tqdm(range(N), ascii=True, desc="coloring"):
#     rg[i, :, g] = colors.copy()
#     rg[:, i, r] = colors.copy()
    # gb[i, :, g] = colors.copy()
    # gb[:, i, b] = colors.copy()
    # br[i, :, b] = colors.copy()
    # br[:, i, r] = colors.copy()


# def make_mesh(mesh, c, i):
#     mesh[:, :, c] = i / 255
#     return mesh


# def rotate(x, y, theta):
#     xr = np.cos(theta) * x + np.sin(theta) * y
#     yr = -np.sin(theta) * x + np.cos(theta) * y
#     return xr, yr


# def cut_mesh(mesh):
#     theta = np.pi / 4
#     blue = mesh[0, 0, b]
#     R = mesh[:, :, r]
#     G = mesh[:, :, g]
#     a = 2.0
#     offset = 0.3
#     Rr, Gr = rotate(R, G, theta)
#     parabel = -20 * (Gr) ** 2 + (a * (blue - offset)) / np.cos(theta)
#     mask = Rr < parabel
#     cut = mesh.copy()
#     cut[~mask] = 0
#     return cut


def layer(mesh):
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize_inches, figsize_inches)
    ax.imshow(mesh, origin="lower", interpolation=None)
    ax.set_xticks([])
    ax.set_yticks(ax.get_xticks())
    fig.tight_layout(pad=0)
    return fig


# def make_gif(figures, filename, fps=10, **kwargs):
#     images = []
#     for fig in tqdm(figures, ascii=True, desc="GIF", total=N):
#         output = BytesIO()
#         fig.savefig(output)
#         plt.close(fig)
#         output.seek(0)
#         images.append(imageio.imread(output))
#     imageio.mimsave(filename, images, fps=fps, **kwargs)


rg_ = (layer(cutter.cut_mesh(painter.paint_mesh(r, g, b, i), r, g)) for i in range(N))
gb_ = (layer(cutter.cut_mesh(painter.paint_mesh(g, b, r, i), r, g)) for i in range(N))
br_ = (layer(cutter.cut_mesh(painter.paint_mesh(b, r, g, i), r, g)) for i in range(N))
# gb_ = (layer(cut_mesh(make_mesh(gb, r, i))) for i in range(N))
# br_ = (layer(cut_mesh(make_mesh(br, g, i))) for i in range(N))

# sites = [rg_, gb_, br_]
sites = [rg_]

for l, layers in zip(
    ["rg_", "gb_", "br_"], tqdm(sites, ascii=True, desc="Sites", total=3)
):
    for i, fig in enumerate(tqdm(layers, ascii=True, desc="GIF", total=N)):
        output = BytesIO()
        fig.savefig("build/{0}{1:03d}.png".format(l, i), dpi=N / figsize_inches)
        plt.close(fig)

print("DONE")
