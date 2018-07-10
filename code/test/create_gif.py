import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import imageio
from io import BytesIO

snapshots = []

path = "build/kepler_moon/"

print("Reading data")
T = np.genfromtxt(path + "T.txt")
print(".", end=" ")
r_i = np.genfromtxt(path + "r_i.txt")
print(".", end=" ")
v_i = np.genfromtxt(path + "v_i.txt")
print(".", end=" ")
r_i_moon = np.genfromtxt(path + "r_i_moon.txt")
print(".", end=" ")
v_i_moon = np.genfromtxt(path + "v_i_moon.txt")
print(".")

print("Done reading data")


def make_snapshot(i=0):
    title = "Keplerbahnen"
    fig = plt.figure(figsize=(5.78, 3.57))
    ax = fig.add_subplot(111)
    ax.scatter(0, 0, label="Sonne", c="C1", marker="o", s=100)

    ax.plot(r_i[0, :][:i], r_i[0 + 1, :][:i], label="Planetenbahn", c="C0")
    ax.scatter(
        r_i[0, :][0], r_i[0 + 1, :][0], label="Startpunkt", marker="x", c="C0"
    )

    ax.plot(
        r_i_moon[0, :][:i], r_i_moon[0 + 1, :][:i], label="Mondbahn", c="C2"
    )
    ax.scatter(r_i_moon[0, :][0], r_i_moon[0 + 1, :][0], marker="x", c="C2")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)

    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])

    fig.tight_layout(pad=0.0)

    return fig


print("Creating plots")
snapshots = (make_snapshot(100 * i) for i in range(1000))


def make_gif(figures, filename, fps=10, **kwargs):
    images = []
    for fig in figures:
        output = BytesIO()
        fig.savefig(output)
        plt.close(fig)
        output.seek(0)
        images.append(imageio.imread(output))
    imageio.mimsave(filename, images, fps=fps, **kwargs)


print("Creating GIF")
make_gif(snapshots, "build/kepler.gif", fps=30)
print("Done")
