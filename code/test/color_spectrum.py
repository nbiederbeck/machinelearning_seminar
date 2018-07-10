import numpy as np
import matplotlib.pyplot as plt


class Spectrum:
    def __init__(self):
        size = 2
        self.fig = plt.figure(figsize=(size * 2, size * 3))
        self.N = 256
        self.colors = np.linspace(0, 1, self.N)
        self.white = np.ones([self.N, self.N, 3])
        self.black = np.zeros([self.N, self.N, 3])

    def _init_empty_colormeshes(self):
        self.red_green__black = self.black.copy()
        self.red_green__white = self.white.copy()

        self.blue_red__black = self.black.copy()
        self.blue_red__white = self.white.copy()

        self.green_blue__black = self.black.copy()
        self.green_blue__white = self.white.copy()

    def _colorize_colormeshes(self):
        self.r = 0
        self.g = 1
        self.b = 2

        for i in range(self.N):
            self.red_green__black[i, :, self.r] = self.red_green__black[
                :, i, self.g
            ] = self.colors.copy()
            self.red_green__white[i, :, self.r] = self.red_green__white[
                :, i, self.g
            ] = self.colors.copy()
            self.blue_red__black[i, :, self.b] = self.blue_red__black[
                :, i, self.r
            ] = self.colors.copy()
            self.blue_red__white[i, :, self.b] = self.blue_red__white[
                :, i, self.r
            ] = self.colors.copy()
            self.green_blue__black[i, :, self.g] = self.green_blue__black[
                :, i, self.b
            ] = self.colors.copy()
            self.green_blue__white[i, :, self.g] = self.green_blue__white[
                :, i, self.b
            ] = self.colors.copy()

        self.colormeshes = {
            "red green blue black": self.red_green__black,
            "red green blue white": self.red_green__white,
            "blue red green black": self.blue_red__black,
            "blue red green white": self.blue_red__white,
            "green blue red black": self.green_blue__black,
            "green blue red white": self.green_blue__white,
        }

    # def _cut_colormeshes(self, cuts=[0, 0, 0]):
    #     self.cut_colormeshes = {}
    #     for name in self.colormeshes:
    #         self.cut_colormeshes[name] = self.colormeshes[name].copy()
    #         mask_r = self.colormeshes[name][:, :, self.r] < cuts[self.r]
    #         mask_g = self.colormeshes[name][:, :, self.g] < cuts[self.g]
    #         mask_b = self.colormeshes[name][:, :, self.b] < cuts[self.b]
    #         self.cut_colormeshes[name][mask_r, self.r] = 0 if name.split()[3] == "white" else 1
    #         self.cut_colormeshes[name][mask_g, self.g] = 0 if name.split()[3] == "white" else 1
    #         self.cut_colormeshes[name][mask_b, self.b] = 0 if name.split()[3] == "white" else 1

    def get_colormesh(self, which="all"):
        if which == "all":
            return self.colormeshes
        else:
            return {which: self.colormeshes[which]}

    def plot_colormesh(self, path, which="all"):
        if which == "all":
            plots = 321
            for i, name in enumerate(self.colormeshes):
                ax = self.fig.add_subplot(plots + i)
                ax.imshow(self.colormeshes[name], origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(name.split()[0])
                ax.set_ylabel(name.split()[1])
                ax.set_title(
                    name.split()[2]
                    + "="
                    + ("0" if name.split()[3] == "black" else "255")
                )
        else:
            name = which
            ax = self.fig.add_subplot(111)
            ax.imshow(self.colormeshes[name], origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(name.split()[0])
            ax.set_ylabel(name.split()[1])
            ax.set_title(
                name.split()[2]
                + "="
                + ("0" if name.split()[3] == "black" else "255")
            )

        self.fig.tight_layout(pad=0)
        self.fig.savefig(path)
        self.fig.clf()

    # def plot_cut_colormesh(self, path, which="all"):
    #     if which == "all":
    #         plots = 321
    #         for i, name in enumerate(self.cut_colormeshes):
    #             ax = self.fig.add_subplot(plots+i)
    #             ax.imshow(self.cut_colormeshes[name], origin="lower")
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             ax.set_xlabel(name.split()[0])
    #             ax.set_ylabel(name.split()[1])
    #             ax.set_title(name.split()[2] + "=" + ("0" if name.split()[3] == "black" else "255"))
    #     else:
    #         name = which
    #         ax = self.fig.add_subplot(111)
    #         ax.imshow(self.cut_colormeshes[name], origin="lower")
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.set_xlabel(name.split()[0])
    #         ax.set_ylabel(name.split()[1])
    #         ax.set_title(name.split()[2] + "=" + ("0" if name.split()[3] == "black" else "255"))
    #     self.fig.tight_layout(pad=0)
    #     self.fig.savefig(path)
    #     self.fig.clf()


def main():
    spectrum = Spectrum()
    spectrum._init_empty_colormeshes()
    spectrum._colorize_colormeshes()
    spectrum.plot_colormesh("rgb.png")

    # spectrum._cut_colormeshes([0, 0, 0])
    # spectrum.plot_cut_colormesh("rgb_cut.png")


if __name__ == "__main__":
    main()
