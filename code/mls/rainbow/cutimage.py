import numpy as np
import matplotlib.pyplot as plt


class RainbowCutter:
    def __init__(self, N):
        self.r, self.b, self.g = np.meshgrid(
            np.linspace(0, 1, N), np.linspace(0, 1, N), np.linspace(0, 1, N)
        )

    def _rotate(self, x, y, theta):
        xr = np.cos(theta) * x + np.sin(theta) * y
        yr = -np.sin(theta) * x + np.cos(theta) * y
        return xr, yr

    def cut_function(self, x, y, x0, y0, theta, scale):

        x_rot, y_rot = self._rotate(x, y, theta)
        x0_rot, y0_rot = self._rotate(x0, y0, theta)

        x = x_rot - x0_rot
        y = y_rot - y0_rot

        mask = y < scale * (x * x)

        return mask

    def mask_cube(self):
        return self.cut_function(self.r, self.g, 1.1*self.b, 1.1*self.b,
                -1*np.pi/3.5
                , -30)


def main():
    N = 255
    point_of_interest= 4

    cutter = RainbowCutter(N)
    mask_cube = cutter.mask_cube()

    r, g = np.meshgrid(
            np.linspace(0, 1, N), np.linspace(0, 1, N)    
            )


    b = np.ones([N,N]) 
    b *= point_of_interest/N

    print('r.shape: ', r.shape)
    print('g.shape: ', g.shape)
    print('b.shape: ', b.shape)

    # r[~mask_cube[point_of_interest, :, :]] = 0
    # g[~mask_cube[point_of_interest, :, :]] = 0
    # b[~mask_cube[point_of_interest, :, :]] = 0
    # b[mask_cube[point_of_interest, :, :]] = point_of_interest/255
    # b=point_of_interest/255

    #this is now an RGB array, 100x100x3 that I want to display
    rgb = np.array([r,g,b]).T
    
    color_tuple = rgb.transpose((1,0,2)).reshape((rgb.shape[0]*rgb.shape[1],rgb.shape[2]))
    
    m = plt.pcolormesh(r, color=color_tuple, linewidth=0)
    m.set_array(None)
    plt.show()


    
    # fig = plt.figure()
    # ax = fig.add_subplot(311)
    # ax.scatter(
    #         r[mask_cube[:, :, point_of_interest]], 
    #         g[mask_cube[:, :, point_of_interest]]
    #         )
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    # ax = fig.add_subplot(312)
    # ax.scatter(
    #         r[mask_cube[:, point_of_interest, :]], 
    #         g[mask_cube[:, point_of_interest, :]]
    #         )
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    # ax = fig.add_subplot(313)
    # ax.scatter(
    #         r[mask_cube[point_of_interest, :, :]],
    #         g[mask_cube[point_of_interest, :, :]]
    #         )
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    # plt.show()

    # # # Alle nicht in der Maske null setzen
    # r[~mask_cube[point_of_interest, :, :]] = 0
    # g[~mask_cube[point_of_interest, :, :]] = 0
    # # r[~mask_cube[:, :, point_of_interest]] = 0
    # # g[~mask_cube[:, :, point_of_interest]] = 0
    # # # Irgendwie an das blaue mesh kommen ... waere wsl. besser das cutter.b zu
    # # # nehmen
    # b = cutter.b[:, :, 0] # richtig??
    # b += point_of_interest/N
    # # auch blaue maske null setzen?
    # # b[~mask_cube[:, :, point_of_interest]] = 0
    # b[~mask_cube[point_of_interest,: , :]] = 0
    
    # # alles in einem "Farben" array speichern
    # c = np.array([r, g, b]).transpose()
    # # grafisch darstellen
    # plt.imshow(c)
    # plt.show()
    


if __name__ == "__main__":
    main()
