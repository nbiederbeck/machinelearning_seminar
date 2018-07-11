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

class plot_cube:
    def __init__(self, mask):
        self.mask = mask 

    def plot_plane(self, xaxis, yaxis, plane):
        x0, x1 = np.meshgrid(
            np.linspace(0, 1, 255), 
            np.linspace(0, 1, 255)    
            )
        cut = np.ones([255,255])  * plane / 255

        x0[~self.mask[plane, :, :]] = 0
        x1[~self.mask[plane, :, :]] = 0
        cut[~self.mask[plane, :, :]] = 0


        rgb = np.array([x0,x1, cut]).T
        
        color_tuple = rgb.transpose((1,0,2)).reshape(
                (rgb.shape[0]*rgb.shape[1],rgb.shape[2]))
        
        m = plt.pcolormesh(x0, color=color_tuple, linewidth=0)
        m.set_array(None)
        plt.show()



def main():
    N = 255

    cutter = RainbowCutter(N)
    mask_cube = cutter.mask_cube()

    pltter = plot_cube(mask_cube)

    pltter.plot_plane(1,1,10)
    pltter.plot_plane(1,1,100)
    pltter.plot_plane(1,1,250)

if __name__ == "__main__":
    main()
