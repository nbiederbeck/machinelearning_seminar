import numpy as np


class RainbowCutter:
    def __init__(self, N=256, theta=-np.pi / 3.5, scale=-10.0, 
            xshift=1.9, yshift=1.9, xbias=-0.1, ybias=0.07,):
        """Initalize a 3d meshgrid with N points per axis.

        Parameters:
            N: int
                Discretisation of axes
            theta: float 0..2pi
                Rotation angle
            scale: float
                Scaling factor for parabola
            xshift: float
            yshift: float
            xbias: float
            ybias: float
        """
        self.theta = theta
        self.scale = scale
        self.xshift = xshift
        self.yshift = yshift
        self.xbias = xbias
        self.ybias = ybias

    def _rotate(self, x, y, theta):
        """Rotate x and y tuple by theta at coordinate origin.

        Parameters:
            x, y: np.ndarray
                Arrays to rotate
            theta: float
                Rotation angle

        Returns:
            xr, xy: np.ndarray
                Rotated input arrays
        """
        xr = np.cos(theta) * x + np.sin(theta) * y
        yr = -np.sin(theta) * x + np.cos(theta) * y
        return xr, yr

    def cut_function(self, x, y, z):
        """Separate background from signal.

        Parameters:
            x, y, z: np.ndarray
                RGB colors

        Returns:
            mask: np.ndarray
                True/False array containing information about
                which points to keep or discard.
        """
        x0 = z * self.xshift + self.xbias
        y0 = z * self.yshift + self.ybias

        x_rot, y_rot = self._rotate(x, y, self.theta)
        x0_rot, y0_rot = self._rotate(x0, y0, self.theta)

        x = x_rot - x0_rot
        y = y_rot - y0_rot

        mask = y < self.scale * (x * x)

        return mask

    def cut_image(self, im):
        """Apply mask to image.

        Parameters:
            im: np.ndarray(PIL.Image.open())
                png/jpg-image to cut

        Returns:
            im: np.ndarray
                Cutted image
        """
        r = im[:, :, 0]
        g = im[:, :, 1]
        b = im[:, :, 2]
        mask = self.cut_function(r, g, b)
        im[~mask] = 0
        return im
