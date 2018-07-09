import numpy as np
import matplotlib.pyplot as plt


def rotated(x, y, phi):
    xs = np.cos(phi) * x + np.sin(phi) * y
    ys = - np.sin(phi) * x + np.cos(phi) * y
    return xs, ys

def squared(x):
    return 4 * x ** 2


xshift = 0.2 
yshift = 0.2
theta = np.pi/2.6

fig = plt.figure()

x = np.linspace(-1,1,31)
y = squared(x)
ax = fig.add_subplot(111)
ax.plot(x + xshift, y + yshift, 'x-', label='org')

ys = squared(x)
xs , ys = rotated(x, ys, theta)
ax.plot(xs + xshift, ys + yshift, 'x-', label='sqrrotated')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.legend()


# comparison
x += xshift
y += yshift
xs += xshift
ys += yshift

# mask = (xs > 1) + (xs < 0) + (ys > 1) + (ys < 0)

# x_random = np.random.uniform(-1.5,1.5,5000)
# y_random = np.random.uniform(-1.5,1.5,5000)
x_random, y_random=np.meshgrid(np.linspace(-1,1.0,103),
        np.linspace(-yshift,1.0,103))
x_random = x_random.flatten()
y_random = y_random.flatten()
x_lin, y_lin= np.meshgrid(np.linspace(-1,1.5,100),np.linspace(-1,1.5,100))


mask = y_lin> squared(x_lin)
xs , ys = rotated(x_lin, y_lin, theta)

from scipy.spatial.distance import cdist
a = np.array([ xs[mask].flatten() ,  ys[mask].flatten() ]).T
b = np.array([ x_random ,  y_random ]).T
dist = cdist(b, a)
Mask = np.min(dist,axis=1) < 0.1

x_random = x_random[Mask.flatten()]
y_random = y_random[Mask.flatten()]

ax.plot(xs[mask] + xshift, ys[mask] + yshift, 'x', label='sqrrotated')
ax.scatter(x_random+ xshift, y_random+ yshift, marker=".")
# ax.scatter(x_random, y_random, marker=".")



plt.show()

