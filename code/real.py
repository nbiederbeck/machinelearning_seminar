import numpy as np
import matplotlib.pyplot as plt

def rotate(x,y, theta):
    xs = np.cos(theta)*x + np.sin(theta)*y
    ys = -np.sin(theta)*x + np.cos(theta)*y
    return xs ,ys

def squared(x, x_0, y_0, a):
    return a*(x-x_0)**2 + y_0

def rotate_func(x, y, theta):
    xs, ys = rotate(x, y, theta)
    x_0 = 0.1
    y_0 = 0.5
    x_0s, y_0s = rotate(x_0, y_0, theta)
    resp = []
    for x_i, y_i in zip(xs, ys):
        resp.append( squared(x_i, x_0s, y_0s, 4) > y_i )
    return np.array(resp)

x, y = np.meshgrid(np.linspace(0,1,256), np.linspace(0,1,256))
x = x.flatten()
y = y.flatten()

mask = rotate_func(x,y, -np.pi/4)

plt.scatter(x, y, marker='.')
plt.scatter(x[mask], y[mask], marker='.')

plt.show()
