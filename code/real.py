import numpy as np
import matplotlib.pyplot as plt

def rotate(x,y, theta):
    xs = np.cos(theta)*x + np.sin(theta)*y
    ys = -np.sin(theta)*x + np.cos(theta)*y
    return xs ,ys

def squared(x, a):
    return a*(x)**2

def rotate_func(x, y, bias_x, bias_y, theta):
    xs, ys = rotate(x, y, theta)
    x_0s, y_0s = rotate(bias_x, bias_y, theta)
    xs -= x_0s
    ys -= y_0s
    resp = []
    for x_i, y_i in zip(xs, ys):
        resp.append( squared(x_i, 4) > y_i )
    return np.array(resp)

x, y = np.meshgrid(np.linspace(0,1,256), np.linspace(0,1,256))
x = x.flatten()
y = y.flatten()

mask = rotate_func(x, y, 0.1, 0.5, -np.pi/4)

plt.scatter(x, y, marker='.')
plt.scatter(x[mask], y[mask], marker='.')

plt.show()
