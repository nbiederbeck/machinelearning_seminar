import numpy as np
import matplotlib.pyplot as plt

def i_f(x, y):
    resp = []
    for x_i, y_i in zip(x, y):
        resp.append( 4*x_i**2 > y_i )
    return resp

def rotate(x,y, theta):
    xs = np.cos(theta)*x + np.sin(theta)*y
    ys = -np.sin(theta)*x + np.cos(theta)*y
    return xs ,ys

x, y = np.meshgrid(np.linspace(0,1,256), np.linspace(0,1,256))
x = x.flatten()
y = y.flatten()


xs, ys = rotate(x, y, -np.pi/3)
mask = i_f(xs,ys)
mask = np.array(mask)

plt.scatter(x, y, marker='.', label='unrotierter Raum1')
plt.scatter(x[mask], y[mask], marker='.', label='unrotiertet Raum3')
plt.xlim(0,1)
plt.legend()

plt.show()
