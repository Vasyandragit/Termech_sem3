import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp


# Geometry
R = 1
xo1 = yo1 = 0
h = OA = 0.5
l0 = 0.6
s = 0.4

# Coords
t = np.linspace(0, 3, 100)
phi = np.linspace(0, 2 * np.pi, 100)
xo = l0 + s + R + np.sin(t)
yo = 1
xa = xo - h * np.sin(phi / 2)
ya = R - h * np.cos(phi / 2)

# Window
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")

# Axes
linex, = ax.plot([0, 5], [0, 0], color=[0, 0, 0])
liney, = ax.plot([0, 0], [0, 3], color=[0, 0, 0])
ArrowX = np.array([-0.2, 0, -0.2])
ArrowY = np.array([0.1, 0, -0.1])
ArrowOX = ax.plot(ArrowX+5, ArrowY, color=[0, 0, 0])
ArrowOY = ax.plot(ArrowY, ArrowX+3, color=[0, 0, 0])

# Circle
O = ax.plot(xo[0], yo, 'o', color=[0, 0, 0])[0]
A = ax.plot(xa[0], ya[0], 'o', color=[0, 0, 0])[0]
circle = ax.plot(xo[0] + R * np.cos(phi), yo + R * np.sin(phi), color=[0, 0, 0])[0]

# Spring
n = 12
hs = 0.1
xs = np.linspace(0, 1, 2 * n + 1)
# sin – наиболее лёгкий способ создания последовательности 0,1,0,-1,0…
ys = hs * np.sin(np.pi / 2 * np.arange(2 * n + 1))
spring = ax.plot(xs * (xo1 + xo[0]), ys + yo, color=[0, 0, 0])[0]

# Ticker
ticker = ax.plot([xo[0], xa[0]], [yo, ya[0]], color=[0, 0, 0])[0]
        
# Action per frame
def kadr(i):

    O.set_data([xo[i]], [yo])
    A.set_data([xa[i]], [ya[i]])
    circle.set_data(xo[i] + R * np.cos(phi), yo + R * np.sin(phi))
    spring.set_data(xs * (xo1 + xo[i]), ys + yo)
    ticker.set_data([xo[i], xa[i]], [yo, ya[i]])
    return [O, A, circle, spring, ticker]


kino = FuncAnimation(fig, kadr, frames=len(t), interval=60, blit=True)

plt.show()