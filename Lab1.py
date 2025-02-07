import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def Rot2D(X, Y, phi):
    X_new = X * np.cos(phi) - Y * np.sin(phi)
    Y_new = X * np.sin(phi) + Y * np.cos(phi)
    return X_new, Y_new

# Coords
t = sp.Symbol('t')
r = 1 + sp.sin(5 * t)
phi = t + 0.3 * sp.sin(30 * t)
x = r * sp.cos(phi)
y = r * sp.sin(phi)

# Velocities
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)

# Funcs
F_x = sp.lambdify(t, x, "numpy")
F_y = sp.lambdify(t, y, "numpy")
F_Vx = sp.lambdify(t, Vx, "numpy")
F_Vy = sp.lambdify(t, Vy, "numpy")
F_Wx = sp.lambdify(t, Wx, "numpy")
F_Wy = sp.lambdify(t, Wy, "numpy")

# Parametres
t = np.linspace(0, 10, 10001)   
x = F_x(t)
y = F_y(t)
Vx = F_Vx(t)
Vy = F_Vy(t)
Wx = F_Wx(t)
Wy = F_Wy(t)
phi = np.arctan2(Vy, Vx)

# Window
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y)
ax.set(xlim=[-3, 3], ylim=[-3, 3])

# Arrows and point
X_arr = np.array([-0.15, 0, -0.15])
Y_arr = np.array([0.05, 0, -0.05])
RX_arr, RY_arr = Rot2D(X_arr, Y_arr, phi[0])
V_arr =  ax.plot(x[0] + Vx[0] + RX_arr, y[0] + Vy[0] + RY_arr, color='r')[0]
W_arr = ax.plot(x[0] + Wx[0] + RX_arr, y[0] + Wy[0] + RY_arr, color='g')[0]
V_line = ax.plot([x[0], x[0] + Vx[0]], [y[0], y[0] + Vy[0]], color='r')[0]
W_line = ax.plot([x[0], x[0] + Wx[0]], [y[0], y[0] + Wy[0]], color ='g')[0]
Curve_line = ax.plot([0, x[0]], [0, y[0]], color='b')[0]
Point = ax.plot(x[0], y[0], 'o', color='b')[0]

# Action per frame
def kadr(i):
    RX_arr, RY_arr = Rot2D(X_arr, Y_arr, phi[i])
    Point.set_data([x[i]], [y[i]])
    V_line.set_data([x[i], x[i] + Vx[i]], [y[i], y[i] + Vy[i]])
    W_line.set_data([x[i], x[i] + Wx[i]], [y[i], y[i] + Wy[i]])
    Curve_line.set_data([0, x[i]], [0, y[i]])
    V_arr.set_data(x[i] + Vx[i] + RX_arr, y[i] + Vy[i] + RY_arr)
    W_arr.set_data(x[i] + Wx[i] + RX_arr, y[i] + Wy[i] + RY_arr)
    return [Point, V_line, V_arr, W_line, W_arr, Curve_line]

kino = FuncAnimation(fig, kadr, interval = t[1] - t[0], frames=len(t))
plt.show()