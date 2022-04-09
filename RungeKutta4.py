

# modified from Source - https://rosettacode.org/wiki/Runge-Kutta_method#Python

# method 1
# def rk4(f):
#     return lambda t, y, dt: (
#         lambda dy1: (
#             lambda dy2: (
#                 lambda dy3: (
#                     lambda dy4: (dy1 + 2* dy2 + 2 * dy3 + dy4) / 6
#                 )(dt * f(t + dt, y + dy3))
#             )(dt * f(t + dt / 2, y + dy2 / 2))
#         )(dt * f(t + dt / 2, y + dy1 / 2))
#     )(dt * f(t, y))
#
#
# def theory(t): return (t ** 2 + 4) ** 2 / 16
#
#
#
# dy = RK4(lambda t, y: t * sqrt(y))
#
# t, y, dt = 0., 1., .1
# while t <= 10:
#     if abs(round(t) - t) < 1e-5:
#         print("y(%2.1f)\t= %4.6f \t error: %4.6g" % (t, y, abs(y - theory(t))))
#     t, y = t + dt, y + dy(t, y, dt)
#


# method 2
import numpy as np
import random as rnd

def RK4(f, t0, x0, t1, n0, h):

    # n0 = time step


    n = np.int((t1-t0)/ n0)
    vt = np.zeros(n)
    vx = np.zeros((n, x0.shape[0]))

    vt[0] = t = t0
    vx[0] = x = x0
    for i in range(0, n ):
        k1 = h * f(t, x)
        k2 = h * f(t + 0.5 * h, x + 0.5 * k1)
        k3 = h * f(t+ 0.5 * h, x + 0.5 * k2)
        k4 = h * f(t + h, x + k3)
        vt[i] = t = t0 + i * h
        vx[i] = x = x + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return vt, vx


def RK4_error(f, t0, x0, t1, n0, h, error, addnoise = True):




    n = np.int((t1-t0)/ n0)
    vt = np.zeros(n)
    vx = np.zeros((n, x0.shape[0]))

    if addnoise:
        ran = error
    else:
        ran = np.zeros([n,40])

    vt[0] = t = t0
    vx[0] = x = x0
    for i in range(0, n ):
        k1 = h * f(t, x)
        k2 = h * f(t + 0.5 * h, x + 0.5 * k1)
        k3 = h * f(t+ 0.5 * h, x + 0.5 * k2)
        k4 = h * f(t + h, x + k3)
        vt[i] = t = t0 + i * h
        vx[i] = x = x + (k1 + k2 + k2 + k3 + k3 + k4) / 6

        # add error
        x = x + ran[i,:]
    return vt, vx





def get_random(t1, n0):
    ran = np.zeros([int(t1/n0),40])
    j = 0
    for j in range(0,40):
        for i in np.arange(0, int(t1/n0)):
            ran[int(i), j] = rnd.gauss(0,1)

    return ran




















