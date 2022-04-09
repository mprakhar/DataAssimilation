
# modified from from wkipedia https://en.wikipedia.org/wiki/Lorenz_96_model

from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import RungeKutta4 as rk
import random as rnd


# these are our constants
N = 40  # number of variables
F = 9 # forcing

x0 = F*np.ones(N) # initial state (equilibrium)
x0[19] += 0.01 # add small perturbation to 20th variable

h = 0.01
t1 = 730
n0 = 0.05


def plot():
    # plot first three variables
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x[:,0],x[:,1],x[:,2])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    plt.title("N:"+str(N) +", F:"+str(F) + ", h:"+str(h) + ", n0:" + str(n0)  + ", t1:" + str(t1))
    plt.show()



def plot_error():
    # plot error
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(np.arange(0, int(t1), n0)[0:-1], error_developing_rate, color="red", lw=1, alpha=0.3, label="RMSE", marker='+')
    plt.title("N:"+str(N) +", F:"+str(F) + ", h:"+str(h) + ", n0:" + str(n0)  + ", t1:" + str(t1))
    plt.show()


def Lorenz96(t, x):

  # compute state derivatives
  d = np.zeros(N)
  # first the 3 edge cases: i=1,2,N
  d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
  d[1] = (x[2] - x[N-1]) * x[0]- x[1]
  d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
  # then the general case
  for i in range(2, N-1):
      d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
  # add the forcing term
  d = d + F

  # return the state derivatives
  return d



# real model values
t, x = rk.RK4(f = Lorenz96, t0 = 0, x0 = x0, t1 = t1 , n0 = n0, h = h)
x_real = x


# get second half of data
x_half = x_real[x_real.shape[0]/2:,:]

#simulate 1000 noise data
x_simnoise = np.zeros([1000,x_real.shape[0],x_real.shape[1]])
x_error = np.zeros([1000,x_real.shape[0],x_real.shape[1]])
for i in range(0,1000):

    # get pre generated error
    ran = np.zeros([40])
    for j in np.arange(0, 40):
        ran[j] = rnd.gauss(0, 1)

    # add random noise to xintegrated dimensions
    x_noise = x0 + ran

    # incorrect code
    #t_err, x_noise = rk.RK4_error(f = Lorenz96, t0 = 0, x0 = x0, t1 = t1 , n0 = n0, h = h, error= ran, addnoise = True)

    t_noise, x_noisy = rk.RK4(f=Lorenz96, t0=0, x0=x_noise, t1=t1, n0=n0, h=h)

    x_simnoise[i] = x_noisy
    x_error[i] = np.abs(x_simnoise[i] - x_real)



# get RMSE
diff = np.abs(x_noise - x)
error_developing_rate = diff[1:,5]/diff[0:-1,5]

