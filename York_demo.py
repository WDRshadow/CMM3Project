# import
import math
import numpy as np
import matplotlib.pyplot as plt
from random import gauss


# the func below is extends the classical Euler method
# 'r' is random numbers with the standard Gaussian probability
# 'u' is the (known) velocity components evaluated at the particle position
def Euler_method(Xp, u, h, D):
    # create a gauss random dx
    r = gauss(0, 1)
    # equation 6
    Xp += u * h + math.sqrt(2 * D) * math.sqrt(h) * r
    return Xp


# this func is used to update the particles after a step time
def go_a_step(x_type0, y_type0, x_type1, y_type1, u, h, D):
    for i in range(len(x_type0)):
        x_type0[i] = Euler_method(x_type0[i], u, h, D)
        y_type0[i] = Euler_method(y_type0[i], u, h, D)
    for i in range(len(x_type1)):
        x_type1[i] = Euler_method(x_type1[i], u, h, D)
        y_type1[i] = Euler_method(y_type1[i], u, h, D)
    return x_type0, y_type0, x_type1, y_type1


# this func is used to setup the particles in one traverse
def setup(x_min, x_max, y_min, y_max, Np):
    # init the list of blue and red particles
    x_type1 = []
    y_type1 = []
    x_type0 = []
    y_type0 = []
    # use np.random to init the particles
    for i in range(Np):
        # temp val for a random particle
        tx = np.random.uniform(x_min, x_max)
        ty = np.random.uniform(y_min, y_max)
        # select the color of this particle
        if np.sqrt(math.pow(tx, 2) + math.pow(ty, 2)) < 0.3:
            x_type1.append(tx)
            y_type1.append(ty)
        else:
            x_type0.append(tx)
            y_type0.append(ty)
    return x_type0, y_type0, x_type1, y_type1


# this func is used to show the plot
def show(x_red, y_red, x_blue, y_blue):
    plt.scatter(x_red, y_red, s=0.5, color='red')
    plt.scatter(x_blue, y_blue, s=0.5, color='blue')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# init data and main code
if __name__ == '__main__':
    # ----------------------------------------
    # init conditions
    # time set
    time_max = 0.4
    dt = 0.0005
    step = time_max / dt
    # diffusivity
    diff = 0.01
    # domain size
    xMin = -1
    xMax = 1
    yMin = -1
    yMax = 1
    # number of point of grid
    Nx = 64
    Ny = 64
    # number of particles
    N_particles = 65536
    # init velocity
    vel = 0
    # -----------------------------------------
    # main code
    # setup the init list of particles
    x_val0, y_val0, x_val1, y_val1 = setup(xMin, xMax, yMin, yMax, N_particles)
    # show init grid
    show(x_val0, y_val0, x_val1, y_val1)
    # cycle in Classic Euler Method step by step
    for x in range(int(step)):
        x_val0, y_val0, x_val1, y_val1 = go_a_step(x_val0, y_val0, x_val1, y_val1, vel, dt, diff)
        show(x_val0, y_val0, x_val1, y_val1)
