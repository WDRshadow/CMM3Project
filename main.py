# import
import math
import numpy as np
import matplotlib.pyplot as plt
from random import gauss
import matplotlib as mpl
from matplotlib.colors import ListedColormap


# for easier setting, I put all the code into a class, developed by York and The Kite
class CMM3(object):
    def __init__(self):
        # ----------------------------------------
        # init conditions / interface
        # time set, "h" is a step time
        self.time_max = 0.4
        self.h = 0.05
        # diffusivity
        self.D = 0.01
        # domain size
        self.x_min = -1
        self.x_max = 1
        self.y_min = -1
        self.y_max = 1
        # number of point of grid
        self.Nx = 64
        self.Ny = 64
        # number of particles
        self.Np = 65536
        # init velocity
        self.vel = 0
        # -----------------------------------------
        # for temp particle position data save
        self.x_blue = []
        self.y_blue = []
        self.x_red = []
        self.y_red = []

    # the func below is extends the classical Euler method, built by York
    def Euler_method(self, Xp):
        # create a gauss random dx
        r = gauss(0, 1)
        # equation 6
        # 'r' is random numbers with the standard Gaussian probability
        # 'vel' is the (known) velocity components evaluated at the particle position
        Xp += self.vel * self.h + math.sqrt(2 * self.D) * math.sqrt(self.h) * r
        return Xp

    # this func is used to update the particles after a step time, built by York
    def go_a_step(self):
        for i in range(len(self.x_blue)):
            self.x_blue[i] = self.Euler_method(self.x_blue[i])
            self.y_blue[i] = self.Euler_method(self.y_blue[i])
        for i in range(len(self.x_red)):
            self.x_red[i] = self.Euler_method(self.x_red[i])
            self.y_red[i] = self.Euler_method(self.y_red[i])

    # this func is used to setup the particles in one traverse, built by York
    def setup(self):
        # use np.random to init the particles
        for i in range(self.Np):
            # temp val for a random particle
            tx = np.random.uniform(self.x_min, self.x_max)
            ty = np.random.uniform(self.y_min, self.y_max)
            # select the color of this particle
            if math.sqrt(math.pow(tx, 2) + math.pow(ty, 2)) < 0.3:
                self.x_blue.append(tx)
                self.y_blue.append(ty)
            else:
                self.x_red.append(tx)
                self.y_red.append(ty)

    # this func is used to show the plot, built by The Kite
    def show(self):
        cmap = ListedColormap(["red", "blue"])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        fig = plt.figure(figsize=(10, 8))
        ax2 = fig.add_subplot(111)
        ax2.scatter(self.x_blue, self.y_blue, s=1, c='b')
        ax2.scatter(self.x_red, self.y_red, s=1, c='r')
        ax2.set_title("2D problem")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
        im3 = plt.scatter(0, 1, c=0, cmap=cmap, norm=norm)
        fig.colorbar(im3)
        plt.show()

    '''
    # This code is built by York, and used to test
    def show(self):
        plt.scatter(self.x_red, self.y_red, s=0.5, color='red')
        plt.scatter(self.x_blue, self.y_blue, s=0.5, color='blue')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    '''

    # main code for this class
    def main(self):
        # setup the init list of particles
        self.setup()
        # show the first grid when t = 0
        self.show()
        # cycle in Classic Euler Method step by step
        for x in range(int(self.time_max / self.h)):
            self.go_a_step()
            self.show()


# main code for the whole project
if __name__ == '__main__':
    # build a instance
    Project = CMM3()
    # to run main code
    Project.main()
