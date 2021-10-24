import math
import numpy as np
import matplotlib.pyplot as plt
from random import gauss
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import seaborn as sns

# the definition of colorbar of gird form
colors1 = [(r, g, b) for (r, g, b) in zip(np.linspace(1, 0.8, 7), np.linspace(0, 0, 7), np.linspace(0, 0.9, 7))]
colors2 = [(r, g, b) for (r, g, b) in zip(np.linspace(0.7, 0, 6), np.linspace(0, 1, 6), np.linspace(1, 0, 6))]
colors3 = [(r, g, b) for (r, g, b) in zip(np.linspace(0, 0, 7), np.linspace(0.7, 0, 7), np.linspace(0, 1, 7))]
colors = colors1 + colors2 + colors3


# boundary condition
def BC(x, y):
    if x < -1:
        x = -2 - x
    elif x > 1:
        x = 2 - x
    if y < -1:
        y = -2 - y
    elif y > 1:
        y = 2 - y
    return x, y


# for easier setting, I put all the code into a class, developed by York and The Kite
class CMM3(object):
    def __init__(self):
        # ----------------------------------------
        # init conditions / interface
        # time set, "h" is a step time
        self.time_max = 1
        self.h = 0.05
        # diffusivity
        self.D = 0.01
        # domain size
        self.x_min = -1
        self.x_max = 1
        self.y_min = -1
        self.y_max = 1
        # number of axial gridlines and total grids
        self.Nx = 64
        self.Ny = 64
        self.grids = self.Nx * self.Ny
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
        # figure and save the interval of the gridlines
        self.ivl_grid_y = (self.y_max - self.y_min) / self.Ny
        self.ivl_grid_x = (self.x_max - self.x_min) / self.Nx
        # data0 is set to count and save the number of red particles and blue particles
        self.data0 = np.zeros((self.Nx, self.Ny, 2))
        # data is set to figure the proportion of blue particles in each grid
        self.data = np.zeros((self.Nx, self.Ny))

    # the func below is extends the classical Euler method
    def Euler_method(self, Xp):
        # create a gauss random dx
        r = gauss(0, 1)
        # equation 6
        # 'r' is random numbers with the standard Gaussian probability
        # 'vel' is the (known) velocity components evaluated at the particle position
        Xp += self.vel * self.h + math.sqrt(2 * self.D) * math.sqrt(self.h) * r
        return Xp

    # this func is used to update the particles after a step time, the movement of each particle is built by York,
    # the operation to data0 is built by TheKite
    def go_a_step(self):
        for i in range(len(self.x_blue)):
            self.x_blue[i] = self.Euler_method(self.x_blue[i])
            self.y_blue[i] = self.Euler_method(self.y_blue[i])
            # use the boundary condition above
            self.x_blue[i], self.y_blue[i] = BC(self.x_blue[i], self.y_blue[i])
            # to locate which grid is the new blue particle in and add it to data0[0]
            ivl_xs = math.ceil((self.x_blue[i] - self.x_min) / self.ivl_grid_x) - 1
            ivl_ys = math.ceil((self.y_blue[i] - self.y_min) / self.ivl_grid_y) - 1
            self.data0[ivl_xs][ivl_ys][0] += 1
        for i in range(len(self.x_red)):
            self.x_red[i] = self.Euler_method(self.x_red[i])
            self.y_red[i] = self.Euler_method(self.y_red[i])
            # use the boundary condition above
            self.x_red[i], self.y_red[i] = BC(self.x_red[i], self.y_red[i])
            # to locate which grid is the new red particle in and add it to data0[1]
            ivl_xs = math.ceil((self.x_red[i] - self.x_min) / self.ivl_grid_x) - 1
            ivl_ys = math.ceil((self.y_red[i] - self.y_min) / self.ivl_grid_y) - 1
            self.data0[ivl_xs][ivl_ys][1] += 1
        # transfer the data0 into data by calculating the proportion of blue particles in each grid
        for i in range(self.Nx):
            for j in range(self.Ny):
                self.data[i][j] = self.data0[i][j][0] / (self.data0[i][j][0] + self.data0[i][j][1])

    # this func is used to setup the particles in one traverse, built by York
    def setup(self):
        # to reset the coordinates and quantities for the circulation in main()
        self.x_blue = []
        self.y_blue = []
        self.x_red = []
        self.y_red = []
        # use np.random to init the particles
        for i in range(self.Np):
            # temp val for a random particle
            tx = np.random.uniform(self.x_min, self.x_max)
            ty = np.random.uniform(self.y_min, self.y_max)
            ivl_xs = math.ceil((tx - self.x_min) / self.ivl_grid_x) - 1
            ivl_ys = math.ceil((ty - self.y_min) / self.ivl_grid_y) - 1
            # select the color of this particle and count each color in data0
            if math.sqrt(math.pow(tx, 2) + math.pow(ty, 2)) < 0.3:
                self.x_blue.append(tx)
                self.y_blue.append(ty)
                self.data0[ivl_xs][ivl_ys][0] += 1
            else:
                self.x_red.append(tx)
                self.y_red.append(ty)
                self.data0[ivl_xs][ivl_ys][1] += 1
        # transfer the data0 into data by calculating the proportion of blue particles in each grid
        for i in range(self.Nx):
            for j in range(self.Ny):
                self.data[i][j] = self.data0[i][j][0] / (self.data0[i][j][0] + self.data0[i][j][1])

    # this func is the visualization built by The Kite
    def show1(self):
        # set the figure and pass in the coordinates of blue and red particles
        plt.figure(figsize=(10, 8))
        plt.scatter(self.x_blue, self.y_blue, s=1, c='b')
        plt.scatter(self.x_red, self.y_red, s=1, c='r')
        # set the layout of axis and title
        plt.title("2D problem", fontname='Arial', fontsize=30, weight='bold')
        plt.xlabel("x", fontname='Arial', fontsize=20, weight='bold')
        plt.ylabel("y", fontname='Arial', fontsize=20, weight='bold')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.xticks([-1 + i * 0.5 for i in range(5)])
        plt.yticks([-1 + i * 0.5 for i in range(5)])
        # set the parameters of the colorbar in particle form and create it
        cmap = ListedColormap(["red", "blue"])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        plt.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        )
        # show
        plt.show()

    # the visualization of grid form
    # set the parameters of the colorbar in grid form
    colors1 = [(r, g, b) for (r, g, b) in zip(np.linspace(1, 0.5, 7), np.linspace(0, 0, 7), np.linspace(0, 0.9, 7))]
    colors2 = [(r, g, b) for (r, g, b) in zip(np.linspace(0.7, 0, 6), np.linspace(0, 1, 6), np.linspace(1, 0, 6))]
    colors3 = [(r, g, b) for (r, g, b) in zip(np.linspace(0, 0, 7), np.linspace(0.7, 0, 7), np.linspace(0, 1, 7))]
    colors = colors1 + colors2 + colors3

    def show2(self):
        # pass in data and create the heatmap
        sns_plot = sns.heatmap(self.data, vmin=0, vmax=1, cmap=colors)
        # set the layout of axis and title
        plt.title("2D problem", fontname='Arial', fontsize=30, weight='bold')
        sns_plot.set_xlabel("x", fontname='Arial', fontsize=20, weight='bold')
        sns_plot.set_ylabel("y", fontname='Arial', fontsize=20, weight='bold')
        plt.axis('off')
        # show
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
        print("This is CMM3 group's project of 2D probelms")
        while True:
            choice = int(input("for particle press 0,for grid press 1,press anything to quit\n"))
            # the visualization of particle form
            if choice == 0:
                # setup the init list of particles
                self.setup()
                # show the first graph when t = 0
                self.show1()
                # cycle in Classic Euler Method step by step
                for i in range(int(self.time_max / self.h)):
                    self.data = np.zeros((self.Nx, self.Ny))
                    self.go_a_step()
                    self.show1()
                # after the iteration recreate data0 and data
                self.data0 = np.zeros((self.Nx, self.Ny, 2))
                self.data = np.zeros((self.Nx, self.Ny))
            # the visualization of grid form
            elif choice == 1:
                # setup the init list of particles, data0 and data
                self.setup()
                # show the first graph when t = 0, data0 and data
                self.show2()
                # cycle in Classic Euler Method step by step
                for i in range(int(self.time_max / self.h)):
                    self.data = np.zeros((self.Nx, self.Ny))
                    self.go_a_step()
                    self.show2()
                # after the iteration recreate data0 and data
                self.data0 = np.zeros((self.Nx, self.Ny, 2))
                self.data = np.zeros((self.Nx, self.Ny))
            # quit the project
            else:
                return


# main code for the whole project
if __name__ == '__main__':
    # build a instance
    Project = CMM3()
    # to run main code
    Project.main()
