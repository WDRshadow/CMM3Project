import math
import numpy as np
import matplotlib.pyplot as plt
from random import gauss
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import seaborn as sns
from sys import exit


# for easier setting, I put all the code into a class
class TaskA(object):
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
        # number of axial gridlines and total grids
        self.Nx = 64
        self.Ny = 64
        # number of particles
        self.Np = 65536
        # init velocity
        self.vel_type = 1
        # -----------------------------------------
        # for temp particle position data save
        # data[0] is the value of red particle, [1] is the value for blue one
        self.x_data = [[], []]
        self.y_data = [[], []]
        # init velocity field data saver, include x, y and velocity
        self.vel_field = np.zeros((32, 32, 2))

    # setup the velocity field
    def velocity_field_setup(self):
        # input the data from out side
        vel = np.loadtxt('velocityCMM3.dat')
        # calculate the length of each field
        gird_l = (self.x_max - self.x_min) / 32
        # add the label for every velocity field
        for i in range(len(vel)):
            field_x = math.ceil((vel[i][0] - self.x_min) / gird_l) - 1
            field_y = math.ceil((vel[i][1] - self.y_min) / gird_l) - 1
            self.vel_field[field_x][field_y][0] = vel[i][2]
            self.vel_field[field_x][field_y][1] = vel[i][3]

    # the func below is the classical Euler method, equation 6
    def EX_Euler_method(self, Xp, u):
        # create a gauss random dx, 'r' is random numbers with the standard Gaussian probability
        r = gauss(0, 1)
        Xp += u * self.h + math.sqrt(2 * self.D) * math.sqrt(self.h) * r
        return Xp

    # boundary condition
    def BC(self, x, y):
        if x < -1:
            x = -2 - x
        else:
            if x > 1:
                x = 2 - x
        if y < -1:
            y = -2 - y
        else:
            if y > 1:
                y = 2 - y
        return x, y

    # this func is used to update the particles after a step time, the movement of each particle is built by York,
    def go_a_step(self):
        gird_l = (self.x_max - self.x_min) / 32
        for i in range(len(self.x_data)):
            for n in range(len(self.x_data[i])):
                # Confirm what field should each particle be in
                field_x = math.ceil((self.x_data[i][n] - self.x_min) / gird_l) - 1
                field_y = math.ceil((self.y_data[i][n] - self.y_min) / gird_l) - 1
                # Use EX Euler method to calculate next position
                self.x_data[i][n] = self.EX_Euler_method(self.x_data[i][n], self.vel_field[field_x][field_y][0])
                self.y_data[i][n] = self.EX_Euler_method(self.y_data[i][n], self.vel_field[field_x][field_y][1])
                # use the boundary condition above
                self.x_data[i][n], self.y_data[i][n] = self.BC(self.x_data[i][n], self.y_data[i][n])

    # this func is used to setup the particles in one traverse, built by York
    def non_vel_2D_setup(self):
        # use np.random to init the particles
        for i in range(self.Np):
            # temp val for a random particle
            tx = np.random.uniform(self.x_min, self.x_max)
            ty = np.random.uniform(self.y_min, self.y_max)
            # select the color of this particle and count each color in data0
            if math.sqrt(math.pow(tx, 2) + math.pow(ty, 2)) < 0.3:
                self.x_data[1].append(tx)
                self.y_data[1].append(ty)
            else:
                self.x_data[0].append(tx)
                self.y_data[0].append(ty)

    # the visualization of particle form, by The Kite
    def show_particle_form(self):
        # set the figure and pass in the coordinates of blue and red particles
        plt.figure(figsize=(10, 8))
        plt.scatter(self.x_data[1], self.y_data[1], s=1, c='b')
        plt.scatter(self.x_data[0], self.y_data[0], s=1, c='r')
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

    # the visualization of grid form, by The Kite
    def show_grid(self):
        # generate some initial conditions and data saver
        # figure and save the interval of the gridlines
        ivl_grid_y = (self.y_max - self.y_min) / self.Ny
        ivl_grid_x = (self.x_max - self.x_min) / self.Nx
        # data0 is set to count and save the number of red particles and blue particles
        data0 = np.zeros((self.Nx, self.Ny, 2))
        # data is set to figure the proportion of blue particles in each grid
        data = np.zeros((self.Nx, self.Ny))
        # calculate the data0 of grid
        for i in range(len(self.x_data)):
            # to locate which grid is the new particle in and add it to data0[i]
            for n in range(len(self.x_data[i])):
                ivl_xs = math.ceil((self.x_data[i][n] - self.x_min) / ivl_grid_x) - 1
                ivl_ys = math.ceil((self.y_data[i][n] - self.y_min) / ivl_grid_y) - 1
                data0[ivl_xs][ivl_ys][i] += 1
        # transfer the data0 into data by calculating the proportion of blue particles in each grid
        for i in range(self.Nx):
            for j in range(self.Ny):
                data[i][j] = data0[i][j][1] / (data0[i][j][0] + data0[i][j][1])
        # the defination of colorbar of gird form
        colors1 = [(r, g, b) for (r, g, b) in zip(np.linspace(1, 0.8, 7), np.linspace(0, 0, 7), np.linspace(0, 0.9, 7))]
        colors2 = [(r, g, b) for (r, g, b) in zip(np.linspace(0.7, 0, 6), np.linspace(0, 1, 6), np.linspace(1, 0, 6))]
        colors3 = [(r, g, b) for (r, g, b) in zip(np.linspace(0, 0, 7), np.linspace(0.7, 0, 7), np.linspace(0, 1, 7))]
        colors = colors1 + colors2 + colors3
        # pass in data and create the heatmap
        sns_plot = sns.heatmap(data, vmin=0, vmax=1, cmap=colors)
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
        plt.scatter(self.x_data[0], self.y_data[0], s=0.5, color='red')
        plt.scatter(self.x_data[1], self.y_data[1], s=0.5, color='blue')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    '''

    # main code for this class
    def main(self):
        print("This is CMM3 group's project of 2D probelms")
        # setup the init list of particles
        self.non_vel_2D_setup()
        # setup the velocity if vel_type = 1
        if self.vel_type == 1:
            self.velocity_field_setup()
        choice = input("For particle enter 0, for grid enter 1, and if you would like to quit, enter anything else.\n")
        try:
            choice = int(choice)
        except:
            print("Thanks for using our code.\nQuitting")
            exit()
        # the visualization of particle form
        if choice == 0:
            # show the first graph when t = 0
            self.show_particle_form()
            # cycle in Classic Euler Method step by step
            for i in range(int(self.time_max / self.h)):
                self.go_a_step()
                self.show_particle_form()
        # the visualization of grid form
        elif choice == 1:
            # show the first graph when t = 0, data0 and data
            self.show_grid()
            # cycle in Classic Euler Method step by step
            for i in range(int(self.time_max / self.h)):
                self.go_a_step()
                self.show_grid()
        print("Thanks for using our code.\nQuitting")
        exit()


# main code for the whole TaskA
if __name__ == '__main__':
    # build a instance
    run = TaskA()
    # to run main code
    run.main()
