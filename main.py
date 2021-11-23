import math
import numpy as np
import matplotlib.pyplot as plt
from random import gauss
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from gui import GUI


# class with main code
class CMM:
    def __init__(self):
        # initial function, use GUI class to generate the initial values
        self.gui = GUI()
        self.x_min = float(self.gui.x_min)
        self.x_max = float(self.gui.x_max)
        self.y_min = float(self.gui.y_min)
        self.y_max = float(self.gui.y_max)
        self.D = float(self.gui.D)
        self.time_max = float(self.gui.time_max)
        self.h = float(self.gui.h)
        self.Nx = int(self.gui.Nx)
        self.Ny = int(self.gui.Ny)
        self.Np = int(self.gui.Np)
        self.r = float(self.gui.r)
        self.r_x = float(self.gui.r_x)
        self.r_y = float(self.gui.r_y)
        self.vel_type = self.gui.vel_type
        self.con = self.gui.con
        del self.gui
        # saves temporary position of particles
        self.x_data = None
        self.y_data = None
        # saves initial velocity field
        self.vel_field = np.zeros((32, 32, 2))

    # velocity field setup
    def velocity_field_setup(self):
        # input the data from out side
        vel = np.loadtxt('velocityCMM3.dat')
        # calculate the length of each field
        grid_l = (self.x_max - self.x_min) / 32
        # add the label for every velocity field
        for i in range(len(vel)):
            field_x = math.ceil((vel[i][0] - self.x_min) / grid_l) - 1
            field_y = math.ceil((vel[i][1] - self.y_min) / grid_l) - 1
            self.vel_field[field_x][field_y][0] = vel[i][2]
            self.vel_field[field_x][field_y][1] = vel[i][3]

    # function to setup the particles
    def setup(self):
        # use np.random to initialize random particles
        self.x_data = [[], []]
        self.y_data = [[], []]
        for i in range(self.Np):
            # temporary value for a random particle
            tx = np.random.uniform(self.x_min, self.x_max)
            ty = np.random.uniform(self.y_min, self.y_max)
            # select the color of this particle and count each color in data0
            if self.con == 0 or self.con == 1 or self.con == 4 or self.con == 5:
                if math.sqrt(math.pow(tx - self.r_x, 2) + math.pow(ty - self.r_y, 2)) < self.r:
                    self.x_data[1].append(tx)
                    self.y_data[1].append(ty)
                else:
                    self.x_data[0].append(tx)
                    self.y_data[0].append(ty)
            elif self.con == 2 or self.con == 3:
                self.D = 0.1
                if tx < 0:
                    self.x_data[1].append(tx)
                    self.y_data[1].append(ty)
                else:
                    self.x_data[0].append(tx)
                    self.y_data[0].append(ty)

    # equation 6, the classical Euler method
    def ex_euler(self, Xp, u):
        # creates a random number 'r' using standard Gaussian probability
        r = gauss(0, 1)
        Xp += u * self.h + math.sqrt(2 * self.D) * math.sqrt(self.h) * r
        return Xp

    # boundary condition
    def bc(self, x, y):
        if x < self.x_min:
            x = 2 * self.x_min - x
        elif x > self.x_max:
            x = 2 * self.x_max - x
        if y < self.y_min:
            y = 2 * self.y_min - y
        elif y > self.y_max:
            y = 2 * self.y_max - y
        return x, y

    # this func is used to update the particles after every step time
    def go_a_step(self):
        for i in range(len(self.x_data)):
            for n in range(len(self.x_data[i])):
                field_x, field_y = 0, 0
                if self.vel_type == 1:
                    # Confirm what field should each particle be in
                    field_x = math.ceil((self.x_data[i][n] - self.x_min) / ((self.x_max - self.x_min) / 32)) - 1
                    field_y = math.ceil((self.y_data[i][n] - self.y_min) / ((self.x_max - self.x_min) / 32)) - 1
                # Use EX Euler method to calculate next position
                self.x_data[i][n] = self.ex_euler(self.x_data[i][n], self.vel_field[field_x][field_y][0])
                self.y_data[i][n] = self.ex_euler(self.y_data[i][n], self.vel_field[field_x][field_y][1])
                # use the boundary condition above
                self.x_data[i][n], self.y_data[i][n] = self.bc(self.x_data[i][n], self.y_data[i][n])

    # graph in particle form
    def show_particle_form(self):
        # creat the graph and add the coordinates of blue and red particles
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(self.x_data[1], self.y_data[1], s=1, c='b')
        plt.scatter(self.x_data[0], self.y_data[0], s=1, c='r')
        # set the layout of axis and title
        plt.title("Particle Form", fontname='Arial', fontsize=30, weight='bold')
        plt.xlabel("x", fontname='Arial', fontsize=20, weight='bold')
        plt.ylabel("y", fontname='Arial', fontsize=20, weight='bold')
        plt.xlim([self.x_min, self.x_max])
        plt.ylim([self.y_min, self.y_max])
        plt.xticks([-1 + i * 0.5 for i in range(5)])
        plt.yticks([-1 + i * 0.5 for i in range(5)])
        # set the parameters of the colorbar in particle form and create it
        cmap = ListedColormap(["red", "blue"])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        im3 = plt.scatter(0, 0, s=0, c=0, cmap=cmap, norm=norm)
        fig.colorbar(im3)
        # show
        cax = plt.gcf().axes[-1]
        cax.axis('off')
        plt.show()

    # calculate the percentage of particles in each grid
    def count_grid(self):
        # calculated the percentage using histogram2d
        x_all = np.concatenate((self.x_data[0], self.x_data[1]))
        y_all = np.concatenate((self.y_data[0], self.y_data[1]))
        x_grid = setup_grid(self.x_min, self.x_max, self.Nx)
        y_grid = setup_grid(self.y_min, self.y_max, self.Ny)
        blue, x_edges, y_edges = np.histogram2d([-i for i in self.y_data[1]], self.x_data[1], bins=[y_grid, x_grid])
        all_p, x_edges, y_edges = np.histogram2d([-i for i in y_all], x_all, bins=[y_grid, x_grid])
        return np.divide(blue, all_p, out=np.zeros_like(blue), where=all_p != 0)

    # the graph in grid form
    def show_grid(self):
        # defintion of the colorbar for grid form
        colors1 = [(r, g, b) for (r, g, b) in zip(np.linspace(1, 0.8, 7), np.linspace(0, 0, 7), np.linspace(0, 0.9, 7))]
        colors2 = [(r, g, b) for (r, g, b) in zip(np.linspace(0.7, 0, 6), np.linspace(0, 1, 6), np.linspace(1, 0, 6))]
        colors3 = [(r, g, b) for (r, g, b) in zip(np.linspace(0, 0, 7), np.linspace(0.7, 0, 7), np.linspace(0, 1, 7))]
        colors = colors1 + colors2 + colors3
        # add data and create the heatmap
        sns_plot = sns.heatmap(self.count_grid(), vmin=0, vmax=1, cmap=colors)
        # set the layout of axis and title
        plt.title("Grid Form", fontname='Arial', fontsize=30, weight='bold')
        sns_plot.set_xlabel("x", fontname='Arial', fontsize=20, weight='bold')
        sns_plot.set_ylabel("y", fontname='Arial', fontsize=20, weight='bold')
        # show
        plt.show()

    # show 1D form if self.con == 1
    def show_1d_form(self, j):
        plt.title("1D problem", fontname='Arial', fontsize=30, weight='bold')
        plt.xlabel("x", fontsize=20)
        plt.ylabel("Ф", fontsize=20)
        plt.xlim([-1, 1])
        plt.ylim([0, 1])
        plt.xticks([-1 + i * 0.5 for i in range(5)])
        plt.yticks([0 + i * 0.2 for i in range(6)])
        if j == 0:
            plt.plot([(i - (self.Nx / 2)) / (self.Nx / 2) for i in range(self.Nx)], self.count_grid()[0],
                     color='purple',
                     label='Np={},run1'.format(self.Np))
            plt.legend()
        if j == 1:
            plt.plot([(i - (self.Nx / 2)) / (self.Nx / 2) for i in range(self.Nx)], self.count_grid()[0], color='blue',
                     label='Np={},run2'.format(self.Np))
            plt.legend()
        if j == 2:
            plt.plot([(i - (self.Nx / 2)) / (self.Nx / 2) for i in range(self.Nx)], self.count_grid()[0],
                     color='orange',
                     label='Np={},run3'.format(self.Np))
            plt.legend()
        if j == 3:
            reference = np.loadtxt('reference_solution_1D.dat')
            r_x = []
            r_y = []
            for i in range(len(reference)):
                r_x.append(reference[i][0])
                r_y.append(reference[i][1])
            plt.plot(r_x, r_y, color='black', label='reference')
            plt.legend()

    # to calculate the root mean square error
    def count_error(self, p1):
        data = self.count_grid()[0]
        x = []
        ivl_grid_x = (self.x_max - self.x_min) / self.Nx
        for i in range(self.Nx):
            x.append(i * ivl_grid_x + ivl_grid_x / 2 - 1)
        return np.sqrt(mean_squared_error(data, p1(x)))

    # mark the trace when the concentration >= 0.3
    def task_d(self, data):
        data0 = self.count_grid()
        for i in range(len(data0)):
            for j in range(len(data0[i])):
                if data0[i][j] > 0.3:
                    data[i][j] = 1
        return data

    # taskE for simulation improvement
    def task_e(self, data):
        x_grid = setup_grid(self.x_min, self.x_max, self.Nx)
        y_grid = setup_grid(self.y_min, self.y_max, self.Ny)
        blue, x_edges, y_edges = np.histogram2d([-i for i in self.y_data[1]], self.x_data[1], bins=[x_grid, y_grid])
        for i in range(self.Nx):
            for j in range(self.Ny):
                # self.Np / (self.Nx * self.Ny) is the average particle quantity in each grid
                if blue[i][j] / (self.Np / (self.Nx * self.Ny)) > 0.3:
                    data[i][j] = 1
        return data

    # main code for this class
    def main(self):
        # include the velocity if vel_type = 1
        if self.vel_type == 1:
            self.velocity_field_setup()
        # if con == 0, run 2D problem in particle form
        if self.con == 0:
            # setup the initial list of particles
            self.setup()
            # show the first graph when t = 0
            self.show_particle_form()
            # loop Classic Euler Method step by step
            for i in range(int(self.time_max / self.h)):
                self.go_a_step()
                self.show_particle_form()

        # if con == 1, run 2D problem in grid form
        elif self.con == 1:
            # setup the initial list of particles
            self.setup()
            # show the first graph when t = 0, data0 and data
            self.show_grid()
            # loop Classic Euler Method step by step
            for i in range(int(self.time_max / self.h)):
                self.go_a_step()
                self.show_grid()

        # if con == 2, means 1D problem
        elif self.con == 2:
            for i in range(4):
                if i < 3:
                    # setup the init list of particles for 3 runs
                    self.setup()
                    # loop Classic Euler Method step by step
                    for j in range(int(self.time_max / self.h)):
                        self.go_a_step()
                        if j * self.h == 0.2:
                            break
                # show the 1D diagram when t = time_max / 2
                self.show_1d_form(i)
            plt.show()

        # if con == 3 run 1D error simulation
        elif self.con == 3:
            # import reference data
            p1 = reference_data_setup()
            # for Np VS E simulation
            # initialize x and y_error to save temporarily
            x_type = 'Np'
            x = []
            y_error = []
            # calculate global error for different Np when h == 0.05
            for i in range(18):  # the range of Np = 2 ^ i
                self.h = 0.05
                self.Np = 2 ** i
                x.append(self.Np)
                # setup initial list of particles
                self.setup()
                # loop Classic Euler Method step by step
                for j in range(int(self.time_max / self.h)):
                    self.go_a_step()
                y_error.append(self.count_error(p1))
            show_error(x, np.log10(y_error), x_type)

            # for h VS E simulation
            # initialize x and y_error to save temporarily
            x_type = 'h'
            x = []
            y_error = []
            # calculate global error for different h when Np == 1024
            for i in np.arange(0.005, 0.2, 0.005):  # the range of h
                self.Np = 1024  # single Np
                self.h = i
                x.append(self.h)
                # setup initial list of particles
                self.setup()
                # loop Classic Euler Method step by step
                for j in range(int(self.time_max / self.h)):
                    self.go_a_step()
                y_error.append(self.count_error(p1))
            show_error(x, np.log10(y_error), x_type)

        # if con == 4 or 5 run TaskD simulation
        elif self.con == 4 or self.con == 5:
            # setup the initial list of particles
            self.setup()
            # data is initialized to find the proportion of blue particles in each grid in TaskD
            data = np.zeros((self.Nx, self.Ny))
            # for normal Task D
            if self.con == 4:
                # get the first data
                data = self.task_d(data)
                # loop Classic Euler Method step by step
                for i in range(int(self.time_max / self.h)):
                    self.go_a_step()
                    data = self.task_d(data)
                    show_oil(data)
            # for Task E， only simulate oil particles
            elif self.con == 5:
                # delete red particles
                self.x_data[0] = []
                self.y_data[0] = []
                # get the first data
                data = self.task_e(data)
                # loop Classic Euler Method step by step
                for i in range(int(self.time_max / self.h)):
                    self.go_a_step()
                    data = self.task_e(data)
                    show_oil(data)


# setup the grid
def setup_grid(xy_min, xy_max, N):
    grid_step = (xy_max - xy_min) / N
    return np.arange(xy_min, xy_max + grid_step, grid_step)


# setup the reference solution of 1D problem
def reference_data_setup():
    reference_data = np.loadtxt('reference_solution_1D.dat')
    x = []
    y = []
    for i in range(len(reference_data)):
        x.append(reference_data[i][0])
        y.append(reference_data[i][1])
    return np.poly1d(np.polyfit(x, y, 20))


# to find fitting constants
def func(N, a, b):
    return a * N ** b


# plotting to show the relationship between the global error with different Np
def show_error(x, y, t):
    plt.scatter(x, y, c='blue', label='original values')
    if t == 'Np':
        # Nonlinear least squares fitting
        popt, pcov = curve_fit(func, x, 10 ** y)
        a = popt[0]
        b = popt[1]
        print('\nFit the error form E = α * Np ^ β, where:')
        print('coefficient α=', a)
        print('coefficient β=', b)
        y1 = func(x, a, b)
        plt.plot(x, np.log10(y1), 'r', label='polyfit values')
    plt.xlabel(t, fontsize=20)
    plt.ylabel('lg E', fontsize=20)
    plt.legend(loc=1)
    plt.title("Root mean square error", fontname='Arial', fontsize=30, weight='bold')
    plt.show()


# plot the oil trace for Task D
def show_oil(data):
    # the defination of colorbar of grid form
    # add data and create the heatmap
    sns_plot = sns.heatmap(data, vmin=0, vmax=1, cbar=False, cmap=ListedColormap(["black", "white"]))
    cax = plt.gcf().axes[-1]
    cax.axis('off')
    plt.title("Mark the trace(white area)", fontname='Arial', fontsize=30, weight='bold')
    sns_plot.set_xlabel("x", fontname='Arial', fontsize=20, weight='bold')
    sns_plot.set_ylabel("y", fontname='Arial', fontsize=20, weight='bold')
    # show
    plt.show()


# main code for the whole TaskA
if __name__ == '__main__':
    # build a instance
    cmm = CMM()
    # run CMM main code
    cmm.main()
