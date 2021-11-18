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


# for easier setting, I put all the code into a class
class TaskA:
    def __init__(self):
        # initial function, use GUI class to generate the initial values
        self.gui = GUI()
        self.gui.main()
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
        self.plot_type = self.gui.plot_type
        self.con = self.gui.con
        del self.gui
        # for temp particle position data save
        self.x_data = None
        self.y_data = None
        # init velocity field data saver
        self.vel_field = None
        # for temp error reference data saver

    # setup the velocity field
    def velocity_field_setup(self):
        self.vel_field = np.zeros((32, 32, 2))
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

    # this func is used to setup the particles in one traverse, built by York
    def setup(self):
        # use np.random to init the particles
        self.x_data = [[], []]
        self.y_data = [[], []]
        if self.con == 3:
            self.D = 0.1
            self.r_x = 0.4
            self.r_y = 0.4
            self.r = 0.1
        for i in range(self.Np):
            # temp val for a random particle
            tx = np.random.uniform(self.x_min, self.x_max)
            ty = np.random.uniform(self.y_min, self.y_max)
            # select the color of this particle and count each color in data0
            if self.con == 0 or self.con == 3:
                if math.sqrt(math.pow(tx - self.r_x, 2) + math.pow(ty - self.r_y, 2)) < self.r:
                    self.x_data[1].append(tx)
                    self.y_data[1].append(ty)
                else:
                    self.x_data[0].append(tx)
                    self.y_data[0].append(ty)
            elif self.con == 1 or self.con == 2:
                self.D = 0.1
                if tx < 0:
                    self.x_data[1].append(tx)
                    self.y_data[1].append(ty)
                else:
                    self.x_data[0].append(tx)
                    self.y_data[0].append(ty)

    # the func below is the classical Euler method, equation 6
    def ex_euler(self, Xp, u):
        # create a gauss random dx, 'r' is random numbers with the standard Gaussian probability
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

    # this func is used to update the particles after a step time, the movement of each particle is built by York,
    def go_a_step(self):
        for i in range(len(self.x_data)):
            for n in range(len(self.x_data[i])):
                if self.vel_type == 0:
                    # Use EX Euler method to calculate next position
                    self.x_data[i][n] = self.ex_euler(self.x_data[i][n], 0)
                    self.y_data[i][n] = self.ex_euler(self.y_data[i][n], 0)
                elif self.vel_type == 1:
                    # Confirm what field should each particle be in
                    field_x = math.ceil((self.x_data[i][n] - self.x_min) / ((self.x_max - self.x_min) / 32)) - 1
                    field_y = math.ceil((self.y_data[i][n] - self.y_min) / ((self.x_max - self.x_min) / 32)) - 1
                    # Use EX Euler method to calculate next position
                    self.x_data[i][n] = self.ex_euler(self.x_data[i][n], self.vel_field[field_x][field_y][0])
                    self.y_data[i][n] = self.ex_euler(self.y_data[i][n], self.vel_field[field_x][field_y][1])
                # use the boundary condition above
                self.x_data[i][n], self.y_data[i][n] = self.bc(self.x_data[i][n], self.y_data[i][n])

    # the visualization of particle form, by The Kite
    def show_particle_form(self):
        # set the figure and pass in the coordinates of blue and red particles
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
                ivl_ys = math.ceil((self.Ny - (self.y_data[i][n] - self.y_min) / ivl_grid_y) - 1)
                ivl_xs = math.ceil(((self.x_data[i][n] - self.x_min) / ivl_grid_x) - 1)
                data0[ivl_ys][ivl_xs][i] += 1
        # transfer the data0 into data by calculating the proportion of blue particles in each grid
        for i in range(self.Nx):
            for j in range(self.Ny):
                if data0[i][j][0] + data0[i][j][1] == 0:
                    data[i][j] = 0
                else:
                    data[i][j] = data0[i][j][1] / (data0[i][j][0] + data0[i][j][1])
        # the defination of colorbar of grid form
        colors1 = [(r, g, b) for (r, g, b) in zip(np.linspace(1, 0.8, 7), np.linspace(0, 0, 7), np.linspace(0, 0.9, 7))]
        colors2 = [(r, g, b) for (r, g, b) in zip(np.linspace(0.7, 0, 6), np.linspace(0, 1, 6), np.linspace(1, 0, 6))]
        colors3 = [(r, g, b) for (r, g, b) in zip(np.linspace(0, 0, 7), np.linspace(0.7, 0, 7), np.linspace(0, 1, 7))]
        colors = colors1 + colors2 + colors3
        # pass in data and create the heatmap
        sns_plot = sns.heatmap(data, vmin=0, vmax=1, cmap=colors)
        # set the layout of axis and title
        plt.title("Grid Form", fontname='Arial', fontsize=30, weight='bold')
        sns_plot.set_xlabel("x", fontname='Arial', fontsize=20, weight='bold')
        sns_plot.set_ylabel("y", fontname='Arial', fontsize=20, weight='bold')
        plt.axis('off')
        # show
        plt.show()

    # show 1D form if self.con == 1, by Ziqing and Zsolt
    def show_1d_form(self, j):
        ivl_grid_x = (self.x_max - self.x_min) / self.Nx
        # data0 is set to count and save the number of red particles and blue particles
        data0 = np.zeros((self.Nx, 2))
        # data is set to figure the proportion of blue particles in each grid
        data = np.zeros(self.Nx)
        # calculate the data0 of grid
        for i in range(len(self.x_data)):
            # to locate which grid is the new particle in and add it to data0[i]
            for n in range(len(self.x_data[i])):
                ivl_xs = math.ceil((self.x_data[i][n] - self.x_min) / ivl_grid_x) - 1
                data0[ivl_xs][i] += 1
        # transfer the data0 into data by calculating the proportion of blue particles in each grid
        for i in range(self.Nx):
            if data0[i][0] + data0[i][1] == 0:
                data[i] = 0
            else:
                data[i] = data0[i][1] / (data0[i][0] + data0[i][1])
        plt.title("1D problem", fontname='Arial', fontsize=30, weight='bold')
        plt.xlabel("x", fontsize=20)
        plt.ylabel("Ф", fontsize=20)
        plt.xlim([-1, 1])
        plt.ylim([0, 1])
        plt.xticks([-1 + i * 0.5 for i in range(5)])
        plt.yticks([0 + i * 0.2 for i in range(6)])
        if j == 0:
            plt.plot([(i - (self.Nx / 2)) / (self.Nx / 2) for i in range(self.Nx)], data, color='purple',
                     label='Np={},run1'.format(self.Np))
            plt.legend()
        if j == 1:
            plt.plot([(i - (self.Nx / 2)) / (self.Nx / 2) for i in range(self.Nx)], data, color='blue',
                     label='Np={},run2'.format(self.Np))
            plt.legend()
        if j == 2:
            plt.plot([(i - (self.Nx / 2)) / (self.Nx / 2) for i in range(self.Nx)], data, color='orange',
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

    # to count the root mean square error
    def root_mean_square_error(self, p1):
        ivl_grid_x = (self.x_max - self.x_min) / self.Nx
        # data0 is set to count and save the number of red particles and blue particles
        data0 = np.zeros((self.Nx, 2))
        # data is set to figure the proportion of blue particles in each grid
        data = np.zeros(self.Nx)
        # calculate the data0 of grid
        for i in range(len(self.x_data)):
            # to locate which grid is the new particle in and add it to data0[i]
            for n in range(len(self.x_data[i])):
                ivl_xs = math.ceil((self.x_data[i][n] - self.x_min) / ivl_grid_x) - 1
                data0[ivl_xs][i] += 1
        # transfer the data0 into data by calculating the proportion of blue particles in each grid
        x = []
        for i in range(self.Nx):
            if data0[i][0] + data0[i][1] == 0:
                data[i] = 0
            else:
                data[i] = data0[i][1] / (data0[i][0] + data0[i][1])
            x.append(i * ivl_grid_x + ivl_grid_x / 2 - 1)
        return np.sqrt(mean_squared_error(data, p1(x)))

    # mark the trace of concentration in which the value>=0.3, by The Kite
    def task_d_mark(self, data):
        ivl_grid_y = (self.y_max - self.y_min) / self.Ny
        ivl_grid_x = (self.x_max - self.x_min) / self.Nx
        # data0 is set to count and save the number of red particles and blue particles
        data0 = np.zeros((self.Nx, self.Ny, 2))
        # calculate the data0 of grid
        for i in range(len(self.x_data)):
            # to locate which grid is the new particle in and add it to data0[i]
            for n in range(len(self.x_data[i])):
                ivl_ys = math.ceil((self.Ny - (self.y_data[i][n] - self.y_min) / ivl_grid_y) - 1)
                ivl_xs = math.ceil(((self.x_data[i][n] - self.x_min) / ivl_grid_x) - 1)
                data0[ivl_ys][ivl_xs][i] += 1
        for i in range(self.Nx):
            for j in range(self.Ny):
                if data0[i][j][0] + data0[i][j][1] == 0:
                    data[i][j] = 0
                elif data0[i][j][1] / (data0[i][j][0] + data0[i][j][1]) > 0.3:
                    data[i][j] = 1
        # the defination of colorbar of grid form
        # pass in data and create the heatmap
        sns_plot = sns.heatmap(data, vmin=0, vmax=1, cmap=ListedColormap(["black", "white"]))
        # set the layout of axis and title
        plt.title("Mark the trace(white area)", fontname='Arial', fontsize=30, weight='bold')
        sns_plot.set_xlabel("x", fontname='Arial', fontsize=20, weight='bold')
        sns_plot.set_ylabel("y", fontname='Arial', fontsize=20, weight='bold')
        plt.axis('off')
        # show
        plt.show()
        return data

    # main code for this class
    def main(self):
        # setup the velocity if vel_type = 1
        if self.vel_type == 1:
            self.velocity_field_setup()
        # if con == 0, means 2D problem
        if self.con == 0:
            # setup the init list of particles
            self.setup()
            # the visualization of particle form
            if self.plot_type == 0:
                # show the first graph when t = 0
                self.show_particle_form()
                # cycle in Classic Euler Method step by step
                for i in range(int(self.time_max / self.h)):
                    self.go_a_step()
                    self.show_particle_form()
                # the visualization of grid form
            elif self.plot_type == 1:
                # show the first graph when t = 0, data0 and data
                self.show_grid()
                # cycle in Classic Euler Method step by step
                for i in range(int(self.time_max / self.h)):
                    self.go_a_step()
                    self.show_grid()

        # if con == 1, means 1D problem
        elif self.con == 1:
            for i in range(4):
                if i < 3:
                    # setup the init list of particles for 3 times
                    self.setup()
                    # cycle in Classic Euler Method step by step
                    for j in range(int(self.time_max / self.h)):
                        self.go_a_step()
                        if j * self.h == 0.2:
                            break
                # show the 1D diagram when t = time_max / 2
                self.show_1d_form(i)
            plt.show()

        # if con == 2, means 1D error simulation
        elif self.con == 2:
            # import reference data
            p1 = reference_data_setup()
            # init x and y_error for temp save
            x_type = 'Np'
            x = []
            y_error = []
            # get globe error in different Np when h == 0.05
            for i in range(18):  # the range of Np = 2 ^ i
                self.h = 0.05  # single h
                self.Np = 2 ** i
                x.append(self.Np)
                # setup init list of particles
                self.setup()
                # cycle in Classic Euler Method step by step
                for j in range(int(self.time_max / (2 * self.h))):
                    self.go_a_step()
                y_error.append(self.root_mean_square_error(p1))
            show_error(x, y_error, x_type)
            # init x and y_error for temp save
            x_type = 'h'
            x = []
            y_error = []
            # get globe error in different h when Np == 1024
            for i in np.arange(0.05, 0.2, 0.005):  # the range of h
                self.Np = 1024  # single Np
                self.h = i
                x.append(i)
                # setup init list of particles
                self.setup()
                # cycle in Classic Euler Method step by step
                for j in range(int(self.time_max / (2 * self.h))):
                    self.go_a_step()
                y_error.append(self.root_mean_square_error(p1))
            show_error(x, y_error, x_type)

        # if con == 3, means TaskD
        elif self.con == 3:
            # setup the init list of particles
            self.setup()
            # data is set to figure the proportion of blue particles in each grid in TaskD
            data = np.zeros((self.Nx, self.Ny))
            data = self.task_d_mark(data)
            for i in range(int(self.time_max / self.h)):
                self.go_a_step()
                data = self.task_d_mark(data)


# setup the reference solution of 1D problem
def reference_data_setup():
    reference_data = np.loadtxt('reference_solution_1D.dat')
    x = []
    y = []
    for i in range(len(reference_data)):
        x.append(reference_data[i][0])
        y.append(reference_data[i][1])
    return np.poly1d(np.polyfit(x, y, 20))


# error form for fit
def func(N, a, b):
    return a * N ** b


# plotting to show the relationship between the global error with different Np
def show_error(x, y, t):
    plt.scatter(x, y, c='blue', label='original values')
    if t == 'Np':
        # Nonlinear least squares fitting
        popt, pcov = curve_fit(func, x, y)
        a = popt[0]
        b = popt[1]
        print('\nFit the error form E = a * Np ^ 2, where:')
        print('coefficient a=', a)
        print('coefficient b=', b)
        y1 = func(x, a, b)
        plt.plot(x, y1, 'r', label='polyfit values')
    plt.xlabel(t, fontsize=20)
    plt.ylabel('E', fontsize=20)
    plt.legend(loc=1)
    plt.title("Root mean square error", fontname='Arial', fontsize=30, weight='bold')
    plt.show()


# main code for the whole TaskA
if __name__ == '__main__':
    # build a instance
    cmm = TaskA()
    # run TaskA main code
    cmm.main()
