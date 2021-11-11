import math
import numpy as np
import matplotlib.pyplot as plt
from random import gauss
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import seaborn as sns
import tkinter as tk
from tkinter.ttk import *
from tkinter import messagebox


# for easier setting, I put all the code into a class
class TaskA(object):
    def __init__(self):
        # ----------------------------------------
        # default conditions
        # time set, "h" is a step time
        self.time_max = 0.4
        self.h = 0.05
        # diffusivity
        self.D = 0.1
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
        self.vel_type = 0
        # blue particles location / radius and Center of circle
        self.r = 0.3
        self.r_x = 0
        self.r_y = 0
        # initial condition, 0 means 2D problem and 1 means 1D problem
        self.con = 0
        # plot type, 0 means particle form, 1 means grid
        self.plot_type = 0
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
        grid_l = (self.x_max - self.x_min) / 32
        # add the label for every velocity field
        for i in range(len(vel)):
            field_x = math.ceil((vel[i][0] - self.x_min) / grid_l) - 1
            field_y = math.ceil((vel[i][1] - self.y_min) / grid_l) - 1
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
        if x < self.x_min:
            x = 2 * self.x_min - x
        else:
            if x > self.x_max:
                x = 2 * self.x_max - x
        if y < self.y_min:
            y = 2 * self.y_min - y
        else:
            if y > self.y_max:
                y = 2 * self.y_max - y
        return x, y

    # this func is used to update the particles after a step time, the movement of each particle is built by York,
    def go_a_step(self):
        for i in range(len(self.x_data)):
            for n in range(len(self.x_data[i])):
                # Confirm what field should each particle be in
                field_x = math.ceil((self.x_data[i][n] - self.x_min) / ((self.x_max - self.x_min) / 32)) - 1
                field_y = math.ceil((self.y_data[i][n] - self.y_min) / ((self.x_max - self.x_min) / 32)) - 1
                # Use EX Euler method to calculate next position
                self.x_data[i][n] = self.EX_Euler_method(self.x_data[i][n], self.vel_field[field_x][field_y][0])
                self.y_data[i][n] = self.EX_Euler_method(self.y_data[i][n], self.vel_field[field_x][field_y][1])
                # use the boundary condition above
                self.x_data[i][n], self.y_data[i][n] = self.BC(self.x_data[i][n], self.y_data[i][n])

    # this func is used to setup the particles in one traverse, built by York
    def setup(self):
        # use np.random to init the particles
        self.x_data = [[], []]
        self.y_data = [[], []]
        for i in range(self.Np):
            # temp val for a random particle
            tx = np.random.uniform(self.x_min, self.x_max)
            ty = np.random.uniform(self.y_min, self.y_max)
            # select the color of this particle and count each color in data0
            if self.con == 0:
                # if tx < 0:  # for test, very interesting
                if math.sqrt(math.pow(tx - self.r_x, 2) + math.pow(ty - self.r_y, 2)) < self.r:
                    self.x_data[1].append(tx)
                    self.y_data[1].append(ty)
                else:
                    self.x_data[0].append(tx)
                    self.y_data[0].append(ty)
            elif self.con == 1:
                if tx < 0:
                    self.x_data[1].append(tx)
                    self.y_data[1].append(ty)
                else:
                    self.x_data[0].append(tx)
                    self.y_data[0].append(ty)

    # the visualization of particle form, by The Kite
    def show_particle_form(self):
        # set the figure and pass in the coordinates of blue and red particles
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(self.x_data[1], self.y_data[1], s=1, c='b')
        plt.scatter(self.x_data[0], self.y_data[0], s=1, c='r')
        # set the layout of axis and title
        plt.title("2D problem", fontname='Arial', fontsize=30, weight='bold')
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
                ivl_xs = math.ceil(self.Nx - ((self.x_data[i][n] - self.x_min) / ivl_grid_x) - 1)
                ivl_ys = math.ceil(((self.y_data[i][n] - self.y_min) / ivl_grid_y) - 1)
                data0[ivl_xs][ivl_ys][i] += 1
        # transfer the data0 into data by calculating the proportion of blue particles in each grid
        for i in range(self.Nx):
            for j in range(self.Ny):
                data[i][j] = data0[i][j][1] / (data0[i][j][0] + data0[i][j][1])
        # the defination of colorbar of grid form
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

    # show 1D form if self.con == 1, by Ziqing and Zsolt
    def show_1D_form(self, j):
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
            for i in range(3):
                # setup the init list of particles for 3 times
                self.setup()
                # cycle in Classic Euler Method step by step
                for j in range(int(self.time_max / (2 * self.h))):
                    self.go_a_step()
                # show the 1D diagram when t = time_max / 2
                self.show_1D_form(i)
            plt.show()


# GUI for input the initial condition by William and George
class GUI(object):
    def __init__(self):
        # generate the window
        self.window = tk.Tk()
        self.window.title("Programme Inputs Group 16")
        self.window.resizable(width=False, height=False)

        # add label
        self.xminlab = tk.Label(text="x_min")
        self.xmaxlab = tk.Label(text='x_max')
        self.yminlab = tk.Label(text='y_min')
        self.ymaxlab = tk.Label(text='y_max')
        self.difflab = tk.Label(text="Diffusivity")
        self.timelab = tk.Label(text="Total time")
        self.steplab = tk.Label(text='Step time')
        self.spillxlab = tk.Label(text='Spill x coordinate')
        self.spillylab = tk.Label(text='Spill y coordinate')
        self.spill_radlab = tk.Label(text='Spill radius')
        self.Nx_lab = tk.Label(text='Nx')
        self.Ny_lab = tk.Label(text='Ny')
        self.Np_lab = tk.Label(text='Number of particles')
        self.init_condition_lab = tk.Label(text='Initial condition')
        self.veltype_lab = tk.Label(text='Velocity Type')
        self.plot_lab = tk.Label(text='Plot type')

        # add import interface
        self.xmininp = tk.Entry()
        self.xmaxinp = tk.Entry()
        self.ymininp = tk.Entry()
        self.ymaxinp = tk.Entry()
        self.diffinp = tk.Entry()
        self.timeinp = tk.Entry()
        self.stepinp = tk.Entry()
        self.spillxinp = tk.Entry()
        self.spillyinp = tk.Entry()
        self.spill_radinp = tk.Entry()
        self.Nxinp = tk.Entry()
        self.Nyinp = tk.Entry()
        self.Npinp = tk.Entry()
        self.init_condition = Combobox(state="readonly")
        self.init_condition["values"] = (
            "For 2D Problem",
            "For 1D Problem"
        )
        self.init_condition.current(0)
        self.veltype = Combobox(state="readonly")
        self.veltype["values"] = (
            "zero vel",
            "read from file"
        )
        self.veltype.current(0)
        self.plot = Combobox(state="readonly")
        self.plot["values"] = (
            "Particle",
            "Grid"
        )
        self.plot.current(0)

        input_var_lab = [self.xminlab, self.xmaxlab, self.yminlab, self.ymaxlab, self.difflab, self.timelab,
                         self.steplab, self.spillxlab, self.spillylab, self.spill_radlab, self.Nx_lab, self.Ny_lab,
                         self.Np_lab, self.init_condition_lab, self.veltype_lab, self.plot_lab]

        input_var_entries = [self.xmininp, self.xmaxinp, self.ymininp, self.ymaxinp, self.diffinp, self.timeinp,
                             self.stepinp, self.spillxinp, self.spillyinp, self.spill_radinp, self.Nxinp, self.Nyinp,
                             self.Npinp, self.init_condition, self.veltype, self.plot]

        for i in range(len(input_var_lab)):
            input_var_lab[i].grid(row=i, column=0, padx=5, pady=5, sticky='NW')

        for i in range(len(input_var_entries)):
            input_var_entries[i].grid(row=i, column=1, padx=5, pady=5, sticky='NW')

        self.submit = tk.Button(text="Submit", command=self.get_val)
        self.submit.grid(row=len(input_var_lab), column=3, padx=5, pady=5, sticky='N')

        # init condition, for temp save
        self.time_max = ''
        self.h = ''
        self.D = ''
        self.x_min = ''
        self.x_max = ''
        self.y_min = ''
        self.y_max = ''
        self.Nx = ''
        self.Ny = ''
        self.Np = ''
        self.vel_type = 0
        self.r = ''
        self.r_x = ''
        self.r_y = ''
        self.plot_type = 0
        self.con = 0

    # for the behave after click 'submit' button
    def get_val(self):
        self.x_min = self.xmininp.get()
        self.x_max = self.xmaxinp.get()
        self.y_min = self.ymininp.get()
        self.y_max = self.ymaxinp.get()
        self.D = self.diffinp.get()
        self.time_max = self.timeinp.get()
        self.h = self.stepinp.get()
        self.Nx = self.Nxinp.get()
        self.Ny = self.Nyinp.get()
        self.Np = self.Npinp.get()
        self.r = self.spill_radinp.get()
        self.r_x = self.spillxinp.get()
        self.r_y = self.spillyinp.get()
        self.vel_type = self.veltype.current()
        self.plot_type = self.plot.current()
        self.con = self.init_condition.current()
        # success message after click button
        messagebox.showinfo("Submit", "Submit successfully")
        self.window.destroy()

    # provide a interface to import data to TaskA class
    def imp_val(self, CMM):
        if len(self.x_min) != 0:
            CMM.x_min = float(self.x_min)
        if len(self.x_max) != 0:
            CMM.x_max = float(self.x_max)
        if len(self.y_min) != 0:
            CMM.y_min = float(self.y_min)
        if len(self.y_max) != 0:
            CMM.y_max = float(self.y_max)
        if len(self.D) != 0:
            CMM.D = float(self.D)
        if len(self.time_max) != 0:
            CMM.time_max = float(self.time_max)
        if len(self.h) != 0:
            CMM.h = float(self.h)
        if len(self.Nx) != 0:
            CMM.Nx = int(self.Nx)
        if len(self.Ny) != 0:
            CMM.Ny = int(self.Ny)
        if len(self.Np) != 0:
            CMM.Np = int(self.Np)
        if len(self.r) != 0:
            CMM.r = float(self.r)
        if len(self.r_x) != 0:
            CMM.r_x = float(self.r_x)
        if len(self.r_y) != 0:
            CMM.r_y = float(self.r_y)
        CMM.vel_type = self.vel_type
        CMM.plot_type = self.plot_type
        CMM.con = self.con

    # main code for the GUI class
    def main(self):
        self.window.mainloop()


# main code for the whole TaskA
if __name__ == '__main__':
    # build a instance
    gui = GUI()
    run = TaskA()
    # to run main code
    gui.main()
    gui.imp_val(run)
    run.main()
