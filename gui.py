import tkinter as tk
from tkinter.ttk import *
from tkinter import messagebox


# GUI for input the initial condition by William and George
class GUI:
    def __init__(self):
        # init condition, for temp save
        self.time_max = 0.4
        self.h = 0.0005
        self.D = 0.01
        self.x_min = -1
        self.x_max = 1
        self.y_min = -1
        self.y_max = 1
        self.Nx = 64
        self.Ny = 64
        self.Np = 65536
        self.vel_type = 0
        self.r = 0.3
        self.r_x = 0
        self.r_y = 0
        self.con = 0

        # loop the window1
        self.window = tk.Tk()
        self.window.title("Programme Inputs Group 16")
        self.window.resizable(width=False, height=False)
        self.title = tk.Label(text="Please select the initial condition:")
        self.init_condition_lab = tk.Label(text='Initial condition')
        self.init_condition = Combobox(state="readonly")
        self.init_condition["values"] = (
            "For 2D Problem Particle Form",
            "For 2D Problem Grid",
            "For 1D Problem",
            "For 1D Problem error simulation",
            "For simulation in TaskD"
        )
        self.init_condition.current(0)
        self.title.grid(row=1, column=0, padx=5, pady=5, sticky='NW')
        self.init_condition_lab.grid(row=2, column=0, padx=5, pady=5, sticky='NW')
        self.init_condition.grid(row=2, column=1, padx=5, pady=5, sticky='NW')
        self.button = tk.Button(text="Continue", command=self.get_con)
        self.button.grid(row=3, column=3, padx=5, pady=5, sticky='N')
        self.window.mainloop()

        # create window2
        self.window2 = tk.Tk()
        self.window2.title("Programme Inputs Group 16")
        self.window2.resizable(width=False, height=False)
        # window label
        self.xminlab = tk.Label(self.window2, text="x_min")
        self.xmaxlab = tk.Label(self.window2, text='x_max')
        self.yminlab = tk.Label(self.window2, text='y_min')
        self.ymaxlab = tk.Label(self.window2, text='y_max')
        self.difflab = tk.Label(self.window2, text="Diffusivity")
        self.timelab = tk.Label(self.window2, text="Total time")
        self.steplab = tk.Label(self.window2, text='Step time')
        self.spillxlab = tk.Label(self.window2, text='Spill x coordinate')
        self.spillylab = tk.Label(self.window2, text='Spill y coordinate')
        self.spill_radlab = tk.Label(self.window2, text='Spill radius')
        self.Nx_lab = tk.Label(self.window2, text='Nx')
        self.Ny_lab = tk.Label(self.window2, text='Ny')
        self.Np_lab = tk.Label(self.window2, text='Number of particles')
        self.veltype_lab = tk.Label(self.window2, text='Velocity Type')

        # add import interface
        self.xmininp = tk.Entry(self.window2)
        self.xmaxinp = tk.Entry(self.window2)
        self.ymininp = tk.Entry(self.window2)
        self.ymaxinp = tk.Entry(self.window2)
        self.diffinp = tk.Entry(self.window2)
        self.timeinp = tk.Entry(self.window2)
        self.stepinp = tk.Entry(self.window2)
        self.spillxinp = tk.Entry(self.window2)
        self.spillyinp = tk.Entry(self.window2)
        self.spill_radinp = tk.Entry(self.window2)
        self.Nxinp = tk.Entry(self.window2)
        self.Nyinp = tk.Entry(self.window2)
        self.Npinp = tk.Entry(self.window2)
        self.veltype = Combobox(self.window2, state="readonly")
        self.veltype["values"] = (
            "zero vel",
            "read from file"
        )
        self.input_val()

    def get_con(self):
        self.con = self.init_condition.current()
        self.window.destroy()

    def input_val(self):
        if self.con == 0 or self.con == 1:
            input_var_lab = [self.xminlab, self.xmaxlab, self.yminlab, self.ymaxlab, self.difflab, self.timelab,
                             self.steplab, self.spillxlab, self.spillylab, self.spill_radlab, self.Nx_lab, self.Ny_lab,
                             self.Np_lab, self.veltype_lab]
            input_var_entries = [self.xmininp, self.xmaxinp, self.ymininp, self.ymaxinp, self.diffinp, self.timeinp,
                                 self.stepinp, self.spillxinp, self.spillyinp, self.spill_radinp, self.Nxinp,
                                 self.Nyinp, self.Npinp, self.veltype]
            for i in range(len(input_var_lab)):
                input_var_lab[i].grid(row=i, column=0, padx=5, pady=5, sticky='NW')
            for i in range(len(input_var_entries)):
                input_var_entries[i].grid(row=i, column=1, padx=5, pady=5, sticky='NW')
            button = tk.Button(text="Submit", command=self.get_val_1)
            button.grid(row=len(input_var_lab), column=3, padx=5, pady=5, sticky='N')
            # Add in the default values so they are visible and can be edited in GUI
            self.xmininp.insert(0, '-1')
            self.xmaxinp.insert(0, '1')
            self.ymininp.insert(0, '-1')
            self.ymaxinp.insert(0, '1')
            self.diffinp.insert(0, '0.01')
            self.timeinp.insert(0, '0.4')
            self.stepinp.insert(0, '0.005')
            self.spillxinp.insert(0, '0')
            self.spillyinp.insert(0, '0')
            self.spill_radinp.insert(0, '0.3')
            self.Nxinp.insert(0, '64')
            self.Nyinp.insert(0, '64')
            self.Npinp.insert(0, '65536')
            self.veltype.current(0)
            self.window2.mainloop()

        if self.con == 2:
            input_var_lab = [self.xminlab, self.xmaxlab, self.yminlab, self.ymaxlab, self.difflab, self.timelab,
                             self.steplab, self.Nx_lab, self.Np_lab, self.veltype_lab]
            input_var_entries = [self.xmininp, self.xmaxinp, self.ymininp, self.ymaxinp, self.diffinp, self.timeinp,
                                 self.stepinp, self.Nxinp, self.Npinp, self.veltype]
            for i in range(len(input_var_lab)):
                input_var_lab[i].grid(row=i, column=0, padx=5, pady=5, sticky='NW')
            for i in range(len(input_var_entries)):
                input_var_entries[i].grid(row=i, column=1, padx=5, pady=5, sticky='NW')
            button = tk.Button(text="Submit", command=self.get_val_2)
            button.grid(row=len(input_var_lab), column=3, padx=5, pady=5, sticky='N')
            # Add in the default values so they are visible and can be edited in GUI
            self.xmininp.insert(0, '-1')
            self.xmaxinp.insert(0, '1')
            self.ymininp.insert(0, '-1')
            self.ymaxinp.insert(0, '1')
            self.diffinp.insert(0, '0.1')
            self.timeinp.insert(0, '0.4')
            self.stepinp.insert(0, '0.005')
            self.Nxinp.insert(0, '64')
            self.Npinp.insert(0, '65536')
            self.veltype.current(0)
            self.window2.mainloop()

        if self.con == 3:
            title = tk.Label(text="Submit successfully")
            title.grid(row=1, column=0, padx=5, pady=5, sticky='NW')
            self.get_val_3()
            self.window2.mainloop()

        if self.con == 4:
            input_var_lab = [self.timelab, self.steplab, self.Nx_lab, self.Ny_lab, self.Np_lab, self.veltype_lab]
            input_var_entries = [self.timeinp, self.stepinp, self.Nxinp, self.Nyinp, self.Npinp, self.veltype]
            for i in range(len(input_var_lab)):
                input_var_lab[i].grid(row=i, column=0, padx=5, pady=5, sticky='NW')
            for i in range(len(input_var_entries)):
                input_var_entries[i].grid(row=i, column=1, padx=5, pady=5, sticky='NW')
            button = tk.Button(text="Submit", command=self.get_val_4)
            button.grid(row=len(input_var_lab), column=3, padx=5, pady=5, sticky='N')
            # Add in the default values so they are visible and can be edited in GUI
            self.timeinp.insert(0, '0.4')
            self.stepinp.insert(0, '0.005')
            self.Nxinp.insert(0, '64')
            self.Nyinp.insert(0, '64')
            self.Npinp.insert(0, '150000')
            self.veltype.current(1)
            self.window2.mainloop()

    # for the behave after click 'submit' button
    def get_val_1(self):
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
        # success message after click button
        messagebox.showinfo("Submit", "Submit successfully")
        self.window2.destroy()

    def get_val_2(self):
        self.x_min = self.xmininp.get()
        self.x_max = self.xmaxinp.get()
        self.y_min = self.ymininp.get()
        self.y_max = self.ymaxinp.get()
        self.D = self.diffinp.get()
        self.time_max = self.timeinp.get()
        self.h = self.stepinp.get()
        self.Nx = self.Nxinp.get()
        self.Np = self.Npinp.get()
        self.vel_type = self.veltype.current()
        # success message after click button
        messagebox.showinfo("Submit", "Submit successfully")
        self.window2.destroy()

    def get_val_3(self):
        self.x_min = -1
        self.x_max = 1
        self.y_min = -1
        self.y_max = 1
        self.D = 0.1
        self.time_max = 0.4
        self.h = 0.0005
        self.Nx = 64
        self.Np = 65536
        self.vel_type = 0
        self.window2.destroy()

    def get_val_4(self):
        self.x_min = -1
        self.x_max = 1
        self.y_min = -1
        self.y_max = 1
        self.D = 0.1
        self.time_max = self.timeinp.get()
        self.h = self.stepinp.get()
        self.r = 0.1
        self.r_x = 0.4
        self.r_y = 0.4
        self.Nx = self.Nxinp.get()
        self.Ny = self.Nyinp.get()
        self.Np = self.Npinp.get()
        self.vel_type = self.veltype.current()
        # success message after click button
        messagebox.showinfo("Submit", "Submit successfully")
        self.window2.destroy()
