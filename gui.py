import tkinter as tk
from tkinter.ttk import *
from tkinter import messagebox


# GUI for input the initial condition by William and George
class GUI:
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
        self.plot_lab = tk.Label(text='Plot type (2D Only)')

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
            "For 1D Problem",
            "For 1D Problem error simulation",
            "For simulation in TaskD"
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

    # main code for the GUI class
    def main(self):
        self.window.mainloop()
