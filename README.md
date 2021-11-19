# CMM3Project - Group 16

#### Introduce

This is a project storage for 2021 Year3 Computational Method and Modeling 3 Group 16 Work in the University of Edinburgh.

To know more about this project, you can watch lecture7.pdy slide on Learn Page.

The [main.py][1] is our main code. The file contains reference demos written by the team members and other groups.

Be warn! The deadline of this project is **23th Nov**.

---

#### Goals of the Project

- Design and implement a simulation code, based on a Lagrangian particle-based method, for the advection and diffusion of a substance
- Use the developed code to perform simulations to address engineering questions

---

#### Tasks

**Tasks A. Develop a python code with some functionality.**
  
In Task A, we will need to generate an invitational code to solve the advetion diffusion problem in a 2D and 1D problem.

After initializing the conditions, we firstly generate the position of the particles position, then use Euler method and Gaussian random numbers to calculate the position of each particle after a step time, and display it with a grid graph. Next step, repeat the position after the next step time Calculate until touch the maximum time. And we created a velocity_field_setup method to input the velocity from a file given by Learn.

**Tasks B. Validation**
  
In Task B, we need to use the code in A to deal with some problems in 2D and 1D problem. And in additional conditions, we may need to input the velocity field. 

Besides, in 1D problem, error simulation is requested. So we add an initial condition to simulate the error in 1D problem. We firstly import the reference data from the file given from Learn Page and use it to fit a func, then compare the real data with the func root to count the global error. We used h = 0.05 to simulate the relationship between Np and E, which Np = 2 ^ i (from 0 to 18) and E is globe error (root_mean_square_error) and used Np = 1024 to simulate the relationship between h and E, which h = (0.0005, 0.2, 0.0005).

In E VS Np simulation, we use method to fix the function E = a * Np ^ b and get the coefficients where b= -0.2385269253404275.

**Tasks C. Develop a user interface**
  
In Task C, we create a GUI window by using tkinter lib. We create 15 interface including initial condition, max time, step time, domain range, diffusivity, number of grids, number of particles, velocity type, radius and Center of circle of blue area and plot type.

To make it easy to use for several times, we set up 5 initial conditions, including 2D problem show in particle form and grid, 1D problem and its simulation, Task D simulation. Each initial condition has its own default condition value. For each task, users just need to select initial condition and click "submit" button if they use the default conditions.

**Tasks D. Use the python code you developed to perform the following engineering simulations**

Like Task A, we need to use the initial condition gaven by the slide and plot it into a grid form. But what is different with Task A grid form is that this form need to keep track. So we rebuilt the show method to meet the request.

---

#### For Issues and Request

If you find any place could be improved, please put your Issues or Request on GitHub or, simply contact with group member who coded it(the name will be sign in the code note). We may deal with it very soon and thanks for your contribution.

---

#### Group Members

- Azureen Ab Wahab

- Zsolt Csonka

- Konstantions Koulas

- William Lauder

- George Panagiotopoulos

- Ziqing Luo

- The Kite (Zekai Tao)

- York Hsu (Yunhao Xu)

[1]: https://github.com/WDRshadow/CMM3Project/blob/master/main.py
