# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 08:54:50 2021

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from random import gauss
import matplotlib as mpl
from matplotlib.colors import ListedColormap

n_particles = 12000    # the number of particles I want
D = 0.01            # the coefficient of diffusivity
h = 0.1             # the step size
n_steps = 30        #the number of steps

#variables below set the division of number of particles given into 2 types of particles
red_particles=int(n_particles/5) 
blue_particles=int(n_particles*(4/5))

#the function below generates a random array of x and y coordinates whithin set bounds    
def setup(x_min,x_max, y_min, y_max):
    #arrays for particles of type 1
    x_coord1 = [np.random.uniform(x_min,x_max) for i in range(red_particles)]
    y_coord1= [np.random.uniform(y_min,y_max) for i in range(red_particles)]
    #arrays for particles of type 2
    x_coord2=[]
    y_coord2=[]
    for i in range(blue_particles):
        x= np.random.uniform(-1,1)
        y= np.random.uniform(-1,1)
        if -0.3<x<0.3  and -0.3<y<0.3:
            continue
        x_coord2.append(x)            
        y_coord2.append(y)
    return x_coord1, y_coord1, x_coord2, y_coord2

#here I am calling the function above to generate an array of x and y coordinates between -0.3 and 0.3
x_coord1, y_coord1, x_coord2, y_coord2 = setup(-0.3, 0.3, -0.3, 0.3)

#the function below updates the coordinates of each particle using the diffusion part of equations 6 and 7
def take_step(x_coord1, y_coord1, x_coord2, y_coord2):
    #type 1 particles
    for i in range(red_particles):
        randx = gauss(0,1)                             #generates a random number from the normal guassian dist
        randy = gauss(0,1)                             #generates a random number from the normal guassian dist
        x_coord1[i] += np.sqrt(2*D)*np.sqrt(h)*randx    #basically a python version of equation 6
        y_coord1[i] += np.sqrt(2*D)*np.sqrt(h)*randy    #a python version of equation 7
        
    #type 2 particles   
    for i in range(len(x_coord2)):    
        randx = gauss(0,1)
        x_coord2[i] += np.sqrt(2*D)*np.sqrt(h)*randx 
    for i in range(len(y_coord2)):
        randy = gauss(0,1)
        y_coord2[i] += np.sqrt(2*D)*np.sqrt(h)*randy 
    return x_coord1, y_coord1, x_coord2, y_coord2           #returns the new (updated) array of x and y coordinates


#this code should customize the colorbar, I didn't manage to make it work yet 
cmap = ListedColormap(["red", "blue"])
norm = norm = mpl.colors.Normalize(vmin=0, vmax=1)


#the function below runs and plots the location of each of the particles at each step
for i in range(n_steps):
    x_coord1, y_coord1, x_coord2, y_coord2 = take_step(x_coord1, y_coord1, x_coord2, y_coord2)     #runs the function above to update the coordinates of the array
    fig=plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)  
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    im1= ax1.scatter(x_coord1, y_coord1, s=10,c="red")
    im2=  ax1.scatter(x_coord2, y_coord2,s=10, c= "blue")
    im3=ax1.scatter(0,0,c=0,cmap=cmap, norm=norm)
    fig.colorbar(im3)
    plt.show()

