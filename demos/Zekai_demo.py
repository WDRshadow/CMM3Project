import numpy as np
import matplotlib.pyplot as plt
from random import gauss
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import math
# x_coord1 = [np.random.uniform(0,100) for i in range(10)]
Np = 65536
Xmin = Ymin = -1
Xmax = Ymax = 1
Particle_b = math.ceil(math.pi * (0.3**2)/1 * Np)
Particle_r = Np - Particle_b
x_pr = []
y_pr = []
for i in range(Particle_r):
    x = np.random.uniform(Xmin,Xmax)
    y = np.random.uniform(Xmin,Xmax)
    if (x**2 + y**2) >= 0.3**2:
        x_pr.append(x)
        y_pr.append(y)

x_pb = []
y_pb = []

for i in range(Particle_b):
    x = np.random.uniform(-0.3,0.3)
    y = np.random.uniform(-0.3,0.3)
    if (x**2 + y**2)<0.3**2:
        x_pb.append(x)
        y_pb.append(y)

cmap = ListedColormap(["red", "blue"])
norm = norm = mpl.colors.Normalize(vmin=0, vmax=1)
fig=plt.figure(figsize=(10,8))
ax2 = fig.add_subplot(111)
ax2.scatter(x_pb,y_pb,s = 1,c = 'b')
ax2.scatter(x_pr,y_pr,s = 1,c = 'r')
ax2.set_title("2D problem")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
plt.xlim([-1,1])
plt.ylim([-1,1])
ax2.set_xticks([-1,-0.5,0,0.5,1])
ax2.set_yticks([-1,-0.5,0,0.5,1])
im3=plt.scatter(0,0,c=0,cmap=cmap, norm=norm)
fig.colorbar(im3)
plt.show()