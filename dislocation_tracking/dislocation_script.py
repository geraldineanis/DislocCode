import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits import mplot3d
import os
import glob
from natsort import natsorted
from disloc_analysis import *

###################################################################
#                                                                 #
#                              Analysis                           #
#                                                                 #
###################################################################

# List frames
disloc_frames = natsorted(glob.glob("./disloc_ppt/disloc_ppt_data_*.txt"))

step = 500*0.001

# Dislocation lines coordinates
x_coords = []
y_coords = []
z_coords = []

# Time
time = []
for i in range(len(disloc_frames)):
    time.append(i*step)

# Position averages
avg = []
avg_abs = []

for i, frame in enumerate(disloc_frames):

    dislocations = read_disloc_data(frame)

    # Process dislocation data
    # Cell extents
    # x_lim = 350.95123763850734      # dense/sparse
    x_lim = 301.17092024297432      # config_3
    y_lim = 172.44407789193573
    z_lim = 241.889

    x_coords_i = []
    y_coords_i = []
    z_coords_i = []

    avg_pos = []

    for dislocation in dislocations:
        # Wrap coordinates
        dislocation = wrap_coords_y(dislocation,y_lim)

        # Get coordinates of dislocation line vertices
        x, y, z = get_disloc_coords(dislocation)
        x_coords_i.append(x)
        y_coords_i.append(y)
        z_coords_i.append(z)

        # Get average position
        avg_pos.append(get_avg_pos(dislocation))

    # Append coordinates
    x_coords.append(x_coords_i)
    y_coords.append(y_coords_i)
    z_coords.append(z_coords_i)

    # Append average positions
    avg.append(np.array(avg_pos))

for i in range(len(avg)):
    avg_abs_i = []
    for j in range(len(avg[i])):
        avg_abs_i.append(math.hypot(avg[i][j][0],avg[i][j][1],avg[i][j][2]))
    avg_abs.append(avg_abs_i)

avg_abs = np.array(avg_abs)

for i in range(len(avg)):
    for j in range(len(avg[i])):
        if avg[i][j][0] > x_lim:
            avg[i][j][0] -= x_lim            
        if avg[i][j][0] < 0.0:
            avg[i][j][0] += x_lim       

# print(np.shape(avg))

# print(avg[0])

# avg_x = np.array([[i,avg[i][j][0]] for i in range(len(avg)) for j in range(len(avg[i]))])

# # for i in range(len(avg)):
# #     for j in range(len(avg[i])):
# #         plt.plot(i*0.001*500,avg[i][j][0], "x")
# plt.plot(avg_x[:,0],avg_x[:,1], "x")
# plt.ylabel("x")
# plt.xlabel("t")
# plt.show()

###################################################################
#                                                                 #
#                        Dislocation Tracking                     #
#                                                                 #
###################################################################
n_frames = len(disloc_frames)

position, t = track_disloc(avg, x_lim, n_frames)

   
# Velocity 
vel = []
for i in range(4):
    vel.append(get_velocity(position[i], del_t=500*0.001))

###################################################################
#                                                                 #
#                        Plots and Animation                      #
#                                                                 #
###################################################################

fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
for i in range(4):
    # ax1.scatter(time[0:len(position[i])],position[i], marker="o", s=2, label=f"Dislocation {i+1}")
    ax1.scatter(t[i],position[i], marker="o", s=2, label=f"Partial disloc. {i+1}")
# ax1.axvline(30.0, ls="--", color="tab:blue") # Both dislocations have gone once through the ppt
ax1.set_xlabel("Time (ps)")
ax1.set_ylabel(r"x $(\mathring A)$")

ax2 = fig.add_subplot(122)
for i in range(4):
    ax2.plot(t[i][2:],vel[i][2:], label=f"Partial disloc. {i+1}")
# ax2.axvline(30.0, ls="--", color="tab:blue") #Both dislocations have gone once through the ppt
ax2.set_xlabel("Time (ps)")
ax2.set_ylabel(r"$v_{x} (\mathring A/ps$)")

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 10}, bbox_to_anchor=(1.0, 1.0))


fig.suptitle("Shear Stress = 250 MPa")
# plt.savefig("250_MPa.png", format="png", dpi=350, bbox="tight")

# ax3 = fig.add_subplot(133)
# for i in range(len(vel)):
#     ax3.plot(ma_vel[i][2:], label=f"Dislocation {i+1}")
# ax3.legend()

plt.show()

# # Precipitate representation as a sphere
# gamma_latt = 3.52
# ppt_radius = 10.5*np.sqrt(2)*gamma_latt/2.0
# u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
# x = ((60.0+40.0)*np.sqrt(2)*gamma_latt/2.0) +  (ppt_radius*np.cos(u)*np.sin(v))
# y = (20.0*np.sqrt(6)*gamma_latt/2.0) + (ppt_radius * np.sin(u) * np.sin(v))
# z = (20.0*np.sqrt(3)*gamma_latt) + (ppt_radius * np.cos(v))

# fig = plt.figure(figsize=(10,10))
# ax = plt.axes(projection="3d")

# def animate(i):
#     """
#     Animation function
#     """
#     ax.clear()

#     for j in range(len(dislocations)):
#         # Dislocation lines
#         ax.scatter(x_coords[i][j], y_coords[i][j], z_coords[i][j],\
#                    s=1, color="tab:blue", label=f"Shockley Partial {j+1}")
#         # # Average position of each Shockley partial
#         # ax.scatter(avg[i][j][0], avg[i][j][1], avg[i][j][2],\
#         #            color="tab:red", label=f"Average Position {j+1}")


#     # ax.plot_wireframe(x, y, z, color="tab:blue",ls="--",lw=0.5, label=r"Ni$_{3}$Al Precipitate")

#     # Title
#     ax.set_title(f"Time = {i*step} ps", pad=20)

#     # Axes labels
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")

#     # # Axes Limits
#     # ax.set_xlim(0,x_lim)
#     # ax.set_ylim(0,y_lim)
#     # ax.set_zlim(0,z_lim)

#     # Show legend
#     # ax.legend()

# anim = animation.FuncAnimation(fig, animate, frames=len(disloc_frames), interval=500)

# # print(len(time))

# # # Plot absolute average position
# # fig1 = plt.figure(figsize=(10,5))
# # ax1 = fig1.add_subplot(121)
# # for i in range(4):
# #     ax1.scatter(time,avg_abs[:,i], color="tab:red")
# # # ax1.scatter(time,avg_abs[:,1])
# # ax1.set_xlabel("Time (ps)")
# # ax1.set_ylabel("Average Position")
# # # ax1.legend()

# # # # Plot dislocation velocity
# # # ax2 = fig1.add_subplot(122)
# # # ax2.scatter(time,vel)
# # # ax2.set_xlabel("Time (ps)")
# # # ax2.set_ylabel(r"Velocity ($\mathring A/ps$)")

# plt.show()
