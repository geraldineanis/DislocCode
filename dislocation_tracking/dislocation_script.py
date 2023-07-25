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
disloc_frames = natsorted(glob.glob("./Ni_disloc/disloc_data_*.txt"))

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


x_lim, y_lim, z_lim = get_cell_lims("dump.shear_unwrap.0")

for i, frame in enumerate(disloc_frames):

    # Read in cell extents from dump file
    dislocations = read_disloc_data(frame)

    # Process dislocation data
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
        # Also save coordinates of all vertices
        avg_pos.append([get_avg_pos(dislocation),list(zip(x,y,z))])

    # Append coordinates
    x_coords.append(x_coords_i)
    y_coords.append(y_coords_i)
    z_coords.append(z_coords_i)

    # Append average positions
    avg.append(np.array(avg_pos, dtype=object))

# for i in range(len(avg)):
#     for j in range(len(avg[i])):
#         if avg[i][j][0] > x_lim:
#             avg[i][j][0] -= x_lim            
#         if avg[i][j][0] < 0.0:
#             avg[i][j][0] += x_lim

# avg_x_pos = [avg[i][j][0][0] for i in range(len(avg)) for j in range(len(avg[i]))]

# for i in range(len(avg_x_pos)):
#     if i <= 30 and avg_x_pos[i] > 0.9*x_lim:
#         avg_x_pos[i] -= x_lim

# plt.plot(avg_x_pos, "o")
# plt.show()

# # Fix for part of dislocation starting beyond pbc
# for i in range(len(avg)):
#     for j in range(len(avg[i])):
#         if i <= 30 and avg[i][j][0][0] > 0.9*x_lim:
#             avg[i][j][0][0] -= x_lim


###################################################################
#                                                                 #
#                        Dislocation Tracking                     #
#                                                                 #
###################################################################
n_frames = len(disloc_frames)

position, vertices, t = track_disloc(avg, x_lim, n_frames)

# ###################################################################
# #                                                                 #
# #                       Calculations and Plots                    #
# #                                                                 #
# ###################################################################
# for i in range(len(position)):
#     plt.scatter(t[i], position[i], label=f"{i}")
# plt.legend()
# plt.show()

# Average coordinates of partials
# Leading dislocation
leading_disloc = perfect_disloc_coords(position[0], position[1])
# # Trailing dislocation
# trailing_disloc = perfect_disloc_coords(position[2], position[3])

perfect_disloc = [leading_disloc]
labels=["Perfect dislocation"]

# Calculate perfect dislocation velocities
velocities = [get_velocity(disloc, step) for disloc in perfect_disloc]

# Plot perfect dislocation position against time
plt.figure(figsize=(7,5))
for i in range(len(perfect_disloc)):
    plt.scatter(t[0][:24], perfect_disloc[i][:24], label=labels[i], s=2)
# plt.title("Dislocation Trajectory")
plt.xlabel("t (ps)")
plt.ylabel(r"x $(\mathring A)$")
plt.legend()
plt.savefig("disloc_pos.png", dpi=350, format="png")
plt.show()

# Plot perfect dislocation velocity against time
plt.figure(figsize=(7,5))
for i in range(len(velocities)):
    plt.plot(t[0], velocities[i], label=labels[i])
# plt.axvline(19.0, ls="--", lw=1, c="tab:red")
# plt.title("Dislocation Velocity")
plt.xlabel("t (ps)")
plt.ylabel(r"$v_{x}$ $(\mathring A/ps)$")
plt.legend()
plt.savefig("disloc_vel.png", dpi=350, format="png")
plt.show()

# # Write coordinates for each dislocation
# Leading dislocation
# filename = "perfect_pos.txt"
# write_prop(filename, perfect_disloc[0], time)
# Trailing dislocation
# filename = "trailing_pos.txt"
# write_prop(filename, perfect_disloc[1], time)

# Write velocities for each dislocation
# Leading dislocation
# filename = "perfect_vel.txt"
# write_prop(filename, velocities[0], time)
# Trailing dislocation
# filename = "trailing_pos.txt"
# write_prop(filename, perfect_disloc[1], time)

