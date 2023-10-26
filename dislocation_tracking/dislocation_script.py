import numpy as np
from matplotlib import pyplot as plt
import glob
from natsort import natsorted
from dislocation_analysis import *

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
        # dislocation = wrap_coords_y(dislocation,y_lim)
        dislocation = wrap_coords(dislocation,"y",y_lim)


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

###################################################################
#                                                                 #
#                      Dislocation Tracking                       #
#                                                                 #
###################################################################
position, vertices, t = track_disloc(avg, x_lim)

###################################################################
#                                                                 #
#                              Plots                              #
#                                                                 #
###################################################################

# Partial Dislocations
partial_1 = position[0]
partial_2 = position[1]

partial_1_vel = get_velocity(partial_1, step)
partial_2_vel = get_velocity(partial_2, step)

labels = ["Partial 1", "Partial 2"]

fig, axs = plt.subplots(1, 2 , figsize=(11,5))
fig.suptitle("Partial Dislocations")

axs[0].set_title("Position")
axs[0].plot(t[0], partial_1, label=labels[0])
axs[0].plot(t[1], partial_2, label=labels[1])

axs[0].set_ylabel(r"$x$ $(\mathring A)$")

# Plot perfect dislocation velocity against time
axs[1].set_title("Velocity")
axs[1].plot(t[0], partial_1_vel, label=labels[0])
axs[1].plot(t[1], partial_2_vel, label=labels[1])
axs[1].set_ylabel(r"$v_{x}$ ($\mathring A$/ps)")

for ax in axs:
    ax.set_xlabel(r"$t$ (ps)")
    ax.legend()
plt.show()
fig.savefig("partials_pos_vel.png", dpi=350, format="png")

# Perfect Dislocation
# Average coordinates of partials
perfect_disloc = perfect_disloc_coords(partial_1, partial_2)

# Calculate perfect dislocation velocities
velocity = get_velocity(perfect_disloc, step)

fig1, axs = plt.subplots(1, 2 , figsize=(11,5))
fig1.suptitle("Perfect Dislocation")

axs[0].set_title("Position")
axs[0].plot(t[0], perfect_disloc)
axs[0].set_ylabel(r"$x$ $(\mathring A)$")


# Plot perfect dislocation velocity against time
axs[1].set_title("Velocity")
axs[1].plot(t[0], velocity)
axs[1].set_ylabel(r"$v_{x}$ ($\mathring A$/ps)")

for ax in axs:
    ax.set_xlabel(r"$t$ (ps)")

plt.show()
fig1.savefig("perfect_pos_vel.png", dpi=350, format="png")

# Write coordinates and velocity of perfect dislocation
filename = "perfect_pos.txt"
write_prop(filename, perfect_disloc, time)

filename = "perfect_vel.txt"
write_prop(filename, velocity, time)