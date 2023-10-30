"""!
Dislocation trajectory post-processing script.

@ref This script demonstrates how the output of the OVITO DXA tool, which is used in ovito_disloc.py, can be post-processed to "track" these dislocations throughout the trajectory. 
The script ovito_disloc.py uses DXA to detect dislocations in Molecular Dynamics (MD) trajectories and outputs the coordinates of the vertices of each detected dislocation line.
However, the OVITO DXA tool does not index detected dislocations consistently across frames i.e. one dislocation can be indexed as dislocation 0
in one frame and dislocation 1 in a different frame. This makes it difficult to follow or "track" a specific dislocation across a trajectory for further analysis, such as studying its dynamics.

@ref This script shows how this can be overcome by using the tools contained in dislocation_analysis.py to extract trajectories (position vs. time data) for each distinct dislocation detected by OVITOS'
DXA. This script takes as input text files containing the output from DXA in the format described in the documentation of ovito_disloc.py.

@ref This script is used to analyse an MD simulation of an a/2<110>{111} edge dislocation in pure Ni. This dislocation dissociates into two partial dislocations. The scripts lmp_pre_strain.py, gen_1disloc_Ni.py,
and gen_111_atoms.py can be used to create the structure and run the MD simulation.

This script can be used to analyse trajectories containing any number of dislocations (or partial dislocations), given the following assumptions:
1. The dislocations are straight, edge dislocations.
2. The dislocations only move in the x direction.
3. The number of dislocations is constant across all frames
i.e. no dislocations are generated nor lost throughout the analysed trajectory.

Some plotting code is provided at the end of the script to illustrate its output. Two figures are generated each showing the average x-position vs. time and x-velocity vs. time plots for the:
1. Partial disocations.
2. Perfect dislocation (the average of the two partials)

@sa track_disloc
"""

import numpy as np
from matplotlib import pyplot as plt
import glob
from natsort import natsorted
from dislocation_analysis import *

# Post-processing OVITO DXA output

# List MD trajectory frames sorted by frame number
# The OVITO DXA output is stored in the disloc_data_*.txt files
# The wildcard * indicates the frame number
disloc_frames = natsorted(glob.glob("./Ni_disloc/disloc_data_*.txt"))

# timestep = writing frequency * MD timestep
timestep = 500*0.001

# Simulation Timesteps
sim_time = [i*timestep for i in range(len(disloc_frames))]

# Arrays to store coordinates of dislocation lines
x_coords = []
y_coords = []
z_coords = []
# Dislocation position averages
avg = []

# Read in simulation cell dimensions from a LAMMPS dump file
# This can be any dump file in the trajectory
x_lim, y_lim, z_lim = get_cell_lims("dump.shear_unwrap.0")

for i, frame in enumerate(disloc_frames):
    dislocations = read_disloc_data(frame)
    # Process dislocation data
    x_coords_i = []
    y_coords_i = []
    z_coords_i = []

    avg_pos = []

    for dislocation in dislocations:
        # Wrap coordinates in y
        dislocation = wrap_coords(dislocation,"y",y_lim)

        # Get coordinates of dislocation line vertices
        # From 
        x, y, z = get_disloc_coords(dislocation)
        x_coords_i.append(x)
        y_coords_i.append(y)
        z_coords_i.append(z)

        # Get the average position of a dislocation
        # Also save the coordinates of all vertices
        avg_pos.append([get_avg_pos(dislocation),list(zip(x,y,z))])

    # Append coordinates
    x_coords.append(x_coords_i)
    y_coords.append(y_coords_i)
    z_coords.append(z_coords_i)

    # Append average positions
    avg.append(np.array(avg_pos, dtype=object))

# Dislocation Tracking
position, vertices, t = track_disloc(avg, x_lim)

###################################################################
#                                                                 #
#                              Plots                              #
#                                                                 #
###################################################################

# Partial Dislocations
partial_1 = position[0]
partial_2 = position[1]

partial_1_vel = get_velocity(partial_1, timestep)
partial_2_vel = get_velocity(partial_2, timestep)

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
velocity = get_velocity(perfect_disloc, timestep)

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
write_prop(filename, perfect_disloc, sim_time)

filename = "perfect_vel.txt"
write_prop(filename, velocity, sim_time)