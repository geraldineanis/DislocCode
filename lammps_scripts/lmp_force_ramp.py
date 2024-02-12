"""! @ref
An example LAMMPS python script to set up and run a dislocation MD simulation in pure Ni.

This script shows how to set up and run an MD dislocation simulation using the LAMMPS Python library. The script sets up a simulation of a single  
in a pure Ni cell, where a shear stress of 10 MPa is applied to the simulation cell in order to move the dislocation. Periodic boundary conditions are applied along x and y, and a shrink 
boundary condition (non-periodic) in z, such that the dislocations are infinitely long. The dislocations lie in the (111) plane and the applied shear stress is applied perpendicular to the 
dislocation such that it moves in the [110] (x) direction.


The starting structure is constructed by reading in three .data files: Ni_start.data (1st padding layer); disloc_Ni.data (cell with dislocation); and Ni_end.data (2nd padding layer).
Once the starting structure is constructed, the script runs the following simulation steps:
- Energy minimization: for the a/2<110>{111} edge dislocation considered here, this is where the expected dissociation into two partial dislocations occurs
- Equilibration (NVT ensemble)
- MD - force ramp: gradually applying a shear force to the simulation cell (NVT).
- MD - constant force: applying a constant shear force to the simulation cell (NVT)

The user can adjust some additional simulation settings listed at the beginning of the script, which include:
- Running a new simulation or continuing a simulation from a LAMMPS restart file (only for MD stage).
- Temperature and pressure
- Applied shear stress
- Number of equilibration and MD timesteps
- loading type - Symmetric or assymetric loading. In the symmetric loading case, both the top few and bottom few layers of the simulation box are fixed and the shear force is applied to each 
in opposite directions. In the assymetric loading case, only the bottom few layers are fixed and the shear force is applied to the top few layers of the simulation box.

The interatomic potential used is the Mishin 2004 potential, which can be changed to any other potential with appropriate adjustment of settings .

This script can be run in parallel using the following command, where X is replaced by the number of processors:
@code mpirun -np <X> python3 lmp_pre_strain.py
"""

from lammps import lammps
from lammps.formats import LogFile as lmp_log
import numpy as np
import os
import sys
from lmp_wrappers import *
from mpi4py import MPI

# Select "Ni" or "Ni3Al" 
sys_type = "Ni"

# Set lattice parameter
if sys_type == "Ni":
    gamma_latt = 3.52   
elif sys_type == "Ni3Al":
    gamma_latt = 3.57
else:
    print("Error: sys_type should be to 'Ni' or 'Ni3Al'")
    sys.exit()

# Set simulation type
# if sim_type = "new" do structural relaxation and equilibration
# if sim_type = "continue" skip to production
# give last timestep to be read from dump file - later change to binary file for precision
sim_type = "new"

# Only change this if doing a restart 
# i.e. sim_type = "continue"
restart_file = "equil_restart.10000"
last_timestep = 0

# Set problem type
# if problem = "relax" perform an energy minimization
# if problem = "single" calculate energy of struc. only
problem = "single"

# Ensemble
# options: "NVE", "NVT", or "NPT"
thermo_ensemble = "NVT"
timestep = 0.001 # timestep (ps)

# User-set parameters
temperature = 300   # Temperature (K)
pressure = 1        # Pressure (bar)
sigma = 100         # Shear stress (bar) - 10 MPa
loading_type = "sym"
pre_strain = np.sqrt(2)*gamma_latt/4.0

# Number of MD timesteps
equil_run = 10000
MD_run_ramp = 20000
MD_run_const = 100000

if sim_type == "continue":
    MD_run_const = MD_run_const - last_timestep

# Energy conversion
eV_to_J = 1602191.7

# Random seed for velocity
vel_seed = np.random.randint(10000,99999)

# Block sizes to set offset in x for different layers
# and simulation box dimensions
x1 = 10
x2 = 21
x3 = 45

y1 = 40
z1 = 50

# Additional options
check_overlap = False

#####################################################
#               System initialization               #
#####################################################

# create a LAMMPS instance
lmp = lammps()

units = "metal"

block = f"""
clear
units {units}
dimension 3
boundary p p s
atom_style atomic
atom_modify map array
"""
lmp.commands_string(block)

# define a simulation box
# Create lattice to setup simulation box
lmp.command(f"lattice none {gamma_latt}")
# Simulation box parameters
box_name = "sim_box" # region-ID
box_dim = [(x1+x2+x3)*np.sqrt(2)/2.0, y1*np.sqrt(6)/2.0, z1*np.sqrt(3)]
species = 1  # number of atom types to use in simulation
# Define simulation box
lmp.command(f"region {box_name} block 0 {box_dim[0]} 0 {box_dim[1]} 0 {box_dim[2]}")
lmp.command(f"create_box {species} {box_name}")

# read in structure from data file
# 1st padding layer
lmp.command(f"read_data Ni_start.data group pad_1 add append")
# Cell containing dislocation(s)
x_shift  = x1*gamma_latt*np.sqrt(2)
lmp.command(f"read_data disloc_Ni.data group disloc shift {x_shift} 0. 0. add append")
# 2nd padding layer
x_shift  = (x1+x2)*gamma_latt*np.sqrt(2)
lmp.command(f"read_data Ni_end.data group pad_2 shift {x_shift} 0. 0. add append")


######################################################
#             Define Interatomic Potential           #
######################################################
potential = "./NiAl_Mishin_2004.eam.alloy"
elements = ["Ni", "Al"]

lmp.command("pair_style eam/alloy")

if sys_type == "Ni":
    lmp.command(f"pair_coeff * * {potential} {elements[0]}")
else:
    lmp.command(f"pair_coeff * * {potential} {elements[0]} {elements[1]}")

lmp.command("neighbor 2.0 bin")
lmp.command(f"neigh_modify delay 10 check yes")

######################################################
#             Check for overlapping atoms            #
######################################################
if check_overlap and (sim_type == "new"):
    lmp.command("delete_atoms overlap 0.3 all all")

######################################################
#    Define lower, upper, and mobile atom groups     #
######################################################
# Group definitions are written to restarts
# Only needed if new simulation otherwise skip

upper_lim = lmp.extract_box()[1][2] - (3.0*gamma_latt*np.sqrt(3))
lower_lim = lmp.extract_box()[0][2] + (3.0*gamma_latt*np.sqrt(3))

lmp.command(f"region upper block INF INF INF INF {upper_lim} INF units box")
lmp.command(f"region lower block INF INF INF INF INF {lower_lim} units box")

# definition of groups
block = f"""
group upper region upper
group lower region lower
group mobile subtract all upper lower
"""
lmp.commands_string(block) 

# define group for boundary fix
lmp.command("group boundaries union lower upper")

#######################################################
#                   Define Computes                   #
#######################################################

block = f"""
timestep {timestep}
compute temp1 mobile temp # Define temperature as that of mobile atoms
"""
lmp.commands_string(block)

# Calculate energy per atom
lmp.command("compute eng mobile pe/atom")
lmp.command("compute pe_mobile mobile reduce sum c_eng")

# Compute stress/atom
lmp.command("compute 1 mobile stress/atom temp1 virial")
lmp.command("compute 2 mobile reduce sum c_1[4]")
lmp.command("compute 3 mobile reduce sum c_1[5]")

######################################################
#                  Energy Minimization               #
######################################################
# Write new log file
lmp.command("log log.minim")

# Fix upper and lower regions
lmp.command(f"fix 2 lower setforce NULL 0.0 0.0")
lmp.command(f"fix 3 upper setforce NULL 0.0 0.0")

# Define dumps
lmp.command("dump 1 all custom 100 dump.minim_1.* id type xs ys zs fx fy fz c_eng")
lmp.command("dump_modify 1 first yes")

# Run minimization
if (problem == "relax"):
    # Perform an energy minimization
    block = """
    reset_timestep 0
    thermo 1
    thermo_style custom step pe lx ly lz press
    min_style cg
    minimize 1e-12 1e-12 5000 10000
    """
elif (problem == "single"):
    # Calculate energy of structure
    block = """
    thermo_style custom step pe lx ly lz press
    run 0
    """
lmp.commands_string(block)

######################################################
#                    Equilibration                   #
######################################################
# Write new log file
lmp.command("log log.equil")

# Undump previous dump (keep fixes)
lmp.command("undump 1")

# Thermo outputs and dumps
lmp.command("thermo 1")
lmp.command("thermo_style custom step pe ke lx ly lz pxx pyy pzz pxy pyz pxz temp c_pe_\
mobile")
lmp.command("dump 1 all custom 500 dump.equilibration.* id type x y z vx vy vz fx fy fz\
 c_eng")

# Set timestep
block = f"""
timestep {timestep}
reset_timestep 0
"""
lmp.commands_string(block)

# Initialize velocities
block = f"""
velocity mobile create {temperature} {vel_seed} units box
velocity mobile zero linear
velocity mobile zero angular
"""

lmp.commands_string(block)

# Temperature/Pressure control (Set only once)
# Ensemble chosen by setting thermo_ensemble variable at beginning
# of script
thermo_type(lmp, thermo_ensemble, temperature, pressure, timestep)


# Run equilibration
lmp.command("restart 5000 equil_restart")
lmp.command(f"run {equil_run}")

######################################################
#                  MD - Shear (Ramp)                 #
######################################################
lmp.command("log log.md_ramp")

# Thermo outputs and dumps
lmp.command("undump 1")

force = "ramp"

lmp.command("thermo 1")
lmp.command("thermo_style custom step pe ke lx ly lz pxx pyy pzz pxy pyz pxz temp c_pe_mobile c_2 c_3")
lmp.command(f"dump 1 all custom 500 dump.shear_u_{force}.* id type xu yu zu vx vy vz fx fy fz c_1[1] c_1[2] c_1[3] c_1[4] c_1[5] c_1[6] c_eng")

# Calculate force
app_force_lower, app_force_upper, nlower, nupper, lx, ly = calculate_force(lmp,sigma,eV_to_J)

# Boundaries
# Boundary conditions
lmp.command("variable t equal 'time'")
lmp.command("variable theta equal '20.0'")
lmp.command(f"variable forceL equal '-({app_force_lower}*v_t)/v_theta'")
lmp.command(f"variable forceU equal '({app_force_upper}*v_t)/v_theta'")

force_lower = '${forceL}'
force_upper = '${forceU}'

block = f"""
fix 4 upper aveforce {force_upper} 0.0 0.0
fix 5 lower aveforce {force_lower} 0.0 0.0

fix 6 boundaries rigid group 2 upper lower
"""
lmp.commands_string(block)

# Temperature/Pressure control - already set in "equilibration"

# Run MD
lmp.command(f"reset_timestep 0")

lmp.command(f"restart 5000 md_ramp_restart")
lmp.command(f"run {MD_run_ramp}")

######################################################
#             MD - Shear (Constant force)            #
######################################################
lmp.command("log log.md_const")

# Thermo outputs and dumps
force = "const"

lmp.command("undump 1")
lmp.command("unfix 4")
lmp.command("unfix 5")

lmp.command(f"dump 1 all custom 500 dump.shear_u_{force}.* id type xu yu zu vx vy vz fx fy fz c_1[1] c_1[2] c_1[3] c_1[4] c_1[5] c_1[6] c_eng")

# Boundaries
# Boundary conditions

lmp.command(f"fix 4 upper aveforce {app_force_upper} 0.0 0.0")
lmp.command(f"fix 5 lower aveforce {-app_force_upper} 0.0 0.0")

# Run MD
lmp.command(f"reset_timestep 0")

lmp.command(f"restart 5000 md_const_restart")
lmp.command(f"run {MD_run_const}")


######################################################
#                    System Shutdown                 #
######################################################
MPI.Finalize()
