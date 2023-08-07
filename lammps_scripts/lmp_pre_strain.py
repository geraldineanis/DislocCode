###############################################################
# LAMMPS input script                        		          #
# MD simulation to move a/2<1-10>{111} dislocations in        #
# pure Ni/Ni3Al                                               #
# Requires 3 input structure files:                           #
#  - 1st padding layer                                        #  
#  - Cell containing dislocation(s)                           #
#  - 2nd padding layer                                        #
# OR can read a LAMMPS restart file                           #  
# Potential: NiAl_Mishin_2004.eam.alloy                       #
###############################################################        

from lammps import lammps
from lammps.formats import LogFile as lmp_log
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
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
restart_file = "md_restart.30000"
last_timestep = 30000

# Set problem type
# if problem = "relax" perform an energy minimization
# if problem = "single" calculate energy of struc. only
problem = "relax"

# Ensemble
# options: "NVE", "NVT", or "NPT"
thermo_ensemble = "NVT"
timestep = 0.001 # timestep (ps)

# User-set parameters
temperature = 300   # Temperature (K)
pressure = 1        # Pressure (bar)
sigma = 500         # Shear stress (bar) - 250 MPa
loading_type = "sym"

# Run times (ps)
equil_run = 2
MD_run = 2

if sim_type == "continue":
    MD_run = MD_run - last_timestep

# Energy conversion
eV_to_J = 1602191.7

# Block sizes to set offset in x for different layers
# and simulation box dimensions
x1 = 10
x2 = 21
x3 = 40

y1 = 20
z1 = 40

# Additional options
check_overlap = False

# Pre-strain
pre_strain = -np.sqrt(2)*gamma_latt/4.0

#####################################################
#             Temperature/Pressure Control          #
#####################################################
def thermo_type(thermo_ensemble):
    if thermo_ensemble == "NVE":
        block = f"""
        fix 1 mobile nve
        fix 2 mobile temp/rescale 1 {temperature} {temperature} 1.0 0.5
        fix_modify 2 temp temp1
        thermo_modify temp temp1
        """ 
    elif thermo_ensemble == "NVT":
        block = f"""
        fix 1 mobile nvt temp {temperature} {temperature} {100.0*timestep}
        fix_modify 1 temp temp1
        thermo_modify temp temp1
        """ 
    elif thermo_ensemble == "NPT":
        block = f"""
        # cannot apply barostat to a non-periodic dimension
        fix 1 mobile npt temp {temperature} {temperature} {100.0*timestep} &
                                x {pressure} {pressure} {1000.0*timestep} &
                                y {pressure} {pressure} {1000.0*timestep}
        fix_modify 1 temp temp1
        thermo_modify temp temp1
        """ 
    else:
        print("Error: thermo_ensemble should be set to 'NVE', 'NVT', or 'NPT'")
        sys.exit()

    lmp.commands_string(block)
#####################################################
#               System initialization               #
#####################################################

# create a LAMMPS instance
lmp = lammps()

if sim_type == "new":

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

elif sim_type == "continue":
    # read in structure from restart file
    lmp.command(f"read_restart {restart_file}")
else:
    print("Error: sim_type should be set to 'new' or 'continue'")
    sys.exit()  
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

if sim_type == "new":
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

#######################################################
#                   Define Computes            	#
#######################################################

# Calculate energy per atom
lmp.command("compute eng mobile pe/atom")
lmp.command("compute pe_mobile mobile reduce sum c_eng")

######################################################
#                   1st Minimization                 #
######################################################
if sim_type == "new":
    # Fix upper and lower regions
    lmp.command(f"fix 1 lower setforce 0.0 NULL 0.0")
    lmp.command(f"fix 2 upper setforce 0.0 NULL 0.0")

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
    else:
        print("Error: problem should be set to 'single' or 'relax'")
        sys.exit()

    lmp.commands_string(block)

######################################################
#                    Displace atoms                  #
######################################################
cell_extents = lmp.extract_box()
z_mid = ((cell_extents[1][2])/2.0)
lmp.command(f"region upper_pre block INF INF INF INF {z_mid} INF units box")

# definition of groups
block = f"""
group upper_pre region upper_pre
group lower_pre subtract all upper_pre
"""
lmp.commands_string(block)

lmp.command(f"displace_atoms upper_pre ramp x 0.0 {pre_strain} z {z_mid} {cell_extents[1][2]}")
lmp.command(f"displace_atoms lower_pre ramp x 0.0 {pre_strain} z {cell_extents[0][2]} {z_mid}")

lmp.command("dump 2 all custom 1 dump.displace.* id type xs ys zs fx fy fz c_eng")

######################################################
#                   2nd Minimization                 #
######################################################
if sim_type == "new":
    # Undump previous dumps
    lmp.command("undump 1")
    lmp.command("undump 2")


    # Fix upper and lower regions
    lmp.command(f"fix 1 lower setforce 0.0 NULL 0.0")
    lmp.command(f"fix 2 upper setforce 0.0 NULL 0.0")

    # Define dumps
    lmp.command("dump 1 all custom 100 dump.minim_2.* id type xs ys zs fx fy fz c_eng")
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
    else:
        print("Error: problem should be set to 'single' or 'relax'")
        sys.exit()

    lmp.commands_string(block)

######################################################
#                    Equilibration                   #
######################################################
if sim_type == "new":
    # Unfix/undump/uncompute previous fixes, dumps, and computes
    block = """
    unfix 1
    unfix 2
    undump 1
    """
    lmp.commands_string(block)

    # Thermo outputs and dumps
    lmp.command("thermo 1")
    lmp.command("thermo_style custom step pe ke lx ly lz pxx pyy pzz pxy pyz pxz temp c_pe_mobile")
    lmp.command("dump 1 all custom 500 dump.equilibration.* id type x y z vx vy vz fx fy fz c_eng")

    block = f"""
    timestep {timestep}
    reset_timestep 0
    compute temp1 mobile temp # Define temperature as that of mobile atoms"""
    lmp.commands_string(block)

    # Initialize velocities
    block = f"""
    velocity mobile create {temperature} 12345 units box
    velocity mobile zero linear
    velocity mobile zero angular
    """
    lmp.commands_string(block)

    # Temperature/Pressure control
    # Ensemble chosen by setting thermo_ensemble variable at beginning
    # of script
    thermo_type(thermo_ensemble)

    # Boundary conditions
    block = """
    fix 3 lower setforce 0.0 NULL 0.0
    fix 4 upper setforce 0.0 NULL 0.0
    """
    lmp.commands_string(block)

    # Run equilibration
    lmp.command(f"restart 1000 equil_restart")
    lmp.command(f"run {equil_run}")

######################################################
#                       MD - Shear                   #
######################################################
# Unfix/undump/uncompute previous fixes, dumps, and computes
# For NVE we have 2 fixes
if sim_type == "new":
    if thermo_ensemble == "NVE":
        block = f"""
        unfix 1
        unfix 2
        unfix 3
        unfix 4
        undump 1
        uncompute temp1
        """
    else:
        block = f"""
        unfix 1
        unfix 3
        unfix 4
        undump 1
        uncompute temp1
        """
    lmp.commands_string(block)

# Thermo outputs and dumps
lmp.command("thermo 1")
lmp.command("thermo_style custom step pe ke lx ly lz pxx pyy pzz pxy pyz pxz temp c_pe_mobile")
lmp.command("dump 1 all custom 5000 dump.shear.* id type x y z vx vy vz fx fy fz c_eng")
lmp.command("dump 2 all custom 500 dump.shear_unwrap.* id type xu yu zu vx vy vz fx fy fz c_eng") # Unwrapped coordinates

block = f"""
timestep {timestep}
compute temp1 mobile temp # Define temperature as that of mobile atoms
"""
lmp.commands_string(block)

block = """
variable nupper equal count(upper)
variable nlower equal count(lower)
variable length_x equal "lx"
variable length_y equal "ly"
"""
lmp.commands_string(block)

nupper = lmp.extract_variable("nupper")
nlower = lmp.extract_variable("nlower")
lx = lmp.extract_variable("length_x")
ly = lmp.extract_variable("length_y")

app_force_upper = (lx*ly)/nupper*sigma/eV_to_J
app_force_lower = (lx*ly)/nlower*sigma/eV_to_J

# Boundaries
# Velocity at boundaries
block = """
velocity upper set 0.0 0.0 0.0 units box
velocity lower set 0.0 0.0 0.0 units box
"""
lmp.commands_string(block)

# Boundary conditions
# Rigid fixes must come before any box changing fix
if loading_type == "asymm":
    block = f"""
    fix 3 upper setforce NULL NULL 0.0
    fix 4 lower setforce 0.0  NULL 0.0
    
    fix 5 upper aveforce {app_force_upper} 0.0 0.0 
    
    fix 6 upper rigid group 1 upper
    """
    lmp.commands_string(block)

elif loading_type == "sym":
    block = f"""
    fix 3 upper setforce NULL NULL 0.0
    fix 4 lower setforce NULL NULL 0.0
    
    fix 5 upper aveforce {app_force_upper} 0.0 0.0 
    fix 6 lower aveforce {app_force_lower} 0.0 0.0 

    fix 7 upper rigid group 1 upper
    fix 7 lower rigid group 1 lower

    """
else:
    print("Error: loading_type should be set to 'sym' or 'asymm'")
    sys.exit()

# Temperature/Pressure control
# Ensemble chosen by setting thermo_ensemble variable at beginning
# of script
thermo_type(thermo_ensemble)

# Run MD
if sim_type == "new":
    lmp.command("reset_timestep 0")
elif sim_type == "continue":
    lmp.command(f"reset_timestep {last_timestep}")

lmp.command(f"restart 1000 md_restart")
lmp.command(f"run {MD_run}")

######################################################
#                    System Shutdown                 #
######################################################
MPI.Finalize()