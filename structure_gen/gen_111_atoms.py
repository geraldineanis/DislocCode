from ase.lattice.compounds import L1_2
from ase.lattice.cubic import FaceCenteredCubic
from ase.build import fcc111, fcc110, surface, bulk, rotate, cut
from ase.visualize import view
from ase import Atoms
import numpy as np

# Structure fle name
out_file = "Ni_end.data"

# Lattice parameters
gamma_latt = 3.52

# Directions for (111)
directions = np.array([[1,-1,0],
                       [1,1,-2],
                       [1,1,1]])

# Layer sizes
x1 = int(45*2)
y1 = int(10*2)
z1 = int(40)

# Box dimensions
cell_x = x1*gamma_latt*np.sqrt(2)/2.0
cell_y = y1*gamma_latt*np.sqrt(6)/2.0
cell_z = z1*gamma_latt*np.sqrt(3)
box_dim = [cell_x,cell_y,cell_z]

###############################################################
#                        Lower Gamma Layer                    #
###############################################################

gamma_1 = FaceCenteredCubic(directions=directions, size=(x1,y1,z1), symbol="Ni", pbc=(1,1,1), latticeconstant=gamma_latt)


gamma_1_atoms = gamma_1.get_positions()
gamma_1_atom_types = gamma_1.get_chemical_symbols()

gamma_1 = list(zip(gamma_1_atom_types, gamma_1_atoms))

###############################################################
#                    Write LAMMPS Data File                   #
###############################################################

with open(out_file, "w") as fdata:

    # Comment line
    fdata.write("This is a test to generate atoms for LAMMPS\n\n")

    ntypes = 1

    # Header
    # Number of atoms and atom types
    fdata.write(f"{len(gamma_1)} atoms\n")
    fdata.write(f"{ntypes} atom types\n")

    # Box dimensions
    fdata.write(f"{0.0} {box_dim[0]} xlo xhi\n")
    fdata.write(f"{0.0} {box_dim[1]} ylo yhi\n")
    fdata.write(f"{0.0} {box_dim[2]} zlo zhi\n")
    fdata.write("\n")

    # Atoms section
    fdata.write(f"Atoms\n\n")

    # Write each position
    # gamma_1
    for i,atom in enumerate(gamma_1):

        fdata.write(f"{i+1} 1 {atom[1][0]} {atom[1][1]} {atom[1][2]}\n") # atom-ID atom-type x y z
