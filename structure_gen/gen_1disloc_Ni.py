"""! @ref
Script to generate a single a/2<110>{111} edge dislocation in a pure FCC Ni cell.

To construct the dislocation, two slabs of pure Ni are created, where one of the slabs has an extra plane (half-plane) of atoms. On relaxing this structure (see lmp_pre_strain.py) 
the correct dislocation core structure, where the dislocation dissociates into two a/6<112> partial dislocations should form.

It is recommended that the cell dimensions here are kept unchanged and creating a bigger cell can be achieved by adding padding using gen_111_atoms.py.

@sa 

lmp_pre_strain.py

gen_111_atoms.py
"""


from ase.lattice.compounds import L1_2
from ase.lattice.cubic import FaceCenteredCubic
from ase.build import fcc111, fcc110, surface, bulk, rotate, cut
from ase.visualize import view
from ase import Atoms
import numpy as np

# Lattice parameters
gamma_latt = 3.52

# Directions for (111)
directions = np.array([[1,-1,0],
                       [1,1,-2],
                       [1,1,1]])

# Layer sizes
# Lower gamma
x1 = int(20.5*2)
y1 = int(10*2)
z1 = 20

# Upper gamma
x2 = int(21*2)
y2 = int(10*2)
z2 = 20

# Box dimensions
cell_x = x2*gamma_latt*np.sqrt(2)/2.0
cell_y = y1*gamma_latt*np.sqrt(6)/2.0
cell_z = (z1+z2)*gamma_latt*np.sqrt(3)
box_dim = [cell_x,cell_y,cell_z]

###############################################################
#                        Lower Gamma Layer                    #
###############################################################

gamma_1 = FaceCenteredCubic(directions=directions, size=(x1,y1,z1), symbol="Ni", pbc=(1,1,1), latticeconstant=gamma_latt)


gamma_1_atoms = gamma_1.get_positions()
gamma_1_atom_types = gamma_1.get_chemical_symbols()

gamma_1 = list(zip(gamma_1_atom_types, gamma_1_atoms))

# Modify atom positions to get correct structure
k = 1.0+(1.0/x1)
# k = 1.0
for atom in gamma_1:
    # # Wrap atoms in x
    # if atom[1][0] < 0.0:
    #     atom[1][0] = atom[1][0] + x1*gamma_latt*np.sqrt(2)/2.0    
    # # Wrap atoms in y
    # if atom[1][1] < 0.0:
    #     atom[1][1] = atom[1][1] + y1*gamma_latt*np.sqrt(6)/2.0
    # Wrap atoms in z
    if atom[1][2] >= z1*gamma_latt*np.sqrt(3):
        atom[1][2] = atom[1][2] - z1*gamma_latt*np.sqrt(3)
    # Elongate atoms in the x direction
    if atom[1][0] > 0.5*gamma_latt*np.sqrt(2)/2.0:
        atom[1][0] = atom[1][0]*k

# Check number of atoms
print("Lower gamma layer")

lx = np.sqrt(2)/2.0
ly = np.sqrt(6)/2.0
lz= np.sqrt(3)

# Expected number of atoms
natoms_th = x1*lx*y1*ly*z1*lz*4
print("Expected number of atoms:", natoms_th)

natoms = len(gamma_1)
print("Actual number of atoms:", natoms)

###############################################################
#                        Upper Gamma Layer                    #
###############################################################
gamma_2 = FaceCenteredCubic(directions=directions, size=(x2,y2,z2), symbol="Ni", pbc=(1,1,1), latticeconstant=gamma_latt)

gamma_2_atoms = gamma_2.get_positions()
gamma_2_atom_types = gamma_2.get_chemical_symbols()

gamma_2 = list(zip(gamma_2_atom_types, gamma_2_atoms))

# Modify atom positions to get correct structure
# k = 1.0-(1.0/(x1+x2))
k = 1.0
for atom in gamma_2:
    # # Wrap atoms in x
    # if atom[1][0] < 0.0:
    #     atom[1][0] = atom[1][0] + x2*gamma_latt*np.sqrt(2)/2.0
    # # Wrap atoms in y
    # if atom[1][1] < 0.0:
    #     atom[1][1] = atom[1][1] + y2*gamma_latt*np.sqrt(6)/2.0                   
    # Wrap atoms in z
    if atom[1][2] >= z2*gamma_latt*np.sqrt(3):
        atom[1][2] = atom[1][2] - z2*gamma_latt*np.sqrt(3) 
    # Shift atoms in z direction
    atom[1][2] = atom[1][2] + z1*gamma_latt*np.sqrt(3)
    # Compress atoms in the x direction
    if atom[1][0] > 0.5*gamma_latt*np.sqrt(2)/2.0:
        atom[1][0] = atom[1][0]*k

# del_atom = [i for i in range(len(gamma_2)) if (gamma_2[i][1][0] > cell_x-(0.5*gamma_latt*np.sqrt(2)/2.0))]
# gamma_2 = np.delete(gamma_2, del_atom, 0)


# Check number of atoms
print("Upper gamma layer")

# Expected number of atoms
natoms_th = x2*lx*y2*ly*z2*lz*4
print("Expected number of atoms:", natoms_th)

natoms = len(gamma_2)
print("Actual number of atoms:", natoms)

###############################################################
#                    Write LAMMPS Data File                   #
###############################################################

with open("disloc_Ni.data", "w") as fdata:

    # Comment line
    fdata.write("This is a test to generate atoms for LAMMPS\n\n")

    ntypes = 1

    # Header
    # Number of atoms and atom types
    fdata.write(f"{len(gamma_1)+len(gamma_2)} atoms\n")
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
    atype=1
    for i,atom in enumerate(gamma_1):
        fdata.write(f"{i+1} {atype} {atom[1][0]} {atom[1][1]} {atom[1][2]}\n") # atom-ID atom-type x y z

    # gamma_2
    for i,atom in enumerate(gamma_2):
        fdata.write(f"{i+1+len(gamma_1)} {atype} {atom[1][0]} {atom[1][1]} {atom[1][2]}\n") # atom-ID atom-type x y z
