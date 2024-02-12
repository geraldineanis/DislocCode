# DislocCode
A collection of python tools to setup, run, and analyse molecular dynamics (MD) simulations of edge dislocations in face-centered cubic (FCC) Ni.


### `structure_gen`
Scripts that generate:
- A single $\langle 110 \rangle\{ 111\}$ edge dislocation in an FCC Ni cell
- Slabs of FCC Ni

### `lammps_scripts`
Script to run MD simulations.

### `dislocation_tracking`
Scripts to post-process dislocation simulation output - detect and track dislocations in an MD simulation to generate a dislocation trajectory for further analysis.

### `de_mcmc`
Script to fit the parameters of an equation of motion to MD dislocation trajectories using differential evolution Monte Carlo (DE-MC). A step-by-step tutorial on how this is done can be found in the `de_mc_analysis.ipynb` notebook.


## Requirements 
LAMMPS Python Module and Shared library:
https://docs.lammps.org/Python_install.html

Ovito python libraries:
https://www.ovito.org/docs/current/python/introduction/installation.html

MC3:
https://mc3.readthedocs.io/en/latest/get_started.html

ASE:
https://wiki.fysik.dtu.dk/ase/install.html

Python libraries:
- Numpy
- Matplotlib
- Scipy
- Pandas
- Seaborn
