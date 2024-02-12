"""! @ref
Script to run Differential Evolution Markov Chain Monte Carlo (DE-MC).

This script sets the settings need to run the \f$\text{MC}^3\f$ code to fit the parameters of the non-dimensional equation of motion \f$x'' + c_{0}x' = c_{1}\f$ using the DE-MC algorithm 
(Braak and Vrugt, https://doi.org/10.1007/s11222-008-9104-9). The full \f$\text{MC}^3\f$ code documentation including a detailed description of the settings used below can be found at 
https://mc3.readthedocs.io/en/latest/mcmc_tutorial.html.

The script reads in the dislocation position vs. time data obtained by post-processing LAMMPS MD trajectories and uses DE-MC to fit the parameters of the above equation of motion. A
directory should be set by setting the out_dir variable, where the DE-MC output can be stored. The scripts, de_mc_analysis.py and mBF_new.py, can be used to analyze the results of the 
fitting procedure, which give the parameter marginal and joint posterior distributions. In addition to parameter distributions, other outputs of the analysis scripts include trace plots and parameter mean and 
standard deviation convergence with DE-MC generation (iteration), which can be used for further optimisation of the DE-MC settings provided here.

The script mBF_new.py provides plotting tools, which can ve used to produce additional plots other than those generated by the \f$\text{MC}^3\f$ code (these can be found in the output directory 
set at the beginning of this script). The de_mcmc.npz file contains the DE_MC accepted samples and is used for further analysis, while the de_mcmc.log and de_mcmc_statistics.txt files contain 
a summary of the DE-MC fitting procedure output. These files can also be found in the user-set directory defined in this script.

@sa

dislocation_script.py

mBF_new.py
"""

import numpy as np
import seaborn as sns
from scipy.integrate import odeint
from scipy.optimize import minimize
import scipy.stats as st
from matplotlib import pyplot as plt
import pandas as pd
import mc3
import os
from funcs import *

# Set directory to store DE-MC ouput files
out_dir = "de_mc_out"

# Set number of DE-MC chains
chains = 6

# Set prior distributions for fitting parameters
# Calibration parameters for model are c0 and c1
prior =    np.array([0.0, 0.0])     # parameters mean vector
priorlow = np.array([1.0, 1.0])     # parameters -1*sigma 
priorup =  np.array([1.0, 1.0])     # parameters +1*sigma

# Read in and non-dimnesionalise dislocation position-time data
# Number of points to discard at beginning of trajectory
data_n = np.loadtxt("data.txt")

def lin_func(t):
    m, b = t
    return m*x + b

# MC3 settings
# Input data
x = data_n[:,0]
data = np.array(data_n[:,1])
uncert = [1.0 for i in range(len(data))]

# Modelling function
func = lin_func

# Fitting Parameters - initial guesses
ndims = 2
params = np.random.random(ndims)

# Parameters stepping behavior
pstep = np.array([1.0, 1.0])

# Parameter names
pnames = ["c0", "c1"]
textnames = [r"c_{0}", r"c_{1}"]

# Sampler algorithm
sampler = "demc"

# MCMC Configuration
nsamples = 1e4          # Number of MCMC iterations
burnin = 0              # Number of burn in samples - This is taken care of in analysis scripts
nchains = chains        # Number of MCMC chains
ncpu = 1                # Number of CPUs 
thinning = 1            # 

# Pre-MCMC setup
kickoff = "normal"      # MCMC initial draw - can also be "uniform"
hsize = 10              # DE-MC snooker pre-MCMC sample size

# Convergence
grtest = True           # Run Gelman-Rubin convergence test
grbreak = 0.0           # Convergence threshold to stop MCMC - 0.0 is no break
grnmin = 0.5

# Fine-tuning
fgamma = 1.0            # Scale factor for DEMC's gamma jump
fepsilon = 0.0          # Jump scale factor for DEMC's "e" distribution

# Logging
log = "de_mcmc.log"     # Log file name

# Output and plotting settings
savefile = 'de_mcmc.npz'
plots = True
theme = 'indigo'
statistics = 'med_central'
rms = True

# DE-MC run
dir = out_dir
os.mkdir(dir)
os.chdir(dir)

output = mc3.sample(
    data=data, uncert=uncert, func=func, params=params, pstep=pstep,
    pnames=pnames, texnames=textnames,
    prior=prior, priorlow=priorlow, priorup=priorup,
    sampler=sampler, nsamples=nsamples,  nchains=nchains,
    ncpu=ncpu, burnin=burnin, thinning=thinning,
    leastsq=None, chisqscale=False,
    grtest=grtest, grbreak=grbreak, grnmin=grnmin, log=log,
    plots=plots, theme=theme, statistics=statistics,
    savefile=savefile, rms=rms
)
