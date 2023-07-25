import numpy as np
import seaborn as sns
from scipy.integrate import odeint
from scipy.optimize import minimize
import scipy.stats as st
from matplotlib import pyplot as plt
import pandas as pd
import mc3
import os

# Test effect of changing prior distribution
# Setting nchains to 14 chains
chains = 14

prior =    [np.array([0.0, 0.0, 0.0])]
priorlow = [np.array([2.0, 2.0, 2.0])]
priorup =  [np.array([2.0, 2.0, 2.0])]

# Calibration parameters for model are c0, c1, and c2
# See notes for non-dimensionalising equation of motion

# Read in and non-dimnesionalise dislocation position-time data

# Number of points to throw away
# Later modify the model to account for lag
# (time needed for correct strain field to develop)
n = 6

data = np.loadtxt("perfect_pos.txt")
t_data = data[:-n,0]
t_range = np.ptp(t_data)
t_data = data[:-n,0]/t_range
x_data = 0.1*data[n:,1]     # Convert from Angstroms to nm
x_range = np.ptp(x_data)
xi = x_data[0]
x_data = (x_data-xi)/x_range

# Model (non-dimensional equation of motion)
def x_derivatives(x,t,c0,c1,c2):
    """
    x: LHS row vector (dependent variable)
    t: time (the independent variable)
    m, B, F: constants
    """
    dxdt = [x[1], np.exp(c0)*(np.exp(c2) - np.exp(c1)*x[1])]
    return dxdt

# Set the initial conditions
# initial position and velocity
x0 = [0.0,0.0]  

def ODE_solution(t):
    c0, c1, c2 = t  # Unpack parameters
    position, velocity = odeint(x_derivatives, x0, t_data, args=(c0,c1,c2)).T
    return position

# Input data
data = np.array(x_data)
uncert = np.ones_like(x_data)*np.random.normal(0.0,0.1)

# Modelling function
func = ODE_solution

# indparams = [x]

# Fitting Parameters
# Initial guess
ndims = 3
params = np.random.random(ndims)

# Parameters stepping behavior
# ADJUST
pstep = np.array([0.5, 1.0, 1.0])

# Parameter priors
# Set Gaussian priors on all parameters
# Set above

# prior = np.array([0.0, 0.0, 0.0])
# priorlow = np.array([1.0, 1.0, 1.0])
# priorup = np.array([1.0, 1.0, 1.0])

# Parameter names
pnames = ["c0", "c1", "c2"]
textnames = [r"c_{0}", r"c_{1}", r"c_{2}"]

# Sampler algorithm
sampler = "snooker"

# MCMC Configuration
nsamples = 1e6
burnin = 0
nchains = chains
ncpu = 7
thinning = 1

# Pre-MCMC setup
kickoff = "normal"      # MCMC initial draw - can also be "uniform"
hsize = 10              # DEMC snooker pre-MCMC sample size

# Convergence
grtest = True             # Run Gelman-Rubin convergence test
grbreak = 0.0             # Convergence threshold to stop MCMC - 0.0 is no break
grnmin = 0.5

# Fine-tuning
fgamma = 1.0              # Scale factor for DEMC's gamma jump
fepsilon = 0.0            # Jump scale factor for DEMC's "e" distribution

# Logging
log = "de_mcmc.log"       # Store screen output into a log file

# Outputs
savefile = 'de_mcmc.npz'
plots = True
theme = 'indigo'
statistics = 'med_central'
rms = True

# MCMC run
for i in range(len(prior)):
    dir = f"de_mc_out"
    os.mkdir(dir)
    os.chdir(dir)

    output = mc3.sample(
        data=data, uncert=uncert, func=func, params=params, pstep=pstep,
        pnames=pnames, texnames=textnames,
        prior=prior[i], priorlow=priorlow[i], priorup=priorup[i],
        sampler=sampler, nsamples=nsamples,  nchains=nchains,
        ncpu=ncpu, burnin=burnin, thinning=thinning,
        leastsq=None, chisqscale=False,
        grtest=grtest, grbreak=grbreak, grnmin=grnmin,
        hsize=hsize, kickoff=kickoff, log=log,
        plots=plots, theme=theme, statistics=statistics,
        savefile=savefile, rms=rms
    )

    os.chdir("../")
