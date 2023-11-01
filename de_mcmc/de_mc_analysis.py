"""! @ref
Script to analyse the ouput from the parameter fitting using DE-MC.

This script uses the functions contained in the funcs.py module to:
- Calculate relevant statistics pertaining to accepted samples obtained from the parameter fitting using DE-MC.
- Visualise the DE-MC output as well as the marginal and joint parameter posterior distributions.
- Produce a trace plot and parameter mean and convergence with DE-MC generation (iteration) to guage the convergence
  of the DE-MC.

The function, stat_convergence, takes a long time to run. Accordingly, it is recommended that it is run only once, 
and its output can be written to a text file using write_out=True in the function argument for further analysis.
"""
import numpy as np
from funcs import *

# Read in and non-dimnesionalise dislocation position-time data
data = np.loadtxt("perfect_pos.txt")   
# Number of points to discard at beginning of trajectory
# Make sure this matches the number that was given in the mc3_de_mcmc.py script
n = 7

if n == 0:
    # time
    t_data = data[:,0]
    t_range = np.ptp(t_data)
    t_data = data[:,0]/t_range
    # position data
    x_data = 0.1*data[:,1]          # Convert from Angstroms to nm
    x_range = np.ptp(x_data)
    xi = x_data[0]
    x_data = (x_data-xi)/x_range
else:
    # time
    t_data = data[:-n,0]
    t_range = np.ptp(t_data)
    t_data = data[:-n,0]/t_range
    # position data
    x_data = 0.1*data[n:,1]                    # Convert from Angstroms to nm
    x_range = np.ptp(x_data)
    xi = x_data[0]
    x_data = (x_data-xi)/x_range

# DE-MCMC results
n_samples = int(1e6)                           # Number of DE-MC iterations
# Number of accepted samples to discard
# Run this script once and determine based on trace plot
burn = int(1e5)
# Fitting parameters                               
dims = 2
params = [r"$c_{0}$", r"$c_{1}$"]
log_params = [r"ln($c_{0}$)", r"ln($c_{1}$)"]

prior =    np.array([0.0, 0.0])                # parameters mean vector
priorlow = np.array([2.0, 2.0])                # parameters -1*sigma 
priorup =  np.array([2.0, 2.0])                # parameters +1*sigma

# Set the directory where the DE-MC output files are stored
out_dir = f"de_mc_out"
data = np.load(f"./{out_dir}/de_mcmc.npz")

print(f"Remaining samples = {n_samples-burn}")

# Note: work with logs of data - log(c_i) ~ Normal
# The model is set up such that the we sample log(c_i)
log_c, log_c_mean, post_cov, log_c_std = get_posterior(data, n_samples, burn, dims)

print("Parameter Means", "\n", log_c_mean)
print("Standard dev.", "\n", log_c_std)
print("Posterior Covariance Matrix", "\n", post_cov)

# Write summary
filename = "de_mc_summary.txt"
write_demc_summary(filename, n_samples, burn, log_c_mean, log_c_std, post_cov)

# # Plots
# # Trace plots
# plot_trace(dims,burn,log_c,params,save_plt=True,show_plt=True)

# # Pairplot
# plot_pairplot(log_c,log_params,save_plt=True,show_plt=True)

# # Marginal parameter posterior disributions
# d_range = np.linspace(-10.0, 10.0, 1001)
# plot_marginal(dims,log_c,log_c_mean,log_c_std,prior,priorlow,d_range,log_params,save_plt=True,show_plt=True)

# Convergence of mean and standard devs with generation
# This takes a very long time to run (for 1e6 iterations ~ 1h)
# To run, uncomment the following line
# means, std_devs = stat_convergence(log_c, dims, write_out=True)