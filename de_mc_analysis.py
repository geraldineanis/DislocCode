import numpy as np
import scipy.stats as st
from scipy.integrate import odeint
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import os
import glob
import time

from funcs import *

# Number of points to discard
n=6

# Use non-dimensionalized data
data = np.loadtxt("perfect_pos.txt")
t_data = data[:-n,0]
t_range = np.ptp(t_data)
t_data = data[:-n,0]/t_range
x_data = 0.1*data[n:,1]     # Convert from Angstroms to nm
x_range = np.ptp(x_data)
xi = x_data[0]
x_data = (x_data-xi)/x_range

xs = x_range                                               # nm
ts = t_range                                               # ps
Fs = (250.0*1e6*1e12*np.sqrt(2)*0.352)/(2.0*(1e9**2))      # pN/nm

# Set the initial conditions
# initial position and velocity
x0 = [0.0,0.0]  

# DE-MCMC results
n_samples = int(1e6)
burn = int(1e5)
dims = 3
params = [r"$c_{0}$", r"$c_{1}$", r"$c_{2}$"]
log_params = [r"ln($c_{0}$)", r"ln($c_{1}$)", r"ln($c_{2}$)"]

prior    = np.array([0.0, 0.0, 0.0])
priorlow = np.array([2.0, 2.0, 2.0])
priorup  = np.array([2.0, 2.0, 2.0])

dirs = f"de_mc_out"
data = np.load(f"./{dirs}/de_mcmc.npz")

print(f"Remaining samples = {n_samples-burn}")

# Note: work with logs of data - log(c_i) ~ Normal
# The model is set up such that the we sample log(c_i)
posterior, log_c, log_c_mean, post_cov, log_c_std = get_posterior(data, n_samples, burn, dims=3)

print("Parameter Means", "\n", log_c_mean)
print("Standard dev.", "\n", log_c_std)
print("Posterior Covariance Matrix", "\n", post_cov)

# Plots
# Trace plots
fig, axs = plt.subplots(1, 3, figsize=(15,5))
# fig.suptitle("Trace plot")
for i in range(dims):
    axs[i].plot(data["posterior"][:,i], lw=1, alpha=0.5)
    axs[i].axvline(burn, c="tab:red", lw=1, ls="--", label=f"Burned Samples ={burn}")
    axs[i].set_ylabel(f"ln({params[i]})")
axs[1].set_xlabel("Generation")
axs[2].legend()
plt.show()
fig.savefig("trace.png", dpi=350, format="png")

# # Pairplot
# sns.pairplot(pd.DataFrame(log_c.T, columns=log_params), diag_kind="kde")
# plt.savefig("pairplot.png", dpi=350, format="png")
# plt.show()

# Plot posterior distribution for each parameter
# For different nchains values
fig, axs = plt.subplots(1,3, figsize=(15,10))
bins = 100
for i in range(dims):
    # Posterior
    axs[i].hist(log_c[i], density=True, bins=bins, color=f"C0", alpha=0.5, label="DE-MC Samples")
    axs[i].plot(np.linspace(-10.0, 10.0, 1001), st.norm.pdf(np.linspace(-10.0, 10.0, 1001), loc=log_c_mean[i], scale=log_c_std[i]),
                color="C0", label=r"$\mathcal{N}($" + f"{round(log_c_mean[i],2)}, {round(log_c_std[i],2)})")    
    # Prior
    axs[i].plot(np.linspace(-10.0, 10.0, 1001), st.norm.pdf(np.linspace(-10.0, 10.0, 1001), loc=prior[i], scale=priorlow[i]),
                color="C3", label=r"Prior - $\mathcal{N}($" + f"{prior[i]}, {priorlow[i]})")

    axs[i].set_xlabel(f"ln({params[i]})")
    axs[i].legend()
axs[0].set_ylabel("Density")
plt.show()
fig.savefig("param_posterior.png", dpi=350, format="png")

# # Convergence of mean and standard devs with generation
# # This takes a very long time to run
# means, std_devs = stat_convergence(log_c, dims)

# fig, axs = plt.subplots(1,2, figsize=(10,5))
# for i in range(dims):
#     # means
#     axs[0].plot(means[i], label=log_params[i])
#     axs[0].set_ylabel("Parameter mean")

#     # std. devs
#     axs[1].plot(std_devs[i], label=log_params[i])
#     axs[1].set_ylabel("Parameter Std. Dev.")

# fig.text(0.5, 0.01, "generation", ha="center")  # Common x-axis label
# fig.legend(labels=log_params, loc="upper right")
# plt.show()

# fig.savefig("convergence.png", dpi=350, format="png")