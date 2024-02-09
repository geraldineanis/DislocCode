"""! @ref
DE-MC analysis module.

This module contains the functions needed to fit an equation of motion to dislocation position 
data using Differential Evolution Monte Carlo (DE-MC)
"""

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as st

# General
def get_nondim_data(filename, discard=0):
    """
    Non-dimensionalise time and position data.
    The user should make sure that the same dislocation data and the number of discareded points
    are used across different analyses tools.
    It is not always necessary to discard any portion of the trajectory and in that case discard should
    be set to zero, which is also the default.

    Parameters
    ----------
    filename : str
               Dislocation position data file name.
    discard  : int
               Number of points to discard at beginning of trajectory if required.
    
    Returns
    -------
    numpy.ndarray
               Original (dimimensional) time data.    
    numpy.ndarray
               Non-dimensional time data.
    numpy.ndarray
               Original (dimimensional) postition data.                   
    numpy.ndarray
               Non-dimensional position data.
    float
               Range of time data.
    float      
               Range of position data.               
    """
    data = np.loadtxt(filename)   

    if discard == 0:
        # time
        t_dim = data[:,0]
        t_range = np.ptp(t_dim)
        t_nondim = t_dim/t_range
        # position data
        x_dim = 0.1*data[:,1]          
        x_range = np.ptp(x_dim)
        xi = x_dim[0]
        x_dim = x_dim - xi
        x_nondim = x_dim/x_range
    else:
        # time
        t_dim = data[:-discard,0]
        t_range = np.ptp(t_dim)
        t_nondim = t_dim/t_range
        # position data
        x_dim = 0.1*data[discard:,1]
        x_range = np.ptp(x_dim)
        xi = x_dim[0]
        x_dim = x_dim-xi
        x_nondim = x_dim/x_range
    
    return t_dim, t_nondim, x_dim, x_nondim, t_range, x_range

# Model
def x_t(t,xi,m, B, F):
    """    
    Analytic solution to the equation of motion mx" + Bx' = F

    Parameters
    ----------
    t  : numpy.ndarray
         Time - times at which position is calculated.
    xi : float
         Initial dislocation position.
    m  : float
         Dislocation effective mass.
    B  : float
         Drag coefficient.
    F  : float
         Force on the dislocation.
    Returns
    -------
    float
         Dislocation position timeseries.    
    """
    return xi + ((F/B)*(t + (m/B)*(np.exp(-(B/m)*t) - 1.0)))

def x_derivatives(x,t,c0,c1):
    """
    Defines the derivatives needed to solve the non-dimensional equation of motion x" + c0*x' = c1
    
    Parameters
    ----------
    x  : numpy.ndarray 
         LHS row vector (dependent variable)
    t  : numpy.ndarray
         Time (the independent variable)
    c0 : float
         Non-dimensional B/m
    c1  : float
         Non-dimensional F/m
    Returns
    -------
    numpy.ndarray
          Derrivates.
    """
    dxdt = [x[1], np.exp(c1)- np.exp(c0)*x[1]]
    return dxdt

def ODE_solution(t,x0,t_data):
    """
    Solves the non-dimensional equation of motion x" + c0*x' = c1
    Requires derivatives to be defined previously.
    
    Parameters
    ----------
    t      : numpy.ndarray
             pNon-dimensional arameter vector
    x0     : numpy.ndarray  
             Initial position and velocity.
    t_data : numpy.ndarray
             Time - times at which equation is solved.    
    Returns
    -------
    numpy.ndarray
             Non-dimensional dislocation position time series.
    numpy.ndarray
             Non-dimensional dislocation velocity time series.
    """
    c0, c1 = t  # Unpack parameters
    position, velocity = odeint(x_derivatives, x0, t_data, args=(c0,c1)).T
    return position, velocity

# Dimensional Model
def x_derivatives_dim(x,t,B_m,F_m):
    """
    Defines the derivatives needed to solve the dimensional equation of motion x" + (B/m)x' = (F/m)
    
    Parameters
    ----------
    x : numpy.ndarray 
        LHS row vector (dependent variable)
    t : numpy.ndarray
        time (the independent variable)
    B : float
        Dimensional B/m.
    F : float
        Dimensional F/m.
    Returns
    -------
    numpy.ndarray
          Derrivates.
    """
    dxdt = [x[1], (F_m - B_m*x[1])]
    return dxdt

def ODE_solution_dim(t,x0,t_data):
    """
    Solves the dimensional equation of motion x" + (B/m)x' = (F/m)
    Requires derivatives to be defined previously.
    
    Parameters
    ----------
    t      : numpy.ndarray
             Dimensional parameter vector
    x0     : numpy.ndarray  
             Initial position and velocity.
    t_data : numpy.ndarray
             Time - times at which equation is solved.    
    Returns
    -------
    numpy.ndarray
             Non-dimensional dislocation position time series.
    numpy.ndarray
             Non-dimensional dislocation velocity time series.
    """
    B_m, F_m = t  # Unpack parameters
    position, velocity = odeint(x_derivatives_dim, x0, t_data, args=(B_m,F_m)).T
    return position, velocity

# Functions to convert from ci to B/m and F/m
def c0_B_m(c0, ts):
    """
    Calculates B/m from its corresponding non-dimensional parameter.

    Parameters
    ----------
    c0 : float
         Non-dimensional parameter corresponding to B/m
    ts : float
         Range of dimensional time data.
    Returns
    -------
    float
         B/m.
    """
    return c0/ts

def c1_F_m(c1, xs, ts):
    """
    Calculates F/m from its corresponding non-dimensional parameter.

    Parameters
    ----------
    c1 : float
         Non-dimensional parameter corresponding to F/m
    ts : float
         Range of dimensional time data.
    xs : float
         Range of dimensional position data.
    Returns
    -------
    float
         F/m.
    """    
    return c1*xs/(ts**2)

# DE-MC Analysis
def get_posterior(data, n_samples, burn, dims=3):
    """
    Calculates parameter posterior distributions from DE-MC accepted samples.

    Parameters
    ----------
    data      : np.ndarray
                DE-MC output loaded from ".npz" file generated by MC3 code.
    n_samples : int
                Number of DE_MC generations.
    burn      : int
                Number of accepted samples to discard.
    dims      : int
                Number of parameters.

    Returns
    -------
    numpy.ndarray
                Marginal parameter posterior distributions.
    numpy.ndarray
                Parameter means.
    numpy.ndarray
                Joint posterior distribution covariance matrix.
    numpy.ndarray
                Parameter standard deviations.
    """
    # All posterior samples
    posterior = np.array(data["posterior"][burn:n_samples])
    # Marginal posterior samples
    c_dist = np.array([np.array(posterior[:,i]) for i in range(dims)])
    # Parameter means
    c_means = np.array([c_dist[i].mean() for i in range(dims)])
    # Posterior covariance matrix
    post_cov = np.array(np.cov(posterior, rowvar=False))
    # Parameter standard deviations
    c_std = np.array([np.sqrt(post_cov[i][i]) for i in range(dims)])
    
    return c_dist, c_means, post_cov, c_std

def stat_convergence(c_dist, dims, write_out=True):
    """
    Function to calculate the mean and standard deviation of a parameter dataset
    with increasing number of samples.

    Parameters
    ----------
    c_dist    : numpy.array
                Accepted samples for each parameter. This should be an N x M 
                array where N is equal to dims and M is the number of DE-MC samples.
    dims      : int
                Number of fitting parameters.
    write_out : bool
                If True, writes out the results to stats_convg.txt
    Returns
    -------
    numpy.ndarray
                Parameter means for n DE-MC generations.
    numpy.ndarray
                Parameter standard deviations for n DE-MC generations.                
    """
    means    = np.array([[c_dist[i][:N].mean() for N in range(1,len(c_dist[i]))] for i in range(dims)])
    std_devs = np.array([[c_dist[i][:N].std() for N in range(1,len(c_dist[i]))] for i in range(dims)])

    if write_out:
       f = open("stats_convg.txt", "w")

       for i in range(len(means)):
           f.write(f"{means[i]}     {std_devs[i]} \n")
       f.close()
       
    return means, std_devs

# Writing output to file
# Statistics
def write_demc_summary(filename, n_samples, burn, c_mean, c_std, post_cov):
    """
    Writes out a summary of the parameter fitting using DE-MC.
    
    Parameters
    ----------
    filename : str
               Output file name.
    n_samples: int
               Number of DE-MC samples.
    burn     : int
               Number of accepted samples to discard.              
    c_mean   : numpy.ndarray
               Parameter means.    
    c_std    : numpy.ndarray
               Parameter standard deviations.
    post_cov : nump.ndarray
               Joint posterior distribution covariance matrix.              
    """
    f = open(f"{filename}", "w")

    f.write("DE-MC parameter fitting summary \n\n")
    f.write("General \n")
    f.write("------- \n")
    f.write(f"Accepted samples  = {n_samples} \n")
    f.write(f"Burned samples    = {burn} \n")
    f.write(f"Remaining samples = {n_samples-burn} \n\n")

    f.write("Parameter Statistics \n")
    f.write("-------------------- \n")
    f.write(f"Parameter Means \n {c_mean} \n")
    f.write(f"Standard dev. \n {c_std} \n")
    f.write(f"Posterior Covariance Matrix \n {post_cov}")    
    
    f.close()

# Plotting
def plot_trace(dims,burn,c_dist,params,save_plt=False,show_plt=True):
    """
    Plots a trace plot using the accepted DE-MC samples.

    Parameters
    ----------
    dims     : int
               Number of fitting parameters.
    burn     : int
               Number of accepted samples to discard.
    c_dist   : numpy.ndarray
               Accepted samples for each parameter. This should be an N x M 
               array where N is equal to dims and M is the number of samples.
    params   : list
               N-dimensional array containingg parameter names as strings for
               plot labels.
    save_plt : bool
               If True, saves the trace plot. Default value is False.
    show_plt : bool
               If True, shows the trace plot. Default value is True.
    """
    fig, axs = plt.subplots(1, dims, figsize=(10,5))
    for i in range(dims):
        axs[i].plot(c_dist[i], lw=1, alpha=0.5)
        axs[i].axvline(burn, c="tab:red", lw=1, ls="--", label=f"Burned Samples = "+r"$10^5$")
        axs[i].set_ylabel(f"ln({params[i]})", fontsize=12)
    for ax in axs:
        ax.set_xlabel("Generation", fontsize=12)
    axs[-1].legend(fontsize=12)
    
    if save_plt:
        fig.savefig("trace.png", dpi=350, format="png")
    if show_plt:
        plt.show()

def plot_pairplot(c_dist, params,save_plt=False,show_plt=True):
    """
    Plots a pairplot of the fitted parameters.

    Parameters
    ----------
    c_dist   : numpy.ndarray
               Accepted samples for each parameter. This should be an N x M 
               array where N is equal to dims and M is the number of samples.
    params   : list
               N-dimensional array containing parameter names as strings for plot 
               labels.
    save_plt : bool
               If True, saves the trace plot. Default value is False.
    show_plt : bool
               If True, shows the trace plot. Default value is True.       
    """
    sns.pairplot(pd.DataFrame(c_dist.T, columns=params), plot_kws={"s": 3})
    
    if save_plt:
        plt.savefig("pairplot.png", dpi=350, format="png")    
    if show_plt:
        plt.show()

def plot_marginal(dims,c_dist,c_mean,c_std,prior_mean,prior_std,d_range,params,save_plt=False,show_plt=True):
    """
    Plots the marginal parameter posterior and parameter prior distributions.
    Assumes parameters are normally distributed.

    Parameters
    ----------
    dims       : int
                 Number of fitting parameters.    
    c_dist     : numpy.ndarray
                 Accepted samples for each parameter. This should be an N x M 
                 array where N is equal to dims and M is the number of samples.
    c_mean     : numpy.ndarray
                 Parameter means.
    c_std      : numpy.ndarray
                 Parameter standard deviations.
    prior_mean : numpy.ndarray
                 Prior means.
    prior_std  : numpy.ndarray
                 N-dimensional array of prior standard deviations.
    d_range    : numpy.ndarray
                 Values at which to evaluate the parameter probability
                 density function (PDF).
    params     : list
                 N-dimensional array containing parameter names as strings for plot 
                 labels.
    save_plt   : bool
                 If True, saves the trace plot. Default value is False.
    show_plt   : bool
                 If True, shows the trace plot. Default value is True.   
    """
    fig, axs = plt.subplots(1,dims, figsize=(10,5))
    
    bins = 100

    for i in range(dims):
        # Posterior
        axs[i].hist(c_dist[i], density=True, bins=bins, color=f"C0", alpha=0.5, label="DE-MC Samples")
        axs[i].plot(d_range, st.norm.pdf(d_range, loc=c_mean[i], scale=c_std[i]),
                    color="C0", label=r"$\mathcal{N}($" + f"{round(c_mean[i],2)}, {round(c_std[i],2)}" +r"$^2$)")    
        # Prior
        axs[i].plot(d_range, st.norm.pdf(d_range, loc=prior_mean[i], scale=prior_std[i]),
                    color="C3", label=r"Prior - $\mathcal{N}($" + f"{prior_mean[i]}, {prior_std[i]}"+r"$^2$)")

        axs[i].set_xlabel(f"{params[i]}", fontsize=12)
        axs[i].legend(loc=2, fontsize=10.5)

    axs[0].set_ylabel("Density", fontsize=12)
   
    if save_plt:
         fig.savefig("param_posterior.png", dpi=350, format="png")
    if show_plt:
        plt.show()

def plot_convg(dims,means,std_devs,params,save_plt=False,show_plt=True):
    """
    Plots the marginal parameter posterior and parameter prior distributions.
    Assumes parameters are normally distributed.

    Parameters
    ----------
    dims     : int
               Number of fitting parameters.    
    means    : numpy.ndarray
               Parameter means for n DE-MC generations.
    std_devs : numpy.ndarray
               Parameter standard deviations for n DE-MC generations.
    params   : list
               N-dimensional array containing parameter names as strings for plot 
               labels.
    save_plt : bool
               If True, saves the trace plot. Default value is False.
    show_plt : bool
               If True, shows the trace plot. Default value is True.
    """                 
    fig, axs = plt.subplots(1,2, figsize=(15,5))
    for i in range(dims):
        # means
        axs[0].plot(means[i], label=params[i])
        axs[0].set_ylabel("Parameter mean", fontsize=12)
        axs[0].set_xlabel("Generation", fontsize=12)

        # std. devs
        axs[1].plot(std_devs[i], label=params[i])
        axs[1].set_ylabel("Parameter Std. Dev.", fontsize=12)
        axs[1].set_xlabel("Generation", fontsize=12)
    
    axs[1].legend(loc=4, fontsize=12)

    for ax in axs:
        ax.tick_params(axis='both', labelsize=11)

    
    if save_plt:
        fig.savefig("convergence.png", dpi=350, format="png", bbox_inches="tight")
    
    if show_plt:
        plt.show()

    
