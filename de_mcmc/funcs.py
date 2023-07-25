import numpy as np
from scipy.integrate import odeint

# Equation of motion solution
def x_t(t,xi,m, B, F):
    """
    Analytic solution to the equation of motion mx" + Bx' = F
    Full dimensional equation
    """
    return xi + ((F/B)*(t + (m/B)*(np.exp(-(B/m)*t) - 1.0)))


# Non-dimensional Model
def x_derivatives(x,t,c0,c1,c2):
    """
    x: LHS row vector (dependent variable)
    t: time (the independent variable)
    m, B, F: constants
    """
    dxdt = [x[1], np.exp(c0)*(np.exp(c2) - np.exp(c1)*x[1])]
    return dxdt

def ODE_solution(t,x0,t_data):
    """
    x0: vector containing initial position and velocity
    """
    c0, c1, c2 = t  # Unpack parameters
    position, velocity = odeint(x_derivatives, x0, t_data, args=(c0,c1,c2)).T
    return position, velocity

# Dimensional Model
def x_derivatives_dim(x,t,m,B,F):
    """
    x: LHS row vector (dependent variable)
    t: time (the independent variable)
    m, B, F: constants
    """
    dxdt = [x[1], (1/m)*(F - B*x[1])]
    return dxdt

def ODE_solution_dim(t,x0,t_data):
    """
    x0: vector containing initial position and velocity
    """
    m, B, F = t  # Unpack parameters
    position, velocity = odeint(x_derivatives_dim, x0, t_data, args=(m,B,F)).T
    return position, velocity

# Functions to convert from non-dimensional parameters
# c0, c1, c2 to m, B, F
def c0_to_m(c0,ts,xs,Fs):
    return ((ts**2)*Fs)/(xs*c0)

def c1_to_B(c1,ts,xs,Fs):
    return (c1*ts*Fs)/xs

def c2_to_F(c2,Fs):
    return Fs*c2

# DE-MC Analysis
def get_posterior(data, n_samples, burn, dims=3):
    """
    Function to get parameter posterior distributions
    Returns posterior samples for each parameter discarding burn samples
    Also returns the multivariate posterior distribution
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
    
    return posterior, c_dist, c_means, post_cov, c_std

def stat_convergence(data, dims):
    """
    Function to calculate the mean and standarde deviation of a parameter dataset
    with increasing sample size
    """
    means    = np.array([[data[i][:N].mean() for N in range(1,len(data[i]))] for i in range(dims)])
    std_devs = np.array([[data[i][:N].std() for N in range(1,len(data[i]))] for i in range(dims)])

    return means, std_devs
