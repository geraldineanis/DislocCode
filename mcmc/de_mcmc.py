"""
Standard DE-MC (Braak and Vrugt (2006))
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(314)     # Set random seed for testing

# Model
# Start with a simple linear model
def lin_func(x,t):
    m, b = t
    return m*x + b

# Generate a straight line with noise
x = np.linspace(0,10,1000)
m = 2.0
b = 1.5
y = lin_func(x, [m,b])

sigma_n = 1.0
noise = np.random.normal(0.0, sigma_n, len(x))

data = y + noise

# plt.plot(x, data)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# Prior
# Assume a d-dimensional normal prior
def log_prior(x, prior_mean, prior_cov):
     return st.multivariate_normal.logpdf(x, mean=prior_mean, cov=prior_cov)

prior_mean = np.array([0.0, 0.0])
prior_cov = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

# Likelihood
def log_likelihood(func, x , t, y, sigma):
     f = func(x,t)
     return -1.0/(2.0*sigma**2)*np.sum((f - y)**2)

# Posterior
def log_posterior(log_likelihood, log_prior):
     return log_likelihood + log_prior

# Standard DE-MC
# Number of DE-MC iterations (generations)
n_iter = 10000

# Number of MCMC chains
N = 6

# Number of parameters (d)
dims = 2

# DE-MC parameters
gamma = 2.38/np.sqrt(2.0*dims)

# Initial parameter values
x0 = np.array(np.random.normal(size=dims))

# Initialise chain state matrix
# This is an N x dims matrix
X = np.zeros((N,dims))
for i in range(N):
    X[i] = np.random.normal(size=dims)

print(X)

# Chain updates
# Generate a proposal: xp = xi + gamma*(x_R1 - x_R2) + e
# Here e is 0
# for i in range(N):
# Choose from current states without replacement
samples = np.zeros((n_iter,N,dims))
samples[0] = X

states = np.arange(N)
counter = np.zeros(N)
for i in range(1,n_iter):                               # Loop through iterations
    X_new = np.zeros((N,dims))
    for j in range(N):                                  # Loop through chains
        x_R = np.random.choice(np.delete(states,j), 2)  # Select 2 chains at random
        R = X[x_R[0]] - X[x_R[1]]                       # Calculate the difference vector
        x_p = X[j] + gamma*R                            # Generate proposal

        # Calculate acceptance ratio, r
        r_i = log_posterior(log_likelihood(lin_func, x, X[j], data, sigma_n), log_prior(X[j], prior_mean, prior_cov))
        r_p = log_posterior(log_likelihood(lin_func, x, x_p, data, sigma_n), log_prior(x_p, prior_mean, prior_cov))

        r = np.exp(r_p - r_i)

        # Accept or reject proposal
        if np.random.rand() <= min(1, r):
            X_new[j] = x_p
            counter[j] = counter[j] + 1
        else:
            X_new[j] = X[j]

    samples[i] = X_new
    X = X_new

# Acceptance Rate
acceptance_rate = [counter[i]/n_iter for i in range(N)]
for i in range(N):
    print(f"Acceptance rate for chain {i+1} = {round(acceptance_rate[i]*100, 2)} %")

print(np.shape(samples))

burn = 1000
samples = samples[burn:]

p1 = np.array([samples[i][:,0] for i in range(n_iter-burn)]).T
p2 = np.array([samples[i][:,1] for i in range(n_iter-burn)]).T

print(np.shape(p1))

# Trace plots for all N chains
fig, axs = plt.subplots(1,2)
for i in range(N):
    # p1
    axs[0].plot(p1[i], alpha=0.3, label=f"p1 - chain {i+1}")
    
    # p2
    axs[1].plot(p2[i], alpha=0.3, label=f"p2 - chain {i+1}")

for ax in axs:
    ax.legend()
plt.show()

# Histogram of accepted samples for all N chains

fig, axs = plt.subplots(1,2)
for i in range(N):
    # p1
    axs[0].hist(p1[i], density=True, bins=100, color=f"C{i}", alpha=0.3, label=f"p1 - chain {i+1}")
    xd1 = np.linspace(min(p1[i]), max(p1[i]), 1000)
    p_cov = np.cov(p1[i], p2[i])
    axs[0].plot(xd1, st.norm.pdf(xd1, loc = np.mean(p1[i]), scale=np.sqrt(p_cov[0,0])), color=f"C{i}")
    axs[0].axvline(np.mean(p1[i]), ls="--", color=f"C{i}")
    # p2
    xd2 = np.linspace(min(p2[i]), max(p2[i]), 1000)
    axs[1].hist(p2[i], density=True, bins=100, color=f"C{i}", alpha=0.3, label=f"p2 - chain {i+1}")
    axs[1].plot(xd2, st.norm.pdf(xd2, loc = np.mean(p2[i]), scale=np.sqrt(p_cov[1,1])), color=f"C{i}")
    axs[1].axvline(np.mean(p2[i]), ls="--", color=f"C{i}")

for ax in axs:
    ax.legend()
plt.show()

# Choose a chain at random for the analysis
c = np.random.choice(states)
print(f"Doing analysis using chain {c}")

# Trace plots
fig, axs = plt.subplots(1,2)
axs[0].plot(p1[c])
axs[1].plot(p2[c])

plt.show()

p_means = np.array([np.mean(p1[c]), np.mean(p2[c])])
p_cov = np.array(np.cov(p1[c], p2[c]))

fit = lin_func(x, [np.mean(p1[c]), np.mean(p2[c])])

xd1 = np.linspace(min(p1[c]), max(p1[c]), 1000)
xd2 = np.linspace(min(p2[c]), max(p2[c]), 1000)

fig, axs = plt.subplots(1,2)
axs[0].hist(p1[c], bins=100, density=True, label="p1 probability density")
axs[0].plot(xd1, st.norm.pdf(xd1, loc =p_means[0], scale = np.sqrt(p_cov[0,0])))
# axs[0].plot(np.linspace(0.0,2.0, 1000), st.norm.pdf(np.linspace(0.0,2.0, 1000), loc=prior_mean[0], scale=np.sqrt(prior_cov[0,0])), label="p1 prior")

axs[1].hist(p2[c], bins=100, density=True, label="p2 probability density")
axs[1].plot(xd2, st.norm.pdf(xd2, loc = p_means[1], scale=np.sqrt(p_cov[1,1])))
# axs[1].plot(np.linspace(0.0,2.0, 1000), st.norm.pdf(np.linspace(0.0,2.0, 1000), loc=prior_mean[1], scale=np.sqrt(prior_cov[1,1])), label="p2 prior")

for ax in axs:
    ax.legend()

plt.show()

# Draw samples from posterior and plot fit
post_samples = st.multivariate_normal.rvs(p_means, p_cov, size=1000)

model_samples = [lin_func(x, s) for s in post_samples]

fig, axs = plt.subplots(1,1)
axs.scatter(x, data, color="tab:red", s=2, zorder=2, label="Data with noise")
axs.plot(x, y, color="tab:red", zorder=3, label="Original Data" )
axs.plot(x, fit, color="k", zorder=3, label="MAP Fit")

for i, s in enumerate(model_samples):
    if i == 1:
        axs.plot(x, s, color="tab:green", alpha=0.3, label="Samples")
    else:
        axs.plot(x, s, color="tab:green", alpha=0.3)

axs.legend()
plt.show()

