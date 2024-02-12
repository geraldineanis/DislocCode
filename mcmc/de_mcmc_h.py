"""
Standard DE-MC  (Braak and Vrugt (2006))
Including noise variance as a fitting parameter - requires a prior on the noise
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

def lin_func_n(x,t):
    m, b, sigma_n = t
    noise = np.random.normal(0.0, sigma_n, len(x))
    return m*x + b + noise

# Generate a straight line with noise
x = np.linspace(0,10,1000)
m = 2.0
b = 1.5
y = lin_func(x, [m,b])

sigma_n = 5.0
noise = np.random.normal(0.0, sigma_n, len(x))

data = y + noise

data_n = lin_func_n(x,[m,b,sigma_n])

# plt.plot(x, data)
# plt.plot(x, data_n)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# Prior(s)
# Assume normal priors for now
def log_prior(x, prior_mean, prior_std):
     return st.norm.logpdf(x, loc=prior_mean, scale=prior_std)

prior_mean = np.array([0.0, 0.0])
prior_std = np.array([1.0, 1.0])

mean_noise = 0.0
sigma_noise = 1.0


# Likelihood
def log_likelihood(func, x , t, y, sigma):
     f = func(x,t)
     return -1.0/(2.0*sigma**2)*np.sum((f - y)**2)

# Posterior
def log_posterior(log_likelihood, log_prior):
     return log_likelihood + log_prior

# Standard DE-MC
# Number of DE-MC iterations (generations)
n_iter = 5000

# Number of MCMC chains
N = 6

# Number of parameters (d)
dims = 3

# DE-MC parameters
gamma = 2.38/np.sqrt(2.0*dims)

# Initial parameter values
x0 = np.array(np.random.normal(size=dims))

# Initialise chain state matrix
# This is an N x dims matrix
X = np.zeros((N,dims))
for i in range(N):
    X[i][:2] = np.random.normal(size=2)
    X[i][2] = np.random.exponential()

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

        if x_p[2] <= 0.0:
            continue
        # Calculate acceptance ratio, r
        r_i = log_likelihood(lin_func_n, x, X[j], data, X[j][2]) + log_prior(X[j][0], prior_mean[0], prior_std[0]) \
                                                                 + log_prior(X[j][1], prior_mean[1], prior_std[1]) \
                                                                 + st.expon.logpdf(X[j][2])
        
        r_p = log_likelihood(lin_func_n, x, x_p, data, x_p[2]) + log_prior(x_p[0], prior_mean[0], prior_std[0]) \
                                                               + log_prior(x_p[1], prior_mean[1], prior_std[1]) \
                                                               + st.expon.logpdf(x_p[2])

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
p3 = np.array([samples[i][:,2] for i in range(n_iter-burn)]).T

print(np.shape(p1))

# Trace plots for all N chains
fig, axs = plt.subplots(1,dims)
for i in range(N):
    # p1
    axs[0].plot(p1[i], alpha=0.3, label=f"p1 - chain {i+1}")
    # p2
    axs[1].plot(p2[i], alpha=0.3, label=f"p2 - chain {i+1}")
    # p3
    axs[2].plot(p3[i], alpha=0.3, label=f"p3 - chain {i+1}")

for ax in axs:
    ax.legend()
plt.show()

# Histogram of accepted samples for all N chains

fig, axs = plt.subplots(1,dims)
for i in range(N):
    # p1
    axs[0].hist(p1[i], density=True, bins=100, color=f"C{i}", alpha=0.3, label=f"p1 - chain {i+1}")
    xd = np.linspace(min(p1[i]), max(p1[i]), 1000)
    p_cov = np.cov(p1[i], p2[i])
    axs[0].plot(xd, st.norm.pdf(xd, loc = np.mean(p1[i]), scale=np.sqrt(p_cov[0,0])), color=f"C{i}")
    axs[0].axvline(np.mean(p1[i]), ls="--", color=f"C{i}")
    # p2
    xd = np.linspace(min(p2[i]), max(p2[i]), 1000)
    axs[1].hist(p2[i], density=True, bins=100, color=f"C{i}", alpha=0.3, label=f"p2 - chain {i+1}")
    axs[1].plot(xd, st.norm.pdf(xd, loc = np.mean(p2[i]), scale=np.sqrt(p_cov[1,1])), color=f"C{i}")
    axs[1].axvline(np.mean(p2[i]), ls="--", color=f"C{i}")
    # p3
    xd = np.linspace(min(p3[i]), max(p3[i]), 1000)
    axs[2].hist(p3[i], density=True, bins=100, color=f"C{i}", alpha=0.3, label=f"p3 - chain {i+1}")
    axs[2].plot(xd, st.norm.pdf(xd, loc = np.mean(p3[i]), scale=np.sqrt(p_cov[1,1])), color=f"C{i}")
    axs[2].axvline(np.mean(p3[i]), ls="--", color=f"C{i}")

for ax in axs:
    ax.legend()
plt.show()

for i in range(N):
    print(np.mean(p1[i]), np.mean(p2[i]), np.mean(p3[i]))

# # # Choose a chain at random for the analysis
# # c = np.random.choice(states)
# # print(f"Doing analysis using chain {c}")

# # # Trace plots
# # fig, axs = plt.subplots(1,dims)
# # axs[0].plot(p1[c])
# # axs[1].plot(p2[c])
# # axs[2].plot(p3[c])

# # plt.show()

# # p_means = np.array([np.mean(p1[c]), np.mean(p2[c]), np.mean(p3[c])])
# # p_cov = np.cov(p1[c], p2[c], p3[c]], rowvar=False)

# # fit = lin_func(x, [np.mean(p1[c]), np.mean(p2[c])])

# # # Draw samples from posterior
# # post_samples = st.multivariate_normal.rvs(p_means, p_cov, size=1000)

# # model_samples = [lin_func(x, s[:2]) for s in post_samples]

# # xd1 = np.linspace(min(p1[c]), max(p1[c]), 1000)
# # xd2 = np.linspace(min(p2[c]), max(p2[c]), 1000)
# # xd2 = np.linspace(min(p2[c]), max(p2[c]), 1000)


# # fig, axs = plt.subplots(1,2)
# # axs[0].hist(p1[c], bins=50, density=True)
# # axs[0].plot(xd1, st.norm.pdf(xd1, loc = p_means[0], scale = p_cov[0,0]))
# # axs[1].hist(p2[c], bins=50, density=True)
# # axs[1].plot(xd2, st.norm.pdf(xd2, loc = p_means[1], scale = p_cov[1,1]))
# # axs[2].hist(p2[c], bins=50, density=True)
# # axs[2].plot(xd2, st.norm.pdf(xd2, loc = p_means[1], scale = p_cov[2,2]))

# # plt.show()

# # fig, axs = plt.subplots(1,1)
# # axs.scatter(x, data, color="tab:red", s=2, zorder=2, label="Data with noise")
# # axs.plot(x, y, color="tab:red", zorder=3, label="Original Data" )
# # axs.plot(x, fit, color="k", zorder=3, label="MAP Fit")

# # for i, s in enumerate(model_samples):
# #     if i == 1:
# #         axs.plot(x, s, color="tab:green", alpha=0.3, label="Samples")
# #     else:
# #         axs.plot(x, s, color="tab:green", alpha=0.3)

# # axs.legend()
# # plt.show()

