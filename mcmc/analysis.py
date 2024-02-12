import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def lin_func(x,t):
    m, b = t
    return m*x + b

data = np.load(f"./de_mc_out/de_mcmc.npz")
lin_f_data = np.loadtxt(f"./data.txt")

x = lin_f_data[:,0]
y = lin_f_data[:,1]

posterior = np.array(data["posterior"][1000:10000]).T

print(np.shape(posterior))

p1 = posterior[0,:]
p2 = posterior[1,:]

p_mean = np.array([np.mean(p1), np.mean(p2)])
p_cov = np.array(np.cov(p1,p2))

print("Parameter means")
print(p_mean)
print("\n")
print("Posterior Covariance Matrix")
print(p_cov)

fit = lin_func(x, p_mean)

post_samples = st.multivariate_normal.rvs(p_mean, p_cov, size=1000)

model_samples = [lin_func(x, s) for s in post_samples]

# Trace plots
fig, axs = plt.subplots(1,2)
axs[0].plot(p1)
axs[1].plot(p2)

axs[0].set_xlabel("p1")
axs[1].set_xlabel("p2")

plt.show()

xd1 = np.linspace(min(p1), max(p1), 1000)
xd2 = np.linspace(min(p2), max(p2), 1000)

# Histograms
fig, axs = plt.subplots(1,2)
axs[0].hist(p1, density=True, bins=100)
axs[0].plot(xd1, st.norm.pdf(xd1, loc =p_mean[0], scale = np.sqrt(p_cov[0,0])))

axs[1].hist(p2, density=True, bins=100)
axs[1].plot(xd2, st.norm.pdf(xd2, loc =p_mean[1], scale = np.sqrt(p_cov[1,1])))


axs[0].set_xlabel("p1")
axs[1].set_xlabel("p2")

for ax in axs:
    ax.set_ylabel("Density")

plt.show()

fig, axs = plt.subplots(1,1)
axs.scatter(x, y, color="tab:red", s=2, zorder=2, label="Data with noise")
axs.plot(x, lin_func(x, [2.0, 1.5]), color="tab:red", zorder=3, label="Original Data" )
axs.plot(x, fit, color="k", zorder=3, label="Mean Fit")

for i, s in enumerate(model_samples):
    if i == 1:
        axs.plot(x, s, color="tab:green", alpha=0.3, label="Samples")
    else:
        axs.plot(x, s, color="tab:green", alpha=0.3)

axs.set_xlabel("x")
axs.set_ylabel("y")

axs.legend()
plt.show()