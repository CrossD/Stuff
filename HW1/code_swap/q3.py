import sys
import numpy as np
from numpy.core.function_base import linspace
import pandas as pd
from scipy.stats.mstats import mquantiles
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt

########################################################################################
## Handle batch job arguments:

nargs = len(sys.argv)
print 'Command line arguments: ' + str(sys.argv)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

########################
#sim_start = 1000
#length_datasets = 200
########################
#
##Note: this only sets the random seed for numpy, so if you intend
##on using other modules to generate random numbers, then be sure
##to set the appropriate RNG here
#
#if (nargs <= 1):
#    sim_num = sim_start + 1
#    np.random.seed(1330931)
#else:
#    # Decide on the job number, usually start at 1000:
#    sim_num = sim_start + int(sys.argv[2])
#    # Set a different random seed for every job number!!!
#    np.random.seed(762*sim_num + 1330931)
#
##Simulation datasets numbered 1001-1200

########################################################################################
########################################################################################


def logit(p):
    return np.log(p) - np.log(1 - p)


def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


def loglik(beta, m, y, X):
    ll = (np.sum(y * np.log(inv_logit(X.dot(beta))) + (m - y) * np.log(1 - inv_logit(X.dot(beta)))) -
          1 / 2 * (beta - beta_0).T.dot(Sigma_0_inv.dot(beta - beta_0)))
    return ll


def metrop_logreg(beta_init, m, y, X, niter=10000,
                  burnin=3000, print_every=1000, retune=100, verbose=False):
    p = X.shape[1]
    tune_period = 300
    samp = [[0] * p] * niter
    tune = np.identity(p) * 1e-1
    samp_acpt = 0.
    beta_cur = beta_init
    for j in range(burnin / tune_period):
        burn_acpt = 0.
        burnin_samp = np.zeros((tune_period, p))
        for i in range(tune_period):
            beta_prop = beta_cur + np.random.multivariate_normal([0] * p, tune)[:, np.newaxis]
            u = loglik(beta_prop, m, y, X) - loglik(beta_cur, m, y, X)
            if (np.log(np.random.uniform()) < u):
                beta_cur = beta_prop
                burn_acpt += 1
            burnin_samp[i, :] = beta_cur.T
        if (burn_acpt / tune_period > .95):
            tune = tune * 10
        elif (burn_acpt / tune_period < .05):
            tune = tune / 10
        else:
            tune = np.cov(burnin_samp.T)
        #print burn_acpt / tune_period
    # Sampling period
    for i in range(niter):
        beta_prop = beta_cur + np.random.multivariate_normal([0] * p, tune)[:, np.newaxis]
        u = loglik(beta_prop, m, y, X) - loglik(beta_cur, m, y, X)
        if (np.log(np.random.uniform()) < u):
            beta_cur = beta_prop
            samp_acpt += 1
        samp[i] = beta_cur
    #print tune
    print samp_acpt / niter
    return np.array(samp).reshape(niter, p)


def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)


def autocor(samp):
    nrows = samp.shape[0]
    ncols = samp.shape[1]
    samp_corr = [0] * ncols
    for i in range(ncols):
        cov_mat = np.cov(samp[0:(nrows - 2), i], samp[1:(nrows - 1), i])
        samp_corr[i] = cov_mat[0, 1] / np.sqrt(cov_mat[0, 0] * cov_mat[1, 1])
    return samp_corr


#################################################
p = 10 + 1
beta_0 = np.zeros((p, 1))
Sigma_0_inv = np.diag(np.ones(p)) * 1000
data = np.genfromtxt("breast_cancer.txt", dtype="f8," * 10 + "S5", names=True)
data = pd.DataFrame(data)
n = len(data)
y = np.array([int(i == "\"M\"") for i in data["diagnosis"]])[:, np.newaxis]
m = np.array([1] * n)[:, np.newaxis]
X = np.array(data.iloc[:, range(p-1)])
X = np.hstack((np.ones((n, 1)), X))
niter = 10000
burnin = 12000
# Set initial value to MLE
beta_init = np.zeros((p, 1))
np.random.seed(1330931)
samp = metrop_logreg(beta_init, m, y, X, niter=niter, burnin=burnin)

# Trace plots
for i in range(p):
    plt.subplot(6, 2, i+1)
    plt.plot(samp[:, i])
    plt.title("beta" + str(i))
plt.show()

# Kernel density estimate plots
for i in range(p):
    plt.subplot(6, 2, i+1)
    tmp_kde = gaussian_kde(samp[:, i].flatten())
    x = np.linspace(min(samp[:, i]), max(samp[:, i]), 200)
    plt.plot(x, tmp_kde(x))
    plt.title("beta" + str(i))
plt.show()

# Propose some new observations given the current observations
thin = 10
new_len = niter / thin
new_samp = np.zeros((new_len, n))
for i in range(new_len):
    p_new = inv_logit(X.dot(samp[i * thin, :].flatten()))
    y_new = [np.random.binomial(n0, p0) for (n0, p0) in zip(m.astype("l8"), p_new)]
    new_samp[i, :] = y_new

# Posterior predictive check
no_of_0 = np.zeros(new_len)
no_of_1 = np.zeros(new_len)
for i in range(new_len):
    count = count_unique(new_samp[i, :])
    no_of_0[i] = count[1][0]
    no_of_1[i] = count[1][1]
count_y = count_unique(y.flatten())

new_row_means = new_samp.mean(axis=1)
new_col_means = new_samp.mean(axis=0)

plt.subplot(311)
plt.hist(no_of_0)
plt.vlines(count_y[1][0], 0, 300, colors="red")
plt.subplot(312)
plt.hist(no_of_1)
plt.vlines(count_y[1][1], 0, 299, colors="red")
plt.subplot(313)
plt.hist(new_row_means)
plt.vlines(y.mean(), 0, 300, colors="red")
plt.show()


# Experiment
U, L, V = np.linalg.svd(X, full_matrices=False) # Note that the V in Numpy is the V' in usual sense
Y = X.dot(V.T)
np.random.seed(1330931)
samp_y = metrop_logreg(beta_init, m, y, Y, niter=10000, burnin=burnin)

for i in range(p):
    plt.subplot(6, 2, i+1)
    plt.plot(samp_y.dot(V)[:, i])
plt.show()

autocor(samp_y)
autocor(samp)