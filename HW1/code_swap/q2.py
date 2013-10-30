import sys
import numpy as np
from scipy.stats.mstats import mquantiles
import matplotlib.pyplot as plt

########################################################################################
## Handle batch job arguments:

nargs = len(sys.argv)
print 'Command line arguments: ' + str(sys.argv)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

#######################
sim_start = 1000
length_datasets = 200
#######################

#Note: this only sets the random seed for numpy, so if you intend
#on using other modules to generate random numbers, then be sure
#to set the appropriate RNG here

if (nargs <= 1):
    sim_num = sim_start + 1
    np.random.seed(1330931)
else:
    # Decide on the job number, usually start at 1000:
    sim_num = sim_start + int(sys.argv[2])
    # Set a different random seed for every job number!!!
    np.random.seed(762*sim_num + 1330931)

#Simulation datasets numbered 1001-1200

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
                  burnin=1000, print_every=1000, retune=100, verbose=False):
    p = X.shape[1]
    burnin_samp = [[0, 0]] * burnin
    samp = [[0, 0]] * niter
    tune = 3e-3
    burn_acpt = 0.
    samp_acpt = 0.

    beta_cur = beta_init

    # Burn-in period
    for i in range(burnin):
        beta_prop = beta_cur + np.random.multivariate_normal([0] * p, tune * np.diag([1] * p))
        u = loglik(beta_prop, m, y, X) - loglik(beta_cur, m, y, X)
        if (np.log(np.random.uniform()) < u):
            beta_cur = beta_prop
            burn_acpt += 1
        burnin_samp[i] = beta_cur

    # Sampling period
    for i in range(niter):
        beta_prop = beta_cur + np.random.multivariate_normal([0] * p, tune * np.diag([1] * p))
        u = loglik(beta_prop, m, y, X) - loglik(beta_cur, m, y, X)
        if (np.log(np.random.uniform()) < u):
            beta_cur = beta_prop
            samp_acpt += 1
        samp[i] = beta_cur
    print burn_acpt / burnin, samp_acpt / niter

    return np.array(samp)

#################################################
p = 2
beta_0 = np.zeros(2).astype("f8")
Sigma_0_inv = np.diag(np.ones(p))
y, m, X1, X2 = np.genfromtxt("blr_data_" + str(sim_num) + ".csv", skip_header=1,  delimiter=",",
                             unpack=True)
X = np.array([X1, X2]).T

samp = metrop_logreg(np.array([0, 0]).astype("f8"), m, y, X, niter=10000)

#plt.subplot(211)
#plt.plot(samp[:, 0])
#plt.subplot(212)
#plt.plot(samp[:, 1])
#plt.show()

res = mquantiles(samp, prob=np.linspace(0.01, 0.99, 99), axis=0)
np.savetxt("blr_res_" + str(sim_num) + ".csv", res, delimiter=",")
