import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append("../models")
from prrr_nb_tfp import fit_rrr
from rrr_tfp_gaussian import fit_rrr as fit_rrr_gaussian
from sklearn.model_selection import train_test_split

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True


n = 200
p = 30
q = 30
k = 2
z = np.random.gamma(2, 1, size=(n, k))
A = np.random.gamma(2, 1, size=(p, k))
B = np.random.gamma(2, 1, size=(q, k))
A_zero_idx = np.random.binomial(n=1, p=0.5, size=(p, k)).astype(bool)
B_zero_idx = np.random.binomial(n=1, p=0.5, size=(q, k)).astype(bool)
A[A_zero_idx] = 0
B[B_zero_idx] = 0

X = np.random.poisson(z @ A.T)
Y = np.random.poisson(z @ B.T)

rrr_results = fit_rrr(X=X, Y=Y, k=k)

A_lognormal_mean = rrr_results['A_mean'].numpy()
A_stddev = rrr_results['A_stddev'].numpy()
B_lognormal_mean = rrr_results['B_mean'].numpy()
B_stddev = rrr_results['B_stddev'].numpy()

A_est = np.exp(A_lognormal_mean + 0.5 * A_stddev**2)
B_est = np.exp(B_lognormal_mean + 0.5 * B_stddev**2)

plt.subplot(121)
sns.heatmap(A)
plt.subplot(122)
sns.heatmap(A_est)
plt.show()

import ipdb; ipdb.set_trace()