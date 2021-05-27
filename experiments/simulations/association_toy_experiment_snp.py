import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

sys.path.append("../../models")
from prrr_nb_tfp import fit_rrr
from sklearn.model_selection import train_test_split

import matplotlib

font = {"size": 25}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


def abline(slope, intercept, label=None, color="red"):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--", color=color, label=label, linewidth=5)


## Generate data
n = 200
p = 2
q = 2
k = 1

# z = np.random.gamma(1, 1, size=(n, k))
# A = np.array([[1.5], [-1.5]])
# B = np.array([[1.5], [1.5]])

# X = np.random.poisson(z @ A.T)
X = np.random.choice([0, 1, 2], size=(n, p))
A = np.random.gamma(1, 1, size=(p, k))
B = np.random.gamma(1, 1, size=(q, k))
Y = np.random.poisson(X @ A @ B.T)

## Fit model
rrr_results = fit_rrr(X=X, Y=Y, k=k)



## Extract parameters
A_mean = rrr_results["A_mean"].numpy()
B_mean = rrr_results["B_mean"].numpy()
A_stddev = rrr_results["A_stddev"].numpy()
B_stddev = rrr_results["B_stddev"].numpy()

A_est = np.exp(A_mean + 0.5 * A_stddev**2)
B_est = np.exp(B_mean + 0.5 * B_stddev**2)

import ipdb; ipdb.set_trace()

## Plot
f, (a1, a2, a3) = plt.subplots(
    1, 3, gridspec_kw={"width_ratios": [1.25, 1, 1.25]}, figsize=(19, 6)
)

# subplot 1
plt.sca(a1)
plt.scatter(X[:, 0], X[:, 1], c=np.mean(Y, axis=1))
cbar = plt.colorbar()
cbar.set_label("Gene 1 + gene 2", rotation=270, size=20, labelpad=15)
cbar.ax.tick_params(labelsize=10)
plt.xlabel("Cell covariate 1")
plt.ylabel("Cell covariate 2")
ax = plt.gca()
ax.text(-0.1, 1.0, "A", transform=ax.transAxes, size=20, weight="bold")
plt.title(r"$X$")
plt.xticks([])
plt.yticks([])

# subplot 2
plt.sca(a2)
plt.scatter(X[:, 0], X[:, 1], c=np.mean(Y, axis=1))
abline(A_est[1, 0] / A_est[0, 0], intercept=0, label="A")
plt.legend(prop={"size": 20})
ax = plt.gca()
ax.text(-0.1, 1.0, "B", transform=ax.transAxes, size=20, weight="bold")
plt.title(r"$X$")
plt.xticks([])
plt.yticks([])
plt.xlabel("Cell covariate 1")
plt.ylabel("Cell covariate 2")

# subplot 3
plt.sca(a3)
plt.scatter(Y[:, 0], Y[:, 1], c=np.squeeze(X @ A_est))
cbar = plt.colorbar()
cbar.set_label(r"Latent projection $XA$", rotation=270, size=20, labelpad=15)
cbar.ax.tick_params(labelsize=10)
abline(B_est[0, 1] / B_est[0, 0], intercept=0, label="B", color="green")
plt.legend(prop={"size": 20})
ax = plt.gca()
ax.text(-0.1, 1.0, "C", transform=ax.transAxes, size=20, weight="bold")
plt.title(r"$Y$")
plt.xticks([])
plt.yticks([])
plt.xlabel("Gene 1")
plt.ylabel("Gene 2")
plt.suptitle(r"$Y=XAB+\epsilon$", y=0.98)
plt.tight_layout()
plt.savefig("../../figures/paper_figures/figure1.png")
plt.savefig("../../figures/paper_figures/figure1.pdf", bbox_inches="tight")
plt.show()

import ipdb

ipdb.set_trace()
