import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

sys.path.append("../../prrr/models/old")
# from prrr_nb_tfp import fit_rrr
from rrr_tfp_gaussian import fit_rrr as fit_rrr_gaussian
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

z = np.random.normal(size=(n, k))
U = np.array([[1.5], [-1.5]])
V = np.array([[1.5], [1.5]])
X = np.random.normal(z @ U.T)
Y = np.random.normal(z @ V.T)

# X = np.random.choice([0, 1, 2], replace=True, size=(n, p))
# U = np.array([[1.0], [-1.0]])
# V = np.array([[1.0, -1.0]])
# Y = np.random.normal(X @ U @ V)

# linear_predictor = X @ U @ V
# # size_factors = np.random.uniform(low=1e-1, high=2, size=n).reshape(-1, 1)
# size_factors = np.ones((n, 1))
# linear_predictor += np.log(size_factors)
# Y_mean = np.exp(linear_predictor)

# Y = np.random.poisson(Y_mean)

# nonzero_idx = np.where(Y.sum(1) > 0)[0]
# X = X[nonzero_idx]
# Y = Y[nonzero_idx]
# size_factors = size_factors[nonzero_idx]

## Fit model
rrr_results = fit_rrr_gaussian(X=X, Y=Y, k=k)

## Extract parameters
A_est = rrr_results["A_mean"].numpy()
B_est = rrr_results["B_mean"].numpy()

# import ipdb; ipdb.set_trace()
## Plot
f, (a1, a2, a3) = plt.subplots(
    1, 3, gridspec_kw={"width_ratios": [1, 1, 1]}, figsize=(19, 6)
)
# plt.figure(figsize=(16, 7))

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
# plt.subplot(121)
plt.scatter(
    X[:, 0] + np.random.normal(size=n, scale=0.1),
    X[:, 1] + np.random.normal(size=n, scale=0.1),
    c=np.mean(Y, axis=1),
)
cbar = plt.colorbar()
cbar.set_label("Gene 1 + gene 2", rotation=270, size=20, labelpad=15)
cbar.ax.tick_params(labelsize=10)
abline(A_est[1, 0] / A_est[0, 0], intercept=0, label="U")
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
# plt.subplot(122)
plt.scatter(Y[:, 0], Y[:, 1], c=np.squeeze(X @ A_est))
cbar = plt.colorbar()
cbar.set_label(r"Latent projection $XU$", rotation=270, size=20, labelpad=15)
cbar.ax.tick_params(labelsize=10)
abline(B_est[0, 1] / B_est[0, 0], intercept=0, label="V", color="green")
plt.legend(prop={"size": 20})
ax = plt.gca()
ax.text(-0.1, 1.0, "C", transform=ax.transAxes, size=20, weight="bold")
plt.title(r"$Y$")
plt.xticks([])
plt.yticks([])
plt.xlabel("Expression, gene 1")
plt.ylabel("Expression, gene 2")
# plt.suptitle(r"$Y=XUV^\top + \epsilon$", y=0.98)
plt.tight_layout()
plt.savefig("../../figures/paper_figures/figure1.png")
plt.savefig("../../figures/paper_figures/figure1.pdf", bbox_inches="tight")
plt.show()

import ipdb

ipdb.set_trace()
