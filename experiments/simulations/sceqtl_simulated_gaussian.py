import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
import sys
from sklearn.metrics import r2_score

sys.path.append("../../prrr/models/")
from grrr import GRRR

import sys

sys.path.append("../../prrr/models/old")
from rrr_tfp_gaussian import fit_rrr as fit_rrr_gaussian

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


## Create synthetic data
n = 400
p = 2
q = 2
r_true = 1
USE_VI = False
# latent_dim = 5


X = np.random.choice([0, 1, 2], replace=True, size=(n, p))  # genotype
U = np.array([[1.0], [-1.0]])
V = np.array([[1.0, -1.0]])
Y = np.random.normal(X @ U @ V)

## Fit model
rrr_results = fit_rrr_gaussian(X=X, Y=Y, k=r_true)

## Extract parameters
A_est = rrr_results["A_mean"].numpy()
B_est = rrr_results["B_mean"].numpy()


def abline(slope, intercept, label, color):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "--", label=label, color=color)


# plt.figure(figsize=(15, 5))
fig = plt.figure(figsize=(15, 5), constrained_layout=True)

eqtl_plot_inner = [
    ["eqtl1"],
    ["eqtl2"],
]
ax_dict = fig.subplot_mosaic(
    [["genotype", "expression", eqtl_plot_inner]],
    gridspec_kw={
        "hspace": 0.5,
    },
)

plt.sca(ax_dict["genotype"])
plt.scatter(
    X[:, 0] + np.random.normal(size=n, scale=0.1),
    X[:, 1] + np.random.normal(size=n, scale=0.1),
    c=Y[:, 0],
)  # , color="black")
plt.xlabel("SNP 1 minor allele count")
plt.ylabel("SNP 2 minor allele count")
plt.colorbar()


plt.sca(ax_dict["expression"])
plt.scatter(
    Y[:, 0],
    Y[:, 1],
    color="black",
)
xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
abline(
    slope=B_est.squeeze()[1] / B_est.squeeze()[0],
    intercept=0,
    label=r"$V$",
    color="red",
)
plt.gca().set_xlim(xlim)
plt.gca().set_ylim(ylim)
plt.xlabel("Expression, gene 1")
plt.ylabel("Expression, gene 2")
plt.legend()

# plt.subplot(133)
plt.sca(ax_dict["eqtl1"])
eqtl_df = pd.DataFrame({"Genotype": X[:, 0], "Expression": Y[:, 0]})
sns.boxplot(data=eqtl_df, x="Genotype", y="Expression")
# plt.xlabel("SNP 1 genotype (MAF)")
plt.xlabel("")
plt.ylabel("Expression,\ngene 1")

plt.sca(ax_dict["eqtl2"])
eqtl_df = pd.DataFrame({"Genotype": X[:, 0], "Expression": Y[:, 1]})
sns.boxplot(data=eqtl_df, x="Genotype", y="Expression")
plt.xlabel("SNP 1 minor allele count")
plt.ylabel("Expression,\ngene 2")

plt.tight_layout()
plt.savefig("./out/toy_example_sceqtl_gaussian.png")
plt.show()

# data_sliced = pd.DataFrame({"Genotype": X[:, top_coeff_idx[0]], "Expression": expression_sliced})
# sns.boxplot(data=data_sliced, x="Genotype", y="Expression")
# plt.show()

# plt.scatter(X[:, 0] + np.random.normal(scale=0.1, size=n), X[:, 1] + np.random.normal(scale=0.1, size=n))
# plt.show()
import ipdb

ipdb.set_trace()
