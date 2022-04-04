import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
import sys

sys.path.append("../../models")
from prrr_nb_tfp import fit_rrr, fit_poisson_regression
from rrr_tfp_gaussian import fit_rrr as fit_rrr_gaussian
from sklearn.model_selection import train_test_split

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

## Simulate data

FIGURE_DIR = "../../figures/plots/"


def association_mapping():

    p = 10
    q = 40
    n = 100
    k = 3

    A_true = np.random.gamma(2, 1, size=(p, k))
    B_true = np.random.gamma(2, 1, size=(k, q))

    # A_true[np.random.binomial(n=1, p=0.5, size=(p, k)).astype(bool)] = 0
    A_true[: p // 2, 0] = 0
    A_true[p // 2 :, 1] = 0
    A_true[p // 4 : 3 * p // 4, 2] = 0
    B_true[0, : 2 * q // 3] = 0
    B_true[1, q // 3 :] = 0
    B_true[2, : q // 3] = 0
    B_true[2, 2 * q // 3 :] = 0
    AB_true = A_true @ B_true
    X = np.exp(np.random.normal(size=(n, p)))
    Y_mean = X @ A_true @ B_true
    Y = np.random.poisson(Y_mean)

    # Fit PRRR
    pr_results = fit_rrr(X=X, Y=Y, k=k)
    A_lognormal_mean = pr_results["A_mean"].numpy()
    A_lognormal_stddev = pr_results["A_stddev"].numpy()
    B_lognormal_mean = pr_results["B_mean"].numpy()
    B_lognormal_stddev = pr_results["B_stddev"].numpy()
    A_est = np.exp(A_lognormal_mean + 0.5 * A_lognormal_stddev ** 2)
    B_est = np.exp(B_lognormal_mean + 0.5 * B_lognormal_stddev ** 2)

    plt.figure(figsize=(21, 12))
    plt.subplot(231)
    sns.heatmap(A_true)
    plt.xlabel("Latent Factors")
    plt.ylabel("Covariates")
    plt.xticks([])
    plt.title(r"True $U$")
    plt.subplot(232)
    sns.heatmap(B_true.T)
    plt.xlabel("Genes")
    plt.ylabel("Latent factors")
    plt.xticks([])
    plt.title(r"True $V$")
    plt.subplot(233)
    sns.heatmap(A_true @ B_true)
    plt.title(r"True $UV^\top$")
    plt.xlabel("Genes")
    plt.ylabel("Covariates")
    plt.xticks([])

    plt.subplot(234)
    sns.heatmap(A_est)
    plt.xlabel("Latent Factors")
    plt.ylabel("Covariates")
    plt.xticks([])
    plt.title(r"Estimated $U$")
    plt.subplot(235)
    sns.heatmap(B_est.T)
    plt.xlabel("Genes")
    plt.ylabel("Latent factors")
    plt.xticks([])
    plt.title(r"Estimated $V$")
    plt.subplot(236)
    sns.heatmap(A_est @ B_est)
    plt.title(r"Estimated $UV^\top$")
    plt.xlabel("Genes")
    plt.ylabel("Covariates")
    plt.tight_layout()
    plt.savefig(
        pjoin("../../figures/paper_figures/", "figure4.pdf"), bbox_inches="tight"
    )
    plt.show()

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":

    association_mapping()
