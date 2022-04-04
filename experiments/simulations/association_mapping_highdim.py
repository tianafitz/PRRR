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
    # A_est = np.exp(A_lognormal_mean - A_lognormal_stddev ** 2)
    # B_est = np.exp(B_lognormal_mean - B_lognormal_stddev ** 2)

    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    sns.heatmap(B_true)
    plt.xlabel("Genes")
    plt.ylabel("Latent factors")
    plt.xticks([])
    plt.title(r"True $V$")
    plt.subplot(122)
    sns.heatmap(B_est)
    plt.xlabel("Genes")
    plt.ylabel("Latent factors")
    plt.xticks([])
    plt.title(r"Estimated $V$")
    plt.tight_layout()
    plt.savefig(
        pjoin("../../figures/paper_figures/", "figure4.pdf"), bbox_inches="tight"
    )
    plt.show()

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":

    association_mapping()
