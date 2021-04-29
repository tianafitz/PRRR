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

font = {"size": 25}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

## Simulate data

FIGURE_DIR = "../../figures/plots/"

def association_mapping():

    p = 10
    q = 40
    n = 100
    k_true = 5

    n_repeats = 5
    r_list = np.arange(1, 11)
    elbo_list = np.zeros((n_repeats, len(r_list)))

    
    X = np.exp(np.random.normal(size=(n, p)))
    
    for ii in range(n_repeats):

        A_true = np.random.gamma(2, 1, size=(p, k_true))
        B_true = np.random.gamma(2, 1, size=(k_true, q))

        AB_true = A_true @ B_true
        
        Y_mean = X @ A_true @ B_true
        Y = np.random.poisson(Y_mean)

        for jj, k in enumerate(r_list):

            # Fit PRRR
            pr_results = fit_rrr(X=X, Y=Y, k=k)
            elbo = -pr_results['loss_trace'].numpy()[-1]
            elbo_list[ii, jj] = elbo

    plt.figure(figsize=(7, 5))
    plt.errorbar(r_list, np.mean(elbo_list, axis=0), yerr=np.std(elbo_list, axis=0))
    plt.axvline(k_true, linestyle="--", color="black")
    plt.xlabel("Rank")
    plt.ylabel("ELBO")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig("../../figures/paper_figures/figure5.pdf", bbox_inches="tight")
    plt.show()

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":

    association_mapping()
