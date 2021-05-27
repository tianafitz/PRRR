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


num_repeats = 3
FIGURE_DIR = "../../figures/plots/"

## Get error for varying values of q
q_list = np.arange(1000, 10001, 1000)
print(q_list)


def compare_fullrank_rrr():

    p = 100
    # q = 1000
    n = 100
    n_train = 50
    k = 3
    mses_fullank = np.zeros((num_repeats, len(q_list)))
    mses_rrr = np.zeros((num_repeats, len(q_list)))

    for ii in range(num_repeats):
        for jj, q in enumerate(q_list):
            A_true = np.random.gamma(2, 1, size=(p, k))
            B_true = np.random.gamma(2, 1, size=(k, q))
            AB_true = A_true @ B_true
            X = np.exp(np.random.normal(size=(n, p)))
            Y_mean = X @ A_true @ B_true
            Y = np.random.poisson(Y_mean)

            X_train, Y_train = X[:n_train, :], Y[:n_train, :]
            X_test, Y_test = X[n_train:, :], Y[n_train:, :]

            # Fit full-rank model
            pr_results = fit_poisson_regression(X=X_train, Y=Y_train)
            B_lognormal_mean = pr_results["B_mean"].numpy()
            B_lognormal_stddev = pr_results["B_stddev"].numpy()
            B_est = np.exp(B_lognormal_mean + 0.5 * B_lognormal_stddev ** 2)

            # Compute MSE on test data
            test_preds = X_test @ B_est
            mse = np.mean((test_preds - Y_test)**2)
            mses_fullank[ii, jj] = mse

            # Fit PRRR
            pr_results = fit_rrr(X=X_train, Y=Y_train, k=k)
            A_lognormal_mean = pr_results["A_mean"].numpy()
            A_lognormal_stddev = pr_results["A_stddev"].numpy()
            B_lognormal_mean = pr_results["B_mean"].numpy()
            B_lognormal_stddev = pr_results["B_stddev"].numpy()
            A_est = np.exp(A_lognormal_mean + 0.5 * A_lognormal_stddev ** 2)
            B_est = np.exp(B_lognormal_mean + 0.5 * B_lognormal_stddev ** 2)

            # Compute MSE on test data
            test_preds = X_test @ A_est @ B_est
            mse = np.mean((test_preds - Y_test)**2)
            mses_rrr[ii, jj] = mse

            # print("MSE PR: {}, MSE RRR: {}".format(round(mses_fullank[ii], 3), round(mses_rrr[ii], 3)))
        

    # mse_df = pd.DataFrame(np.vstack([mses_fullank, mses_rrr]), columns=q_list)
    # mse_df['model'] = np.concatenate([["fullrank"] * num_repeats, ["prrr"] * num_repeats])
    # import ipdb; ipdb.set_trace()
    # mse_df = pd.melt()
    plt.figure(figsize=(10, 8))
    plt.errorbar(q_list, np.mean(mses_fullank, axis=0), yerr=np.std(mses_fullank, axis=0), label="Full-rank")
    plt.errorbar(q_list, np.mean(mses_rrr, axis=0), yerr=np.std(mses_rrr, axis=0), label="PRRR")
    plt.xlabel(r"$q$")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(FIGURE_DIR, "fullrank_vs_prrr.png"))
    plt.savefig("../../figures/paper_figures/figure3.pdf", bbox_inches="tight")
    plt.show()

    
    # plt.show()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":

    compare_fullrank_rrr()
