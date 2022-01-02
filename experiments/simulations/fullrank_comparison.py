import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
import sys
import functools
from tensorflow_probability import distributions as tfd
from sklearn.metrics import r2_score

sys.path.append("../../models")
from prrr_nb_tfp import fit_rrr, fit_poisson_regression
from grrr_tfp import fit_grrr, grrr
from rrr_tfp_gaussian import fit_rrr as fit_rrr_gaussian
from sklearn.model_selection import train_test_split

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

## Simulate data


num_repeats = 5
FIGURE_DIR = "../../figures/plots/"

## Get error for varying values of q
# q_list = np.arange(1, 50, 10)
# q_list = np.linspace(1, 10, 3).astype(int)
q_list = np.array([1, 10, 100, 1000])


def compare_fullrank_rrr():

    p = 30
    # q = 1000
    n = 100
    n_train = 80
    
    mses_fullank = np.zeros((num_repeats, len(q_list)))
    mses_rrr = np.zeros((num_repeats, len(q_list)))

    for ii in range(num_repeats):
        for jj, q in enumerate(q_list):

            if q == 1:
                k = 1
            else:
                k = 2

            ## Generate data from model
            X = np.random.normal(size=(n, p))
            rrr_model = functools.partial(grrr, X=X, q=q, k=k, size_factors=None)
            model = tfd.JointDistributionCoroutineAutoBatched(rrr_model)
            A_true, B_true, Y = model.sample()
            A_true = A_true.numpy()
            B_true = B_true.numpy()
            Y = Y.numpy()

            ## Split into train and test
            X_train, Y_train = X[:n_train, :], Y[:n_train, :]
            X_test, Y_test = X[n_train:, :], Y[n_train:, :]

            ###### Fit full-rank model ######
            pr_results = fit_poisson_regression(X=X_train, Y=Y_train, use_vi=False, k_initialize=k)
            B = pr_results["B"].numpy()

            # Compute MSE on test data
            test_preds = np.exp(X_test @ B)
            mse = np.mean((test_preds - Y_test)**2)
            print("MSE, PR: {}".format(round(mse, 2)))
            mses_fullank[ii, jj] = mse

            ###### Fit PRRR ######
            results = fit_grrr(X=X_train, Y=Y_train, k=k, use_vi=False)
            A = results["A"].numpy()
            B = results["B"].numpy()
            AB_est = A @ B

            # Compute MSE on test data
            test_preds = np.exp(X_test @ AB_est)
            mse = np.mean((test_preds - Y_test)**2)
            # mse = r2_score(test_preds, Y_test)
            mses_rrr[ii, jj] = mse

            print("MSE, GRRR: {}".format(round(mse, 2)))

            # plt.scatter(np.ndarray.flatten(test_preds), np.ndarray.flatten(Y_test))
            # plt.show()
            # import ipdb; ipdb.set_trace()
            

            # print("MSE PR: {}, MSE RRR: {}".format(round(mses_fullank[ii], 3), round(mses_rrr[ii], 3)))
        

    fullrank_df = pd.melt(pd.DataFrame(mses_fullank, columns=q_list))
    fullrank_df['model'] = ["Full rank"] * fullrank_df.shape[0]
    prrr_df = pd.melt(pd.DataFrame(mses_rrr, columns=q_list))
    prrr_df['model'] = ["PRRR"] * prrr_df.shape[0]
    mse_df = pd.concat([fullrank_df, prrr_df], axis=0)

    # mse_df = pd.DataFrame({"Full rank": mses_fullank, "PRRR": mses_rrr})
    # mse_df['model'] = np.concatenate([["fullrank"] * num_repeats, ["prrr"] * num_repeats])
    # import ipdb; ipdb.set_trace()
    # mse_df = pd.melt()
    plt.figure(figsize=(10, 8))
    # plt.errorbar(q_list, np.mean(mses_fullank, axis=0), yerr=np.std(mses_fullank, axis=0), label="Full-rank")
    # plt.errorbar(q_list, np.mean(mses_rrr, axis=0), yerr=np.std(mses_rrr, axis=0), label="PRRR")
    sns.lineplot(data=mse_df, x="variable", y="value", hue="model")
    plt.xlabel("Number of outcome variables")
    plt.ylabel("Test MSE")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pjoin(FIGURE_DIR, "fullrank_vs_prrr.png"))
    # plt.savefig("../../figures/paper_figures/figure3.pdf", bbox_inches="tight")
    plt.show()

    
    # plt.show()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":

    compare_fullrank_rrr()
