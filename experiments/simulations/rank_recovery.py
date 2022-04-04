import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
import sys
import functools
from tensorflow_probability import distributions as tfd

sys.path.append("../../prrr/models/")
from grrr import GRRR

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

## Simulate data

FIGURE_DIR = "../../figures/plots/"


def association_mapping():

    n = 200
    p = 20
    q = 20
    r_true = 3
    frac_train = 0.8

    n_repeats = 30
    latent_dim_list = np.arange(1, 11)
    elbo_list = np.zeros((n_repeats, len(latent_dim_list)))

    for ii in range(n_repeats):

        ## Generate data from model
        X = np.random.uniform(low=-5, high=5, size=(n, p))
        U = np.random.normal(loc=0.01, scale=0.25, size=(p, r_true))
        V = np.random.normal(loc=0.01, scale=0.25, size=(r_true, q))

        linear_predictor = X @ U @ V
        size_factors = np.ones((n, 1))
        linear_predictor += np.log(size_factors)
        Y_mean = np.exp(linear_predictor)

        Y = np.random.poisson(Y_mean)

        for jj, latent_dim in enumerate(latent_dim_list):

            # Fit PRRR
            grrr = GRRR(latent_dim=latent_dim)
            grrr.fit(
                X=X,
                Y=Y,
                use_vi=True,
                n_iters=5000,
                learning_rate=1e-2,
                size_factors=size_factors,
            )
            elbo = -grrr.loss_trace.numpy()[-1]
            elbo_list[ii, jj] = elbo
            print(elbo)

    results_df = pd.melt(pd.DataFrame(elbo_list, columns=latent_dim_list))
    results_df.to_csv("./out/rank_recovery_experiment.csv")

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        latent_dim_list, np.mean(elbo_list, axis=0), yerr=np.std(elbo_list, axis=0)
    )
    plt.axvline(r_true, linestyle="--", color="black")
    plt.xlabel("Rank")
    plt.ylabel("ELBO")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig("./out/rank_recovery_experiment.png")
    plt.show()

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":

    association_mapping()
