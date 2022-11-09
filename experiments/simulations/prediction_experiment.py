import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
import sys
from sklearn.metrics import r2_score
import subprocess
import os
import glob

sys.path.append("../../prrr/models/")
from grrr import GRRR
from prrr import PRRR
from gaussian_rrr import GaussianRRR

## Create synthetic data
n = 200
p = 20
q = 20
r_true = 3
frac_train = 0.8
USE_VI = False

model = "GRRR"
data_generating_model = "GRRR"


def centered_r2_score(y_test, preds):
    return r2_score(y_test - y_test.mean(0), preds - preds.mean(0))


n_repeats = 2
latent_dim_list = [1, 2, 3, 4, 5, 10, min(p, q)]
results = np.zeros((n_repeats, len(latent_dim_list)))
results_gaussrrr = np.zeros((n_repeats, len(latent_dim_list)))
results_glmnet = np.zeros(n_repeats)

for ii in range(n_repeats):

    if data_generating_model == "GRRR":
        X = np.random.uniform(low=-5, high=5, size=(n, p))
        U = np.random.normal(loc=0.01, scale=0.25, size=(p, r_true))
        V = np.random.normal(loc=0.01, scale=0.25, size=(r_true, q))
        linear_predictor = X @ U @ V
        size_factors = np.ones((n, 1))
        linear_predictor += np.log(size_factors)
        Y_mean = np.exp(linear_predictor)
        Y = np.random.poisson(Y_mean)

    elif data_generating_model == "PRRR":

        X = np.random.uniform(low=0, high=6, size=(n, p))
        size_factors = np.ones((n, 1))
        U_true = np.random.gamma(1.0, size=(p, r_true))
        V_true = np.random.gamma(1.0, size=(r_true, q))

        Y_mean = X @ U_true @ V_true
        Y = np.random.poisson(Y_mean)

    for jj, latent_dim in enumerate(latent_dim_list):

        train_idx = np.random.choice(np.arange(n), size=int(frac_train * n))
        test_idx = np.setdiff1d(np.arange(n), train_idx)

        X_train, y_train = X[train_idx], Y[train_idx]
        X_test, y_test = X[test_idx], Y[test_idx]
        size_factors_train, size_factors_test = (
            size_factors[train_idx],
            size_factors[test_idx],
        )

        # size_factors = np.ones((train_idx.shape[0], 1))

        if model == "GRRR":
            model_object = GRRR(latent_dim=latent_dim)
        elif model == "PRRR":
            model_object = PRRR(latent_dim=latent_dim)
        model_object.fit(
            X=X_train,
            Y=y_train,
            use_vi=USE_VI,
            n_iters=1000,
            learning_rate=1e-2,
            size_factors=size_factors_train,
        )

        if USE_VI:
            U_est = model_object.param_dict["U"].numpy()
            V_est = model_object.param_dict["V"].numpy()
        else:
            U_est = model_object.param_dict["U"].numpy()
            V_est = model_object.param_dict["V"].numpy()

        if model == "GRRR":
            test_preds = np.exp(X_test @ U_est @ V_est + np.log(size_factors_test))
        elif model == "PRRR":
            test_preds = X_test @ U_est @ V_est

        curr_r2 = centered_r2_score(y_test, test_preds)
        print(latent_dim, flush=True)
        print(curr_r2, flush=True)
        print("\n", flush=True)
        results[ii, jj] = curr_r2

        ## Gaussian RRR
        model_object = GaussianRRR(latent_dim=latent_dim)
        model_object.fit(
            X=X_train,
            Y=np.log(y_train + 1),
            use_vi=USE_VI,
            n_iters=1000,
            learning_rate=1e-2,
            size_factors=size_factors_train,
            # size_factors=np.zeros((len(X_train), 1)),
        )

        U_est = model_object.param_dict["U"].numpy()
        V_est = model_object.param_dict["V"].numpy()
        v0_est = model_object.param_dict["v0"].numpy()
        test_preds = X_test @ U_est @ V_est + v0_est
        # import ipdb; ipdb.set_trace()

        curr_r2 = centered_r2_score(np.log(y_test + 1), test_preds)
        print(curr_r2, flush=True)
        print("\n", flush=True)
        results_gaussrrr[ii, jj] = curr_r2

    ## Sparse Gaussian regression
    files = glob.glob("./tmp/*")
    for f in files:
        os.remove(f)

    pd.DataFrame(X_train).to_csv("./tmp/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("./tmp/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("./tmp/Y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("./tmp/Y_test.csv", index=False)

    process = subprocess.Popen(["Rscript", "run_glmnet.R"])
    process.wait()
    test_preds = pd.read_csv("./tmp/glmnet_preds.csv", index_col=0).values

    # import ipdb; ipdb.set_trace()
    curr_r2 = centered_r2_score(np.log(y_test + 1), test_preds)
    print(curr_r2, flush=True)
    print("\n", flush=True)
    results_glmnet[ii] = curr_r2

    files = glob.glob("./tmp/*")
    for f in files:
        os.remove(f)
    # import ipdb; ipdb.set_trace()

# import ipdb; ipdb.set_trace()
results_df_rrr = pd.melt(pd.DataFrame(results, columns=latent_dim_list))
results_df_rrr["method"] = "PRRR"
results_df_gauss = pd.melt(pd.DataFrame(results_gaussrrr, columns=latent_dim_list))
results_df_gauss["method"] = "Gaussian RRR"

results_df_glmnet = pd.DataFrame(
    {"variable": np.repeat([1, 20], 2), "value": np.tile(results_glmnet, 2)}
)
# results_df_glmnet["variable"] = np.repeat([1, 20], 2)
results_df_glmnet["method"] = "LASSO"

results_df = pd.concat([results_df_rrr, results_df_gauss, results_df_glmnet], axis=0)
results_df.index = np.arange(results_df.shape[0])
results_df.to_csv(
    "./out/prediction_experiment_results_{}{}.csv".format(
        model, "_VI" if USE_VI else ""
    )
)
plt.figure(figsize=(10, 5))
# import ipdb; ipdb.set_trace()
# sns.boxplot(data=results_df, x="variable", y="value")
sns.lineplot(data=results_df, x="variable", y="value", hue="method")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("Rank")
plt.ylabel("R^2")
plt.tight_layout()
plt.show()
import ipdb

ipdb.set_trace()
