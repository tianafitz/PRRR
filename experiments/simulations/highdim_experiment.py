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
from poisson_glm import fit_poisson_regression

## Create synthetic data
n = 200
p = 20
# q_list = [10, 50, 100, 1_000] #, 10_000]
dim_list = [10, 50, 100] #, 200] #, 10_000]
r_true = 5
frac_train = 0.8
USE_VI = False

model = "GRRR"
data_generating_model = "GRRR"

def centered_r2_score(y_test, preds):
    return r2_score(y_test - y_test.mean(0), preds - preds.mean(0))

n_repeats = 20
results = np.zeros((n_repeats, len(dim_list)))
results_fullrank = np.zeros((n_repeats, len(dim_list)))
results_glmnet = np.zeros((n_repeats, len(dim_list)))

for ii in range(n_repeats):

    for jj, dim in enumerate(dim_list):

        if data_generating_model == "GRRR":
            X = np.random.uniform(low=-5, high=5, size=(n, p))
            U = np.random.normal(loc=0.0, scale=0.25, size=(p, r_true))
            V = np.random.normal(loc=0.0, scale=0.25, size=(r_true, dim))
            linear_predictor = X @ U @ V
            size_factors = np.ones((n, 1))
            linear_predictor += np.log(size_factors)
            Y_mean = np.exp(linear_predictor)
            Y = np.random.poisson(Y_mean)

        elif data_generating_model == "PRRR":

            X = np.random.uniform(low=0, high=6, size=(n, p))
            size_factors = np.ones((n, 1))
            U_true = np.random.gamma(1.0, size=(p, r_true))
            V_true = np.random.gamma(1.0, size=(r_true, dim))

            Y_mean = X @ U_true @ V_true
            Y = np.random.poisson(Y_mean)

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
            model_object = GRRR(latent_dim=r_true)
        elif model == "PRRR":
            model_object = PRRR(latent_dim=r_true)
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
        print(dim, flush=True)
        print(curr_r2, flush=True)
        print("\n", flush=True)
        results[ii, jj] = curr_r2

        ## Full-rank
        model_object = GRRR(latent_dim=dim)
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
        
        curr_r2 = centered_r2_score(np.log(y_test + 1), test_preds)
        print(curr_r2, flush=True)
        print("\n", flush=True)
        results_fullrank[ii, jj] = curr_r2


        ## Sparse Gaussian regression (LASSO)
        files = glob.glob('./tmp/*')
        for f in files:
            os.remove(f)

        pd.DataFrame(X_train).to_csv("./tmp/X_train.csv", index=False)
        pd.DataFrame(X_test).to_csv("./tmp/X_test.csv", index=False)
        pd.DataFrame(y_train).to_csv("./tmp/Y_train.csv", index=False)
        pd.DataFrame(y_test).to_csv("./tmp/Y_test.csv", index=False)

        process = subprocess.Popen(['Rscript', 'run_glmnet.R'])
        process.wait()
        test_preds = pd.read_csv("./tmp/glmnet_preds.csv", index_col=0).values

        curr_r2 = centered_r2_score(np.log(y_test + 1), test_preds)
        print(curr_r2, flush=True)
        print("\n", flush=True)
        results_glmnet[ii, jj] = curr_r2

        files = glob.glob('./tmp/*')
        for f in files:
            os.remove(f)

# import ipdb; ipdb.set_trace()
results_df_rrr = pd.melt(pd.DataFrame(results, columns=dim_list))
results_df_rrr["method"] = "PRRR"
results_df_fullrank = pd.melt(pd.DataFrame(results_fullrank, columns=dim_list))
results_df_fullrank["method"] = "Full-rank"

results_df_glmnet = pd.melt(pd.DataFrame(results_glmnet, columns=dim_list))
results_df_glmnet["method"] = "LASSO"


results_df = pd.concat([results_df_rrr, results_df_fullrank, results_df_glmnet], axis=0)
# results_df = pd.concat([results_df_rrr, results_df_fullrank], axis=0)
results_df.index = np.arange(results_df.shape[0])
results_df.to_csv("./out/highdim_experiment.csv")
plt.figure(figsize=(10, 5))
# import ipdb; ipdb.set_trace()
# sns.boxplot(data=results_df, x="variable", y="value")
sns.lineplot(data=results_df, x="variable", y="value", hue="method")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel(r"$q$")
# plt.xscale("log")
plt.ylabel("R^2")
plt.tight_layout()
plt.show()
import ipdb

ipdb.set_trace()
