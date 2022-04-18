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
from prrr import PRRR

## Create synthetic data
n = 200
p = 20
q = 20
r_true = 3
frac_train = 0.8
USE_VI = True

model = "PRRR"
data_generating_model = "PRRR"

n_repeats = 2
latent_dim_list = [1, 2, 3, 4, 5, 10, 20]
results = np.zeros((n_repeats, len(latent_dim_list)))

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
        U_true = np.random.gamma(1., size=(p, r_true))
        V_true = np.random.gamma(1., size=(r_true, q))

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

        curr_r2 = r2_score(y_test, test_preds)
        print(latent_dim, flush=True)
        print(curr_r2, flush=True)
        print("\n", flush=True)
        results[ii, jj] = curr_r2

        # plt.scatter(y_test[:, 0], test_preds[:, 0])
        # plt.show()
        # import ipdb; ipdb.set_trace()

results_df = pd.melt(pd.DataFrame(results, columns=latent_dim_list))
results_df.to_csv(
    "./out/prediction_experiment_results_{}{}.csv".format(model, "_VI" if USE_VI else "")
)
plt.figure(figsize=(5, 5))
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel("Rank")
plt.ylabel("R^2")
plt.show()
import ipdb

ipdb.set_trace()
