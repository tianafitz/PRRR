import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
import sys
from sklearn.metrics import r2_score
import sklearn.linear_model
from sklearn.linear_model import LinearRegression

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


sys.path.append("../../prrr/models/")
from grrr import GRRR
from prrr import PRRR
from gaussian_rrr import GaussianRRR

## Create synthetic data
n = 1000
p = 20
q = 20
r_true = 3
frac_train = 0.8
USE_VI = False

data_generating_model = "PRRR"

n_repeats = 2
results_prrr = np.zeros(n_repeats)
results_grrr = np.zeros(n_repeats)
results_gaussrrr = np.zeros(n_repeats)
results_fullrank = np.zeros(n_repeats)


def centered_r2_score(y_test, preds):
    return r2_score(y_test - y_test.mean(0), preds - preds.mean(0))


def _fit_rrr_no_intercept_all_ranks(
    X: np.ndarray, Y: np.ndarray, alpha: float, solver: str
):
    ridge = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=False, solver=solver)
    beta_ridge = ridge.fit(X, Y).coef_
    Lambda = np.eye(X.shape[1]) * np.sqrt(np.sqrt(alpha))
    X_star = np.concatenate((X, Lambda))
    Y_star = X_star @ beta_ridge.T
    _, _, Vt = np.linalg.svd(Y_star, full_matrices=False)
    return beta_ridge, Vt


def _fit_rrr_no_intercept(
    X: np.ndarray, Y: np.ndarray, alpha: float, rank: int, solver: str, memory=None
):
    memory = sklearn.utils.validation.check_memory(memory)
    fit = memory.cache(_fit_rrr_no_intercept_all_ranks)
    beta_ridge, Vt = fit(X, Y, alpha, solver)
    return Vt[:rank, :].T @ (Vt[:rank, :] @ beta_ridge)


class ReducedRankRidge(
    sklearn.base.MultiOutputMixin,
    sklearn.base.RegressorMixin,
    sklearn.linear_model._base.LinearModel,
):
    def __init__(
        self, alpha=1.0, fit_intercept=True, rank=None, ridge_solver="auto", memory=None
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.rank = rank
        self.ridge_solver = ridge_solver
        self.memory = memory

    def fit(self, X, y):
        if self.fit_intercept:
            X_offset = np.average(X, axis=0)
            y_offset = np.average(y, axis=0)
            # doesn't modify inplace, unlike -=
            X = X - X_offset
            y = y - y_offset
        self.coef_ = _fit_rrr_no_intercept(
            X, y, self.alpha, self.rank, self.ridge_solver, self.memory
        )
        self.rank_ = np.linalg.matrix_rank(self.coef_)
        if self.fit_intercept:
            self.intercept_ = y_offset - X_offset @ self.coef_.T
        else:
            self.intercept_ = np.zeros(y.shape[1])
        return self

    def predict(self, X):
        return X @ model_object.coef_.T + model_object.intercept_


for ii in range(n_repeats):

    if data_generating_model == "GRRR":
        X = np.random.uniform(low=0, high=3, size=(n, p))
        # # U = np.random.normal(loc=0.1, scale=0.1, size=(p, r_true))
        # # V = np.random.normal(loc=0.1, scale=0.1, size=(r_true, q))
        # U = np.random.uniform(low=-0.0, high=0.1, size=(p, r_true))
        # V = np.random.uniform(low=-0.0, high=0.1, size=(r_true, q))
        # linear_predictor = X @ U @ V
        size_factors = np.ones((n, 1))

        # # linear_predictor += np.log(size_factors)
        # Y_mean = np.exp(linear_predictor)
        # Y = np.random.poisson(Y_mean).astype(int)
        # # size_factors = Y.sum(1).reshape(-1, 1)
        Y = np.random.poisson(1, size=(n, q))

    elif data_generating_model == "PRRR":

        X = np.random.uniform(low=0, high=3, size=(n, p))
        size_factors = np.ones((n, 1))
        U_true = np.random.gamma(0.5, size=(p, r_true))
        V_true = np.random.gamma(0.5, size=(r_true, q))

        Y_mean = X @ U_true @ V_true
        Y = np.random.poisson(Y_mean)

    # import ipdb; ipdb.set_trace()
    Y = Y * np.random.choice([0, 1], p=[0.1, 0.9], size=Y.shape)

    train_idx = np.random.choice(np.arange(n), size=int(frac_train * n))
    test_idx = np.setdiff1d(np.arange(n), train_idx)

    X_train, y_train = X[train_idx], Y[train_idx]
    X_test, y_test = X[test_idx], Y[test_idx]
    size_factors_train, size_factors_test = (
        size_factors[train_idx],
        size_factors[test_idx],
    )

    ### PRRR
    model_object = PRRR(latent_dim=r_true)
    model_object.fit(
        X=X_train,
        Y=y_train,
        use_vi=USE_VI,
        n_iters=1000,
        learning_rate=1e-2,
        # size_factors=size_factors_train,
        use_total_counts_as_size_factors=True,
        # size_factors=np.zeros((len(X_train), 1)),
    )

    U_est = model_object.param_dict["U"].numpy()
    V_est = model_object.param_dict["V"].numpy()
    size_factors_est = model_object.param_dict["size_factors"].numpy()

    linreg = LinearRegression()
    linreg.fit(y_train.mean(1).reshape(-1, 1), size_factors_est.squeeze())
    size_factors_est_test = linreg.predict(y_test.mean(1).reshape(-1, 1))

    test_preds = (X_test @ U_est @ V_est) * size_factors_est_test.reshape(-1, 1)
    
    curr_r2 = centered_r2_score(y_test, test_preds)
    print("PRRR")
    print(curr_r2, flush=True)
    print("\n", flush=True)
    results_prrr[ii] = curr_r2
    import ipdb; ipdb.set_trace()

    ### GRRR
    model_object = GRRR(latent_dim=r_true)
    model_object.fit(
        X=X_train,
        Y=y_train,
        use_vi=USE_VI,
        n_iters=1000,
        learning_rate=1e-2,
        size_factors=size_factors_train,
        # size_factors=np.zeros((len(X_train), 1)),
    )

    U_est = model_object.param_dict["U"].numpy()
    V_est = model_object.param_dict["V"].numpy()
    test_preds = np.exp(X_test @ U_est @ V_est + np.log(size_factors_test))

    curr_r2 = centered_r2_score(y_test, test_preds)
    print("GRRR")
    print(curr_r2, flush=True)
    print("\n", flush=True)
    results_grrr[ii] = curr_r2

    ### "Gaussian" RRR
    # model_object = model_object = ReducedRankRidge(
    #     rank=r_true, fit_intercept=True, alpha=0.0
    # )
    # # y_train_stddized = np.log(y_train + 1)
    # y_train_stddized = (y_train - y_train.mean(0)) / y_train.std(0)

    # # y_test_stddized = np.log(y_test + 1)
    # y_test_stddized = (y_test - y_test.mean(0)) / y_test.std(0)
    # model_object.fit(
    #     X=X_train,
    #     y=y_train,
    # )
    # test_preds = model_object.predict(X_test)
    # curr_r2 = centered_r2_score(y_test, test_preds)
    # print(curr_r2, flush=True)
    # print("\n", flush=True)
    # results_gaussrrr[ii] = curr_r2

    model_object = GaussianRRR(latent_dim=r_true)
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
    results_gaussrrr[ii] = curr_r2

    ### Full-rank regression (equivalent to univariate regression for every pair)
    model_object = model_object = ReducedRankRidge(
        rank=min(p, q), fit_intercept=True, alpha=0.0
    )
    # y_train_stddized = (y_train - y_train.mean(0)) / y_train.std(0)

    # y_test_stddized = (y_test - y_test.mean(0)) / y_test.std(0)
    model_object.fit(
        X=X_train,
        y=y_train,
    )
    test_preds = model_object.predict(X_test)
    curr_r2 = centered_r2_score(y_test, test_preds)
    print(curr_r2, flush=True)
    print("\n", flush=True)
    results_fullrank[ii] = curr_r2
    
    # import ipdb; ipdb.set_trace()


results_df = pd.melt(
    pd.DataFrame(
        {"GRRR": results_grrr, "PRRR": results_prrr, "Gaussian\nRRR": results_gaussrrr, "Full-rank": results_fullrank}
        # {"PRRR": results_prrr, "Gaussian\nRRR": results_gaussrrr, "Full-rank": results_fullrank}
    )
)
# import ipdb; ipdb.set_trace()

# results_df = pd.melt(pd.DataFrame(results, columns=latent_dim_list))
results_df.to_csv("./out/prediction_experiment_results_methods_comparison.csv")
plt.figure(figsize=(7, 5))
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel("")
plt.ylabel(r"$R^2$")
plt.tight_layout()
plt.show()
import ipdb

ipdb.set_trace()
