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

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


## Create synthetic data
n = 400
p = 2
q = 2
r_true = 1
USE_VI = False
# latent_dim = 5


X = np.random.choice([0, 1, 2], replace=True, size=(n, p)) # genotype
# U = np.random.normal(loc=0., scale=1., size=(p, r_true))
# V = np.random.normal(loc=0., scale=1., size=(r_true, q))
U = np.array([[1.], [-1]])
V = np.array([[1., -1]])

linear_predictor = X @ U @ V
# size_factors = np.random.uniform(low=1e-1, high=2, size=n).reshape(-1, 1)
size_factors = np.ones((n, 1))
linear_predictor += np.log(size_factors)
Y_mean = np.exp(linear_predictor)

Y = np.random.poisson(Y_mean)

nonzero_idx = np.where(Y.sum(1) > 0)[0]
X = X[nonzero_idx]
Y = Y[nonzero_idx]
size_factors = size_factors[nonzero_idx]
n = nonzero_idx.shape[0]


grrr = GRRR(latent_dim=r_true)
grrr.fit(X=X, Y=Y, use_vi=USE_VI, n_iters=5000, learning_rate=1e-2, size_factors=size_factors)


if USE_VI:
    U_est = grrr.param_dict["A_mean"].numpy()
    V_est = grrr.param_dict["B_mean"].numpy()
else:
    U_est = grrr.param_dict["A"].numpy()
    V_est = grrr.param_dict["B"].numpy()

coeff_mat = U_est @ V_est
top_coeff_idx = np.unravel_index(coeff_mat.argmax(), coeff_mat.shape)

expression_normalized = np.log(Y / Y.sum(1).reshape(-1, 1) + 1)
expression_normalized = (expression_normalized - expression_normalized.mean(0)) / expression_normalized.std(0)

expression_logged_stdized = np.log(Y + 1)
expression_logged_stdized = (expression_logged_stdized - expression_logged_stdized.mean(0)) / expression_logged_stdized.std(0)
expression_logged_stdized_jittered = expression_logged_stdized + np.random.normal(size=(n, q), scale=0.1)


expression_sliced = expression_normalized[:, top_coeff_idx[1]]
Y_jittered = Y + np.random.normal(scale=0.1, size=(n, q))

def abline(slope, intercept, label, color):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', label=label, color=color)

# plt.figure(figsize=(15, 5))
fig = plt.figure(figsize=(15, 5), constrained_layout=True)

eqtl_plot_inner = [
    ["eqtl1"],
    ["eqtl2"],
]
ax_dict = fig.subplot_mosaic(
    [
        ["genotype", "expression", eqtl_plot_inner]
    ],
)

plt.sca(ax_dict["genotype"])
plt.scatter(X[:, 0] + np.random.normal(size=n, scale=0.1), X[:, 1] + np.random.normal(size=n, scale=0.1), c=expression_logged_stdized[:, 0]) #, color="black")
plt.xlabel("SNP 1 genotype (MAF)")
plt.ylabel("SNP 2 genotype (MAF)")


plt.sca(ax_dict["expression"])
plt.scatter(expression_logged_stdized_jittered[:, 0], expression_logged_stdized_jittered[:, 1], color="black")
abline(slope=V_est.squeeze()[1] / V_est.squeeze()[0], intercept=0, label=r"$V$", color="red")
plt.xlabel("Expression, gene 1")
plt.ylabel("Expression, gene 2")
plt.legend()

# plt.subplot(133)
plt.sca(ax_dict["eqtl1"])
eqtl_df = pd.DataFrame({"Genotype": X[:, 0], "Expression": expression_normalized[:, 0]})
sns.boxplot(data=eqtl_df, x="Genotype", y="Expression")
# plt.xlabel("SNP 1 genotype (MAF)")
plt.xlabel("")
plt.ylabel("Normalized exp.,\ngene 1")

plt.sca(ax_dict["eqtl2"])
eqtl_df = pd.DataFrame({"Genotype": X[:, 0], "Expression": expression_normalized[:, 1]})
sns.boxplot(data=eqtl_df, x="Genotype", y="Expression")
plt.xlabel("SNP 1 genotype (MAF)")
plt.ylabel("Normalized exp.,\ngene 2")

plt.tight_layout()
plt.savefig("./out/toy_example_sceqtl.png")
plt.show()

# data_sliced = pd.DataFrame({"Genotype": X[:, top_coeff_idx[0]], "Expression": expression_sliced})
# sns.boxplot(data=data_sliced, x="Genotype", y="Expression")
# plt.show()

# plt.scatter(X[:, 0] + np.random.normal(scale=0.1, size=n), X[:, 1] + np.random.normal(scale=0.1, size=n))
# plt.show()
import ipdb; ipdb.set_trace()



