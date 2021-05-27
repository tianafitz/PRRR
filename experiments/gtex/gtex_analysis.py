import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

sys.path.append("../../models")
from prrr_nb_tfp import fit_rrr
from rrr_tfp_gaussian import fit_rrr as fit_rrr_gaussian
from sklearn.model_selection import train_test_split
from collections import Counter

import matplotlib
font = {"size": 15}
matplotlib.rc("font", **font)

EXPRESSION_FILE = "../../data/gtex_expression_Thyroid_clean.csv"
METADATA_PATH = (
    "../../data/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
)
NUM_GENES = 5000
TISSUE = "Thyroid"

expression_data = pd.read_csv(EXPRESSION_FILE, index_col=0)
v8_metadata = pd.read_table(METADATA_PATH)

## Align subjects
subj_ids_expression = expression_data.index.str.split("-").str[:2].str.join("-")
expression_data["SUBJID"] = subj_ids_expression
shared_subj_ids = np.intersect1d(subj_ids_expression, v8_metadata.SUBJID.values)
v8_metadata.index = v8_metadata.SUBJID.values
v8_metadata = v8_metadata.transpose()[shared_subj_ids].transpose()

expression_data = expression_data.drop_duplicates("SUBJID")
expression_data.index = expression_data["SUBJID"]
expression_data = expression_data.transpose()[shared_subj_ids].transpose()

expression_data = expression_data.drop("SUBJID", axis=1)

assert np.array_equal(v8_metadata.index.values, expression_data.index.values)

# Get most variable genes
n_genes = 200
Y = expression_data.values.astype(int)
Ylog = np.log(Y + 1)
gene_vars = Ylog.std(0)
sorted_idx = np.argsort(-gene_vars)

Y = Y[:, sorted_idx[:n_genes]]

# Pick a few metadata columns
metadata_cols = ["SEX", "RACE", "ETHNCTY", "HGHT", "WGHT"]
dummy_vars = ["SEX", "RACE", "ETHNCTY"]
nondummy_vars = np.setdiff1d(metadata_cols, dummy_vars)
dummy_list = []
for dummy_var in dummy_vars:
    curr_dummies = pd.get_dummies(v8_metadata[dummy_var])
    dummy_list.append(curr_dummies)

X_df = pd.concat(dummy_list, axis=1)

for nondummy_var in nondummy_vars:
    X_df = pd.concat([X_df, v8_metadata[nondummy_var]], axis=1)


# Fit model
k = 2
cell_total_counts = np.expand_dims(Y.sum(1).astype("float32"), 1)
# import ipdb; ipdb.set_trace()
rrr_results = fit_rrr(X=X_df.values, Y=Y, k=k, size_factors=cell_total_counts)
A_lognormal_mean = rrr_results["A_mean"].numpy()
A_stddev = rrr_results["A_stddev"].numpy()
B_lognormal_mean = rrr_results["B_mean"].numpy()
B_stddev = rrr_results["B_stddev"].numpy()

A_est = np.exp(A_lognormal_mean + 0.5 * A_stddev ** 2)
B_est = np.exp(B_lognormal_mean + 0.5 * B_stddev ** 2)
AB_est = A_est @ B_est

plt.figure(figsize=(7, 7))
plt.scatter(AB_est[0, :], AB_est[1, :])
plt.xlabel("Gene coefficients (male)", fontsize=25)
plt.ylabel("Gene coefficients (female)", fontsize=25)
plt.tight_layout()
plt.savefig("../../figures/plots/gtex_male_vs_female.png")
plt.show()
import ipdb; ipdb.set_trace()

