import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from os.path import join as pjoin
import sys

sys.path.append("../../prrr/models/")
from prrr import PRRR


def convert_to_one_hot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b


import socket

if socket.gethostname() == "andyjones":
    # GTEX_EXPRESSION_FILE = "../../data/gtex/gtex_expression_Muscle - Skeletal.csv"
    # GTEX_EXPRESSION_FILE = "../../data/gtex/gtex_expression_Liver.csv"
    GTEX_EXPRESSION_FILE = "../../data/gtex/gtex_expression_Thyroid.csv"
    # GTEX_GENOTYPE_FILE = "../../data/gtex/genotype/Muscle_Skeletal/genotype.csv"
    GTEX_GENOTYPE_FILE = "./data/tissues/Thyroid/genotype.csv"
else:
    # GTEX_EXPRESSION_FILE = "/tigress/aj13/rrr/data_for_prrr/expression/gtex_expression_Muscle - Skeletal.csv"
    # GTEX_GENOTYPE_FILE = "/tigress/aj13/rrr/data_for_prrr/genotype/genotype.csv"
    GTEX_EXPRESSION_FILE = (
        "/tigress/aj13/rrr/data_for_prrr/expression/gtex_expression_Liver.csv"
    )
    GTEX_GENOTYPE_FILE = "/tigress/aj13/rrr/gtex_genotype/tissues/Liver/genotype.csv"


# n_snps = 10_000
n_snps = 2_000
# n_snps = 5
if n_snps is not None:
    n_snps_total = pd.read_csv(GTEX_GENOTYPE_FILE, usecols=[0]).shape[0]
    rows_to_keep = np.random.choice(np.arange(n_snps_total), replace=False, size=n_snps)
    rows_to_keep = np.append(rows_to_keep, 0)
    genotype = pd.read_csv(
        GTEX_GENOTYPE_FILE, index_col=0, skiprows=lambda x: x not in rows_to_keep
    ).transpose()
else:
    genotype = pd.read_csv(GTEX_GENOTYPE_FILE, index_col=0).transpose()

expression = pd.read_csv(GTEX_EXPRESSION_FILE, index_col=0)  # , nrows=10)


expression_subject_ids = expression.index.str.split("-").str[:2].str.join("-")
expression["subject_id"] = expression_subject_ids.values

expression = expression.drop_duplicates(subset=["subject_id"])
expression = expression.set_index("subject_id")

subject_ids_shared = np.intersect1d(genotype.index.values, expression.index.values)
expression = expression.transpose()[subject_ids_shared].transpose()
genotype = genotype.transpose()[subject_ids_shared].transpose()

assert np.array_equal(expression.index.values, genotype.index.values)
X = genotype.values

# import ipdb; ipdb.set_trace()

n_genes = None
# n_genes = 10
if n_genes is not None:
    gene_idx = np.random.choice(
        np.arange(expression.shape[1]), replace=False, size=n_genes
    )
else:
    gene_idx = np.arange(expression.shape[1])

Y = expression.values[:, gene_idx].astype(float)


latent_dim = 10
prrr = PRRR(latent_dim=latent_dim)
prrr.fit(
    X=X, Y=Y, use_vi=False, use_total_counts_as_size_factors=True, n_iters=10_000
)  # 50_000)
# grrr.fit(X=X, Y=Y, use_vi=False, use_total_counts_as_size_factors=True, n_iters=50)

pd.DataFrame(prrr.loss_trace.numpy()).to_csv("./out/loss_trace.csv")

A_est = pd.DataFrame(prrr.param_dict["U"].numpy(), index=genotype.columns.values)
B_est = pd.DataFrame(
    prrr.param_dict["V"].numpy().T, index=expression.columns.values[gene_idx]
)

coeff_mat = pd.DataFrame(
    A_est @ B_est.T,
    index=genotype.columns.values,
    columns=expression.columns.values[gene_idx],
)
coeff_mat.to_csv("./out/eqtl_coeff_matrix_grrr.csv")
pd.DataFrame(A_est).to_csv("./out/eqtl_A_grrr.csv")
pd.DataFrame(B_est).to_csv("./out/eqtl_B_grrr.csv")


import ipdb

ipdb.set_trace()
