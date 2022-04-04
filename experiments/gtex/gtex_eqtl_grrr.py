import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from os.path import join as pjoin
import sys

sys.path.append("../../prrr/models/")
from grrr import GRRR


def convert_to_one_hot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b

import socket

if socket.gethostname() == "andyjones":
    GTEX_EXPRESSION_FILE = "../../data/gtex/gtex_expression_Muscle - Skeletal.csv"
    GTEX_GENOTYPE_FILE = "../../data/gtex/genotype/Muscle_Skeletal/genotype.csv"
else:
    GTEX_EXPRESSION_FILE = "/tigress/aj13/rrr/data_for_prrr/expression/gtex_expression_Muscle - Skeletal.csv"
    GTEX_GENOTYPE_FILE = "/tigress/aj13/rrr/data_for_prrr/genotype/genotype.csv"


n_snps = 10000
if n_snps is not None:
    n_snps_total = pd.read_csv(GTEX_GENOTYPE_FILE, usecols=[0]).shape[0]
    rows_to_keep = np.random.choice(np.arange(n_snps_total), replace=False, size=n_snps)
    genotype = pd.read_csv(GTEX_GENOTYPE_FILE, index_col=0, skiprows=lambda x: x not in rows_to_keep).transpose()
else:
    genotype = pd.read_csv(GTEX_GENOTYPE_FILE, index_col=0).transpose()

expression = pd.read_csv(GTEX_EXPRESSION_FILE, index_col=0)


expression_subject_ids = expression.index.str.split("-").str[:2].str.join("-")
expression["subject_id"] = expression_subject_ids.values

expression = expression.drop_duplicates(subset=["subject_id"])
expression = expression.set_index("subject_id")

subject_ids_shared = np.intersect1d(genotype.index.values, expression.index.values)
expression = expression.transpose()[subject_ids_shared].transpose()
genotype = genotype.transpose()[subject_ids_shared].transpose()

assert np.array_equal(expression.index.values, genotype.index.values)

X = genotype.values

n_genes = None
if n_genes is not None:
    gene_idx = np.random.choice(np.arange(expression.shape[1]), replace=False, size=n_genes)
else:
    gene_idx = np.arange(expression.shape[1])

Y = expression.values[:, gene_idx].astype(float)


latent_dim = 10
grrr = GRRR(latent_dim=latent_dim)
grrr.fit(X=X, Y=Y, use_vi=False, use_total_counts_as_size_factors=True, n_iters=50_000)

pd.DataFrame(grrr.loss_trace.numpy()).to_csv("./out/loss_trace.csv")

A_est = pd.DataFrame(grrr.param_dict["A"].numpy(), index=genotype.columns.values)
B_est = pd.DataFrame(grrr.param_dict["B"].numpy(), columns=expression.columns.values[gene_idx])

coeff_mat = pd.DataFrame(
    A_est @ B_est, index=genotype.columns.values, columns=expression.columns.values[gene_idx]
)
coeff_mat.to_csv("./out/eqtl_coeff_matrix_grrr.csv")
pd.DataFrame(A_est).to_csv("./out/eqtl_A_grrr.csv")
pd.DataFrame(B_est).to_csv("./out/eqtl_B_grrr.csv")


import ipdb; ipdb.set_trace()
