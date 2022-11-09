import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from os.path import join as pjoin
import sys

sys.path.append("../../prrr/models/")
from prrr import PRRR as PRRR


def convert_to_one_hot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b


PBMC_DATA_DIR = "../../data/pancreas"

data_file = pjoin(PBMC_DATA_DIR, "pancreas_counts.h5ad")
adata = sc.read_h5ad(data_file)


cell_types_ints, cell_types = pd.factorize(np.array(adata.obs.cell_type.values))
X = convert_to_one_hot(cell_types_ints)

n_genes = None
if n_genes is not None:
    gene_idx = np.random.choice(np.arange(adata.shape[1]), replace=False, size=n_genes)
else:
    gene_idx = np.arange(adata.shape[1])

Y = adata.X[:, gene_idx]

latent_dim = 5
prrr = PRRR(latent_dim=latent_dim)
prrr.fit(
    X=X, Y=Y, use_vi=False, use_total_counts_as_size_factors=True, n_iters=10_000
)  # 10_000)

A_est = prrr.param_dict["U"].numpy()
B_est = prrr.param_dict["V"].numpy()

coeff_mat = pd.DataFrame(
    A_est @ B_est, index=cell_types, columns=adata.var.index.values[gene_idx]
)
coeff_mat.to_csv("./out/coeff_matrix_prrr.csv")
pd.DataFrame(A_est).to_csv("./out/A_prrr.csv")
pd.DataFrame(B_est).to_csv("./out/B_prrr.csv")


plt.scatter(A_est[:, 0], A_est[:, 1])
n_cell_types = len(cell_types)
for ii in range(n_cell_types):
    plt.text(A_est[ii, 0], A_est[ii, 1], s=cell_types[ii])
plt.show()


import ipdb

ipdb.set_trace()
