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
grrr = GRRR(latent_dim=latent_dim)
grrr.fit(X=X, Y=Y, use_vi=False, use_total_counts_as_size_factors=True, n_iters=15000)

A_est = grrr.param_dict["A"].numpy()
B_est = grrr.param_dict["B"].numpy()

coeff_mat = pd.DataFrame(
    A_est @ B_est, index=cell_types, columns=adata.var.index.values[gene_idx]
)
coeff_mat.to_csv("./out/coeff_matrix_grrr.csv")
pd.DataFrame(A_est).to_csv("./out/A_grrr.csv")
pd.DataFrame(B_est).to_csv("./out/B_grrr.csv")


plt.scatter(A_est[:, 0], A_est[:, 1])
n_cell_types = len(cell_types)
for ii in range(n_cell_types):
    plt.text(A_est[ii, 0], A_est[ii, 1], s=cell_types[ii])
plt.show()


import ipdb

ipdb.set_trace()
