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


DATA_DIR = "../../data/simulated"

cell_types_file = pjoin(DATA_DIR, "splatter_cell_types.csv")
data_file = pjoin(DATA_DIR, "splatter_cell_types_gex.csv")
data = pd.read_csv(data_file, index_col=0)
cell_types = pd.read_csv(cell_types_file, index_col=0)


cell_types_ints, cell_types = pd.factorize(cell_types.iloc[:, 0])
X = convert_to_one_hot(cell_types_ints)

n_genes = 100
if n_genes is not None:
    gene_idx = np.random.choice(np.arange(data.shape[1]), replace=False, size=n_genes)
else:
    gene_idx = np.arange(data.shape[1])

Y = data.iloc[:, gene_idx].values


latent_dim = 5
grrr = GRRR(latent_dim=latent_dim)
grrr.fit(X=X, Y=Y, use_vi=False, use_total_counts_as_size_factors=True, n_iters=5000)

A_est = grrr.param_dict["A"].numpy()
B_est = grrr.param_dict["B"].numpy()

coeff_mat = pd.DataFrame(
    A_est @ B_est, index=cell_types, columns=data.columns.values[gene_idx]
)
coeff_mat.to_csv("./out/coeff_matrix_simcelltypes.csv")
# pd.DataFrame(A_est).to_csv("./out/A_prrr.csv")
# pd.DataFrame(B_est).to_csv("./out/B_prrr.csv")

import ipdb

ipdb.set_trace()


plt.scatter(A_est[:, 0], A_est[:, 1])
n_cell_types = len(cell_types)
for ii in range(n_cell_types):
    plt.text(A_est[ii, 0], A_est[ii, 1], s=cell_types[ii])
plt.show()


import ipdb

ipdb.set_trace()
