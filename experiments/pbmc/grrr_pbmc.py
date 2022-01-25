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

PBMC_DATA_DIR = "../../data/pbmc"

data_file = pjoin(PBMC_DATA_DIR, "pbmc3k_counts.h5ad")
adata = sc.read_h5ad(data_file)

Y = adata.X.todense()[:, :200]

cell_types_ints, cell_types = pd.factorize(np.array(adata.obs.leiden.values))
X = convert_to_one_hot(cell_types_ints)

latent_dim = 2
grrr = GRRR(latent_dim=latent_dim)
grrr.fit(X=X, Y=Y, use_vi=False, use_total_counts_as_size_factors=True)

A_est = grrr.param_dict["A"].numpy()
plt.scatter(A_est[:, 0], A_est[:, 1])

n_cell_types = len(cell_types)
for ii in range(n_cell_types):
	plt.text(A_est[ii, 0], A_est[ii, 1], s=cell_types[ii])
plt.show()

# A_est, B_est = grrr.param_dict["A"].numpy(), grrr.param_dict["B"].numpy()
# preds = np.exp(X @ A_est @ B_est + np.log(size_factors))

# plt.scatter(Y_mean[:, 0], preds[:, 0])
# plt.show()


import ipdb; ipdb.set_trace()


