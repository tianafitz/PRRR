import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from os.path import join as pjoin
import sys
from sklearn.decomposition import PCA
import seaborn as sns

sys.path.append("../../prrr/models/")
from grrr import GRRR

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams["xtick.labelsize"] = 10
# matplotlib.rcParams["ytick.labelsize"] = 10


def convert_to_one_hot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b


PBMC_DATA_DIR = "../../data/pancreas"

data_file = pjoin(PBMC_DATA_DIR, "pancreas_counts.h5ad")
adata = sc.read_h5ad(data_file)


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    # adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # sc.pp.filter_cells(adata, min_counts=5000)
    # sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    # sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata


cell_types_ints, cell_types = pd.factorize(np.array(adata.obs.cell_type.values))
X = convert_to_one_hot(cell_types_ints)

cell_types = np.array([" ".join(x.split("_")).capitalize() for x in cell_types])

adata = process_data(adata)
Y = adata.X
Y = Y - Y.mean(0)


pca = PCA(n_components=2)
Y_reduced = pca.fit_transform(Y)
Y_reduced_df = pd.DataFrame(Y_reduced, columns=["PC1", "PC2"])
Y_reduced_df["Cell type"] = cell_types[cell_types_ints]
plt.figure(figsize=(10, 7))
sns.scatterplot(data=Y_reduced_df, x="PC1", y="PC2", hue="Cell type")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=20)
plt.tight_layout()
plt.savefig("./out/pancreas_pca_plot.png")
plt.show()

import ipdb

ipdb.set_trace()

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
