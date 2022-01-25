import numpy as np
import pandas as pd
import scanpy as sc
from os.path import join as pjoin

PBMC_DATA_DIR = "../../data/pbmc"

adata = sc.read_10x_mtx(
    pjoin(PBMC_DATA_DIR, "filtered_gene_bc_matrices", "hg19"),
    var_names="gene_symbols",
    cache=True,
)
adata.var_names_make_unique()

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)


adata.var["mt"] = adata.var_names.str.startswith(
    "MT-"
)  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)

adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

raw_counts_adata = adata.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
sc.pp.scale(adata, max_value=10)


sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
# sc.tl.paga(adata)
# sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
# sc.tl.umap(adata, init_pos='paga')

# sc.tl.umap(adata)

sc.tl.leiden(adata)
# sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
# sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
# sc.tl.rank_genes_groups(adata, 'leiden', method='logreg')


marker_genes = [
    "IL7R",
    "CD79A",
    "MS4A1",
    "CD8A",
    "CD8B",
    "LYZ",
    "CD14",
    "LGALS3",
    "S100A8",
    "GNLY",
    "NKG7",
    "KLRB1",
    "FCGR3A",
    "MS4A7",
    "FCER1A",
    "CST3",
    "PPBP",
]

new_cluster_names = [
    "CD4 T",
    "CD14 Monocytes",
    "B",
    "CD8 T",
    "NK",
    "FCGR3A Monocytes",
    "Dendritic",
    "Megakaryocytes",
]
adata.rename_categories("leiden", new_cluster_names)
# sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# sc.pl.umap(
#     adata, color="leiden", legend_loc="on data", title="", frameon=False, save=".pdf"
# )


raw_counts_adata = raw_counts_adata[:, adata.var.highly_variable.index.values]
raw_counts_adata.obs["leiden"] = adata.obs.leiden

data_file = pjoin(PBMC_DATA_DIR, "pbmc3k_counts.h5ad")
raw_counts_adata.write(data_file)
import ipdb

ipdb.set_trace()
