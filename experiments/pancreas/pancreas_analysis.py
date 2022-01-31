import scanpy as sc
import pandas as pd
import seaborn as sns
import anndata

DATA_FILE = "../../data/pancreas/GSE84133_RAW/GSM2230757_human1_umifm_counts.csv"

data_df = pd.read_csv(DATA_FILE, index_col=0)
expression_df = data_df.iloc[:, 2:]

## Make AnnData object
adata = anndata.AnnData(expression_df)
adata.obs["barcode"] = data_df.barcode.values
adata.obs["cell_type"] = data_df.assigned_cluster.values

## Filter low-count genes and cells
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

## Remove mitochondrial genes
adata.var["mt"] = adata.var_names.str.startswith("MT-")
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
# sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
# sc.pp.scale(adata, max_value=10)

raw_counts_adata = raw_counts_adata[:, adata.var.highly_variable.index.values]

data_file = "../../data/pancreas/pancreas_counts.h5ad"
raw_counts_adata.write(data_file)
import ipdb

ipdb.set_trace()
