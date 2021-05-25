import pandas as pd
import os
import subprocess
import sys
sys.path.append("../../util")


def gsea_fisher(hit_genes):
	# hit_gene_path = "../data/gtex_expression_Thyroid_clean.csv"
	out_path = "../util/tmp_output.csv"

	# hit_genes_df = pd.DataFrame(hit_genes)
	# hit_genes_df.to_csv(hit_gene_path)
	subprocess.call("../util/gsea_fisher_tfedited.R")
	gsea_out = pd.read_csv(out_path, index_col=0)

	# os.remove(hit_gene_path)
	# os.remove(out_path)

	return gsea_out
