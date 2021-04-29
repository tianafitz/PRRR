import pandas as pd
import os
import subprocess

def gsea_fisher(hit_genes):

	hit_gene_path = "../util/tmp/tmp_hit_genes.csv"
	out_path = "../util/tmp/tmp_output.csv"

	try:
		hit_genes_df = pd.DataFrame(hit_genes)
		hit_genes_df.to_csv(hit_gene_path)
		subprocess.call("../util/gsea_fisher.R")
		gsea_out = pd.read_csv(out_path, index_col=0)

	finally:
		os.remove(hit_gene_path)
		os.remove(out_path)

	return gsea_out