import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os.path import join as pjoin


import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

gsea_dir = "./out/gsea"
gsea_filenames = os.listdir(gsea_dir)
table_dfs = []
for ii, ff in enumerate(gsea_filenames):
	print(ff)
	res = pd.read_csv(pjoin(gsea_dir, ff), index_col=0)
	curr_table_df = res[res.padj < 0.01]
	curr_table_df["factor_num"] = ii + 1
	curr_table_df.pathway = curr_table_df.pathway.str.split("_").str[1:].str.join(" ")
	for jj in range(len(curr_table_df)):
		curr_table_df.leadingEdgeList.values[jj] = ",".join(["emph{" + x + "}" for x in curr_table_df.leadingEdgeList.values[jj].split(", ")[:3]])
	# import ipdb; ipdb.set_trace()
	curr_table_df = curr_table_df[["factor_num", "pathway", "padj", "NES", "leadingEdgeList"]]
	curr_table_df.columns = ["Factor", "Pathway", "Adjusted p-val", "NES", "Leading edge"]
	table_dfs.append(curr_table_df)
	
	res["neglog10pval"] = -np.log10(res.padj.values)
	# print(res.sort_values("padj").head(3), "\n")
	# print(res.sort_values("NES", ascending=False).head(3), "\n")

	# plt.figure(figsize=(7, 7))
	plt.subplot(2, 5, ii + 1)
	sns.scatterplot(data=res, x="NES", y="neglog10pval")
	plt.title("Component {}".format(ff.split(".")[0].split("_")[-1]))

	plt.tight_layout()
# plt.show()
plt.close()


table_df = pd.concat(table_dfs)
table_df.to_latex(index=False, buf="./table.txt", float_format="%.2e")
import ipdb; ipdb.set_trace()
component_num = 9
n_gene_sets = 2
res = pd.read_csv(
	pjoin(gsea_dir, "gsea_results_component_{}.csv".format(component_num)), index_col=0
)
res["neglog10pval"] = -np.log10(res.padj.values)

# plt.figure(figsize=(7, 7))
# plt.subplot(2, 5, ii + 1)

plt.figure(figsize=(7, 7))
sns.scatterplot(data=res, x="NES", y="neglog10pval", edgecolor=None, color="black")

plot_df = res.sort_values("NES", ascending=False).head(n_gene_sets)
for ii in range(n_gene_sets):
	curr_gs_name = " ".join(plot_df.pathway.values[ii].split("_")[1:])
	plt.text(
		x=plot_df.NES.values[ii] - 0.15,
		y=plot_df.neglog10pval.values[ii],
		s=curr_gs_name,
		ha="right",
		fontsize=20,
	)
	plt.scatter(
		plot_df.NES.values[ii],
		plot_df.neglog10pval.values[ii],
		color="red",
		s=100,
	)

plt.title("Component {}".format(component_num))
plt.xlabel("Enrichment score")
plt.ylabel(r"$-\log_{10}$(p-val)")

plt.tight_layout()
plt.savefig("./out/gsea_plot_enrichment.png	")
plt.show()
plt.close()
import ipdb

ipdb.set_trace()
