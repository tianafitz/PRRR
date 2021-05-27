import sys

sys.path.append("../../models")
from prrr_nb_tfp import fit_rrr
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
font = {"size": 30}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True



EXPRESSION_FILE = "../../data/zeisel/zeisel_data_for_prrr.csv"
# METADATA_PATH = "../data/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
CELL_TYPES_PATH = "../../data/zeisel/zeisel_types.csv"
CELL_TYPES_LAYER2_PATH = "../../data/zeisel/zeisel_types_47.csv"

# Read in expression file
expression_data = pd.read_csv(EXPRESSION_FILE, index_col=0)

# Read in cell types
cell_info = pd.read_csv(CELL_TYPES_PATH, index_col=0)

# Second-layer cell tyeps
cell_types_layer2 = pd.read_csv(CELL_TYPES_LAYER2_PATH, header=None)

assert np.all(cell_info['cell.id'].values == expression_data.index.values)

# Cast to integers
expression_data = expression_data.values.astype(int)

# Compute total count per cell
cell_total_counts = expression_data.sum(0)

# Convert cell types to one hot
cell_types_dummies = pd.get_dummies(cell_info['cell.type'])
cell_types_oh = cell_types_dummies.values
cell_types_names = cell_types_dummies.columns.values

cell_types_layer2_dummies = pd.get_dummies(cell_types_layer2.iloc[:, 0])
cell_types_layer2_oh = cell_types_layer2_dummies.values
cell_types_layer2_names = cell_types_layer2_dummies.columns.values

# Combine both levels of the hierarchy
cell_types_oh_concat = np.concatenate([cell_types_layer2_dummies, cell_types_dummies], axis=1)
# import ipdb; ipdb.set_trace()

# Fit model
k = 2
rrr_results = fit_rrr(Y=expression_data, X=cell_types_oh_concat, k=k, size_factors=cell_total_counts)
A_lognormal_mean = rrr_results["A_mean"].numpy()
A_stddev = rrr_results["A_stddev"].numpy()
B_lognormal_mean = rrr_results["B_mean"].numpy()
B_stddev = rrr_results["B_stddev"].numpy()

A_est = np.exp(A_lognormal_mean + 0.5 * A_stddev ** 2)
B_est = np.exp(B_lognormal_mean + 0.5 * B_stddev ** 2)
AB_est = A_est @ B_est

# xticks = ["Latent dim {}".format(x + 1) for x in range(k)]
# plt.figure(figsize=(5, 7))
# sns.heatmap(A_est, xticklabels=xticks, yticklabels=cell_types_names)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(7, 5))
# sns.heatmap(AB_est[:, np.argsort(-AB_est.sum(0))], yticklabels=cell_types_names)
# plt.tight_layout()
# plt.show()

# plt.scatter(A_est[:, 0], A_est[:, 1])
labels = []
for ii in range(cell_types_layer2_oh.shape[1]):
	curr_name = cell_types_layer2_names[ii]
	if "Astro" in curr_name:
		curr_label = "Astrocyte"
	elif "CA1Pyr" in curr_name:
		curr_label = "CA1 Pyramidal"
	elif "Choroid" in curr_name:
		curr_label = "Choroid"
	elif "Epend" in curr_name:
		curr_label = "Epend"
	elif "Int" in curr_name:
		curr_label = "Interneuron"
	elif "Mgl" in curr_name:
		curr_label = "Microglia"
	elif "Oligo" in curr_name:
		curr_label = "Oligodendrocyte"
	elif "Peric" in curr_name:
		curr_label = "Peric"
	elif "Pvm" in curr_name:
		curr_label = "Pvm"
	elif "S1Pyr" in curr_name:
		curr_label = "S1Pyr"
	elif "SubPyr" in curr_name:
		curr_label = "SubPyr"
	elif "Vend" in curr_name:
		curr_label = "Vend"
	elif "Vsmc" in curr_name:
		curr_label = "Vsmc"
	else:
		curr_label = "None"
	# plt.annotate(curr_name, (A_est[ii, 0], A_est[ii, 1]), label=curr_label)
	# plt.scatter(A_est[ii, 0], A_est[ii, 1], label=curr_label)
	labels.append(curr_label)

plt.figure(figsize=(10, 7))
plot_df = pd.DataFrame(A_est[:-cell_types_dummies.shape[1], :], columns=["A1", "A2"])
plot_df['cell_type'] = labels
# plot_df['cell_type'] = cell_types_layer2_names
sns.scatterplot(data=plot_df, x="A1", y="A2", hue="cell_type", s=80)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15)
plt.xlabel("Latent dim 1")
plt.ylabel("Latent dim 2")
plt.tight_layout()
plt.savefig("../../figures/plots/zeisel_lowD_celltypes.pdf", bbox_inches="tight")
plt.show()


import ipdb; ipdb.set_trace()

