import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

A = pd.read_csv("./out/eqtl_A_grrr.csv", index_col=0)
B = pd.read_csv("./out/eqtl_B_grrr.csv", index_col=0)
# import ipdb; ipdb.set_trace()
AB = A.values @ B.values.T
sorted_idx = np.dstack(np.unravel_index(np.argsort(AB.ravel()), AB.shape)).squeeze()
# for ii in range(sorted_idx.shape[0]):
# 	print(A.index[sorted_idx[ii][0]], B.columns[sorted_idx[ii][1]])
# 	import ipdb; ipdb.set_trace()

top_idx = np.argsort(B.var(1))[:2000]

sns.clustermap(A)
plt.savefig("./out/A_heatmap_prrr.png")
plt.show()

plt.close()


sns.clustermap(B[:500])
plt.savefig("./out/B_heatmap_prrr.png")
plt.show()
plt.close()
import ipdb; ipdb.set_trace()

import socket

if socket.gethostname() == "andyjones":
    GTEX_EXPRESSION_FILE = "../../data/gtex/gtex_expression_Muscle - Skeletal.csv"
    GTEX_GENOTYPE_FILE = "../../data/gtex/genotype/Muscle_Skeletal/genotype.csv"
else:
    GTEX_EXPRESSION_FILE = "/tigress/aj13/rrr/data_for_prrr/expression/gtex_expression_Muscle - Skeletal.csv"
    GTEX_GENOTYPE_FILE = "/tigress/aj13/rrr/data_for_prrr/genotype/genotype.csv"


n_snps = None
if n_snps is not None:
    n_snps_total = pd.read_csv(GTEX_GENOTYPE_FILE, usecols=[0]).shape[0]
    rows_to_keep = np.random.choice(np.arange(n_snps_total), replace=False, size=n_snps)
    genotype = pd.read_csv(
        GTEX_GENOTYPE_FILE, index_col=0, skiprows=lambda x: x not in rows_to_keep
    ).transpose()
else:
    genotype = pd.read_csv(GTEX_GENOTYPE_FILE, index_col=0).transpose()

expression = pd.read_csv(GTEX_EXPRESSION_FILE, index_col=0)


expression_subject_ids = expression.index.str.split("-").str[:2].str.join("-")
expression["subject_id"] = expression_subject_ids.values

expression = expression.drop_duplicates(subset=["subject_id"])
expression = expression.set_index("subject_id")

subject_ids_shared = np.intersect1d(genotype.index.values, expression.index.values)
expression = expression.transpose()[subject_ids_shared].transpose()
genotype = genotype.transpose()[subject_ids_shared].transpose()

# expression_normalized = expression.astype(float).div(expression.astype(float).sum(axis=1), axis=0)
expression_normalized = np.log(expression.astype(float) + 1)
expression_normalized = (
    expression_normalized - expression_normalized.mean(0)
) / expression_normalized.std(0)


n_associations_to_plot = 3
plt.figure(figsize=(5 * n_associations_to_plot, 5))
for ii in range(n_associations_to_plot):
    plt.subplot(1, n_associations_to_plot, ii + 1)
    curr_df = pd.DataFrame(
        {
            "Genotype": genotype[A.index[sorted_idx[ii][0]]].astype(float).values,
            "Expression": expression_normalized[B.columns[sorted_idx[ii][1]]]
            .astype(float)
            .values,
        }
    )
    sns.boxplot(data=curr_df, x="Genotype", y="Expression")
    plt.xlabel(A.index.values[sorted_idx[ii][0]])
    plt.ylabel(B.columns.values[sorted_idx[ii][1]])
    # plt.scatter(genotype[A.index[sorted_idx[ii][0]]].astype(float), expression[B.columns[sorted_idx[ii][1]]].astype(float))
plt.savefig("./out/gtex_eqtl_boxplots.png")
plt.show()


import ipdb

ipdb.set_trace()
