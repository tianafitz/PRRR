import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib

font = {"size": 40}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

coeff_mat = pd.read_csv("./out/coeff_matrix_grrr.csv", index_col=0)
A = pd.read_csv("./out/A_grrr.csv", index_col=0).values
B = pd.read_csv("./out/B_grrr.csv", index_col=0).values

rank = A.shape[1]
cell_types = coeff_mat.index.values
n_cell_types = len(cell_types)

A_df = pd.DataFrame(A)
A_df.columns = ["Dim. {}".format(xx) for xx in range(1, rank + 1)]
islets = ["alpha", "beta", "delta", "epsilon", "gamma"]
A_df["Islet"] = [x in islets for x in cell_types]
sns.pairplot(
    data=A_df, hue="Islet", palette=["black", "red"], plot_kws={"s": 200}
)  # , height=1.5)
# plt.tight_layout()
plt.savefig("./out/pairplot_grrr.png")
plt.show()


import ipdb

ipdb.set_trace()
