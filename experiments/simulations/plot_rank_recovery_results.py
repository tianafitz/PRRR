import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
import sys
from sklearn.metrics import r2_score
from plottify import autosize

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


plt.figure(figsize=(7, 5))

# plt.title("MLE")
results_df = pd.read_csv("./out/rank_recovery_experiment.csv", index_col=0)
sns.lineplot(data=results_df, x="variable", y="value", color="black", err_style="bars")
plt.xlabel("Rank")
plt.ylabel("ELBO")
plt.tight_layout()
plt.xticks(np.sort(results_df.variable.unique().astype(int)))
plt.axvline(3, linestyle="--", color="gray", label="True rank")
plt.legend()
plt.savefig("./out/rank_recovery_experiment.png")
plt.show()
import ipdb

ipdb.set_trace()
