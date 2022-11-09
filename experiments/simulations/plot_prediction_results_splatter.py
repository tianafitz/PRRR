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

results_df = pd.read_csv(
    "./out/prediction_experiment_results_methods_comparison_splatter.csv"
)
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel("")
plt.ylabel(r"$R^2$")
plt.ylim([0, np.max(results_df.value.values) + 0.1])
plt.tight_layout()
plt.savefig("./out/prediction_experiment_results_splatter.png")
plt.show()
import ipdb

ipdb.set_trace()
