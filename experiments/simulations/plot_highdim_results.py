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

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


plt.figure(figsize=(10, 5))


results_df = pd.read_csv("./out/highdim_experiment.csv", index_col=0)
sns.lineplot(data=results_df, x="variable", y="value", hue="method")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel(r"$q$ (Number of outcome vars.)")
plt.ylabel(r"$R^2$")
plt.tight_layout()
plt.savefig("./out/highdim_experiment.png")
plt.show()
import ipdb; ipdb.set_trace()

