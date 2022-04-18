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


plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.title("MLE")
results_df = pd.read_csv("./out/prediction_experiment_results.csv")
# results_df = results_df[results_df > 0]
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel("Rank")
plt.ylabel(r"$R^2$")
plt.ylim([0, 1])

plt.subplot(122)
plt.title("VI")
results_df = pd.read_csv("./out/prediction_experiment_results_VI.csv")
# results_df = results_df[results_df > 0]
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel("Rank")
plt.ylabel(r"$R^2$")
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("./out/prediction_experiment_results.png")
plt.show()



plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.title("MLE")
results_df = pd.read_csv("./out/prediction_experiment_results_PRRR.csv")
# results_df = results_df[results_df > 0]
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel("Rank")
plt.ylabel(r"$R^2$")
plt.ylim([0, 1])

plt.subplot(122)
plt.title("VI")
results_df = pd.read_csv("./out/prediction_experiment_results_PRRR_VI.csv")
# results_df = results_df[results_df > 0]
sns.boxplot(data=results_df, x="variable", y="value")
plt.xlabel("Rank")
plt.ylabel(r"$R^2$")
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("./out/prediction_experiment_results_PRRR.png")
plt.show()
import ipdb

ipdb.set_trace()
