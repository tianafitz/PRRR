import sys
sys.path.append("../../scGeneFit-python-master/examples")
sys.path.append("../../models")
from scGeneFit.functions import *
from prrr_nb_tfp import fit_rrr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

[data, labels, names] = load_example_data("zeisel")
labels = labels[1].T
labels = pd.get_dummies(labels)

rrr_results = fit_rrr(Y=data, X=labels, k=5)
A_lognormal_mean = rrr_results["A_mean"].numpy()
A_stddev = rrr_results["A_stddev"].numpy()
B_lognormal_mean = rrr_results["B_mean"].numpy()
B_stddev = rrr_results["B_stddev"].numpy()

A_est = np.exp(A_lognormal_mean + 0.5 * A_stddev ** 2)
B_est = np.exp(B_lognormal_mean + 0.5 * B_stddev ** 2)
AB_est = A_est @ B_est

sns.heatmap(AB_est)
plt.show()

# import ipdb; ipdb.set_trace()