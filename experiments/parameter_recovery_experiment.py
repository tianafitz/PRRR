import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append("../models")
from prrr_nb_tfp import fit_rrr
from rrr_tfp_gaussian import fit_rrr as fit_rrr_gaussian
from sklearn.model_selection import train_test_split

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

## Simulate data
p = 30
q = 20
k = 5

num_repeats = 5

## Get error for varying values of n
n_list = [50, 100, 200]


def test_rrr_gaussian_varyn():
	mses_AB = np.zeros((num_repeats, len(n_list)))

	for ii in range(num_repeats):
		for jj, n in enumerate(n_list):
			A_true = np.random.normal(size=(p, k))
			B_true = np.random.normal(size=(k, q))
			AB_true = A_true @ B_true
			X = np.random.normal(size=(n, p))
			Y_mean = X @ A_true @ B_true
			Y = Y_mean + np.random.normal(size=(n, q))

			# Fit model
			rrr_results = fit_rrr_gaussian(X=X, Y=Y, k=k)

			# Compute MSE
			A_est = rrr_results['A_mean'].numpy()
			B_est = rrr_results['B_mean'].numpy()
			AB_est = A_est @ B_est
			mse_AB = np.mean((AB_est - AB_true)**2)
			print("n: {}, AB MSE: {}".format(n, round(mse_AB, 3)))

			mses_AB[ii, jj] = mse_AB

	mse_AB_df = pd.melt(pd.DataFrame(mses_AB, columns=n_list))


	plt.figure(figsize=(10, 8))
	sns.boxplot(data=mse_AB_df, x="variable", y="value")
	plt.xlabel(r'$n$')
	plt.ylabel(r'$\|\widehat{A} \widehat{B}-A^\star B^\star\|_2$')
	plt.title("Gaussian RRR")
	plt.tight_layout()
	plt.savefig("../figures/plots/rrr_gaussian_MSE_varyn.png")
	plt.show()

def test_rrr_poisson_varyn():
	mses_AB = np.zeros((num_repeats, len(n_list)))

	for ii in range(num_repeats):
		for jj, n in enumerate(n_list):
			A_true = np.random.gamma(1, 1, size=(p, k))
			B_true = np.random.gamma(1, 1, size=(k, q))
			AB_true = A_true @ B_true
			X = np.random.gamma(1, 1, size=(n, p))
			Y_mean = X @ A_true @ B_true
			Y = np.random.poisson(Y_mean)

			# Fit model
			rrr_results = fit_rrr(X=X, Y=Y, k=k)

			# Compute MSE
			A_est = rrr_results['A_concentration'].numpy() / rrr_results['A_rate'].numpy()
			B_est = rrr_results['B_concentration'].numpy() / rrr_results['B_rate'].numpy()
			AB_est = A_est @ B_est
			mse_AB = np.mean((AB_est - AB_true)**2)
			print("n: {}, AB MSE: {}".format(n, round(mse_AB, 3)))

			mses_AB[ii, jj] = mse_AB


	mse_AB_df = pd.melt(pd.DataFrame(mses_AB, columns=n_list))


	plt.figure(figsize=(10, 8))
	sns.boxplot(data=mse_AB_df, x="variable", y="value")
	plt.xlabel(r'$n$')
	plt.ylabel(r'$\|\widehat{A} \widehat{B}-A^\star B^\star\|_2$')
	plt.title("PRRR")
	plt.tight_layout()
	plt.savefig("../figures/plots/rrr_gaussian_MSE_varyn.png")
	plt.show()


if __name__ == "__main__":

	# test_rrr_gaussian_varyn()
	test_rrr_poisson_varyn()



