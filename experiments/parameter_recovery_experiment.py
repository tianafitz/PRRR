import numpy as np
import matplotlib.pyplot as plt
from rrr_tfp import fit_rrr
from sklearn.model_selection import train_test_split


## Simulate data
n = 300
p = 30
q = 20
k = 5
A_true = np.random.normal(size=(p, k))
B_true = np.random.normal(size=(k, q))
X = np.random.normal(size=(n, p))
Y = np.random.normal(size=(n, q))

# Fit model
rrr_results = fit_rrr(X=X, Y=Y, k=k)

# Compute MSE
mse_A = np.mean((rrr_results['A_mean'].numpy() - A_true)**2)
mse_B = np.mean((rrr_results['B_mean'].numpy() - B_true)**2)
print("A MSE: {}, B MSE: {}".format(round(mse_A, 3), round(mse_B, 3)))

## Get error for varying values of n
n_list = [10, 30, 100, 200]
mse_list_A = []
mse_list_B = []
for n in n_list:
	A_true = np.random.normal(size=(p, k))
	B_true = np.random.normal(size=(k, q))
	X = np.random.normal(size=(n, p))
	Y = np.random.normal(size=(n, q))

	# Fit model
	rrr_results = fit_rrr(X=X, Y=Y, k=k)

	# Compute MSE
	mse_A = np.mean((rrr_results['A_mean'].numpy() - A_true)**2)
	mse_B = np.mean((rrr_results['B_mean'].numpy() - B_true)**2)
	print("n: {}, A MSE: {}, B MSE: {}".format(n, round(mse_A, 3), round(mse_B, 3)))

	mse_list_A.append(mse_A)
	mse_list_B.append(mse_B)

plt.figure(figsize=(14, 6))
plt.subplot(121)
plt.bar(n_list, mse_list_A)
plt.xlabel("n")
plt.ylabel("MSE")
plt.subplot(122)
plt.bar(n_list, mse_list_B)
plt.xlabel("n")
plt.ylabel("MSE")
plt.show()

import ipdb; ipdb.set_trace()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) #, stratify=tissues)
