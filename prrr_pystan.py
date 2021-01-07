#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:59:07 2020

@author: tianafitz
"""

import pystan
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from hashlib import md5
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from scipy.special import logsumexp

plt.style.use("ggplot")
model_code = """
data {
    int<lower=0> N; // number of data points
    int<lower=0> N_test; // number of data points
    int<lower=0> P; // number of features in X
    int<lower=0> Q; // number of features in Y
    int<lower=1> R; // number of dimensions
    matrix<lower=0>[N, P] X; // covariate data
    matrix<lower=0>[N_test, P] X_test; // test covariate data
    int<lower=0> Y[N, Q]; // response data
    int<lower=0> Y_test[N_test, Q]; // test response data
}
parameters {
    matrix<lower=0>[P, R] A; // tall and skinny
    matrix<lower=0>[R, Q] B; // long
}
model {
    for (p in 1:P) 
        A[p,] ~ gamma(2, 1); //
    for (q in 1:Q) 
        B[,q] ~ gamma(2, 1); // 
    for (x in 1:Q) 
        Y[,x] ~ poisson(X * (A * B)[,x]); //
}
generated quantities {
	matrix[N_test, Q] ll; // log-likelihood of test data
	for (n in 1:N_test) {
		for (q in 1:Q) {
			ll[n,q] = Y_test[n, q] .* log((X_test * (A * B)[,q])[n]) -  
            (X_test * (A * B)[,q])[n] - lgamma(Y_test[n, q] + 1); 
            // this is the log of the poisson probability mass function (PMF)
		}
	}
}
"""

# Simulated Data
n = 500 # number of samples
p = 8 # number of cell types
q = 20 # number of genes
real_r = 5
n_iter = 100

A_sim = np.random.gamma(2, 1, size=(p, real_r))
B_sim = np.random.gamma(2, 1, size=(real_r, q))
X_sim = np.zeros((n, p))
for ii in range(n):
    cell_type_idx = np.random.choice(np.arange(p))
    X_sim[ii, cell_type_idx] = 1
Y_sim = np.random.poisson(X_sim @ (A_sim @ B_sim))

# import ipdb; ipdb.set_trace()
sns.heatmap(A_sim @ B_sim)
plt.title("Randomized Parameters")
plt.figure()

test_ranks = [1, 2, 3, 4, 5, 6, 10, 20]

all_ll = []

# PBMC DATA
'''X_sim = pd.read_csv('pbmcX.csv').to_numpy()
Y_sim = pd.read_csv('pbmcY.csv').to_numpy()
n = 2700
p = 9
q = 200
n_iter = 100'''

for r in test_ranks:
    cur_ll = []
    
    for i in range(5):
        
        train_idx, test_idx = train_test_split(np.arange(n), test_size=0.25, random_state=42)
        Y_train, Y_test = Y_sim[train_idx, :], Y_sim[test_idx, :]
        X_train, X_test = X_sim[train_idx, :], X_sim[test_idx, :]
        n_train, n_test = train_idx.shape[0], test_idx.shape[0]
        
        # Format data for Stan model
        dat = {'N': n_train, 'N_test': n_test, 'P': p, 'Q': q, "R": r, 
               'X': X_train, 'X_test': X_test, 'Y': Y_train, 'Y_test': Y_test}
        sm = pystan.StanModel(model_code=model_code)
        # Fit Stan model
        fit = sm.sampling(data=dat, iter=n_iter)
        # Look at model coefficients
        A_est = np.mean(fit.extract()['A'], axis=0)
        B_est = np.mean(fit.extract()['B'], axis=0)
        
        sns.heatmap(A_est @ B_est)
        plt.title("Sampled Parameters")
        plt.xlabel('gene')
        plt.ylabel('cluster')
        plt.figure()
            
        # Compute log-likelihood of test data
        def test_ll(sample_mat):
        	# Sample mat is (# samples x # datapoints x # features)
        	S = sample_mat.shape[0] # Number of samples
        	LL_mat = np.sum(sample_mat, axis=2) # sum over genes (independent)
        	LL_mat_per_sample = logsumexp(LL_mat, axis=0) - np.log(S)
        	return LL_mat_per_sample
        
        log_likelihood_per_cell = test_ll(fit.extract()['ll'])
        LL = np.mean(log_likelihood_per_cell)  
        cur_ll.append(LL)
    all_ll.append(cur_ll)

plot_ll = np.mean(all_ll, axis=1)
plt.xlabel('Rank for RRR')
plt.ylabel('Average Log Likelihood')
plt.title('RRR on PMBC Data')
plt.plot(test_ranks, plot_ll)
plt.title('Sampled Parameters')
sns.heatmap(A_est @ B_est)
    
    