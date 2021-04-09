#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 01:47:27 2021

@author: tianafitz
"""
import sys

sys.path.append("../models")
from prrr_nb_tfp import fit_rrr
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def clean(df):
    df[df < 0] = np.nan
    df = df.dropna(axis=1)

    # Drop columns with zero variance
    col_variances = np.var(df.values, 0)
    nonzero_variance_cols = df.columns.values[np.where(col_variances > 0)[0]]
    df = df[nonzero_variance_cols]
    cols = df.columns

    # Remove columns with negative values
    nonnegative_columns = np.where(np.sum(df < 0, 0) == 0)[0]
    df = df.values[:, nonnegative_columns]
    return df, cols


EXPRESSION_FILE = "../data/gtex_expression_Thyroid_clean.csv"
# METADATA_PATH = "../data/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
GENOTYPE_PATH = "../data/thyroid_genotype.csv"

# Read in expression file
# This is samples x genes
# The index of this dataframe is the sample ID
# The columns are the Ensemble gene names
expression_data = pd.read_csv(EXPRESSION_FILE, index_col=0)

# Read in metadata
# v8_metadata = pd.read_table(METADATA_PATH)
v8_metadata = pd.read_csv(GENOTYPE_PATH, index_col=0)

sub = [x[0:10] for x in expression_data.index]
for i in range(len(sub)):
    if sub[i][9] == '-':
        sub[i] = sub[i][:-1]

cnt = Counter(sub)
meta = pd.DataFrame(columns=v8_metadata.columns)

for x in cnt:
    if x not in v8_metadata.index:
        rem = [i for i, s in enumerate(expression_data.index) if s.startswith(x)]
        expression_data = expression_data.drop(expression_data.index[rem])
        continue
    # df_try = v8_metadata[v8_metadata['SUBJID'] == x]
    df_try = v8_metadata[v8_metadata.index == x]
    meta = meta.append([df_try] * cnt[x])
meta = pd.get_dummies(meta)

meta, meta_cols = clean(meta)

# meta = meta[["HGHT", "WGHT", "BMI"]]

# Cast to integers
expression_data = expression_data.values.astype(int)

# Fit model
rrr_results = fit_rrr(Y=expression_data, X=meta, k=5)
A_lognormal_mean = rrr_results["A_mean"].numpy()
A_stddev = rrr_results["A_stddev"].numpy()
B_lognormal_mean = rrr_results["B_mean"].numpy()
B_stddev = rrr_results["B_stddev"].numpy()

A_est = np.exp(A_lognormal_mean + 0.5 * A_stddev ** 2)
B_est = np.exp(B_lognormal_mean + 0.5 * B_stddev ** 2)
AB_est = A_est @ B_est

sns.heatmap(AB_est)
plt.show()

# sex_associated_genes = AB_est[0, :]
# plt.scatter(np.arange(len(sex_associated_genes)), -np.sort(-sex_associated_genes))
# plt.show()

# plt.scatter(np.arange(A_est.shape[0]), -np.sort(-A_est[:, 0]))
# plt.xlabel("Metadata variable index")
# plt.ylabel("Component enrichment")
# plt.show()

# import ipdb
# ipdb.set_trace()
