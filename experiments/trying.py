#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 01:47:27 2021

@author: tianafitz
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append("../models")
from prrr_nb_tfp import fit_rrr
from rrr_tfp_gaussian import fit_rrr as fit_rrr_gaussian
from sklearn.model_selection import train_test_split
from collections import Counter

def clean(df):
  ##generally, remove all data has no variance
  variables = df.columns

  for x in variables:
    unique_values = len(df[x].unique())
    if unique_values ==1: #no variance -- extereme case that captures also strings
        df.drop(x, axis=1, inplace=True)
  len(df.columns)
  return df

EXPRESSION_FILE = "../data/gtex_expression_thyroid.csv"
METADATA_PATH = "../data/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"

# Read in expression file
# This is samples x genes
# The index of this dataframe is the sample ID
# The columns are the Ensembl gene names
expression_data = pd.read_csv(EXPRESSION_FILE, index_col=0)

# Read in metadata
v8_metadata = pd.read_table(METADATA_PATH)

sub = [x[0:10] for x in expression_data.index]
cnt = Counter(sub)
#meta = pd.DataFrame(columns = v8_metadata.columns)

#for x in cnt:
    #df_try  = v8_metadata[v8_metadata['SUBJID'] == x]
    #meta = meta.append([df_try]*cnt[x], ignore_index=True)
#meta  =  pd.get_dummies(meta)

meta = pd.read_csv('../data/meta.csv', index_col=0)
meta = clean(meta)


rrr_results = fit_rrr(Y=expression_data, X=meta, k=5)
A_lognormal_mean = rrr_results['A_mean'].numpy()
A_stddev = rrr_results['A_stddev'].numpy()
B_lognormal_mean = rrr_results['B_mean'].numpy()
B_stddev = rrr_results['B_stddev'].numpy()

A_est = np.exp(A_lognormal_mean + 0.5 * A_stddev**2)
B_est = np.exp(B_lognormal_mean + 0.5 * B_stddev**2)
AB_est = A_est @ B_est

sns.heatmap(AB_est)