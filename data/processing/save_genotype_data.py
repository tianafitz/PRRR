import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

genotype_file = "../gtex_thyroid_genotype_medium.tsv"
eqtl_file = "../Thyroid.v8.egenes.txt"

QVAL_THRESHOLD = 0.05

genotype_data = pd.read_table(genotype_file, index_col=0)
eqtl_data = pd.read_table(eqtl_file, index_col=0)

# Only take significant pairs
eqtl_data = eqtl_data[eqtl_data.qval <= QVAL_THRESHOLD]

# Only take genotypes of variants that are significant eQTLs
eqtl_variant_ids = eqtl_data.variant_id.values
shared_variants = np.intersect1d(eqtl_data.variant_id.values, genotype_data.index.values)

# Subset genotype data
genotype_data = genotype_data.loc[shared_variants]

# Put subjects on rows
genotype_data = genotype_data.transpose()

# Save
genotype_data.to_csv("../thyroid_genotype.csv")

import ipdb; ipdb.set_trace()

