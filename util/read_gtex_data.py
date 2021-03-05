import pandas as pd
import numpy as np

EXPRESSION_FILE = "../data/gtex_expression_thyroid.csv"
METADATA_PATH = "../data/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"

# Read in expression file
# This is samples x genes
# The index of this dataframe is the sample ID
# The columns are the Ensembl gene names
expression_data = pd.read_csv(EXPRESSION_FILE, index_col=0)

# Read in metadata
v8_metadata = pd.read_table(METADATA_PATH)

# The SAMPID column contains the sample IDs in the metadata
# Join the expression data's index and the metadata's SAMPID column to get paired samples

import ipdb; ipdb.set_trace()
