import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

EXPRESSION_FILE = "../gtex_expression_thyroid.csv"
METADATA_PATH = (
    "../GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
)
NUM_GENES = 5000
TISSUE = "Thyroid"

expression_data = pd.read_csv(EXPRESSION_FILE, index_col=0)
gene_variances = expression_data.var(0).sort_values(ascending=False)
top_genes = gene_variances.index.values[:NUM_GENES]
expression_data_clean = expression_data[top_genes]
expression_data_clean.to_csv("../../data/gtex_expression_{}_clean.csv".format(TISSUE))
# import ipdb; ipdb.set_trace()