import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

sys.path.append("../../models")
from prrr_nb_tfp import fit_rrr
from rrr_tfp_gaussian import fit_rrr as fit_rrr_gaussian
from sklearn.model_selection import train_test_split
from collections import Counter

EXPRESSION_FILE = "../../data/gtex_expression_Thyroid_clean.csv"
METADATA_PATH = (
    "../../data/GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
)
NUM_GENES = 5000
TISSUE = "Thyroid"

expression_data = pd.read_csv(EXPRESSION_FILE, index_col=0)
v8_metadata = pd.read_table(METADATA_PATH)
import ipdb; ipdb.set_trace()

