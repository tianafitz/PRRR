import matplotlib.pyplot as plt
import socket
import pandas as pd
import numpy as np
from os.path import join as pjoin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# EXPRESSION_PATH = "../expression_small.gct"
EXPRESSION_PATH = "../GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_gene_expected_count.gct"
SUBJECT_METADATA_PATH = "../GTEx_Analysis_2017-06-05_v8_Annotations_SubjectPhenotypesDS.txt"
SAMPLE_METADATA_PATH = "../GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"

NUM_GENES = 2000
TISSUE = "Thyroid"

def main():

	# ---------------- Load data ----------------

	# Metadata
	subject_metadata = pd.read_table(SUBJECT_METADATA_PATH)
	sample_metadata = pd.read_table(SAMPLE_METADATA_PATH)
	metadata_sample_ids = np.array(["-".join(x.split("-")[:3]) for x in sample_metadata.SAMPID.values])
	metadata_subject_ids = np.array(["-".join(x.split("-")[:2]) for x in sample_metadata.SAMPID.values])
	sample_metadata['SAMPID'] = metadata_sample_ids
	sample_metadata['SUBJID'] = metadata_subject_ids
	assert len(subject_metadata.SUBJID.value_counts().unique()) == 1

	# ------- Load expression ---------
	# Get sample names of expression data
	expression_ids = pd.read_table(
	    EXPRESSION_PATH, skiprows=2, index_col=0, nrows=1
	).columns.values[1:]

	expression_subject_ids = np.array(["-".join(x.split("-")[:2]) for x in expression_ids])
	expression_sample_ids = np.array(["-".join(x.split("-")[:3]) for x in expression_ids])
	# assert np.all(np.unique(expression_sample_ids, return_counts=True)[1] == 1)

	tissue_sample_metadata = sample_metadata[sample_metadata.SMTSD == TISSUE]

	# Drop duplicate samples for the same subject
	tissue_sample_metadata = tissue_sample_metadata.drop_duplicates(subset="SUBJID")
	assert len(tissue_sample_metadata.SUBJID.value_counts().unique()) == 1
	tissue_idx = np.where(np.isin(expression_sample_ids, tissue_sample_metadata.SAMPID.values))[0]
	assert np.all(np.unique(expression_sample_ids[tissue_idx], return_counts=True)[1] == 1)
	

	# Load expression
	tissue_expression = pd.read_table(
	    EXPRESSION_PATH, skiprows=2, index_col=0, usecols=np.insert(tissue_idx, 0, 0)
	)

	tissue_expression = tissue_expression.transpose()

	tissue_sample_ids = np.array(["-".join(x.split("-")[:3]) for x in tissue_expression.index.values])
	assert np.all(np.unique(tissue_sample_ids, return_counts=True)[1] == 1)
	tissue_expression.to_csv("../gtex_expression_{}.csv".format(TISSUE))
	import ipdb; ipdb.set_trace()

	

if __name__ == "__main__":
	main()