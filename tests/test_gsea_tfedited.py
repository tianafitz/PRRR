import sys
sys.path.append("../util")
from gsea_tfedited import gsea_fisher
import pandas as pd

def test_gsea():
	# hit_genes = ["BRAF"]
	tst = pd.read_csv("../data/gtex_expression_Thyroid_clean.csv")
	tst = list(tst.columns)
	hit_genes = tst
	gsea_out = gsea_fisher(hit_genes)
	
if __name__ == "__main__":
	test_gsea()
