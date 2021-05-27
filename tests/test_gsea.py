import sys
try:
	sys.path.append("../util")
	from gsea import gsea_fisher
except:
	from util.gsea import gsea_fisher

def test_gsea():
	hit_genes = ["BRAF"]
	gsea_out = gsea_fisher(hit_genes)
	
if __name__ == "__main__":
	test_gsea()