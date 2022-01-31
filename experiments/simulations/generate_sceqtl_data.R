# Load library and the test data for SCeQTL
library(SCeQTL)
data(test)

write.csv(x = gene, file = "~/Documents/beehive/rrr/PRRR/data/simulated/sceqtl_gene_exp.csv")
write.csv(x = snp, file = "~/Documents/beehive/rrr/PRRR/data/simulated/sceqtl_genotype.csv")

# check whether the non-zero part could fit negative binomial well
# This function may fail since It's possible that the random picked gene can't be fit to negative bionomial distribution, all zero value for example
# checkdist(gene)
# 
# # normalize gene matrix using DEseq method
# normalize(gene)
# 
# # Detecting the SCeQTL
# result <- cal.pvalue(gene, snp)
# 
# # Picking one sample to visualize
# check.sample(gene[1,], snp[1,])

