library(magrittr)
library(dplyr)
library(readr)


## ----- Load files for GSEA -------
source("~/Documents/beehive/multimodal_bio/gsea/gsea.R")
source("~/Documents/beehive/multimodal_bio/util/ensembl_to_gene_symbol.R")

geneset_file <- "~/Documents/beehive/gtex_data_sample/gene_set_collections/GO_biological_process.gmt"

gsc_bp <- piano::loadGSC(geneset_file)



## ----- Load PRRR results --------

coeff_mat <- read.csv("~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/coeff_matrix_grrr.csv", row.names = 1)
B_mat <- read.csv("~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/B_grrr.csv", row.names = 1)
colnames(B_mat) <- colnames(coeff_mat)


## ----- Run GSEA -------

rank <- nrow(B_mat)
for (rr in seq(1, rank)) {
  genestats <- B_mat[rr,]
  genestats <- as.double(genestats)
  names(genestats) <- colnames(coeff_mat)
  gsea_out <- run_permutation_gsea(gsc = gsc_bp, gene_stats = genestats)
  
  gsea_out %<>%
    as.data.frame() %>%
    dplyr::select("pathway", "pval", "padj") %>%
    dplyr::arrange(padj) 
  print(gsea_out %>% head(5))
}

# hyper_out %>%
#   write_tsv(file.path("out", "gsea", "results", sprintf("gsea_results_hallmark_var%s.tsv", toString(var_ii))))
# print(hyper_out %>% head())
  

