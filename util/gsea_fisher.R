#! /Library/Frameworks/R.framework/Resources/bin/Rscript
library(magrittr)

ensembl_to_gene_symbol <- function(ensembl_ids) {
  library("org.Hs.eg.db")
  mapIds(org.Hs.eg.db,
         keys = ensembl_ids,
         keytype = "ENSEMBL",
         column = "SYMBOL"
  ) %>% as.character()
}

#' Run Fisher's exact test (hypergeometric) for a set of "hit genes"
#'
#' Runs a Fisher's exact test for overrepresentation of hit genes in gene sets.
#'

hit_genes <- read.csv("../util/tmp/tmp_hit_genes.csv", row.names = 1) %>% 
  as.character()

# Uncomment this line if gene names are already gene symbols
# hit_genes <- ensembl_to_gene_symbol(hit_genes)

gsc_file <- "../util/h.all.v7.1.symbols.gmt"
gsc <- piano::loadGSC(gsc_file)
all_genes <- unique(unlist(gsc$gsc) %>% as.character())

# Run hypergeometric test (Fisher's exact)
gsea_hyper_res <- piano::runGSAhyper(
  genes = hit_genes,
  gsc = gsc,
  universe = all_genes
)

# Use different column names that are more friendly for string manipulation and dataframe indexing
result_colnames <- c(
  "pathway",
  "pval",
  "adj_pval",
  "significant_in_gs",
  "nonsignificant_in_gs",
  "significant_notin_gs",
  "nonsignificant_notin_gs"
)

out_file <- "../util/tmp/tmp_output.csv"
out <- gsea_hyper_res$resTab %>%
  as.data.frame() %>%
  tibble::rownames_to_column("pathway") %>%
  set_colnames(result_colnames)

write.csv(x = out, file = out_file)

