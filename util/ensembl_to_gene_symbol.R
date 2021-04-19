#' Convert list of ensembl IDs to gene symbols
#'
#'	Uses "org.Hs.eg.db" to make the conversion.
#'
ensembl_to_gene_symbol <- function(ensembl_ids) {
  library(magrittr)
  library("org.Hs.eg.db")
  mapIds(org.Hs.eg.db,
    keys = ensembl_ids,
    keytype = "ENSEMBL",
    column = "SYMBOL"
  ) %>% as.character()
}


### EXAMPLE (uncomment to run) ----------------

# # Load gene expression marix
# gene_exp_dir <- "~/Documents/beehive/gtex_data_sample/gene_expression"
# gene_exp_path <- list.files(gene_exp_dir)[1]
# gene_exp <- read.table(file.path(
#   gene_exp_dir,
#   gene_exp_path
# ),
# header = T
# )
# 
# # Gene ensembl IDs
# ensembl_ids_raw <- gene_exp$ID %>% as.character()
# ensembl_ids <- lapply(
#   ensembl_ids_raw,
#   function(x) {
#     strsplit(x, "[.]")[[1]][1]
#   }
# ) %>%
#   as.character()
# 
# gene_symbols <- ensembl_to_gene_symbol(ensembl_ids)
