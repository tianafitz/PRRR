library(magrittr)
library(ggplot2)
library(dplyr)
source("~/Documents/beehive/rrr/PRRR/prrr/util/ensembl_to_gene_symbol.R")
source("~/Documents/beehive/rrr/PRRR/prrr/util/gsea_shuffle.R")

results_dir <-
  "/Users/andrewjones/Documents/beehive/rrr/PRRR/experiments/gtex/out"
save_dir <- file.path(results_dir, "gsea")
U_path <- file.path(results_dir, "eqtl_A_grrr.csv")
V_path <- file.path(results_dir, "eqtl_B_grrr.csv")

GO_BP_FILE <-
  "~/Documents/beehive/gtex_data_sample/gene_set_collections/GO_biological_process.gmt"
HALLMARK_FILE <-
  "~/Documents/beehive/gtex_data_sample/gene_set_collections/h.all.v7.1.symbols.gmt"

gsc_bp <- piano::loadGSC(GO_BP_FILE)
gsc_hallmark <- piano::loadGSC(HALLMARK_FILE)

U <- read.csv(U_path, row.names = 1)
V <- read.csv(V_path, row.names = 1)

# V = V[,2:ncol(V)]

n_components <- ncol(V)

ensembl_names <-
  lapply(rownames(V), function(x) {
    strsplit(x, ".", fixed = T)[[1]][1]
  }) %>% as.character()
gene_names <- ensembl_to_gene_symbol(ensembl_names)
not_nan_idx <- which(!is.na(gene_names))

V <- V[not_nan_idx, ]
gene_names <- gene_names[not_nan_idx]

for (ii in seq(1, n_components)) {
  # Extract component
  # curr_component <- scale(V[,ii] %>% as.double(), center = T, scale = T)
  curr_component <- V[, ii] %>% as.double()
  names(curr_component) <- gene_names
  
  # Run GSEA on this component
  gsea_results <-
    run_permutation_gsea(
      gsc_file = HALLMARK_FILE,
      gene_stats = curr_component,
      nperm = 100000,
      gsc = gsc_hallmark,
      eps = 0
    )
  
  gsea_results["leadingEdgeList"] <- lapply(X = gsea_results$leadingEdge, FUN = function(x) {paste(x, collapse = ", ")}) %>% as.character()
  
  write.csv(x = gsea_results[, c("pathway", "pval", "padj", "ES", "NES", "leadingEdgeList")], file = file.path(
    save_dir,
    sprintf("gsea_results_component_%s.csv", toString(ii))
  ))
  
  print(gsea_results[,c("pathway", "pval", "padj", "NES")] %>% arrange(padj) %>% head(5))
  
  ## Run Fisher's exact test
  # n_genes <- 30
  # hit_genes <-
  #   names(curr_component)[order(curr_component)][1:n_genes]
  # gsea_out <-
  #   run_fisher_exact_gsea(
  #     gsc_file = geneset_file,
  #     gsc = gsc_hallmark,
  #     hit_genes = hit_genes %>% unique(),
  #     all_genes = names(curr_component) %>% unique()
  #   )
  # gsea_out <- gsea_out[gsea_out$adj_pval <= 0.05, ]
  # if (ncol(gsea_out) > 0) {
  #   print(gsea_out[, c("pathway", "adj_pval")])
  # }
  # 
  # 
  # hit_genes <-
  #   # names(curr_component)[order(-curr_component)][1:n_genes]
  #   names(curr_component)[which(curr_component > 0)] #[1:n_genes]
  # gsea_out <-
  #   run_fisher_exact_gsea(
  #     gsc_file = geneset_file,
  #     gsc = gsc_hallmark,
  #     hit_genes = hit_genes %>% unique(),
  #     all_genes = names(curr_component) %>% unique()
  #   )
  # gsea_out <- gsea_out %>% filter(adj_pval <= 0.05)
  # if (ncol(gsea_out) > 0) {
  #   print(gsea_out[, c("pathway", "adj_pval")])
  # }
}


curr_component <- V[, 9] %>% as.double()
names(curr_component) <- gene_names

# Run GSEA on this component
gsea_results <-
  run_permutation_gsea(
    gsc_file = HALLMARK_FILE,
    gene_stats = curr_component,
    nperm = 100000,
    gsc = gsc_hallmark,
    eps = 0
  )

curr_leading_edge <- gsea_results$leadingEdge[gsea_results$pathway == "HALLMARK_COMPLEMENT"][[1]]
newcol <- rep(FALSE, nrow(V))
newcol[which(gene_names %in% curr_leading_edge)] <- TRUE
V["in_gene_set"] <- newcol
ggplot(data = V) + geom_point(aes(x=X2, y=X8, color=in_gene_set)) + 
  theme_bw()


pheatmap::pheatmap(V[order(-rowSums(abs(V[,1:10])))[1:1000], 1:10], cluster_cols = F, show_rownames = F)
pheatmap::pheatmap(U[order(-rowSums(abs(U)))[1:1000],], cluster_cols = F, show_rownames = F)

