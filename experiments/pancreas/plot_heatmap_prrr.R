library(ggplot2)
library(magrittr)
library(dplyr)
library(ggrepel)

coeff_mat <- read.csv("~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/coeff_matrix_prrr.csv", row.names = 1)

A_mat <- read.csv("~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/A_prrr.csv", row.names = 1)
rownames(A_mat) <- rownames(coeff_mat)
pheatmap::pheatmap(log(A_mat))

B_mat <- read.csv("~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/B_prrr.csv", row.names = 1)
# rownames(B_mat) <- rownames(coeff_mat)
pheatmap::pheatmap(B_mat, show_colnames = F)



pheatmap::pheatmap(log(coeff_mat), 
                   show_colnames = F)


coeff_mat_zerod <- coeff_mat
# coeff_mat_zerod[coeff_mat_zerod < 1] <- 0
paletteLength <- 10
myColor <- colorRampPalette(c("white", "red"))(paletteLength)
# length(breaks) == length(paletteLength) + 1
# use floor and ceiling to deal with even/odd length pallettelengths
myBreaks <- c(seq(0, length.out=ceiling(paletteLength/2) + 1), 
              seq(max(coeff_mat_zerod)/paletteLength, max(coeff_mat_zerod), length.out=floor(paletteLength/2)))


pheatmap::pheatmap(coeff_mat_zerod,
                   show_colnames = F, color=myColor, breaks=myBreaks,
                   main = "PRRR coefficients",
                   height = 5,
                   width = 10,
                   filename = "~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/coeff_matrix_sparse_prrr_heatmap.png")


pheatmap::pheatmap(log(coeff_mat), 
                   show_colnames = F,
                    main = "PRRR coefficients",
                    height = 5,
                    width = 10,
                    filename = "~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/coeff_matrix_prrr_heatmap.png")


simpleCap <- function(x) {
  s <- strsplit(x, " ")[[1]]
  paste(toupper(substring(s, 1,1)), substring(s, 2),
        sep="", collapse=" ")
}

cell_types <- lapply(X = rownames(coeff_mat), FUN = function(x) {simpleCap(paste(strsplit(x, "_")[[1]], collapse = " "))})
cell_types[which(cell_types == "t Cell")] <- "T Cell"
rownames(A_mat) <- cell_types #rownames(coeff_mat)
rank <- ncol(A_mat)
comp_names <- lapply(X = seq(1, rank), FUN = function(x) {paste("Factor", toString(x), collapse = " ")})
colnames(A_mat) <- comp_names

## Get marker genes for each cell type
n_top_genes <- 20
gene_names <- coeff_mat %>% colnames()
marker_genes_list <- c()
for (cc in seq(length(cell_types))) {
  curr_coeffs <- coeff_mat[cc,]
  sorted_idx <- order(-curr_coeffs)
  curr_top_genes <- gene_names[sorted_idx][1:n_top_genes]
  # new_marker_genes <- paste(curr_top_genes, collapse = ", ")
  new_marker_genes <- paste(paste(paste(paste(curr_top_genes[1:10], collapse = ", "), "\n", collapse = "")), paste(curr_top_genes[11:20], collapse = ", "))
  marker_genes_list <- c(marker_genes_list, new_marker_genes)
}

marker_genes_df <- data.frame("Cell types" = cell_types %>% as.character(), "Marker genes" = marker_genes_list)

library(gridExtra)
tab <- tableGrob(marker_genes_df, cols = c("Cell type", "Marker genes"))
g <- arrangeGrob(tab)
ggsave(file = "~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/pancreas_marker_genes_table.png", plot = g, height = 8, width = 9)


## Plot cell type loadings onto factors
A_mat["cell_type"] <- rownames(A_mat)
A_mat_melted <- reshape2::melt(A_mat)

for (rr in seq(1, rank)) {
  curr_component <- A_mat[,rr]
  top_cell_types <- rownames(A_mat)[order(-curr_component)]
  print(top_cell_types)
  print("")
  
}

ggplot(A_mat_melted, aes(x=cell_type, y=value)) + 
  geom_bar(stat="identity") + 
  facet_wrap(~variable, ncol = 3) +
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
  xlab("") + 
  ylab("")

ggsave("~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/pancreas_factor_loadings_cell_types.png", height = 5, width = 7)


##### GSEA

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


colnames(B_mat) <- colnames(coeff_mat)
rank <- nrow(B_mat)
for (rr in seq(1, rank)) {
  curr_component <- B_mat[rr,] %>% as.double()
  curr_component <- (curr_component - mean(curr_component))
  names(curr_component) <- colnames(B_mat)
  
  # Run GSEA on this component
  gsea_results <-
    run_permutation_gsea(
      gsc_file = HALLMARK_FILE,
      gene_stats = curr_component,
      # nperm = 100000,
      gsc = gsc_hallmark,
      eps = 0
    )
  print(gsea_results %>% arrange(padj) %>% select(pathway, padj, NES) %>% head(5))
  
  ggplot(gsea_results) + geom_point(aes(x=NES, y=-log10(padj))) + 
    theme_bw() + 
    geom_label_repel(data=gsea_results %>% arrange(padj) %>% head(2), aes(x=NES, y=-log10(padj), label = pathway),
                     box.padding   = 0.35, 
                     point.padding = 0.5,
                     segment.color = 'grey50') + 
    ggtitle(paste("Factor", toString(rr), collapse = " "))
  
  ggsave(paste("~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/pancreas_factor_loadings_gsea_comp", toString(rr), ".png", collapse = ""), height = 5, width = 5)
  
  
  # write.csv(x = gsea_results[, c("pathway", "pval", "padj", "ES", "NES")], file = file.path(
  #   save_dir,
  #   sprintf("gsea_results_component_%s.csv", toString(ii))
  # ))
}

