library(ggplot2)
library(magrittr)
library(dplyr)
library(ggrepel)



A = read.csv("/Users/andrewjones/Documents/beehive/rrr/PRRR/experiments/gtex/out/eqtl_A_grrr.csv", row.names = 1)
B = read.csv("/Users/andrewjones/Documents/beehive/rrr/PRRR/experiments/gtex/out/eqtl_B_grrr.csv", row.names = 1)

# top_idx = np.argsort(B.var(1))[1:2000]

# coeff_mat_zerod <- coeff_mat
# coeff_mat_zerod[coeff_mat_zerod < 1] <- 0
paletteLength <- 10
myColor <- colorRampPalette(c("white", "red"))(paletteLength)
# length(breaks) == length(paletteLength) + 1
# use floor and ceiling to deal with even/odd length pallettelengths
myBreaks <- c(seq(0, length.out=ceiling(paletteLength/2) + 1), 
              seq(max(A)/paletteLength, max(A), length.out=floor(paletteLength/2)))


pheatmap::pheatmap(A,
                   show_colnames = F, color=myColor, breaks=myBreaks,
                   main = "nn-PRRR coefficients, U",
                   height = 8,
                   width = 6,
                   show_rownames = F,
                   filename = "~/Documents/beehive/rrr/PRRR/experiments/gtex/out/coeff_matrix_sparse_prrr_heatmap_A.png")


pheatmap::pheatmap(B[1:2000,],
                   show_colnames = F, color=myColor, breaks=myBreaks,
                   main = "nn-PRRR coefficients, V",
                   height = 8,
                   width = 6,
                   show_rownames = F,
                   filename = "~/Documents/beehive/rrr/PRRR/experiments/gtex/out/coeff_matrix_sparse_prrr_heatmap_B.png")


