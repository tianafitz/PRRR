coeff_mat <- read.csv("~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/coeff_matrix_grrr.csv", row.names = 1)
coeff_mat[coeff_mat < -15] = -15

pheatmap::pheatmap(coeff_mat, 
                   show_colnames = F) #, 
                   # main = "GRRR coefficients", 
                   # height = 5,
                   # width = 10,
                   # filename = "~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/coeff_matrix_grrr_heatmap.png")

A_mat <- read.csv("~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/A_grrr.csv", row.names = 1)
rownames(A_mat) <- rownames(coeff_mat)
pheatmap::pheatmap(A_mat)

B_mat <- read.csv("~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/B_grrr.csv", row.names = 1)
# rownames(B_mat) <- rownames(coeff_mat)
pheatmap::pheatmap(B_mat, show_colnames = F)


coeff_mat <- read.csv("~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/coeff_matrix_prrr.csv", row.names = 1)

pheatmap::pheatmap(coeff_mat, 
                   show_colnames = F) #, 
                   # main = "GRRR coefficients", 
                   # height = 5,
                   # width = 10,
                   # filename = "~/Documents/beehive/rrr/PRRR/experiments/pancreas/out/coeff_matrix_grrr_heatmap.png")

