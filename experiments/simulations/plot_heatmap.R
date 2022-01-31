coeff_mat <- read.csv("~/Documents/beehive/rrr/PRRR/experiments/simulations/out/coeff_matrix_simcelltypes.csv", row.names = 1)

pheatmap::pheatmap(coeff_mat, 
                   show_colnames = F)
