library(splatter)
library(magrittr)
library(scater)

# group_probs <- c(0.05, 0.1, 0.2, 0.2, 0.45)
n_cell_types <- 10
group_probs <- runif(n=n_cell_types, min=0, max=1)
group_probs <- group_probs / sum(group_probs)
sim.groups <- splatSimulate(group.prob = group_probs, method = "groups",
                            verbose = FALSE, nGenes = 2000, batchCells = c(5000))

data <- sim.groups@assays@data$counts %>% t()
write.csv(file = "~/Documents/beehive/rrr/PRRR/data/simulated/splatter_cell_types_gex.csv", x=data)
write.csv(file = "~/Documents/beehive/rrr/PRRR/data/simulated/splatter_cell_types.csv", x=as.data.frame(sim.groups$Group))


sim.groups <- logNormCounts(sim.groups)
sim.groups <- runPCA(sim.groups)
plotPCA(sim.groups, colour_by = "Group")
