library(magrittr)

#' Run Fisher's exact test (hypergeometric) for a set of "hit genes"
#'
#' Runs a Fisher's exact test for overrepresentation of hit genes in gene sets.
#'
run_fisher_exact_gsea <- function(gsc_file, hit_genes, all_genes, gsc = NULL) {

  
  
  if (!is.null(gsc)) {
    # Run hypergeometric test (Fisher's exact)
    gsea_hyper_res <- piano::runGSAhyper(
      genes = hit_genes,
      gsc = gsc,
      universe = all_genes
    )
  } else {
    # Load gene set collection file
    gsc <- piano::loadGSC(gsc_file)
    
    # Run hypergeometric test (Fisher's exact)
    gsea_hyper_res <- piano::runGSAhyper(
      genes = hit_genes,
      gsc = gsc,
      universe = all_genes
    )
  }

  

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

  return(gsea_hyper_res$resTab %>%
    as.data.frame() %>%
    tibble::rownames_to_column("pathway") %>%
    set_colnames(result_colnames))
}

#' Runs a permutation test using gene-wise statistics
#'
#' Runs a permutation test to see whether certain gene sets have high or lower stat values (e.g., differential expression) than expected.
#'
run_permutation_gsea <- function(gsc_file, gene_stats, nperm = 1000, gsc = NULL) {
  library(magrittr)
  
  if (!is.null(gsc)) {
    # Run hypergeometric test (Fisher's exact)
    gsea_out <- fgsea::fgseaMultilevel(
      pathways = gsc$gsc,
      stats = gene_stats,
      # nperm = nperm
    )
  } else {
    # Load gene set collection file
    gsc <- piano::loadGSC(gsc_file)
    
    # Run permutation test with fgsea
    gsea_out <- fgsea::fgseaMultilevel(
      pathways = gsc$gsc,
      stats = gene_stats,
      # nperm = nperm
    )
  }

  

  # Return dataframe of results
  return(gsea_out %>% as.data.frame())
}
