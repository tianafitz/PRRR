import numpy as np
import pandas as pd
from os.path import join as pjoin
import socket
import os
from sklearn.decomposition import PCA


if socket.gethostname() == "andyjones":
    METADATA_PATH = "/Users/andrewjones/Documents/beehive/gtex/v8_metadata/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    GENOTYPE_PATH = "/Users/andrewjones/Documents/beehive/gtex_data_sample/genotypes/thyroid_genotype_small.tsv"
    GENOTYPE_DIR = "/Users/andrewjones/Documents/beehive/gtex_data_sample/genotypes"
    GENE_NAME_DIR = (
        "/Users/andrewjones/Documents/beehive/multimodal_bio/qtl/data/gene_snp_names"
    )
    EQTL_DIR = "/Users/andrewjones/Documents/beehive/gtex_image_analysis/data_processing/qtl/data"
    SAVE_DIR = "./data"
else:
    METADATA_PATH = "/tigress/BEE/gtex/dbGaP_index/v8_data_sample_annotations/GTEx_Analysis_2017-06-05_v8_Annotations_SampleAttributesDS.txt"
    GENOTYPE_DIR = "/tigress/BEE/RNAseq/RNAseq_dev/Data/Genotype/gtex_v8/dosage"
    GENE_NAME_DIR = "/scratch/gpfs/aj13/qtl/data/gene_snp_names"
    EQTL_DIR = "/tigress/aj13/gtexv8/eqtls/GTEx_Analysis_v8_eQTL"
    SAVE_DIR = "/tigress/aj13/rrr/gtex_genotype"

# Metadata
v8_metadata = pd.read_table(METADATA_PATH)
v8_metadata["sample_id"] = [
    "-".join(x.split("-")[:3]) for x in v8_metadata.SAMPID.values
]

# Get all unique tissues
if socket.gethostname() == "andyjones":
    metadata_tissues = ["Thyroid"]
else:
    metadata_tissues = v8_metadata.SMTSD.unique()
    metadata_tissues = metadata_tissues[~pd.isna(metadata_tissues)]

# Get list of genotype files available
genotype_files = os.listdir(GENOTYPE_DIR)
genotype_tissues = np.array(
    [x.split(".")[1] for x in genotype_files if x.endswith(".tsv")]
)

# Get list of files containing QTL hits
eqtl_files = os.listdir(EQTL_DIR)

# For each tissue, put together data
for curr_tissue in ["Thyroid"]:  # metadata_tissues:
    # for curr_tissue in ['Thyroid']:

    print("Loading {}...".format(curr_tissue))

    # Make tissue name match the QTL filenames
    curr_genotype_tissue = "_".join(
        curr_tissue.replace(" - ", "_").replace("(", "").replace(")", "").split(" ")
    )

    save_dir_tissue = pjoin(SAVE_DIR, "tissues", curr_genotype_tissue.strip())

    # Skip if we've already processed this tissue
    # if os.path.isdir(save_dir_tissue) and socket.gethostname() != "andyjones":
    #     continue
    # else:
    #     os.makedirs(save_dir_tissue)

    # Get list
    continue_flag = False
    if socket.gethostname() == "andyjones":
        genotype_file_path = "/Users/andrewjones/Documents/beehive/gtex_data_sample/genotypes/thyroid_genotype_small.tsv"
        eqtl_file_path = "Thyroid.v8.signif_variant_gene_pairs.txt"
    else:
        genotype_file_path = [
            x for x in genotype_files if (curr_genotype_tissue in x) & ("final" in x)
        ]
        genotype_file_path = pjoin(GENOTYPE_DIR, genotype_file_path[0])

        eqtl_file_path = [
            x
            for x in eqtl_files
            if (curr_genotype_tissue in x) & ("signif_variant_gene_pairs" in x)
        ][0]

    if len(genotype_file_path) == 0:
        continue

    print(genotype_file_path)

    eqtl_file_path = pjoin(EQTL_DIR, eqtl_file_path)

    # Load significant QTLs for this tissue
    curr_tissue_eqtls = pd.read_table(eqtl_file_path)

    # Load SNPs with genotype data for this tissue
    genotype_snp_df = pd.read_table(genotype_file_path, usecols=["constVarID"])

    # Get names of SNPs that are known eQTLs in this tissue
    # curr_snp_ids = np.intersect1d(
    #     genotype_snp_df["constVarID"].values, curr_tissue_eqtls["variant_id"].values
    # )
    curr_snp_ids = curr_tissue_eqtls["variant_id"].unique()

    # curr_snp_ids = np.unique(curr_tissue_eqtls.variant_id.values[inset_idx])
    qtl_snp_df = pd.DataFrame({"qtl_snps": curr_snp_ids})

    # Join QTL hit data and subject-specific genotype data
    shared_snps_df = genotype_snp_df.merge(
        qtl_snp_df, right_on="qtl_snps", left_on="constVarID", how="left"
    )

    assert shared_snps_df.shape[0] == genotype_snp_df.shape[0]
    assert np.array_equal(
        genotype_snp_df.constVarID.values, shared_snps_df.constVarID.values
    )

    # Get indices where QTL snps had a match
    idx_to_keep = np.where(~shared_snps_df.qtl_snps.isna())[0] + 1

    # If no QTLs, skip this component
    if idx_to_keep.shape[0] == 0:
        continue

    # Indices of SNPs to skip when reading the genotype data
    idx_to_skip = np.setdiff1d(np.arange(genotype_snp_df.shape[0]), idx_to_keep)

    # Which subjects to get genotype data for
    # cols_to_read = np.append(shared_subject_ids, "constVarID")

    # Load the genotype data
    rows_to_skip = np.delete(idx_to_skip, np.argwhere(idx_to_skip == 0))
    curr_genotypes = pd.read_table(
        genotype_file_path, index_col="constVarID", skiprows=rows_to_skip
    )  # , usecols=cols_to_read)

    # Drop last row (weird pandas thing)
    curr_genotypes = curr_genotypes.iloc[:-1, :]
    assert np.all(curr_genotypes.index.isin(qtl_snp_df.qtl_snps) == True)

    nona_rows = np.where(curr_genotypes.isna().sum(1) == 0)[0]
    curr_genotypes = curr_genotypes.iloc[nona_rows, :]

    curr_genotypes.to_csv(pjoin(save_dir_tissue, "genotype.csv"))

    import ipdb

    ipdb.set_trace()
