from tqdm import tqdm
import pandas as pd
import json
from pykeen.datasets import PrimeKG
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import os
import pickle

"""
    prepare the text template of (cell line description, geneA with Omics, geneB with Omics)
    output file: 
    {save_path}/{cell_line}_gene_omics_text_template.json
"""
device = torch.device("cuda:0")
print(device)

### load SL pairs data
labels_df = pd.read_csv('../data/SLKB/rawSL.csv')
# Define cell line DataFrames
cell_line_dfs = {
    "PK1": labels_df[labels_df['cell_line_origin'] == 'PK1'].copy(),
    "IPC298": labels_df[labels_df['cell_line_origin'] == 'IPC298'].copy(),
    "MEWO": labels_df[labels_df['cell_line_origin'] == 'MEWO'].copy(),
    "A549": labels_df[labels_df['cell_line_origin'] == 'A549'].copy(),
    "22RV1": labels_df[labels_df['cell_line_origin'] == '22RV1'].copy(),
    "GI1": labels_df[labels_df['cell_line_origin'] == 'GI1'].copy(),
    "HS936T": labels_df[labels_df['cell_line_origin'] == 'HS936T'].copy(),
}

### prepare omics data
# Function to load omics data
def load_omics_data():
    # Load required omics data
    with open('../data/CCLE/name2ens.json', 'r') as file:
        name_id_mapping = json.load(file)

    ge_data = pd.read_csv('../data/CCLE/GE.csv')
    mu_data = pd.read_csv('../data/CCLE/MU.csv', low_memory=False)
    cn_gene_data = pd.read_csv('../data/CCLE/CNGene.csv')

    # Set index for GE and CNV data
    ge_data.set_index('Unnamed: 0', inplace=True)
    cn_gene_data.set_index('Unnamed: 0', inplace=True)

    return name_id_mapping, ge_data, mu_data, cn_gene_data

# Load labels and omics data
name_id_mapping, ge_data, mu_data, cn_gene_data = load_omics_data()


### function for omics data and template

def if_mutation(mu_data, cell_line_id, gene_id):
    filtered_rows = mu_data[mu_data['ModelID'] == cell_line_id]
    # mu_event = filtered_rows[filtered_rows.iloc[:, -1].astype(str).isin([gene_id, f"{gene_id}.0"])]
    mu_event = filtered_rows[filtered_rows['EnsemblGeneID'].astype(str) == gene_id]
    
    return (1 if not mu_event.empty else 0)
def get_omics_data(ge_data, cn_gene_data, mu_data, cell_line_id, gene_name, gene_id):
    ge_level = None
    cnv_gene = None
    # Gene Expression (GE)
    try:
        gene_columns = [col for col in ge_data.columns if gene_name in col]
        matched_column = gene_columns[0] 
        ge_level = ge_data.loc[cell_line_id, matched_column]
    except (IndexError, KeyError):
        ge_level = None 
    # Copy Number Variation (CNV)
    try:
        gene_columns = [col for col in cn_gene_data.columns if gene_name in col]
        matched_column = gene_columns[0]
        cnv_gene = cn_gene_data.loc[cell_line_id, matched_column]
        if pd.isna(cnv_gene):
            cnv_gene = None
    except (IndexError, KeyError):
        cnv_gene = None
    # MU
    mu_gene = if_mutation(mu_data, cell_line_id, gene_id)

    return [ge_level, cnv_gene, mu_gene]

def format_cell_line_description(cell_line, cell_lines_info, gene_name, omics_data):

    ge_level, cnv_gene, mu_gene = omics_data
    mutation_status = "mutated" if mu_gene == 1 else "not mutated"
    template = (
        # f"{cell_lines_info[cell_line]['description']} "
        f"In the {cell_line} cell line, {gene_name} has a "
        f"gene expression level of {ge_level}, "
        f"a copy number variation of {cnv_gene}, "
        f"and is {mutation_status}."
    )
    return template


cell_lines_info = {
    "A549": {
        "cell_line_id": "ACH-000681",
        "description": "A549 cell line is derived from human lung carcinoma, used in research of lung cancer and chemotherapeutic responses. It is responsible for the study of cancerous transformations and treatment efficacy in pulmonary cells. These cells are characterized by an epithelial-like structure, making them useful for studying ion transport across alveolar surfaces and drug metabolism.",
        "description_w_genes": "A549 cell line is derived from human lung carcinoma, used in research of lung cancer and chemotherapeutic responses. It is responsible for the study of cancerous transformations and treatment efficacy in pulmonary cells. These cells are characterized by an epithelial-like structure, making them useful for studying ion transport across alveolar surfaces and drug metabolism. Commonly overexpressed marker genes include: TM4SF20,NPY4R2,NPY4R,RSPO3,GRIN2B,FGA,INSL4,GABRA5,GKN2,C11orf86,GLP2R,FGB,FGG,BAAT,TUBA4B,AKR1B10,ST6GAL2,MYOCD,IFNA13,C5."
    },
    "PK1": {
        "cell_line_id": "ACH-000307",
        "description": "PK1 cell line is derived from porcine kidney tissue, used in research of viral infections and nephrotoxicity. It is responsible for the diffusion of substances across cellular membranes, vital for understanding pathogenic interactions. These cells exhibit high levels of enzymatic activity, important for studying kidney function and drug-induced renal damage.",
        "description_w_genes": "PK1 cell line is derived from porcine kidney tissue, used in research of viral infections and nephrotoxicity. It is responsible for the diffusion of substances across cellular membranes, vital for understanding pathogenic interactions. These cells exhibit high levels of enzymatic activity, important for studying kidney function and drug-induced renal damage. Commonly overexpressed marker genes include: DIPK2B,C8orf74,PSG8,GPR78,LMOD3,OR10V1,PADI3,AC007906.2,KRT13,DNAH2,LY6D,NIPAL4,DRD1,UPK3B,GJB4,SPRR1B,PLA2G4F,DHRS9,PHACTR3,SERPINF2."
    },
    "IPC298": {
        "cell_line_id": "ACH-000915",
        "description": "IPC-298 cell line is derived from human malignant melanoma, used in research of skin cancer mechanisms and drug efficacy. It is responsible for exploring cellular responses to oncogenic stress. These cells provide a model for understanding melanoma biology and testing novel therapeutic strategies.",
        "description_w_genes": "IPC-298 cell line is derived from human malignant melanoma, used in research of skin cancer mechanisms and drug efficacy. It is responsible for exploring cellular responses to oncogenic stress. These cells provide a model for understanding melanoma biology and testing novel therapeutic strategies. Commonly overexpressed marker genes include: TAC1,SPATS1,KRTAP15-1,SLC17A6,DEFB134,IL24,PRDM7,PMP2,KLHL34,OR6V1,KRTAP13-2,PROKR1,MRGPRX4,MMP8,OR7A10,HTR2B,LGI3,POTEC,CPN1,OCA2."
    },
    "MEWO": {
        "cell_line_id": "ACH-000987",
        "description": "MEWO cell line is derived from human skin (melanoma), used in research of melanoma progression and treatment. It is responsible for modeling melanoma spread and response to therapies. These cells are characterized by their ability to form tumors and metastasize, which are key features for oncological studies.",
        "description_w_genes": "MEWO cell line is derived from human skin (melanoma), used in research of melanoma progression and treatment. It is responsible for modeling melanoma spread and response to therapies. These cells are characterized by their ability to form tumors and metastasize, which are key features for oncological studies. Commonly overexpressed marker genes include: SERTM2,MORC1,PNLDC1,PAEP,SPANXN2,PRY,AATK,TRPM1,MUC7,ATP6V0D2,MMP8,BIRC7,CACNA1E,IRX6,SLAMF9,TRIM63,SLC45A2,DCT,NR4A3,PMEL."
    },
    "22RV1": {
        "cell_line_id": "ACH-000956",
        "description": "22RV1 cell line is derived from a human prostate carcinoma, used in research of prostate cancer and hormone-responsive cancer therapies. It is responsible for studying tumor growth, androgen receptor signaling, and metastasis in prostate cells. These cells are characterized by their ability to proliferate in the presence of androgens, making them a valuable model for investigating prostate cancer progression and the efficacy of androgen-deprivation therapies.",
        "description_w_genes": "22RV1 cell line is derived from a human prostate carcinoma, used in research of prostate cancer and hormone-responsive cancer therapies. It is responsible for studying tumor growth, androgen receptor signaling, and metastasis in prostate cells. These cells are characterized by their ability to proliferate in the presence of androgens, making them a valuable model for investigating prostate cancer progression and the efficacy of androgen-deprivation therapies. Commonly overexpressed marker genes include: RBMY1D,CFC1,CHRNA2,CFC1B,PPP3R2,KLK4,PON1,CHRNE,OR8G1,SSX3,FOXD4,FAM47C,GRIN3A,UGT2B15,KLK2,GUCY2D,ANKRD62,RBMY1B,OR8A1,GLYATL1."
    },
    "GI1": {
        "cell_line_id": "ACH-000756",
        "description": "GI1 cell line is derived from human glioblastoma multiforme (GBM), a highly malignant tumor of the central nervous system. It is used in research on brain cancer, tumor biology, and therapeutic resistance. These cells exhibit high invasiveness and resistance to radiation, making them an important model for studying the mechanisms of glioma progression and developing targeted therapies for GBM.",
        "description_w_genes": "GI1 cell line is derived from human glioblastoma multiforme (GBM), a highly malignant tumor of the central nervous system. It is used in research on brain cancer, tumor biology, and therapeutic resistance. These cells exhibit high invasiveness and resistance to radiation, making them an important model for studying the mechanisms of glioma progression and developing targeted therapies for GBM. Commonly overexpressed marker genes include: OR8B8,GOLGA6L22,OR8B4,CGB3,ADCY8,UTS2,TAFA3,CAMK2A,KCTD8,PLG,NPBWR1,OR10V1,GUCY1A2,KCNK13,MYBPH,HHIPL2,FAM216B,WNT10B,APCDD1L,GIP."
    },
    "HS936T": {
        "cell_line_id": "ACH-000801",
        "description": "HS936T cell line is derived from human melanoma, used in research of skin cancer and metastatic mechanisms. It is employed to study the transformation of melanocytes, tumor invasion, and resistance to treatments. These cells exhibit aggressive proliferation and metastatic potential, making them an essential model for understanding melanoma progression and testing targeted therapies.",
        "description_w_genes": "HS936T cell line is derived from human melanoma, used in research of skin cancer and metastatic mechanisms. It is employed to study the transformation of melanocytes, tumor invasion, and resistance to treatments. These cells exhibit aggressive proliferation and metastatic potential, making them an essential model for understanding melanoma progression and testing targeted therapies. Commonly overexpressed marker genes include: GAGE12G,GAGE10,NRAP,CTLA4,TRPM1,SLC24A4,OR6N1,GAGE13,PRY,GAGE1,CYSLTR2,PRDM7,PAGE1,DEFB126,WFDC1,MAT1A,GAGE2E,CT55,SCML4,TUBA8."
    },
}


def process_cell_line_data(cell_line, cell_lines_info, cell_line_id, target_df, name_id_mapping, ge_data, mu_data, cn_gene_data, save_path):
    
    # Get unique genes
    genes_in_df = pd.unique(target_df[['gene_1', 'gene_2']].values.ravel())
    print(f"Processing {cell_line}...{genes_in_df.size}")
    
    # Dictionary to store gene descriptions
    gene_descriptions = {}
    for gene_name in tqdm(genes_in_df, desc=f"Processing genes for {cell_line}"):
        gene_id = name_id_mapping.get(gene_name)
        if gene_id is None:
            continue
        # Retrieve omics data
        omics_data = get_omics_data(ge_data, cn_gene_data, mu_data, cell_line_id, gene_name, gene_id)
        if any(data is None for data in omics_data):
            continue

        # Generate description
        gene_descriptions[gene_name] = format_cell_line_description(cell_line, cell_lines_info, gene_name, omics_data)
        
    # Save the dictionary to a file
    save_file = f"{save_path}/{cell_line}_gene_omics_text_template.json"
    with open(save_file, "w", encoding="utf-8") as file:
        json.dump(gene_descriptions, file, indent=4, ensure_ascii=False)
    
    print(f"Saved gene descriptions for {cell_line} to {save_file}")




# Directory to save outputs
save_path = "../data/text_template"
os.makedirs(save_path, exist_ok=True)

# Process each cell line
for cell_line_case_name, target_df in cell_line_dfs.items():
    cell_line_name = cell_line_case_name.split("_")[0]
    cell_line_id = cell_lines_info[cell_line_name]["cell_line_id"]
    process_cell_line_data(cell_line_name, cell_lines_info, cell_line_id, target_df, name_id_mapping, ge_data, mu_data, cn_gene_data, save_path)


# python -m src.data_prepare



