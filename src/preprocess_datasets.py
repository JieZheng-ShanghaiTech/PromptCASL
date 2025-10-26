import pandas as pd
import pickle
import json
from pykeen.datasets import PrimeKG
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import os
from src.data_loader import GenePairDataset
import random
from collections import defaultdict


"""
1. concate cell line description, geneA Omics, geneB Omics for text, get token embedding sequence from bioLLM
2. GenePT emb of geneA and geneB, KG emb of geneA and geneB
3. KG embeddings for geneA neighbor list, geneB neighbor list, and shared neighbor list
4. generate 5-fold dataset_F1.pt, dataset_F2.pt... based on the fold_asignment/ data split csv file

"""

device = torch.device("cuda:0")
torch.cuda.device(device)
print(device)

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

with open("../data/multi_omics/cell_line_omics_data.json", "r") as f:
    cell_line_omics = json.load(f)



## step1: concate cell line description, geneA Omics, geneB Omics for text, get token embedding sequence from bioLLM
def get_token_embeddings(model, tokenizer, text, device, max_length=512):
    """
    Generate token embeddings for a given text using a HuggingFace model.

    Args:
        model: The HuggingFace model (BioBERT, Qwen3, etc.).
        tokenizer: Corresponding tokenizer.
        text (str): Input text for embedding.
        device: torch.device
        max_length (int): max input length

    Returns:
        input_ids (torch.Tensor): [1, seq_len]
        attention_mask (torch.Tensor): [seq_len]
        token_embeddings (torch.Tensor): [seq_len, hidden_dim]
        tokens (list): Tokens corresponding to the embeddings.
    """
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Choose embedding method based on model type
    with torch.no_grad():
        if hasattr(model, "embeddings"):
            # For BERT, BioBERT, BioMedBERT, etc.
            token_embeddings = model.embeddings(input_ids=input_ids).squeeze(0) # Shape: [seq_len, hidden_dim]
        elif hasattr(model, "embed_tokens"):
            # For Qwen, Llama, etc.
            token_embeddings = model.embed_tokens(input_ids).squeeze(0)
        else:
            raise ValueError("Model does not have recognized embedding layer: 'embeddings' or 'embed_tokens'")
    return input_ids, attention_mask.squeeze(0), token_embeddings, tokens

def get_biobert_token_emb_for_pair(cell_line, geneA, geneB, cell_lines_info, gene_omics_desc, biobert, tokenizer, device):
    """
    For a single gene pair and cell line, return BioBERT token embeddings for the concatenated text.
    Args:
        cell_line (str): Name of the cell line.
        geneA (str): Name of geneA.
        geneB (str): Name of geneB.
        cell_lines_info (dict): Cell line info dictionary.
        gene_omics_desc (dict): Gene omics description dictionary for this cell line.
        biobert, tokenizer, device: Model and device.
    Returns:
        input_ids, attention_mask, token_embeddings
    """
    cell_line_desc = cell_lines_info[cell_line]["description"]
    # cell_line_desc = cell_lines_info[cell_line]["description_w_genes"]
    descA = gene_omics_desc.get(geneA, f"No description for {geneA}")
    descB = gene_omics_desc.get(geneB, f"No description for {geneB}")
    # full_text = f"{cell_line_desc} [SEP] {descA} [SEP] {descB}" 
    full_text = f"{cell_line_desc} {descA} {descB}"
    input_ids, attention_mask, token_embeddings, _ = get_token_embeddings(biobert, tokenizer, full_text, device)
    return input_ids, attention_mask, token_embeddings


## step2: GenePT emb of geneA and geneB
### load GenePT embeddings
with open("../data/GenePT/GenePT_gene_protein_embedding_model_3_text.pickle.", 'rb') as file:
    genept_gene_embeddings = pickle.load(file) # 3072, text-embedding-3-large embeddings
for gene, emb in genept_gene_embeddings.items():
    genept_gene_embeddings[gene] = np.array(emb, dtype=np.float32)

## step3: KG embeddings for geneA neighbor list, geneB neighbor list, and shared neighbor list
#### PrimeKG Emb
kg_dataset = PrimeKG(create_inverse_triples=True)
kg_entity_embeddings = np.load('../data/KG_ConvE/conv_primekg_entity_embeddings.npy')
kg_entity_to_id = kg_dataset.training.entity_to_id

def get_kg_neighbor_emb_for_pair(geneA, geneB, cell_line, entity_to_id, kg_entity_embeddings, neighbor_dir="../data/KG_neighbors", min_len=100, rd_seed=42):
    """
    For a given geneA, geneB, and cell_line, extract the neighbor lists from the cell line's neighbor .tsv file,
    and return the KG embeddings for each neighbor list.
    Returns:
        geneA_neighbor_embs: [N, emb_dim] numpy array
        geneB_neighbor_embs: [N, emb_dim] numpy array
        shared_neighbor_embs: [M, emb_dim] numpy array
    """
    # Map gene names to IDs
    geneA_id = entity_to_id.get(geneA)
    geneB_id = entity_to_id.get(geneB)
    if geneA_id is None or geneB_id is None:
        raise ValueError(f"GeneA or GeneB not found in entity_to_id: {geneA}, {geneB}")

    # Read the cell line neighbor file
    neighbor_file = f"{neighbor_dir}/{cell_line}_neighbors.tsv"
    df = pd.read_csv(neighbor_file, sep='\t')

    # Find the row for this gene pair (by ID)
    row = df[(df['geneA'] == geneA_id) & (df['geneB'] == geneB_id)]
    if row.empty:
        raise ValueError(f"Gene pair ({geneA}, {geneB}) not found in {neighbor_file}")

    row = row.iloc[0]
    # Parse neighbor lists (comma-separated string of IDs)
    geneA_neighbors = [int(x) for x in str(row['geneA_neighbors']).split(',') if x]
    geneB_neighbors = [int(x) for x in str(row['geneB_neighbors']).split(',') if x]
    shared_neighbors = [int(x) for x in str(row['shared_neighbors']).split(',') if x]

    rng = np.random.default_rng(rd_seed)
    selected_geneA = []
    selected_geneB = []

    # if len(shared_neighbors) >= min_len:
    #     selected_shared_A = rng.choice(shared_neighbors, min_len, replace=False)
    #     geneA_selected_ids = selected_shared_A
    #     selected_shared_B = rng.choice(shared_neighbors, min_len, replace=False)
    #     geneB_selected_ids = selected_shared_B
    # else: # shared neighbors less than min_len
    #     # For geneA
    #     selected_shared_A = shared_neighbors
    #     remaining_A = min_len - len(selected_shared_A)
    #     unique_geneA_neighbors = [n for n in geneA_neighbors if n not in shared_neighbors]
    #     if len(unique_geneA_neighbors) >= remaining_A:
    #         selected_geneA = rng.choice(unique_geneA_neighbors, remaining_A, replace=False)
    #     else:
    #         selected_geneA = unique_geneA_neighbors  # take all if not enough
    #     geneA_selected_ids = list(selected_shared_A) + list(selected_geneA)
    #     # For geneB
    #     selected_shared_B = shared_neighbors
    #     remaining_B = min_len - len(selected_shared_B)
    #     unique_geneB_neighbors = [n for n in geneB_neighbors if n not in shared_neighbors]
    #     if len(unique_geneB_neighbors) >= remaining_B:
    #         selected_geneB = rng.choice(unique_geneB_neighbors, remaining_B, replace=False)
    #     else:
    #         selected_geneB = unique_geneB_neighbors  # take all if not enough
    #     geneB_selected_ids = list(selected_shared_B) + list(selected_geneB)
        
    half_min_len = min_len // 2

    if len(shared_neighbors) >= half_min_len:
        selected_shared = rng.choice(shared_neighbors, half_min_len, replace=False)
    else:
        selected_shared = shared_neighbors
    num_shared = len(selected_shared)
    remain_A = min_len - num_shared
    remain_B = min_len - num_shared

    unique_geneA_neighbors = [n for n in geneA_neighbors if n not in selected_shared]
    if len(unique_geneA_neighbors) >= remain_A:
        selected_geneA = rng.choice(unique_geneA_neighbors, remain_A, replace=False)
    else: 
        selected_geneA = unique_geneA_neighbors  

    unique_geneB_neighbors = [n for n in geneB_neighbors if n not in selected_shared]
    if len(unique_geneB_neighbors) >= remain_B:
        selected_geneB = rng.choice(unique_geneB_neighbors, remain_B, replace=False)
    else:
        selected_geneB = unique_geneB_neighbors 

    geneA_selected_ids = list(selected_shared) + list(selected_geneA)
    geneB_selected_ids = list(selected_shared) + list(selected_geneB)

    geneA_neighbor_embs = kg_entity_embeddings[np.array(geneA_selected_ids)]
    geneB_neighbor_embs = kg_entity_embeddings[np.array(geneB_selected_ids)]
    
    return geneA_neighbor_embs, geneB_neighbor_embs

## step4: form the datasets
def build_folds_GenePairDataset(
    fold_assign_file, 
    cell_line, 
    cell_lines_info, 
    gene_omics_desc, 
    genept_embeddings, 
    kg_entity_to_id, 
    kg_entity_embeddings, 
    fold_output_dir,
    biobert, 
    tokenizer, 
    device, 
    neighbor_dir="../data/KG_neighbors",
    min_kg_neighbors=100
):
    """
    For each fold in the assignment file, build a GenePairDataset with all features.
    Returns: dict {fold_number: GenePairDataset}
    """
    df = pd.read_csv(fold_assign_file)
    fold2rows = defaultdict(list)
    for idx, row in df.iterrows():
        fold = int(row['fold'])
        fold2rows[fold].append(row)
    
    # Get omics vectors for the current cell line
    omics_data = cell_line_omics.get(cell_line)
    if omics_data is None:
        print(f"Missing omics data for {cell_line}")
    omics_ge = np.array(omics_data['GE'], dtype=np.float32)
    omics_cnv = np.array(omics_data['CNV'], dtype=np.float32)
    omics_mu = np.array(omics_data['MU'], dtype=np.float32)

    fold2dataset = {}
    for fold, rows in fold2rows.items():
        embeddings = []
        labels = []
        gene_names = []
        for row in tqdm(rows, desc=f"Processing fold {fold}"):
            gene_1 = row['gene_1']
            gene_2 = row['gene_2']
            # 1. Text token sequence (BioBERT)
            input_ids, attention_mask, token_emb = get_biobert_token_emb_for_pair(
                cell_line, gene_1, gene_2, cell_lines_info, gene_omics_desc, biobert, tokenizer, device
            )
            # 2. GenePT embeddings
            geneA_genept = genept_embeddings.get(gene_1)
            geneB_genept = genept_embeddings.get(gene_2)
            if geneA_genept is None or geneB_genept is None:
                print("missing genept")
                continue  # skip if missing
            # 3. KG embeddings for A and B
            geneA_kgid = kg_entity_to_id.get(gene_1)
            geneB_kgid = kg_entity_to_id.get(gene_2)
            if geneA_kgid is None or geneB_kgid is None:
                print("missing kg id")
                continue
            geneA_kg_emb = kg_entity_embeddings[geneA_kgid]
            geneB_kg_emb = kg_entity_embeddings[geneB_kgid]
            # 4. KG neighbor embeddings (100 for A, 100 for B)
            try:
                geneA_neighbors_emb, geneB_neighbors_emb = get_kg_neighbor_emb_for_pair(
                    gene_1, gene_2, cell_line, kg_entity_to_id, kg_entity_embeddings, neighbor_dir=neighbor_dir, min_len=min_kg_neighbors
                )
            except Exception as e:
                print(f"Skipping {gene_1}, {gene_2} in {cell_line} due to neighbor error: {e}")
                continue
            # 5. Label
            label = 1 if str(row['SL_or_not']).strip().upper() == 'SL' else 0
            # 6. Aggregate all features for this pair
            embedding = (
                token_emb.cpu().numpy(),
                # input_ids.cpu().numpy(),  # qwen06
                attention_mask.cpu().numpy(),
                geneA_genept,
                geneA_kg_emb,
                geneB_genept,
                geneB_kg_emb,
                geneA_neighbors_emb,
                geneB_neighbors_emb,
                omics_ge,       
                omics_cnv,      
                omics_mu        
            )
            embeddings.append(embedding)
            label_tensor = torch.tensor(label).long()
            labels.append(label_tensor)
            # Save both gene_pair and cell_line name for each sample
            gene_names.append((row["gene_pair"], cell_line))  # tuple: (gene_pair, cell_line)
            # if len(embeddings) < 3:  # print for first 3 samples
            #     print(f"Example for {gene_1}, {gene_2}:")
            #     print("  token_emb:", token_emb.shape)
            #     # print("  input_ids:", input_ids.shape) # should be [1, seq_len]
            #     print("  attention_mask:", attention_mask.shape)
            #     print("  geneA_genept:", geneA_genept.shape)
            #     print("  geneA_kg_emb:", geneA_kg_emb.shape)
            #     print("  geneB_genept:", geneB_genept.shape)
            #     print("  geneB_kg_emb:", geneB_kg_emb.shape)
            #     print("  geneA_neighbors_emb:", geneA_neighbors_emb.shape)
            #     print("  geneB_neighbors_emb:", geneB_neighbors_emb.shape)
        # Build dataset for this fold
        fold_dataset = GenePairDataset(embeddings, labels, gene_names)
        fold_pt_path = os.path.join(fold_output_dir, f"{cell_line}_F{fold}.pt")
        torch.save(fold_dataset, fold_pt_path)
        print(f"saved in {fold_pt_path}")
    return 

# Example usage:
### BioBERT
# model_name = "../data/biobert-base-cased-v1.2"
model_name = "../data/BiomedBERT"
# model_name = "../data/Qwen3-Embedding-0.6B"
with torch.cuda.device(device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    biobert = AutoModel.from_pretrained(model_name).to(device)
    
# fold_output_dir = "../data/datasets/cell_line_qwen06_kg50shared"
fold_output_dir = "../data/datasets/cell_line_w_omics_biomedbert_kg50shared"
os.makedirs(fold_output_dir, exist_ok=True)
cell_line_names = ["22RV1", "PK1", "IPC298", "MEWO",  "A549", "GI1", "HS936T"]

for cell_line_case in cell_line_names:
    print("Processing ", cell_line_case)
    with open(f"../data/text_template/{cell_line_case}_gene_omics_descriptions.json", "r") as f:
        gene_omics_desc = json.load(f)
    build_folds_GenePairDataset(
        f"../data/folds_assignment/{cell_line_case}_filtered_NP1_F.csv",
        cell_line_case,
        cell_lines_info,
        gene_omics_desc,
        genept_gene_embeddings,
        kg_entity_to_id,
        kg_entity_embeddings,
        fold_output_dir,
        biobert,
        tokenizer,
        device
    )

# Example for CDC6, MAP2K1:
#   token_emb: torch.Size([512, 768])
#   attention_mask: torch.Size([512])
#   geneA_genept: (3072,)
#   geneA_kg_emb: (768,)
#   geneB_genept: (3072,)
#   geneB_kg_emb: (768,)
#   geneA_neighbors_emb: (100, 768)
#   geneB_neighbors_emb: (100, 768)


## python -m src.preprocess_datasets
