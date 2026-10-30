"""Path configuration for PromptCASL (BIBM 2025 version).

Third-party resources are not redistributed with this repository. Their locations are
read from environment variables and default to `data/external/` (and `data/BiomedBERT`).
Everything that ships with the repository (fold definitions, prompt templates, cell-line
omics vectors) is resolved relative to the repository root.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "data", "external")

# ── third-party inputs (override with environment variables) ──────────────────
CCLE_DIR = os.environ.get("PROMPTCASL_CCLE_DIR", os.path.join(EXTERNAL_DIR, "CCLE"))
SLKB_RAW = os.environ.get("PROMPTCASL_SLKB_RAW", os.path.join(EXTERNAL_DIR, "rawSL.csv"))
GENEPT_PKL = os.environ.get(
    "PROMPTCASL_GENEPT_PKL",
    os.path.join(EXTERNAL_DIR, "GenePT_gene_protein_embedding_model_3_text.pickle"),
)
PRIMEKG_EMB = os.environ.get(
    "PROMPTCASL_PRIMEKG_EMB",
    os.path.join(EXTERNAL_DIR, "conv_primekg_entity_embeddings.npy"),
)
BIOMEDBERT = os.environ.get(
    "PROMPTCASL_BIOMEDBERT", os.path.join(PROJECT_ROOT, "data", "BiomedBERT")
)
KG_NEIGHBOR_DIR = os.environ.get(
    "PROMPTCASL_KG_NEIGHBORS", os.path.join(PROJECT_ROOT, "data", "KG_neighbors")
)

# ── in-repository artefacts ──────────────────────────────────────────────────
FOLDS_DIR = os.path.join(PROJECT_ROOT, "data", "folds_assignment")
TEXT_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "data", "text_template_genes")
OMICS_JSON = os.path.join(PROJECT_ROOT, "data", "multi_omics", "cell_line_omics_data.json")

# ── outputs ──────────────────────────────────────────────────────────────────
DATASETS_DIR = os.environ.get(
    "PROMPTCASL_DATASETS_DIR", os.path.join(PROJECT_ROOT, "data", "datasets")
)
DATASET_NAME = "cell_line_biomedbert_kg50shared"
DATASET_DIR = os.path.join(DATASETS_DIR, DATASET_NAME)
SAVE_DIR = os.path.join(PROJECT_ROOT, "save")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

CELL_LINES = ["A549", "PK1", "IPC298", "MEWO", "22RV1", "GI1", "HS936T"]
