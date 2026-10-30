# Data dictionary

Files shipped in `data/`. Third-party inputs are listed in the README section "External data".

## Cross-validation splits — `data/folds_assignment/`

One row per (gene pair × cell line) instance. Labels come from the SLKB database export
(`rawSL.csv`); each row keeps the originating study's PMID in `study_origin`, together with the
raw SL score, the statistical score and both cut-offs. The file name encodes three qualifiers:

| Token | Meaning |
|---|---|
| `filtered` | Only pairs whose features can be constructed: both genes have a prompt entry (expression / copy-number / mutation values), a GenePT embedding and a PrimeKG entity. |
| `NP1` | Negative-to-positive ratio of 1: all positives plus an equal number of negatives, sampled once with `random_state=42`. |
| `_F` | Carries the five-fold assignment in the `fold` column (`KFold(5, shuffle=True, random_state=42)`). |

Columns: `gene_pair`, `study_origin`, `cell_line_origin`, `gene_1`, `gene_2`, `SL_or_not`,
`SL_score`, `statistical_score`, `SL_score_cutoff`, `statistical_score_cutoff`, `fold`.

| File | Rows | Positives |
|---|---:|---:|
| `A549_filtered_NP1_F.csv` | 1264 | 632 |
| `PK1_filtered_NP1_F.csv` | 1276 | 638 |
| `IPC298_filtered_NP1_F.csv` | 1170 | 585 |
| `MEWO_filtered_NP1_F.csv` | 920 | 460 |
| `GI1_filtered_NP1_F.csv` | 854 | 427 |
| `HS936T_filtered_NP1_F.csv` | 1178 | 589 |
| `22RV1_filtered_NP1_F.csv` | 450 | 225 |

One further file family:

| File | Meaning |
|---|---|
| `{cell_line}_full.csv` | Full imbalanced set of filtered pairs. The `fold` column is constant. |

## Prompt templates — `data/text_template_genes/`

`{cell_line}_gene_omics_descriptions.json` maps a gene symbol to a sentence describing that
gene's expression, copy-number and mutation state in the cell line, e.g.

> In the A549 cell line, A3GALT2 has a gene expression level of 0.0, a copy number variation of 0.812766033180691, and is not mutated.

The cell-line description is defined in `src/preprocess_datasets.py::cell_lines_info` and is
prepended to the two gene sentences.

## Cell-line omics — `data/multi_omics/`

* `cell_line_omics_data.json`: `cell_line → {"GE": [942], "CNV": [942], "MU": [942], "cell_line_id": str}`,
  consumed by the three omics encoders (942 → 256 each).
* `multi_omics_emb_3_256.pkl`: pre-computed auto-encoder omics embeddings, for the `--omics_data 1` variant.
