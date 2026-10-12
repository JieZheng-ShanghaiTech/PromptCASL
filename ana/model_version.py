"""Select which run and fold split the interpretability scripts read.

    SPLIT  (env PROMPTCASL_SPLIT = 'std' | 'gplo')
        Fold split used by the "main-model" artifacts. GPLO is the leakage-controlled
        split in which all cell-line instances of a gene pair share a fold.

Each artifact family exposes stable per-split names (PRED_STD / PRED_GPLO, ...) plus an
active alias (PRED_FULL / ATTN_PKL / ...) resolved by SPLIT. Consumers:
lit_mech_casestudy.py, mechanism_alignment.py, text_pathway_attn.py,
gen_llm_selfattn_cache.py, and the analysis notebooks. Set PROMPTCASL_SPLIT before
importing this module.
"""

import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLIT = os.environ.get('PROMPTCASL_SPLIT', 'std').lower()
assert SPLIT in ('std', 'gplo'), f"PROMPTCASL_SPLIT must be 'std'|'gplo', got {SPLIT!r}"
GPLO = (SPLIT == 'gplo')


def _p(*parts):
    return os.path.join(BASE, *parts)


# --- per-split main-model artifacts ----------------------------------------
CKPT_STD        = _p('save/MCL/mixed-6cl/20260720_212923_pair_pathway_evidence_reactome_K16_ppe_K16_pk1fix')
D1_FROZEN_STD   = _p('save/MCL/mixed-6cl/20260717_013129_pair_pathway_evidence_reactome_K16_abl_ppe_D1_frozenllm_pk1fix')
PRED_STD        = _p('results/ablation_eval/20260720_212923_pair_pathway_evidence_reactome_K16_ppe_K16_pk1fix/test_predictions.csv')
ATTN_STD        = _p('ana/attn_pair_K16_6cl_pk1fix_0720.pkl')
ATTN_D1FRZ_STD  = _p('ana/attn_pair_K16_D1frozen_6cl_pk1fixed.pkl')
MECH_ALIGN_STD  = _p('ana/mechanism_alignment_std.csv')

CKPT_GPLO       = _p('save/MCL/mixed-6cl-gplo/20260722_200007_pair_pathway_evidence_reactome_K16_gplo_ppe_K16_pk1fix')
D1_FROZEN_GPLO  = _p('save/MCL/mixed-6cl-gplo/20260718_011745_pair_pathway_evidence_reactome_K16_gplo_ppe_K16_D1_frozenllm_pk1fix')
PRED_GPLO       = _p('results/ablation_eval_gplo_pk1fix/20260722_200007_pair_pathway_evidence_reactome_K16_gplo_ppe_K16_pk1fix/test_predictions.csv')
ATTN_GPLO       = _p('ana/attn_pair_K16_6cl_gplo_pk1fix_0720.pkl')
ATTN_D1FRZ_GPLO = _p('ana/attn_pair_K16_D1frozen_6cl_gplo_pk1fix.pkl')
MECH_ALIGN_GPLO = _p('ana/mechanism_alignment_gplo.csv')

# no-context blind (A1) baseline, per split.
PRED_BLIND_STD  = _p('results/ablation_eval/20260623_133348_abl_A1_nocontext/test_predictions.csv')
PRED_BLIND_GPLO = _p('results/ablation_eval_gplo_pk1fix/20260629_173633_pair_pathway_evidence_reactome_K16_gplo_ppe_K16_A1_nocontext/test_predictions.csv')

# --- active aliases (chosen by SPLIT) --------------------------------------
CKPT_FULL         = CKPT_GPLO       if GPLO else CKPT_STD
STD_CKPT          = CKPT_FULL
D1_FROZEN_CKPT    = D1_FROZEN_GPLO  if GPLO else D1_FROZEN_STD
PRED_FULL         = PRED_GPLO       if GPLO else PRED_STD
PRED_BLIND        = PRED_BLIND_GPLO if GPLO else PRED_BLIND_STD
ATTN_PKL          = ATTN_GPLO       if GPLO else ATTN_STD
ATTN_PKL_D1FROZEN = ATTN_D1FRZ_GPLO if GPLO else ATTN_D1FRZ_STD
MECH_ALIGN_CSV    = MECH_ALIGN_GPLO if GPLO else MECH_ALIGN_STD

# --- per-split dataset shards (token_emb .pt) ------------------------------
DATASET_STD  = _p('data/datasets/cell_line_biomedbert_kg50shared')
DATASET_GPLO = _p('data/datasets/cell_line_biomedbert_kg50shared_GPLO')
DATASET_DIR  = DATASET_GPLO if GPLO else DATASET_STD

# --- split-independent / std-only analysis caches --------------------------
MECH_EMB_NPZ         = _p('ana/mech_align_emb_pk1fixed.npz')
SELFATTN_NPZ         = _p('ana/llm_selfattn_bycat_pk1fixed_gplo.npz' if GPLO
                          else 'ana/llm_selfattn_bycat_pk1fixed.npz')
TEXT_EMB_TRAINED_PPE = _p('ana/text_emb_pool_v1_trained_ppe_pk1fixed_gplo.npz' if GPLO
                          else 'ana/text_emb_pool_v1_trained_ppe_pk1fixed.npz')

# --- GPLO: causal omics-swap + text-swap -----------------------------------
_SW  = _p('results/omics_swap')
_TSW = _p('results/text_swap')
OMICS_SWAP_FULL = os.path.join(_SW,  '20260721_024529_pair_pathway_evidence_reactome_K16_gplo_ppe_K16_pk1fix__swapmatrix.csv')
TEXT_SWAP_FULL  = os.path.join(_TSW, '20260721_024529_pair_pathway_evidence_reactome_K16_gplo_ppe_K16_pk1fix__textswapmatrix.csv')
OMICS_SWAP_NOTX = os.path.join(_SW,  '20260707_201936_pair_pathway_evidence_reactome_K16_gplo_ppe_K16_no_text__swapmatrix.csv')

# signed-margin references on the GPLO conflict subset.
REF_FULL = 0.132
REF_NOTX = 0.029

# --- GPLO: per-subset modality contribution --------------------------------
_EV_GPLO = _p('results/ablation_eval_gplo_pk1fix')
ARM_FULL = os.path.join(_EV_GPLO, '20260715_233058_pair_pathway_evidence_reactome_K16_gplo_ppe_K16_pk1fix')
ARM_B1   = os.path.join(_EV_GPLO, '20260717_094130_gplo_ppe_K16_B1_noomics_pk1fix')
ARM_A1   = os.path.join(_EV_GPLO, '20260629_173633_pair_pathway_evidence_reactome_K16_gplo_ppe_K16_A1_nocontext')
ARM_NOTEXT = _p('save/MCL/mixed-6cl-gplo/20260707_201936_pair_pathway_evidence_reactome_K16_gplo_ppe_K16_no_text')


def _mtime(path):
    try:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')
    except OSError:
        return 'MISSING'


def summary():
    lines = [f"model_version: SPLIT={SPLIT!r}  (active alias -> {SPLIT})"]
    for name in ('CKPT_FULL', 'PRED_FULL', 'PRED_BLIND', 'ATTN_PKL', 'MECH_ALIGN_CSV', 'MECH_EMB_NPZ'):
        p = globals()[name]
        lines.append(f"  {name:14s} = {p}   [{_mtime(p)}]")
    lines.append(f"  stable refs: PRED_STD={_mtime(PRED_STD)}  PRED_GPLO={_mtime(PRED_GPLO)}  "
                 f"ATTN_STD={_mtime(ATTN_STD)}  ATTN_GPLO={_mtime(ATTN_GPLO)}")
    return '\n'.join(lines)


if __name__ == '__main__':
    print(summary())
