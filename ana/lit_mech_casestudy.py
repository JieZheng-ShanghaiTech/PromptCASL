"""Build the literature-mechanism case-study table from SynLethHub evidence."""
import os, re
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys as _sys; _sys.path.insert(0, f'{BASE}/ana')
import model_version as MV                       # active run + split selection

# PPE-K16 full model; split (std vs GPLO) resolved by model_version.SPLIT
PRED_FULL  = MV.PRED_FULL
# no-context blind baseline (shared A1 ablation) — for the conflict-flip contrast
PRED_BLIND = MV.PRED_BLIND
SYNLETH    = f'{BASE}/SynLethHub_evidence/sl_integrated_long.csv'
# re-exported so notebook cells read the version-correct cache paths off `L`
ATTN_PKL       = MV.ATTN_PKL
MECH_ALIGN_CSV = MV.MECH_ALIGN_CSV

CELL_LINES = ['A549', 'PK1', 'IPC298', 'MEWO', 'GI1', 'HS936T']

# mechanism-theme keyword rules (first match wins); order matters
THEMES = [
    ('DNA repair / HR-PARP',      r'\bHR\b|homologous recomb|PARP|DNA damage|DNA repair|double-strand|replication (?:stress|fork)|\bfork\b|BRCA'),
    ('MAPK / RAF-ERK',            r'MAPK|ERK|\bRAF\b|\bMEK\b|BRAF|RAF1|RAF kinase'),
    ('PI3K-AKT-mTOR',             r'\bAKT\b|mTOR|PI3K|\bS6K\b|RPS6K'),
    ('Cell cycle / CDK',          r'\bCDK\b|cyclin|cell cycle|G1/S|G2/M|mitosi'),
    ('Transcription / splicing',  r'transcription|splicing|Pol[- ]?II|\bCTD\b|elongation|helicase|spliceosome'),
    ('Apoptosis / BH3',           r'apopto|\bBH3\b|BCL2|MCL1|\bBIM\b|cytochrome'),
    ('Ubiquitin / proteasome',    r'ubiquit|proteasome|NEDD|SUMO|myristoyl'),
    ('Metabolism',                r'metabol|fatty acid|glycol|\bCTP\b|synthase|de novo|glucose|nucleotide'),
    ('Chromatin / epigenetic',    r'HDAC|chromatin|NuRD|methylation|super-enhancer|histone'),
]


_CONF_RANK = {'high': 3, 'medium': 2, 'low': 1}


def _norm_pair(a, b):
    return '|'.join(sorted([str(a), str(b)]))


def theme_of(text):
    t = str(text)
    for name, pat in THEMES:
        if re.search(pat, t, re.I):
            return name
    return 'Other / paralog-generic'


def load_predictions(path):
    p = pd.read_csv(path)
    p['pk'] = p.gene_pair.map(lambda g: _norm_pair(*str(g).split('|')))
    return p


def pair_profile(pred):
    """Per-pair subset + per-cell-line prediction summary from a predictions frame."""
    prof = pred.groupby('pk').agg(
        n_cl=('cell_line', 'nunique'),
        n_lab=('label', 'nunique'),
        mean_lab=('label', 'mean'),
    ).reset_index()
    prof['subset'] = np.where(prof.n_cl < 2, 'single',
                              np.where(prof.n_lab > 1, 'conflict', 'consistent'))
    # SL-instance mean prob (want high) and non-SL-instance mean prob (want low)
    g = pred.groupby('pk')
    sl = pred[pred.label == 1].groupby('pk').prob.mean().rename('sl_prob')
    ns = pred[pred.label == 0].groupby('pk').prob.mean().rename('ns_prob')
    prof = prof.merge(sl, on='pk', how='left').merge(ns, on='pk', how='left')
    return prof


def load_lit_mechanism():
    """SynLethHub literature-source rows that carry a mechanism annotation."""
    df = pd.read_csv(SYNLETH, low_memory=False)
    lit = df[df.source == 'literature'].copy()
    lit['pk'] = lit.apply(lambda r: _norm_pair(r.gene_a, r.gene_b), axis=1)
    return lit[lit.mechanism.notna()].copy()


def build_candidates():
    """Return the case-study candidate table: evaluated pairs × literature mechanism."""
    pred = load_predictions(PRED_FULL)
    prof = pair_profile(pred)
    litmech = load_lit_mechanism()

    inter = sorted(set(prof.pk) & set(litmech.pk))
    prof = prof[prof.pk.isin(inter)].copy()

    lm = litmech[litmech.pk.isin(inter)].copy()
    # representative mechanism = longest text; representative sentence = longest sentence
    lm['mlen'] = lm.mechanism.str.len()
    mrep = lm.sort_values('mlen', ascending=False).groupby('pk').first().reset_index()
    lm_s = lm[lm.sentence.notna()].copy()
    lm_s['slen'] = lm_s.sentence.str.len()
    srep = (lm_s.sort_values('slen', ascending=False).groupby('pk')
            .first().reset_index()[['pk', 'sentence', 'pmid']])
    npmid = lm.groupby('pk').pmid.nunique().rename('n_pmid')

    # ── evidence-quality provenance (mechanism is LLM-synthesized: detail=phase1_llm) ──
    # evidence_type / confidence let us keep only solid pairs; no_sl_observed rows are NEGATIVE
    # findings (paper reports NO synthetic lethality) and must be excluded from positive controls.
    qual = lm.groupby('pk').agg(
        evidence_types=('evidence_type', lambda s: ';'.join(sorted(set(s.dropna())))),
        best_confidence=('confidence', lambda s: max((_CONF_RANK.get(x, 0) for x in s.dropna()), default=0)),
        has_experimental=('evidence_type', lambda s: 'experimental' in set(s)),
        no_sl_observed=('evidence_type', lambda s: 'no_sl_observed' in set(s)),
    ).reset_index()
    qual['best_confidence'] = qual.best_confidence.map({3: 'high', 2: 'medium', 1: 'low', 0: 'NA'})

    prof = (prof.merge(mrep[['pk', 'mechanism']], on='pk', how='left')
                .merge(srep, on='pk', how='left')
                .merge(npmid, on='pk', how='left')
                .merge(qual, on='pk', how='left'))
    prof['theme'] = prof.mechanism.map(theme_of)
    # "solid": experimental support, confidence >= medium, and NOT a negative (no_sl_observed) finding
    prof['solid'] = (prof.has_experimental & prof.best_confidence.isin(['high', 'medium'])
                     & ~prof.no_sl_observed)

    # per-cell-line P(SL) for the pair (full model), as a compact dict-string
    plong = pred[pred.pk.isin(inter)]
    percl = (plong.groupby(['pk', 'cell_line'])
             .agg(prob=('prob', 'mean'), label=('label', 'first')).reset_index())

    def cl_summary(pk):
        d = percl[percl.pk == pk]
        return '; '.join(f"{r.cell_line}:{r.prob:.2f}({int(r.label)})"
                         for _, r in d.sort_values('cell_line').iterrows())
    prof['per_cell_line'] = prof.pk.map(cl_summary)

    # correctness flags
    prof['hit'] = np.where(prof.mean_lab >= 0.5,
                           prof.sl_prob > 0.5,           # SL-majority pair -> want sl_prob high
                           prof.ns_prob < 0.5)           # non-SL majority -> want ns_prob low
    cols = ['pk', 'subset', 'theme', 'n_cl', 'mean_lab', 'sl_prob', 'ns_prob', 'hit',
            'solid', 'has_experimental', 'no_sl_observed', 'best_confidence', 'evidence_types',
            'per_cell_line', 'mechanism', 'sentence', 'pmid', 'n_pmid']
    return prof[cols].sort_values(['solid', 'subset', 'sl_prob'],
                                  ascending=[False, True, False]).reset_index(drop=True)


if __name__ == '__main__':
    tab = build_candidates()
    out = f'{BASE}/ana/lit_mechanism_casestudy_candidates.csv'
    tab.to_csv(out, index=False)
    print(f'wrote {len(tab)} candidate pairs -> {out}')
    print('by subset:', tab.subset.value_counts().to_dict())
    print('by theme :', tab.theme.value_counts().to_dict())
    print(f'overall pair-level hit-rate: {tab.hit.mean():.1%}')
