"""Text / pathway token-category attention analysis from the cached gene->context attention."""
import sys, json, os, re, pickle, string, numpy as np, pandas as pd
from collections import defaultdict
from functools import lru_cache
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)          # so torch.load of .pt datasets can resolve `src.*` classes
sys.path.insert(0, f'{BASE}/ana')
import model_version as MV
from src.config import BIOMEDBERT
BIOBERT = BIOMEDBERT
PW_START = 512
N_EVID = 16                       # PPE K16
CELL_LINES = ['A549', 'PK1', 'IPC298', 'MEWO', 'GI1', 'HS936T']

# cell-line descriptions — verbatim from src/preprocess_datasets.py cell_lines_info[cl]["description"]
CL_DESC = {
 "A549":"A549 cell line is derived from human lung carcinoma, used in research of lung cancer and chemotherapeutic responses. It is responsible for the study of cancerous transformations and treatment efficacy in pulmonary cells. These cells are characterized by an epithelial-like structure, making them useful for studying ion transport across alveolar surfaces and drug metabolism.",
 "PK1":"PK1 (PK-1) cell line is derived from human pancreatic ductal adenocarcinoma, used in research of pancreatic cancer and KRAS-driven oncogenic signaling. It is responsible for modeling tumor proliferation, invasion, and therapeutic resistance in pancreatic epithelial cells. These cells are characterized by activating KRAS (G12D) and TP53 mutations, making them a valuable model for studying pancreatic cancer progression and targeted therapies.",
 "IPC298":"IPC-298 cell line is derived from human malignant melanoma, used in research of skin cancer mechanisms and drug efficacy. It is responsible for exploring cellular responses to oncogenic stress. These cells provide a model for understanding melanoma biology and testing novel therapeutic strategies.",
 "MEWO":"MEWO cell line is derived from human skin (melanoma), used in research of melanoma progression and treatment. It is responsible for modeling melanoma spread and response to therapies. These cells are characterized by their ability to form tumors and metastasize, which are key features for oncological studies.",
 "GI1":"GI1 cell line is derived from human glioblastoma multiforme (GBM), a highly malignant tumor of the central nervous system. It is used in research on brain cancer, tumor biology, and therapeutic resistance. These cells exhibit high invasiveness and resistance to radiation, making them an important model for studying the mechanisms of glioma progression and developing targeted therapies for GBM.",
 "HS936T":"HS936T cell line is derived from human melanoma, used in research of skin cancer and metastatic mechanisms. It is employed to study the transformation of melanocytes, tumor invasion, and resistance to treatments. These cells exhibit aggressive proliferation and metastatic potential, making them an essential model for understanding melanoma progression and testing targeted therapies.",
}
CL_ALIASES = {'A549':['A549'],'PK1':['PK1'],'IPC298':['IPC-298','IPC298'],'MEWO':['MEWO'],'GI1':['GI1'],'HS936T':['HS936T']}
LINEAGE = ['lung','carcinoma','melanoma','skin',
           'glioblastoma','gbm','brain','glioma','pancrea','melanocyte','nervous',
           'pulmonary','oncogenic','tumor','tumors','cancer','metasta','malignant']

tok = AutoTokenizer.from_pretrained(BIOBERT)
GENE_DESC = {cl: json.load(open(f'{BASE}/data/text_template/v1/{cl}_gene_omics_descriptions.json')) for cl in CELL_LINES}

CATS = ['CLNAME','LINEAGE','CELLBG','GENENAME','EXPR','CNV','MUT','GENEBG','SPECIAL']  # heatmap text columns
CAT_LABEL = {'CLNAME':'cell-line name','LINEAGE':'tissue / lineage','CELLBG':'cell-line prose',
             'GENENAME':'gene name','EXPR':'expression phrase','CNV':'copy-number phrase',
             'MUT':'mutation status','GENEBG':'gene prose','SPECIAL':'[CLS]/[SEP]','PATHWAY':'pathway evidence'}

# ── prompt reconstruction & position labelling ────────────────────────────────
def _ranges(pat, s, flags=re.I):
    return [(m.start(), m.end()) for m in re.finditer(pat, s, flags)]

def build_full_and_spans(cl, gA, gB):
    """Reconstruct the exact v1 prompt and per-category char ranges (absolute offsets in `full`)."""
    cd = CL_DESC[cl]
    dA = GENE_DESC[cl].get(gA, f"No description for {gA}")
    dB = GENE_DESC[cl].get(gB, f"No description for {gB}")
    full = f"{cd} {dA} {dB}"
    ca = len(cd); da0 = ca + 1; da1 = da0 + len(dA); db0 = da1 + 1; db1 = db0 + len(dB)
    rng = {c: [] for c in CATS}
    for al in CL_ALIASES[cl]:
        rng['CLNAME'] += _ranges(re.escape(al), full)
    for kw in LINEAGE:
        rng['LINEAGE'] += _ranges(r'\b' + re.escape(kw), full)
    for (o0, o1), g in [((da0, da1), gA), ((db0, db1), gB)]:
        seg = full[o0:o1]
        for (s, e) in _ranges(re.escape(g), seg):                    rng['GENENAME'].append((o0+s, o0+e))
        for (s, e) in _ranges(r'gene expression level of [\d.]+', seg): rng['EXPR'].append((o0+s, o0+e))
        for (s, e) in _ranges(r'copy number variation of [\d.]+', seg): rng['CNV'].append((o0+s, o0+e))
        for (s, e) in _ranges(r'is (?:not )?mutated', seg):            rng['MUT'].append((o0+s, o0+e))
    return full, (ca, da0, da1, db0, db1), rng

def _overlap(pos, ranges):
    s, e = pos
    return any(s < b and e > a for (a, b) in ranges)

PRIO = ['GENENAME','MUT','EXPR','CNV','CLNAME','LINEAGE']   # priority on multi-range overlap

@lru_cache(maxsize=None)   # pure fn of (cl,gA,gB); all callers read-only -> safe to memoize
def label_positions(cl, gA, gB):
    """array[512] of category strings aligned 1:1 to cross_attn_w[:, :512] (PAD for padding)."""
    full, (ca, da0, da1, db0, db1), rng = build_full_and_spans(cl, gA, gB)
    enc = tok(full, truncation=True, max_length=512, padding='max_length', return_offsets_mapping=True)
    offs = enc['offset_mapping']; mask = np.array(enc['attention_mask'])
    cats = np.array(['PAD'] * 512, dtype=object)
    for i, (s, e) in enumerate(offs):
        if mask[i] == 0:      cats[i] = 'PAD'; continue
        if s == e:            cats[i] = 'SPECIAL'; continue
        hit = next((c for c in PRIO if _overlap((s, e), rng[c])), None)
        if hit is not None:   cats[i] = hit; continue
        cats[i] = 'CELLBG' if (s + e) / 2 < ca else 'GENEBG'
    return cats

@lru_cache(maxsize=None)
def tokens_for(cl, gA, gB):
    full, *_ = build_full_and_spans(cl, gA, gB)
    enc = tok(full, truncation=True, max_length=512, padding='max_length')
    return tok.convert_ids_to_tokens(enc['input_ids']), np.array(enc['attention_mask'])

def load_cache(key='pair_K16'):
    # route to the SPLIT-aware active alias (std vs gplo) — a SUF-only name would silently read
    # the std cache in gplo mode. Fallback keeps any other key working.
    path = {'pair_K16': MV.ATTN_PKL, 'pair_K16_D1frozen': MV.ATTN_PKL_D1FROZEN}.get(
        key, f'{BASE}/ana/attn_{key}_6cl{MV.SUF}.pkl')
    return pickle.load(open(path, 'rb'))

def verify_alignment(records, dataset_dir=f'{BASE}/data/datasets/cell_line_biomedbert_kg50shared', n=5):
    """Sanity-check: reconstructed tokenization mask == stored .pt attention_mask (elementwise)."""
    import torch
    ok = 0; checked = 0
    for fold in range(1, 6):
        for cl in CELL_LINES:
            pt = f'{dataset_dir}/{cl}_F{fold}.pt'
            if not os.path.exists(pt): continue
            ds = torch.load(pt, map_location='cpu', weights_only=False)
            for i in range(min(len(ds), n)):
                gp, c = ds[i][2]; gA, gB = gp.split('|')
                _, m = tokens_for(c, gA, gB)
                if (m == ds[i][0][1].numpy()).all(): ok += 1
                checked += 1
            if checked >= n: break
        if checked >= n: break
    return ok, checked

# ── 2.4.1 : per-category enrichment + block-level mass split ───────────────────
def category_enrichment(records, n_evid=N_EVID):
    """Per-category attention for geneA-query / geneB-query / mean.
    Baseline = the AVERAGE VALID (non-pad) token so 1.0 = a typical attended token (removes the
    padding-dilution inflation of a 1/ctx baseline); values read as preference AMONG real tokens.
    Returns (df[A,B,mean] x [CATS+PATHWAY] enrichment, mean-token-counts, N, mean_valid)."""
    acc = {r: defaultdict(float) for r in ['A', 'B']}; cnt = defaultdict(float)
    npath_mass = {'A': 0.0, 'B': 0.0}; npath_cnt = 0.0; valid_tot = 0.0; N = 0
    for rec in records:
        gA, gB = rec['gene_pair'].split('|'); cats = label_positions(rec['cell_line'], gA, gB)
        cw = rec['cross_attn_w']; nv = int(rec.get('n_evid_valid', n_evid))
        for ri, r in enumerate(['A', 'B']):
            ta = cw[ri, :PW_START]
            for c in CATS:
                m = (cats == c)
                if m.any(): acc[r][c] += float(ta[m].sum())
            ev = cw[ri, PW_START:PW_START + n_evid]
            npath_mass[r] += float(ev[:nv].sum()) if nv else float(ev.sum())
        for c in CATS: cnt[c] += int((cats == c).sum())
        npath_cnt += nv; valid_tot += int((cats != 'PAD').sum()) + nv; N += 1
    mean_valid = valid_tot / N; base = 1.0 / mean_valid
    rows = {}
    for r in ['A', 'B']:
        d = {c: ((acc[r][c] / cnt[c]) / base if cnt[c] > 0 else np.nan) for c in CATS}
        d['PATHWAY'] = (npath_mass[r] / npath_cnt) / base if npath_cnt > 0 else np.nan
        rows[r] = d
    df = pd.DataFrame(rows).T; df.loc['mean'] = df.mean()
    counts = {c: cnt[c] / N for c in CATS}; counts['PATHWAY'] = npath_cnt / N
    return df, counts, N, mean_valid

def mass_split(records, n_evid=N_EVID):
    """Total gene-query attention MASS (fraction summing to ~1) on the text block vs the pathway block,
    and the within-text breakdown by category. Honest complement to per-token enrichment."""
    text_by_cat = defaultdict(float); path_mass = 0.0; tot = 0.0
    for rec in records:
        gA, gB = rec['gene_pair'].split('|'); cats = label_positions(rec['cell_line'], gA, gB)
        cw = rec['cross_attn_w']; nv = int(rec.get('n_evid_valid', n_evid))
        avg = (cw[0] + cw[1]) / 2.0
        for c in CATS:
            m = (cats == c)
            if m.any(): text_by_cat[c] += float(avg[:PW_START][m].sum())
        path_mass += float(avg[PW_START:PW_START + n_evid][:nv].sum()) if nv else float(avg[PW_START:].sum())
        tot += float(avg[:PW_START].sum()) + (float(avg[PW_START:PW_START + n_evid][:nv].sum()) if nv else 0.0)
    share = {c: text_by_cat[c] / tot for c in CATS}; share['PATHWAY'] = path_mass / tot
    return share, tot / len(records)

# ── 2.4.2 : full vs frozen-LLM (D1) paired comparison of attention allocation ──
def _record_key(rec):
    return (rec['gene_pair'], rec['cell_line'], rec.get('fold'))

def record_category_mass(rec, n_evid=N_EVID):
    """Fraction of one record's total valid gene-query attention (mean of geneA/geneB queries) landing on
    each text category + PATHWAY. Denominator = valid text tokens + valid pathway tokens (PAD excluded).
    The token->category partition is model-independent (same prompt + tokenizer), so comparing this vector
    between full and frozen on the SAME record isolates attention redistribution from tokenization."""
    gA, gB = rec['gene_pair'].split('|')
    cats = label_positions(rec['cell_line'], gA, gB)
    cw = rec['cross_attn_w']; nv = int(rec.get('n_evid_valid', n_evid))
    avg = (cw[0] + cw[1]) / 2.0
    text = avg[:PW_START]; path = avg[PW_START:PW_START + n_evid][:nv]
    tot = float(text[cats != 'PAD'].sum()) + float(path.sum())
    if tot <= 0: return {c: 0.0 for c in CATS + ['PATHWAY']}
    out = {c: (float(text[cats == c].sum()) / tot if (cats == c).any() else 0.0) for c in CATS}
    out['PATHWAY'] = float(path.sum()) / tot
    return out

def compare_caches(R_full, R_frozen, n_evid=N_EVID, subset_keys=None):
    """Paired full-vs-frozen (D1 frozen-LLM) comparison of per-category gene-query attention MASS.
    Records matched by (gene_pair, cell_line, fold). `subset_keys`: optional set of '|'-sorted pair
    strings (e.g. conflict pairs) to restrict to. Returns (DataFrame indexed by category with columns
    full_mean / frozen_mean / delta / pct_full_gt / rank_biserial / wilcoxon_p, N_paired).
    NOTE at N~6.6k Wilcoxon p is tiny for trivial shifts -> read rank_biserial & pct_full_gt for effect."""
    from scipy.stats import wilcoxon
    fz = {_record_key(r): r for r in R_frozen}
    cols = CATS + ['PATHWAY']; paired = {c: [] for c in cols}; N = 0
    for r in R_full:
        k = _record_key(r)
        if k not in fz: continue
        if subset_keys is not None and '|'.join(sorted(r['gene_pair'].split('|'))) not in subset_keys:
            continue
        mf = record_category_mass(r, n_evid); mz = record_category_mass(fz[k], n_evid)
        for c in cols: paired[c].append((mf[c], mz[c]))
        N += 1
    rows = []
    for c in cols:
        a = np.array([x for x, _ in paired[c]]); b = np.array([y for _, y in paired[c]]); d = a - b
        try:    p = float(wilcoxon(a, b, zero_method='wilcox').pvalue) if np.any(d != 0) else 1.0
        except ValueError: p = 1.0
        pos = int((d > 0).sum()); neg = int((d < 0).sum())
        rows.append(dict(category=c, label=CAT_LABEL[c], full_mean=a.mean(), frozen_mean=b.mean(),
                         delta=d.mean(), pct_full_gt=float((d > 0).mean()),
                         rank_biserial=((pos - neg) / (pos + neg) if (pos + neg) else 0.0), wilcoxon_p=p))
    return pd.DataFrame(rows).set_index('category'), N

# ── 2.4.3 : BiomedBERT self-attention concentration, stock vs fine-tuned ───────
# consumes ana/llm_selfattn_bycat.npz produced by ana/gen_llm_selfattn_cache.py:
#   enr_stock / enr_ft : [N, n_layers, n_cats] per-token attention-received enrichment (1.0 = avg token)
BIO_CATS = ('GENENAME', 'EXPR', 'CNV', 'MUT', 'LINEAGE', 'CLNAME')   # keyword-defined biological tokens

def load_selfattn_cache(path=None):
    path = path or MV.SELFATTN_NPZ
    return np.load(path, allow_pickle=True)

def selfattn_layer_profile(d, bio=BIO_CATS):
    """Per-layer mean (over records) enrichment averaged over the biological categories, stock vs ft.
    Returns (n_layers,)-arrays (stock, finetuned)."""
    cats = list(d['cats']); bi = [cats.index(c) for c in bio]
    return d['enr_stock'][:, :, bi].mean(axis=(0, 2)), d['enr_ft'][:, :, bi].mean(axis=(0, 2))

def selfattn_compare(d, layer=-1):
    """Paired stock-vs-fine-tuned per-category self-attention enrichment at one layer (default last).
    Returns DataFrame: stock / finetuned / delta / pct_ft_gt / rank_biserial / wilcoxon_p (read the
    DELTA — absolute values are [CLS]/[SEP]/punctuation-sink dominated in both conditions)."""
    from scipy.stats import wilcoxon
    cats = list(d['cats']); es = d['enr_stock'][:, layer, :]; ef = d['enr_ft'][:, layer, :]
    rows = []
    for ci, c in enumerate(cats):
        a = ef[:, ci]; b = es[:, ci]; dd = a - b
        try:    p = float(wilcoxon(a, b).pvalue) if np.any(dd != 0) else 1.0
        except ValueError: p = 1.0
        pos = int((dd > 0).sum()); neg = int((dd < 0).sum())
        rows.append(dict(category=c, label=CAT_LABEL.get(c, c), stock=b.mean(), finetuned=a.mean(),
                         delta=dd.mean(), pct_ft_gt=float((dd > 0).mean()),
                         rank_biserial=((pos - neg) / (pos + neg) if (pos + neg) else 0.0), wilcoxon_p=p))
    return pd.DataFrame(rows).set_index('category')

# ── 2.4.4 : per-pair content-word decode (SL cell lines) ──────────────────────
_PUNC = set(string.punctuation)
# structural / template words carried by every prompt — masked so only content words remain
STOP = set("""a an the of and is are be in on for to from by with it its these this that has have as
used research study responsible making them useful across drug cell line derived human responses
level number variation copy gene expression not characterized responsible exhibit provide model
important making studying testing understanding developing treatment therapies cancerous mechanisms
efficacy cells structure transport surfaces metabolism tissue substances membranes interactions""".split())

def is_content(t):
    if t in STOP: return False
    if all(ch in _PUNC for ch in t): return False
    return True

def by_pair_index(records):
    d = defaultdict(list)
    for r in records: d['|'.join(sorted(r['gene_pair'].split('|')))].append(r)
    return d

def case_content_tokens(records_by_pair, pair, topn=15, only_sl=True):
    """Top CONTENT text tokens (stopwords/punct removed) by share of text-attn mass, over the pair's
    (SL) cell lines; plus per-category text-mass share. Raw attention is syntax-dominated, so content
    filtering is what exposes the biologically nameable words — reported as illustration, not discovery."""
    key = '|'.join(sorted(pair.split('|')))
    recs = [r for r in records_by_pair.get(key, []) if (r['label'] == 1 if only_sl else True)]
    massW = defaultdict(float); catmass = defaultdict(float); tot = 0.0
    for r in recs:
        gA, gB = r['gene_pair'].split('|')
        cats = label_positions(r['cell_line'], gA, gB); toks, mask = tokens_for(r['cell_line'], gA, gB)
        text = (r['cross_attn_w'][0, :PW_START] + r['cross_attn_w'][1, :PW_START]) / 2.0
        for i in range(PW_START):
            if mask[i] == 0: continue
            massW[toks[i]] += float(text[i]); catmass[cats[i]] += float(text[i]); tot += float(text[i])
    top = [(t, v / tot) for t, v in sorted(massW.items(), key=lambda x: -x[1]) if is_content(t)][:topn]
    catshare = {k: v / tot for k, v in sorted(catmass.items(), key=lambda x: -x[1])}
    cls = sorted(set(r['cell_line'] for r in recs))
    return recs, top, catshare, cls

# ── classification-scheme documentation (for the paper's methods) ─────────────
# Two-level rule: (1) split the prompt into 3 char-bounded segments — cell-line desc / geneA desc /
# geneB desc; (2) within a segment, assign a category by keyword regex (priority order below); tokens
# matching nothing fall back to the segment's prose bucket (CELLBG / GENEBG).
SCHEME_RULES = [
    ('CLNAME',   'cell-line name',    "cell-line aliases, e.g. 'IPC-298' / 'IPC298'",          'any'),
    ('LINEAGE',  'tissue / lineage',  r"lineage keyword list: melanoma|skin|lung|kidney|cancer|oncogenic|malignant|glioblastoma|…", 'any'),
    ('GENENAME', 'gene name',         "the gene symbol itself (DUSP4 / DUSP6), within its own gene segment", 'gene seg'),
    ('EXPR',     'expression phrase', r"'gene expression level of <number>' (phrase + all value subtokens)", 'gene seg'),
    ('CNV',      'copy-number phrase',r"'copy number variation of <number>' (phrase + all value subtokens)", 'gene seg'),
    ('MUT',      'mutation status',   r"'is (not )?mutated'",                                    'gene seg'),
    ('CELLBG',   'cell-line prose',   "fallback: cell-line-segment tokens with no keyword hit (is derived from, used in research, …)", 'cell seg'),
    ('GENEBG',   'gene prose',        "fallback: gene-segment template glue (In the … cell line, has a, and, .)", 'gene seg'),
    ('SPECIAL',  '[CLS]/[SEP]',       "tokenizer special tokens (empty char offset)",           '—'),
    ('PAD',      'padding',           "padding positions (attention_mask==0)",                   '—'),
]
def scheme_table():
    """The classification rule table (category / meaning / rule / scope) as a DataFrame."""
    return pd.DataFrame(SCHEME_RULES, columns=['category', 'meaning', 'rule', 'scope'])

def token_table(cl, gA, gB, segment='gene'):
    """Per-token annotation table for one prompt: pos / token / category / matched-text.
    segment='gene' -> the geneA+geneB region (most illustrative); 'cell' -> cell-line segment; 'all'."""
    full, (ca, da0, da1, db0, db1), _ = build_full_and_spans(cl, gA, gB)
    cats = label_positions(cl, gA, gB)
    enc = tok(full, truncation=True, max_length=512, padding='max_length', return_offsets_mapping=True)
    offs = enc['offset_mapping']; mask = np.array(enc['attention_mask'])
    lo, hi = {'gene': (da0, db1), 'cell': (0, ca), 'all': (0, db1)}[segment]
    rows = []
    for i in range(512):
        if mask[i] == 0: continue
        s, e = offs[i]
        if not (s < hi and e > lo):   # keep tokens overlapping the requested segment
            if not (s == e and lo <= 0 and segment == 'all'): continue
        rows.append({'pos': i, 'token': tok.convert_ids_to_tokens([enc['input_ids'][i]])[0],
                     'category': cats[i], 'matched_text': full[s:e]})
    return pd.DataFrame(rows)

# ── annotated full-sentence render for one concrete example ───────────────────
_PW_BUILDER = None
def _builder():
    global _PW_BUILDER
    if _PW_BUILDER is None:
        from src.pair_pathway import PairPathwayEvidenceBuilder
        _PW_BUILDER = PairPathwayEvidenceBuilder(k=N_EVID)
    return _PW_BUILDER

def selected_pathways(gene_pair, cell_line):
    """Reactome pathway names selected for this (pair, cell_line), in evidence-token order
    (aligned to record['evid_attn'] / cross_attn_w[:,512:]). Deterministic, offline."""
    b = _builder(); g1, _, g2 = str(gene_pair).partition('|')
    pa = b.gene2pws.get(g1, np.array([], np.int64)); pb = b.gene2pws.get(g2, np.array([], np.int64))
    u = np.union1d(pa, pb)
    if u.size == 0: return []
    sh = np.intersect1d(pa, pb, assume_unique=True)
    return [b.pathway_names[int(i)] for i in b._select_topk(u, sh, cell_line)]

# text color per biological category (background encodes attention intensity)
CAT_COLOR = {'GENENAME':'#d62728','LINEAGE':'#1a7a1a','CLNAME':'#7a0177','MUT':'#e6844f',
             'EXPR':'#1f5fbf','CNV':'#0aa0a0','CELLBG':'#333333','GENEBG':'#333333','SPECIAL':'#999999'}
CAT_BOLD = {'GENENAME','LINEAGE','CLNAME','MUT'}

def pick_example(records_by_pair, pair, prefer_sl=True, cell_line=None):
    recs = records_by_pair['|'.join(sorted(pair.split('|')))]
    if cell_line is not None:
        return next(r for r in recs if r['cell_line'] == cell_line)
    if prefer_sl:
        sl = [r for r in recs if r['label'] == 1]
        if sl: return max(sl, key=lambda r: r['pred_score'])   # most confident SL
    return recs[0]

def render_example(rec, ncol=16, cmap='YlOrRd', pctl=99, show_pathway=15):
    """Full annotated figure for one record: (top) the complete input sentence, each token's background
    = gene-query cross-attention, text colour = biological category; (bottom) the pair-conditioned
    Reactome pathway evidence tokens with their attention. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    gA, gB = rec['gene_pair'].split('|'); cl = rec['cell_line']
    cats = label_positions(cl, gA, gB); toks, mask = tokens_for(cl, gA, gB)
    attn = (rec['cross_attn_w'][0, :PW_START] + rec['cross_attn_w'][1, :PW_START]) / 2.0
    valid = np.where(mask == 1)[0]
    toks_v = [toks[i] for i in valid]; attn_v = attn[valid]; cats_v = cats[valid]
    vmax = np.percentile(attn_v, pctl); norm = Normalize(0, vmax)
    cm = plt.get_cmap(cmap)
    nrow = int(np.ceil(len(toks_v) / ncol))

    pws = selected_pathways(rec['gene_pair'], cl); nv = int(rec.get('n_evid_valid', N_EVID))
    pw_attn = rec['evid_attn'][:len(pws)]
    fig = plt.figure(figsize=(16, nrow * 0.52 + 4.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[nrow * 0.52, 3.4], hspace=0.28)
    axT = fig.add_subplot(gs[0]); axP = fig.add_subplot(gs[1])

    # (top) sentence token heatmap
    for k, (t, a, c) in enumerate(zip(toks_v, attn_v, cats_v)):
        row, col = divmod(k, ncol); y = -row
        axT.add_patch(plt.Rectangle((col, y), 1, 1, facecolor=cm(norm(a)), edgecolor='#dddddd', lw=.4))
        axT.text(col + 0.5, y + 0.5, t, ha='center', va='center', fontsize=6.4,
                 color=CAT_COLOR.get(c, '#333333'),
                 fontweight='bold' if c in CAT_BOLD else 'normal')
    axT.set_xlim(0, ncol); axT.set_ylim(-nrow, 1); axT.axis('off')
    axT.set_title(f'{rec["gene_pair"]} @ {cl}   label={"SL" if rec["label"]==1 else "non-SL"}  '
                  f'P(SL)={rec["pred_score"]:.3f}   —   full input sentence, background = gene-query '
                  f'cross-attention', fontweight='bold', fontsize=10.5, loc='left')
    sm = ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axT, fraction=0.025, pad=0.01); cb.set_label('cross-attention', fontsize=8)
    # category legend
    import matplotlib.patches as mpatches
    leg = [mpatches.Patch(color=CAT_COLOR[c], label=CAT_LABEL[c]) for c in
           ['GENENAME','LINEAGE','CLNAME','MUT','EXPR','CNV']]
    axT.legend(handles=leg, loc='lower center', bbox_to_anchor=(0.5, -0.16), ncol=6, fontsize=7.5,
               frameon=False, handlelength=1.0, title='text-token category (font colour)', title_fontsize=8)

    # (bottom) pathway evidence tokens
    if pws:
        idx = np.argsort(pw_attn)[::-1][:show_pathway]
        names = [pws[i].replace('REACTOME_', '')[:52] for i in idx][::-1]
        vals = [pw_attn[i] for i in idx][::-1]
        colr = ['#7a0177' if 'MAPK' in pws[i] or 'ERK' in pws[i] or 'RAF' in pws[i] else '#9e9ac8' for i in idx][::-1]
        axP.barh(range(len(names)), vals, color=colr)
        axP.set_yticks(range(len(names))); axP.set_yticklabels(names, fontsize=7)
        axP.set_xlabel('cross-attention', fontsize=8)
        axP.set_title(f'Pair-conditioned Reactome evidence tokens (K={nv}) — attention '
                      f'(MAPK/ERK/RAF highlighted)', fontweight='bold', fontsize=9.5, loc='left')
    else:
        axP.axis('off'); axP.text(.5, .5, 'no pathway evidence selected', ha='center')
    return fig

if __name__ == '__main__':
    R = load_cache('pair_K16')
    ok, ch = verify_alignment(R); print(f"alignment check: {ok}/{ch} masks elementwise-equal")
    df, counts, N, mv = category_enrichment(R)
    print(f"\nPPE K16  N={N}  mean valid tokens={mv:.0f} — enrichment vs avg valid token (1.0)")
    print(df.round(2).to_string())
    share, _ = mass_split(R)
    print("\nattention MASS share (text cats + pathway):", {k: round(v, 3) for k, v in share.items()})
    bp = by_pair_index(R)
    for pair in ['DUSP4|DUSP6', 'AKT2|RPS6KB1']:
        _, top, cs, cls = case_content_tokens(bp, pair)
        print(f"\n{pair} SL@{cls} top content tokens:", [f'{t}:{v*100:.1f}%' for t, v in top[:8]])
        print("   cat share:", {k: round(v, 2) for k, v in cs.items()})
