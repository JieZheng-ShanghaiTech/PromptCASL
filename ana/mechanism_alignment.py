"""Compare the attended Reactome pathways against literature SL mechanisms."""
import os, sys, re, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, f'{BASE}/ana')
import lit_mech_casestudy as L          # noqa: E402
import model_version as MV              # noqa: E402  active run + split selection
from attn_substrate import build_substrate  # noqa: E402
from src.config import BIOMEDBERT, MULTI_OMICS_DIR  # noqa: E402

with open(os.path.join(MULTI_OMICS_DIR, 'reactome_pathways.json')) as _f:
    PNAMES = json.load(_f)['pathway_names']          # Reactome pathway names (pidx-indexed)
BIOBERT = BIOMEDBERT
CACHE_EMB = MV.MECH_EMB_NPZ
ATTN_PKL  = MV.ATTN_PKL
OUT_CSV   = MV.MECH_ALIGN_CSV

# semantic-alignment thresholds
SEM_PCTILE_ALIGNED = 0.90   # top-attended pathway among top-10% most mechanism-similar
KW_TOPN = 5                 # keyword hit counts if in the top-N attended pathways
MAXK_NULL_B = 1000          # bootstrap draws for the top-k random null (see null_topk below)
_NULL_RNG = np.random.default_rng(0)


def humanize(name):
    return name[len('REACTOME_'):].replace('_', ' ').lower()


def _encode(texts, device='cuda:0', bs=32):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BIOBERT)
    model = AutoModel.from_pretrained(BIOBERT).to(device).eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=256,
                  return_tensors='pt').to(device)
        with torch.no_grad():
            hs = model(**enc).last_hidden_state
        m = enc.attention_mask.unsqueeze(-1).float()
        out.append(((hs * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def build_embeddings(device='cuda:0', force=False):
    """Encode pathway names + solid-pair mechanism texts in ONE common BiomedBERT space.
    Center by the pathway-bank mean (matches reactome_desc_emb de-anisotropization), L2."""
    cand = L.build_candidates()
    solid = cand[cand.solid].reset_index(drop=True)
    if os.path.exists(CACHE_EMB) and not force:
        z = np.load(CACHE_EMB, allow_pickle=True)
        return z['pathway_emb'], z['mech_emb'], list(z['pk']), solid
    pw_texts = [humanize(n) for n in PNAMES]
    mech_texts = solid.mechanism.fillna('').tolist()
    emb = _encode(pw_texts + mech_texts, device)
    center = emb[:len(pw_texts)].mean(0, keepdims=True)      # center on pathway bank
    emb = (emb - center)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    pw_emb, mech_emb = emb[:len(pw_texts)], emb[len(pw_texts):]
    np.savez_compressed(CACHE_EMB, pathway_emb=pw_emb, mech_emb=mech_emb, pk=np.array(solid.pk))
    print(f'cached embeddings -> {CACHE_EMB}  pathways={pw_emb.shape} mech={mech_emb.shape}')
    return pw_emb, mech_emb, list(solid.pk), solid


def _theme_pattern(theme):
    for name, pat in L.THEMES:
        if name == theme:
            return pat
    return None


def build_alignment(device='cuda:0', force=False):
    pw_emb, mech_emb, pks, solid = build_embeddings(device, force)
    solid = solid.set_index('pk')
    mech_by_pk = {pk: mech_emb[i] for i, pk in enumerate(pks)}

    _, _, PDF = build_substrate(ATTN_PKL, 16)
    PDF['pk'] = PDF.pair.map(lambda g: '|'.join(sorted(str(g).split('|'))))

    rows = []
    for pk in pks:
        sub = PDF[(PDF.pk == pk) & (PDF.label == 1)]     # SL cell lines only
        if len(sub) == 0:
            continue
        a = sub.groupby('pidx').attn.mean()               # mean attention per selected pathway
        pidx = a.index.values.astype(int)
        attn = a.values.astype(float)
        if len(pidx) == 0:
            continue
        mvec = mech_by_pk[pk]
        cos = pw_emb[pidx] @ mvec                          # cosine mechanism <-> each selected pathway
        allcos = pw_emb @ mvec                             # vs all 1052 pathways (null reference)
        pct = np.array([(allcos < c).mean() for c in cos]) # percentile of each selected pathway

        order = np.argsort(-attn)                          # attention-descending
        top1 = order[0]
        top1_cos = float(cos[top1])
        top1_pctile = float(pct[top1])                     # how mechanism-specific the #1 attention is
        topk = order[:min(KW_TOPN, len(order))]
        sem_topk_pctile = float(max(pct[o] for o in topk))
        # proper "vs chance" reference for the LENIENT metric: max percentile of KW_TOPN RANDOM
        # selected pathways. Chance for a max-of-k is ~k/(k+1) (≈0.83 for k=5), NOT 0.5 — comparing
        # sem_topk_pctile to 0.5 is a mechanical artifact. Per-pair mean over MAXK_NULL_B draws.
        kk = min(KW_TOPN, len(pct))
        if len(pct) > kk:
            null_topk = float(np.mean([pct[_NULL_RNG.choice(len(pct), kk, replace=False)].max()
                                       for _ in range(MAXK_NULL_B)]))
        else:
            null_topk = float(pct.max())
        # does attention up-weight mechanism-aligned pathways?
        wcos = float((attn * cos).sum() / attn.sum())
        ucos = float(cos.mean())
        sem_gain = wcos - ucos
        rho = float(spearmanr(attn, cos)[0]) if len(attn) >= 4 else np.nan

        # keyword scoring (theme keyword set on attended pathway names, attention order)
        pat = _theme_pattern(solid.loc[pk, 'theme'])
        kw_rank = np.nan
        if pat:
            for rank, o in enumerate(order, 1):
                if re.search(pat, humanize(PNAMES[pidx[o]]), re.I):
                    kw_rank = rank
                    break
        rows.append(dict(
            pk=pk, subset=solid.loc[pk, 'subset'], theme=solid.loc[pk, 'theme'],
            sl_prob=solid.loc[pk, 'sl_prob'], hit=bool(solid.loc[pk, 'hit']),
            n_sel=len(pidx),
            top1_pathway=PNAMES[pidx[top1]].replace('REACTOME_', ''),
            top1_cos=round(top1_cos, 3), top1_pctile=round(top1_pctile, 3),
            sem_topk_pctile=round(sem_topk_pctile, 3), null_topk=round(null_topk, 4),
            sem_gain=round(sem_gain, 4), attn_cos_rho=round(rho, 3) if rho == rho else np.nan,
            kw_rank=kw_rank,
            sem_aligned=top1_pctile >= SEM_PCTILE_ALIGNED,
            kw_aligned=(kw_rank == kw_rank) and (kw_rank <= KW_TOPN),
        ))
    tab = pd.DataFrame(rows)
    tab['both_aligned'] = tab.sem_aligned & tab.kw_aligned
    tab['either_aligned'] = tab.sem_aligned | tab.kw_aligned
    return tab.sort_values(['sem_aligned', 'top1_pctile'], ascending=[False, False]).reset_index(drop=True)


def pair_detail(pk, device='cuda:0'):
    """Per-pair selected-pathway table for plotting: name, mean attention (SL cells),
    mechanism-cosine, and its percentile among all pathways. Sorted by attention desc."""
    pw_emb, mech_emb, pks, solid = build_embeddings(device)
    if pk not in pks:
        return pd.DataFrame()
    mvec = mech_emb[pks.index(pk)]
    _, _, PDF = build_substrate(ATTN_PKL, 16)
    PDF['pk'] = PDF.pair.map(lambda g: '|'.join(sorted(str(g).split('|'))))
    sub = PDF[(PDF.pk == pk) & (PDF.label == 1)]
    if len(sub) == 0:
        return pd.DataFrame()
    a = sub.groupby('pidx').attn.mean()
    pidx = a.index.values.astype(int)
    allcos = pw_emb @ mvec
    cos = allcos[pidx]
    df = pd.DataFrame(dict(
        pathway=[PNAMES[i].replace('REACTOME_', '') for i in pidx],
        attn=a.values, mech_cos=cos,
        cos_pctile=[float((allcos < c).mean()) for c in cos],
    )).sort_values('attn', ascending=False).reset_index(drop=True)
    return df


if __name__ == '__main__':
    from scipy.stats import wilcoxon
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda:0'
    tab = build_alignment(device=dev, force='--force' in sys.argv)
    tab.to_csv(OUT_CSV, index=False)
    n = len(tab)
    print(f'\n[SPLIT={MV.SPLIT}] wrote {n} solid pairs with attention -> {OUT_CSV}')
    print(f'  semantic-aligned (top1 in top-{int((1-SEM_PCTILE_ALIGNED)*100)}% mechanism-similar): '
          f'{tab.sem_aligned.sum()} ({tab.sem_aligned.mean():.0%})')
    print(f'  keyword-aligned  (theme hit in top-{KW_TOPN} attended):                    '
          f'{tab.kw_aligned.sum()} ({tab.kw_aligned.mean():.0%})')
    print(f'  both:  {tab.both_aligned.sum()}   either: {tab.either_aligned.sum()}')
    # STRICT: top1 percentile vs chance 0.50
    p_top1 = wilcoxon(tab.top1_pctile - 0.5, alternative='greater').pvalue
    print(f'  STRICT  mean top1_pctile      = {tab.top1_pctile.mean():.3f}  (chance 0.50)   Wilcoxon p={p_top1:.2e}')
    # LENIENT: top-5 percentile vs the PROPER max-of-5 random null (per-pair), not 0.5
    p_topk = wilcoxon(tab.sem_topk_pctile - tab.null_topk, alternative='greater').pvalue
    print(f'  LENIENT mean sem_topk_pctile  = {tab.sem_topk_pctile.mean():.3f}  '
          f'(max-of-{KW_TOPN} null {tab.null_topk.mean():.3f})   Wilcoxon p={p_topk:.2e}')
    print(f'  mean sem_gain (attn-weighted - uniform cosine): {tab.sem_gain.mean():+.4f}')
