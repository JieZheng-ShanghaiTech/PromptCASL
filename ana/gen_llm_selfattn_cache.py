"""Extract per-category BiomedBERT self-attention (stock vs fine-tuned)."""
import os, sys, argparse, time
import numpy as np
import torch
from transformers import AutoModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'ana'))
from src.model import LLMEncoder
from src.config import BIOMEDBERT
import text_pathway_attn as T          # label_positions, CATS, CELL_LINES (same category scheme)
import model_version as MV             # active run + split selection

BIOBERT = BIOMEDBERT
DATASET_DIR = MV.DATASET_DIR        # SPLIT-aware: std vs GPLO shards (must match FT_CKPT's split)
FT_CKPT = MV.STD_CKPT               # active alias -> CKPT_GPLO under PROMPTCASL_SPLIT=gplo
CELL_LINES = T.CELL_LINES
FOLDS = list(range(1, 6))
CATS = T.CATS


def make_llm(state_dict, device):
    base = AutoModel.from_pretrained(BIOBERT)
    llm = LLMEncoder(base, model_type='bert').to(device)
    if state_dict is not None:
        llm.load_state_dict(state_dict)     # fine-tuned encoder; None -> stock BiomedBERT
    llm.eval()
    return llm


def encoder_attentions(llm, token_emb, mask):
    """Return tuple(len L) of [B,H,S,S] attention weights from the BERT encoder (LLMEncoder's mask rule)."""
    ext = mask[:, None, None, :].to(dtype=token_emb.dtype)
    ext = (1.0 - ext) * -10000.0
    out = llm.model.encoder(token_emb, attention_mask=ext, output_attentions=True, return_dict=True)
    return out.attentions


def enrich_batch(attns, maskb_np, catmask, catn, nv_np):
    """attns: tuple(L) of [B,H,S,S] on device. Returns [B, L, C] per-token enrichment (numpy)."""
    B = maskb_np.shape[0]; L = len(attns)
    out = np.zeros((B, L, len(CATS)), np.float32)
    for li, Al in enumerate(attns):
        Ah = Al.mean(1).detach().cpu().numpy()           # [B,S,S] mean over heads
        for j in range(B):
            n = nv_np[j]
            if not n:
                continue
            recv = Ah[j][maskb_np[j]].sum(0) / n          # [S] attention received per key
            for ci in range(len(CATS)):
                nc = catn[j][ci]
                if nc:
                    out[j, li, ci] = recv[catmask[j][ci]].sum() / nc * n
    return out


def main():
    global DATASET_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--batch', type=int, default=8)     # attns are big; keep small
    ap.add_argument('--out', default=MV.SELFATTN_NPZ)
    ap.add_argument('--dataset_dir', default=None,
                    help='per-fold {cl}_F{fold}.pt dir; default = MV.DATASET_DIR (SPLIT-aware)')
    a = ap.parse_args()
    if a.dataset_dir:
        DATASET_DIR = a.dataset_dir if os.path.isabs(a.dataset_dir) else os.path.join(BASE, a.dataset_dir)
    print(f'SPLIT={MV.SPLIT}\n  ckpt   = {FT_CKPT}\n  dataset= {DATASET_DIR}\n  out    = {a.out}')
    device = f'cuda:{a.device}'

    stock = make_llm(None, device)                       # off-the-shelf BiomedBERT (fold-independent)
    acc = {k: [] for k in ['enr_stock', 'enr_ft', 'cell_line', 'gene_pair', 'label', 'fold']}
    t0 = time.time()
    for fold in FOLDS:
        ck = torch.load(f'{FT_CKPT}/model_MCL_F{fold}.pt', map_location='cpu', weights_only=False)
        ft = make_llm(ck['llm_encoder'], device)
        for cl in CELL_LINES:
            pt = f'{DATASET_DIR}/{cl}_F{fold}.pt'
            if not os.path.exists(pt):
                print(f'  [skip] {pt}'); continue
            ds = torch.load(pt, map_location='cpu', weights_only=False)
            n = len(ds)
            for s in range(0, n, a.batch):
                batch = [ds[i] for i in range(s, min(s + a.batch, n))]
                emb = [it[0] for it in batch]; metas = [it[2] for it in batch]
                labels = [int(it[1].item()) if torch.is_tensor(it[1]) else int(it[1]) for it in batch]
                tok = torch.stack([e[0] for e in emb]).to(device)
                mask = torch.stack([e[1] for e in emb]).to(device)
                maskb_np = (mask == 1).detach().cpu().numpy()
                nv_np = maskb_np.sum(1).astype(int)
                # per-record category masks (label_positions is lru_cached on (cl,gA,gB))
                catmask, catn = [], []
                for mt in metas:
                    cats = T.label_positions(mt[1], *mt[0].split('|'))
                    cm = [(cats == c) for c in CATS]
                    catmask.append(cm); catn.append([int(m.sum()) for m in cm])
                with torch.no_grad():
                    e_s = enrich_batch(encoder_attentions(stock, tok, mask), maskb_np, catmask, catn, nv_np)
                    e_f = enrich_batch(encoder_attentions(ft, tok, mask), maskb_np, catmask, catn, nv_np)
                acc['enr_stock'].append(e_s); acc['enr_ft'].append(e_f)
                acc['cell_line'] += [mt[1] for mt in metas]
                acc['gene_pair'] += [mt[0] for mt in metas]
                acc['label'] += labels; acc['fold'] += [fold] * len(batch)
            del ds
        del ft
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f'  fold {fold} done  ({time.time() - t0:.0f}s)')

    np.savez_compressed(
        a.out,
        enr_stock=np.concatenate(acc['enr_stock']).astype(np.float32),
        enr_ft=np.concatenate(acc['enr_ft']).astype(np.float32),
        cats=np.array(CATS), cell_line=np.array(acc['cell_line']),
        gene_pair=np.array(acc['gene_pair']), label=np.array(acc['label'], np.int64),
        fold=np.array(acc['fold'], np.int64),
    )
    print(f'saved {np.concatenate(acc["enr_stock"]).shape} -> {a.out}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
