"""Generate the gene->context attention cache for a pair_pathway_evidence model."""
import os, sys, argparse, pickle
import numpy as np
import torch
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import (LLMEncoder, GeneAttentionAggregator, GenePairContextAttention,
                       Predictor, TextPathwayFusion, PairPathwayEvidenceEncoder)
from src.pair_pathway import PairPathwayEvidenceBuilder
from src.config import BIOMEDBERT, DATASETS_DIR

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIOBERT_PATH = BIOMEDBERT
DATASET_DIR = os.path.join(DATASETS_DIR, 'cell_line_biomedbert_kg50shared')
CELL_LINES = ['A549', 'PK1', 'IPC298', 'MEWO', 'GI1', 'HS936T']
FOLDS = list(range(1, 6))
PW_START = 512
GENEAGG_IN, GENEAGG_HIDDEN, GENEAGG_FF = 3072, 768, 1024
PRED_IN, PRED_HIDDEN, DROPOUT = 5376, 1024, 0.1


class GenePairContextAttentionWithAttn(GenePairContextAttention):
    """Subclass that additionally returns the cross/self attention weights;
    the state_dict is fully compatible with the parent class."""
    def forward(self, expressed_geneA, geneA_agg, expressed_geneB, geneB_agg,
                context_tokens, context_mask):
        geneA_token = self.gene_token_norm(expressed_geneA + geneA_agg)
        geneB_token = self.gene_token_norm(expressed_geneB + geneB_agg)
        gene_query = torch.stack([geneA_token, geneB_token], dim=1)
        sa_out, self_attn_w = self.gene_self_attn(gene_query, gene_query, gene_query, need_weights=True)
        gene_query = self.norm_self(gene_query + sa_out)
        ctx_out, cross_attn_w = self.gene_cross_attn(
            gene_query, context_tokens, context_tokens,
            key_padding_mask=context_mask, need_weights=True)
        gene_query = self.norm_cross(gene_query + ctx_out)
        gene_query = self.norm_ff(gene_query + self.ff(gene_query))
        return gene_query, cross_attn_w, self_attn_w


def load_models(model_dir, k, device):
    builder = PairPathwayEvidenceBuilder(k=k)
    out = {'builder': builder, 'folds': {}}
    for fold in FOLDS:
        ckpt = torch.load(f'{model_dir}/model_MCL_F{fold}.pt', map_location=device, weights_only=False)
        base = AutoModel.from_pretrained(BIOBERT_PATH)
        llm = LLMEncoder(base, model_type='bert').to(device); llm.load_state_dict(ckpt['llm_encoder'])
        agg = GeneAttentionAggregator(GENEAGG_IN, GENEAGG_HIDDEN, GENEAGG_FF).to(device)
        agg.load_state_dict(ckpt['gene_agg'])
        ctx = GenePairContextAttentionWithAttn(hidden_dim=GENEAGG_HIDDEN, ff_dim=GENEAGG_FF, dropout=DROPOUT).to(device)
        ctx.load_state_dict(ckpt['gene_ctx_attn'])
        pred = Predictor(PRED_IN, PRED_HIDDEN).to(device); pred.load_state_dict(ckpt['predictor'])
        fusion = TextPathwayFusion(hidden_dim=768, nhead=8, ff_dim=512).to(device)
        fusion.load_state_dict(ckpt['text_pathway_fusion'])
        pw_enc = PairPathwayEvidenceEncoder(
            n_features=builder.n_features, desc_dim=builder.desc_dim, output_dim=768).to(device)
        pw_enc.load_state_dict(ckpt['pathway_encoder'])
        for m in [llm, agg, ctx, pred, fusion, pw_enc]:
            m.eval()
        out['folds'][fold] = dict(llm=llm, agg=agg, ctx=ctx, pred=pred, fusion=fusion, pw_enc=pw_enc)
    return out


def run_fold(builder, fold, M, k, device, batch_size=32):
    llm, agg, ctx, pred = M['llm'], M['agg'], M['ctx'], M['pred']
    fusion, pw_enc = M['fusion'], M['pw_enc']
    recs = []
    for cl in CELL_LINES:
        pt = f'{DATASET_DIR}/{cl}_F{fold}.pt'
        if not os.path.exists(pt):
            print(f'  [skip] {pt}'); continue
        ds = torch.load(pt, map_location='cpu', weights_only=False)
        for start in range(0, len(ds), batch_size):
            batch = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            B = len(batch)
            embs = [it[0] for it in batch]
            labels = torch.stack([it[1] for it in batch]); metas = [it[2] for it in batch]
            tok = torch.stack([e[0] for e in embs]).to(device)
            mask = torch.stack([e[1] for e in embs]).to(device)
            gA_gpt = torch.stack([e[2] for e in embs]).to(device); gA_kg = torch.stack([e[3] for e in embs]).to(device)
            gB_gpt = torch.stack([e[4] for e in embs]).to(device); gB_kg = torch.stack([e[5] for e in embs]).to(device)
            gA_nb = torch.stack([e[6] for e in embs]).to(device); gB_nb = torch.stack([e[7] for e in embs]).to(device)
            gp_list = [gp for gp, _ in metas]; cl_list = [c for _, c in metas]
            with torch.no_grad():
                token_seq = llm(tok, mask); text_pad = (mask == 0)
                pp_num, pp_desc, pp_pad = builder.build_batch(gp_list, cl_list, device)
                pw_tok = pw_enc(pp_num, pp_desc, pp_pad)
                ctx_tokens = fusion(token_seq, pw_tok, text_pad, pathway_padding_mask=pp_pad)
                ctx_mask = torch.cat([text_pad, pp_pad], dim=1)
                valid_evid = (~pp_pad)
                exp_gA, gA_agg = agg(gA_gpt, gA_nb, gB_nb, gA_kg, gB_kg)
                exp_gB, gB_agg = agg(gB_gpt, gB_nb, gA_nb, gB_kg, gA_kg)
                gq, cross_w, self_w = ctx(exp_gA, gA_agg, exp_gB, gB_agg, ctx_tokens, ctx_mask)
                a = gq[:, 0, :]; b = gq[:, 1, :]
                pair_repr = torch.cat([a + b, (a - b).abs(), a * b], dim=-1)
                final_repr = torch.cat([pair_repr, exp_gA, gA_agg, exp_gB, gB_agg], dim=-1)
                probs = torch.softmax(pred(final_repr), dim=1)[:, 1]
            cw = cross_w.cpu().numpy(); sw = self_w.cpu().numpy(); ve = valid_evid.cpu().numpy()
            for j in range(B):
                ev_avg = (cw[j, 0, PW_START:] + cw[j, 1, PW_START:]) / 2.0
                ev_avg = ev_avg * ve[j]
                er = float(ev_avg.sum())
                recs.append({'gene_pair': metas[j][0], 'cell_line': metas[j][1],
                             'label': int(labels[j].item()), 'pred_score': float(probs[j].item()),
                             'fold': fold, 'evidence_ratio': er, 'text_ratio': 1.0 - er,
                             'n_evid_valid': int(ve[j].sum()), 'evid_attn': ev_avg,
                             'cross_attn_w': cw[j], 'self_attn_w': sw[j]})
    return recs


def main():
    global DATASET_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--dataset_dir', default=None,
                    help='Per-fold {cl}_F{fold}.pt directory; default = standard split. For GPLO pass '
                         'data/datasets/cell_line_biomedbert_kg50shared_GPLO')
    a = ap.parse_args()
    if a.dataset_dir:
        DATASET_DIR = a.dataset_dir if os.path.isabs(a.dataset_dir) else os.path.join(BASE, a.dataset_dir)
        print(f'dataset_dir override -> {DATASET_DIR}')
    device = f'cuda:{a.device}'
    print(f'loading {a.model_dir}  (K={a.k}) ...')
    M = load_models(a.model_dir, a.k, device)
    recs = []
    for fold in FOLDS:
        print(f'  fold {fold} ...')
        recs.extend(run_fold(M['builder'], fold, M['folds'][fold], a.k, device))
    with open(a.out, 'wb') as f:
        pickle.dump(recs, f)
    print(f'cached {len(recs)} -> {a.out}')


if __name__ == '__main__':
    main()
