"""Build the per-(gene pair, cell line) pathway-attention substrate.

Reads a cached gene->context attention pickle and, for every record, expands the
selected Reactome pathway-evidence tokens into a tidy table used by the mechanism
analysis. Also exposes the ordered Reactome pathway name list and the pathway
activity / membership tables.
"""

import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import MULTI_OMICS_DIR

MELANOMA = {"IPC298", "MEWO", "HS936T"}

with open(f"{MULTI_OMICS_DIR}/reactome_pathways.json") as _f:
    _pw = json.load(_f)
PNAMES = _pw["pathway_names"]
PSIZE = np.asarray(_pw["pathway_sizes"], dtype=np.int64)
GENE2PW = {g: np.asarray(sorted(v), dtype=np.int64) for g, v in _pw["gene2pathways"].items()}

with open(f"{MULTI_OMICS_DIR}/reactome_activity.json") as _f:
    ACTIVITY = {cl: np.asarray(v["scores"], dtype=np.float32) for cl, v in json.load(_f).items()}

_EMPTY = np.empty(0, dtype=np.int64)


def _select_topk(union, shared, cell_line, k):
    """Deterministic top-k pathway selection (shared first, then |expr|, then size)."""
    if len(union) <= k:
        return union
    expr = ACTIVITY[cell_line][union, 0]
    is_shared = np.isin(union, shared)
    size = PSIZE[union]
    order = np.lexsort((union, size, -np.abs(expr), (~is_shared).astype(np.int8)))
    return union[order[:k]]


def build_substrate(cache, k):
    """Expand a cached attention pickle into (records, subset_map, long DataFrame).

    Each output row is one selected pathway-evidence token for one record, with its
    attention weight, pathway size, shared-membership flag and |activity|.
    """
    records = pickle.load(open(cache, "rb"))

    pair_cl_label = defaultdict(lambda: defaultdict(list))
    for r in records:
        pair_cl_label[r["gene_pair"]][r["cell_line"]].append(r["label"])
    subset = {
        gp: (
            "single"
            if len(m) < 2
            else "conflict"
            if len({int(round(np.mean(v))) for v in m.values()}) > 1
            else "consistent"
        )
        for gp, m in pair_cl_label.items()
    }

    rows = []
    for r in records:
        g1, _, g2 = r["gene_pair"].partition("|")
        pa = GENE2PW.get(g1, _EMPTY)
        pb = GENE2PW.get(g2, _EMPTY)
        union = np.union1d(pa, pb)
        if union.size == 0:
            continue
        shared = np.intersect1d(pa, pb, assume_unique=True)
        selected = _select_topk(union, shared, r["cell_line"], k)
        attn = r["evid_attn"][: len(selected)]
        for pos, (pidx, a) in enumerate(zip(selected, attn)):
            rows.append(
                (
                    r["gene_pair"],
                    r["cell_line"],
                    "mel" if r["cell_line"] in MELANOMA else "nonmel",
                    r["label"],
                    subset[r["gene_pair"]],
                    pos,
                    int(pidx),
                    float(a),
                    int(PSIZE[pidx]),
                    int(pidx in shared),
                    float(abs(ACTIVITY[r["cell_line"]][pidx, 0])),
                )
            )

    pdf = pd.DataFrame(
        rows,
        columns=["pair", "cl", "lin", "label", "subset", "pos", "pidx", "attn", "size", "shared", "abs_expr"],
    )
    return records, subset, pdf
