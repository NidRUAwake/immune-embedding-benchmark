#!/usr/bin/env python3
"""Embed the real CDR3vec (CDR3-only pretrained TCR2vec variant) for McPAS+VDJdb.
Anchor-included CDR3 input (model was pretrained on C...F CDR3), stripped CDR3 as offline key.
Run in tcr2vec_env. Writes {ds}_level1_cdr3vec_mean.npy (+ metadata)."""
import sys, os, json, time, numpy as np, pandas as pd, torch
torch.set_num_threads(6)
from tcr2vec.model import TCR2vec
from tcr2vec.dataset import TCRLabeledDset
from torch.utils.data import DataLoader
from tcr2vec.utils import get_emb
W = os.environ.get("TCR2VEC_WEIGHTS","weights/tcr2vec_weights")+"/cdr3vec_dir/CDR3vec_120"
inp, dataset, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(inp)
cdr3s = df["CDR3.beta"].astype(str).tolist()   # stripped -> offline key
AA = set('ACDEFGHIKLMNPQRSTVWY')
def add_anchors(c):
    c = str(c).upper()
    if not c: return c
    if c[0] != 'C': c = 'C' + c
    if c[-1] not in 'FW': c = c + 'F'
    return ''.join(ch for ch in c if ch in AA) or 'A'
model_in = [add_anchors(c) for c in cdr3s]
m = TCR2vec(W); m.eval()
d = TCRLabeledDset(list(model_in), only_tcr=True)
emb = get_emb(m, DataLoader(d, batch_size=128, collate_fn=d.collate_fn, shuffle=False), detach=True).astype(np.float32)
stem = f"{dataset}_level1_cdr3vec_mean"
np.save(os.path.join(outdir, stem + ".npy"), emb)
json.dump({"dataset": dataset, "level": "level1", "model": "cdr3vec", "pooling": "mean",
           "n_sequences": len(cdr3s), "embedding_dim": int(emb.shape[1]), "sequence_order": cdr3s},
          open(os.path.join(outdir, stem + "_metadata.json"), "w"))
print(f"[{dataset}] cdr3vec {emb.shape} wrote {stem}", flush=True)
