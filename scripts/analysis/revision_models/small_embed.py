#!/usr/bin/env python3
"""Embed TCR2vec_small (full-length, 128-d) and CDR3vec_small (CDR3-only, 128-d).
Anchor-included model input (C..F restored); IMGT-stripped CDR3 as offline key. tcr2vec_env.
Usage: python small_embed.py <input_csv> <dataset:mcpas|tcr> <outdir>"""
import sys, os, json, time, numpy as np, pandas as pd, torch
torch.set_num_threads(6)
from tcr2vec.model import TCR2vec
from tcr2vec.dataset import TCRLabeledDset
from torch.utils.data import DataLoader
from tcr2vec.utils import get_emb
from tcr2vec.cdr3_to_full_seq import to_full_seq
import tcr2vec
GENE_DIR = os.path.join(os.path.dirname(tcr2vec.__file__), "data", "TCR_gene_segment_data")
W_FULL = os.environ.get("TCR2VEC_WEIGHTS","weights/tcr2vec_weights")+"/tcr2vec_small_dir/TCR2vec_small_128"
W_CDR3 = os.environ.get("TCR2VEC_WEIGHTS","weights/tcr2vec_weights")+"/cdr3vec_small_dir/CDR3vec_small_128"
inp, dataset, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(inp)
cdr3s = df["CDR3.beta"].astype(str).tolist()          # stripped -> key
Vs, Js = df["V"].tolist(), df["J"].tolist()
AA = set('ACDEFGHIKLMNPQRSTVWY')
def sani(x): return ''.join(c for c in str(x).upper() if c in AA) or 'A'
def add_anchors(c):
    c = str(c).upper()
    if not c: return c
    if c[0] != 'C': c = 'C' + c
    if c[-1] not in 'FW': c = c + 'F'
    return c
def recon(v, j, c):
    try:
        if not isinstance(v, str) or not isinstance(j, str): return c
        out = to_full_seq(GENE_DIR, v, j, c); fs = str(out[0] if isinstance(out, tuple) else out)
        return fs if (fs and c in fs) else c
    except Exception: return c
anch = [add_anchors(c) for c in cdr3s]
fulls = [sani(recon(v, j, c)) for v, j, c in zip(Vs, Js, anch)]
cdr3_in = [sani(c) for c in anch]

def embed(W, seqs):
    m = TCR2vec(W); m.eval()
    d = TCRLabeledDset(list(seqs), only_tcr=True)
    return get_emb(m, DataLoader(d, batch_size=128, collate_fn=d.collate_fn, shuffle=False), detach=True).astype(np.float32)

for tag, W, seqs in [("tcr2vec-small", W_FULL, fulls), ("cdr3vec-small", W_CDR3, cdr3_in)]:
    t = time.time(); emb = embed(W, seqs)
    stem = f"{dataset}_level1_{tag}_mean"
    np.save(os.path.join(outdir, stem + ".npy"), emb)
    json.dump({"dataset": dataset, "level": "level1", "model": tag, "pooling": "mean",
               "n_sequences": len(cdr3s), "embedding_dim": int(emb.shape[1]), "sequence_order": cdr3s},
              open(os.path.join(outdir, stem + "_metadata.json"), "w"))
    print(f"[{dataset}] {tag} {emb.shape} wrote {stem} ({time.time()-t:.0f}s)", flush=True)
print(f"[{dataset}] SMALL_EMBED_DONE", flush=True)
