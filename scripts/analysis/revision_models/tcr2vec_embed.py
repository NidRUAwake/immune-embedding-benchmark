#!/usr/bin/env python3
"""Generate TCR2vec embeddings (offline .npy + metadata) for a TCR working set.
Run in the isolated tcr2vec_env. Two conditions, both keyed by CDR3 (level1 order):
  - tcr2vec-full : embed the reconstructed full-length V-CDR3-J sequence (native TCR2vec input)
  - tcr2vec-cdr3 : embed the CDR3 only, same model (input ablation -> isolates germline-V/J context)
Usage: python tcr2vec_embed.py <input_csv> <dataset:tcr|mcpas> <outdir>
  input_csv cols: CDR3.beta, V, J
Writes <outdir>/{dataset}_level1_tcr2vec-full_mean.npy (+ _metadata.json) and _tcr2vec-cdr3_.
"""
import sys, os, json, time, numpy as np, pandas as pd
import torch
torch.set_num_threads(6)
from tcr2vec.model import TCR2vec
from tcr2vec.dataset import TCRLabeledDset
from torch.utils.data import DataLoader
from tcr2vec.utils import get_emb
from tcr2vec.cdr3_to_full_seq import to_full_seq
import tcr2vec
PKG = os.path.dirname(tcr2vec.__file__)
GENE_DIR = os.path.join(PKG, "data", "TCR_gene_segment_data")
W = os.environ.get("TCR2VEC_WEIGHTS","weights/tcr2vec_weights")+"/tcr2vec_full_dir/TCR2vec_120"

inp, dataset, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)
df = pd.read_csv(inp)
cdr3s = df["CDR3.beta"].astype(str).tolist()   # IMGT-stripped (105-117): the pipeline/split KEY
Vs = df["V"].tolist(); Js = df["J"].tolist()
t0 = time.time()

# TCR2vec/CDR3vec were pretrained on anchor-INCLUDED CDR3 (C...F), like SCEPTR (scratch/run_sceptr_embed.py
# clean_cdr3). Restore the 104C / 118F anchors before feeding the MODEL, but keep the stripped CDR3 as the
# offline-lookup key so retrieval aligns to the canonical clone-aware split exactly as every other method.
def add_anchors(c):
    c = str(c).upper()
    if not c: return c
    if c[0] != 'C': c = 'C' + c
    if c[-1] not in 'FW': c = c + 'F'
    return c
cdr3_model = [add_anchors(c) for c in cdr3s]   # anchor-included, model input

# 1. reconstruct full-length (fallback to CDR3 if reconstruction fails/empty)
def recon(v, j, c):
    try:
        if not isinstance(v, str) or not isinstance(j, str):
            return c
        out = to_full_seq(GENE_DIR, v, j, c)
        fs = str(out[0] if isinstance(out, tuple) else out)
        return fs if (fs and c in fs) else c
    except Exception:
        return c
AA = set('ACDEFGHIKLMNPQRSTVWY')
def sani(x):
    return ''.join(ch for ch in str(x).upper() if ch in AA) or 'A'
fulls = [sani(recon(v, j, c)) for v, j, c in zip(Vs, Js, cdr3_model)]
nfail = sum(1 for f, c in zip(fulls, cdr3_model) if f == sani(c))
print(f"[{dataset}] reconstructed {len(fulls)} full-length ({nfail} fell back to CDR3) in {time.time()-t0:.0f}s", flush=True)

model = TCR2vec(W); model.eval()
print(f"[{dataset}] model loaded {time.time()-t0:.0f}s", flush=True)

def embed(seqs, tag):
    ts = time.time()
    d = TCRLabeledDset(list(seqs), only_tcr=True)
    loader = DataLoader(d, batch_size=128, collate_fn=d.collate_fn, shuffle=False)
    emb = get_emb(model, loader, detach=True).astype(np.float32)
    print(f"[{dataset}] {tag}: {emb.shape} in {time.time()-ts:.0f}s", flush=True)
    return emb

for tag, seqs in [("tcr2vec-full", fulls), ("tcr2vec-cdr3", [sani(c) for c in cdr3_model])]:
    emb = embed(seqs, tag)
    stem = f"{dataset}_level1_{tag}_mean"
    np.save(os.path.join(outdir, stem + ".npy"), emb)
    json.dump({"dataset": dataset, "level": "level1", "model": tag, "pooling": "mean",
               "n_sequences": len(cdr3s), "embedding_dim": int(emb.shape[1]),
               "sequence_order": cdr3s},
              open(os.path.join(outdir, stem + "_metadata.json"), "w"))
    print(f"[{dataset}] wrote {stem}.npy", flush=True)
print(f"[{dataset}] DONE {time.time()-t0:.0f}s", flush=True)
