#!/usr/bin/env python3
"""Embed TCR2vec (TCRdb-pretrained, 120-d, full-length) for McPAS+VDJdb.
Anchor-included full-length reconstruction; IMGT-stripped CDR3 as key. tcr2vec_env.
Usage: python tcrdb_embed.py <input_csv> <dataset> <outdir>"""
import sys, os, json, numpy as np, pandas as pd, torch
torch.set_num_threads(6)
from tcr2vec.model import TCR2vec
from tcr2vec.dataset import TCRLabeledDset
from torch.utils.data import DataLoader
from tcr2vec.utils import get_emb
from tcr2vec.cdr3_to_full_seq import to_full_seq
import tcr2vec
GENE_DIR = os.path.join(os.path.dirname(tcr2vec.__file__), "data", "TCR_gene_segment_data")
W = os.environ.get("TCR2VEC_WEIGHTS","weights/tcr2vec_weights")+"/tcr2vec_tcrdb_dir/tcrdb_1.0"
inp, dataset, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(inp)
cdr3s = df["CDR3.beta"].astype(str).tolist(); Vs, Js = df["V"].tolist(), df["J"].tolist()
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
m = TCR2vec(W); m.eval()
d = TCRLabeledDset(list(fulls), only_tcr=True)
emb = get_emb(m, DataLoader(d, batch_size=128, collate_fn=d.collate_fn, shuffle=False), detach=True).astype(np.float32)
stem = f"{dataset}_level1_tcr2vec-tcrdb_mean"
np.save(os.path.join(outdir, stem + ".npy"), emb)
json.dump({"dataset": dataset, "level": "level1", "model": "tcr2vec-tcrdb", "pooling": "mean",
           "n_sequences": len(cdr3s), "embedding_dim": int(emb.shape[1]), "sequence_order": cdr3s},
          open(os.path.join(outdir, stem + "_metadata.json"), "w"))
print(f"[{dataset}] tcr2vec-tcrdb {emb.shape} wrote {stem}", flush=True)
