#!/usr/bin/env python3
"""Embed EXACTLY the canonical ESM2-150M sequence lists for a BCR model.
Guarantees offline-lookup coverage (same sequence_order) and minimal compute.
Usage: python bcr_embed_focused.py <model> <ds1:level1,level3,...> [<ds2:...>]
Writes outputs/embeddings/{ds}_{level}_{model}_mean.npy (+ _metadata.json).
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "6"); os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("HF_HOME", "hf_cache")
sys.path.insert(0, "scripts")
import numpy as np, torch
torch.set_num_threads(6)
from benchmark.embeddings import TransformerEmbedder
EMB = "outputs/embeddings"
model = sys.argv[1]
class A: batch_size = 32; pooling = "mean"; local_files_only = False
emb = None
for spec in sys.argv[2:]:
    ds, levels = spec.split(":"); levels = levels.split(",")
    for level in levels:
        stem_out = f"{ds}_{level}_{model}_mean"
        if os.path.exists(f"{EMB}/{stem_out}.npy"):
            print(f"skip {stem_out} (exists)", flush=True); continue
        meta_p = f"{EMB}/{ds}_{level}_esm2-150m_mean_metadata.json"
        seqs = json.load(open(meta_p))["sequence_order"]
        if emb is None:
            t = time.time(); emb = TransformerEmbedder(model, A())
            print(f"loaded {model} in {time.time()-t:.0f}s", flush=True)
        t = time.time()
        vecs = emb.encode([str(s) for s in seqs]).astype(np.float32)
        np.save(f"{EMB}/{stem_out}.npy", vecs)
        json.dump({"dataset": ds, "level": level, "model": model, "pooling": "mean",
                   "n_sequences": len(seqs), "embedding_dim": int(vecs.shape[1]),
                   "sequence_order": seqs}, open(f"{EMB}/{stem_out}_metadata.json", "w"))
        print(f"wrote {stem_out} {vecs.shape} ({len(seqs)} seqs, {time.time()-t:.0f}s)", flush=True)
print("BCR_EMBED_DONE", flush=True)
