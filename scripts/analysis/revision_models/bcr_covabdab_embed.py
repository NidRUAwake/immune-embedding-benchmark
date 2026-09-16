#!/usr/bin/env python3
"""Cap CoV-AbDab neutralization set to a balanced working set, embed VH + CDRH3 with
esm2-150m / antiberta2 / igbert. Saves offline .npy keyed by sequence. Main env."""
import os, sys, json, numpy as np, pandas as pd
os.environ.setdefault("OMP_NUM_THREADS","6"); os.environ.setdefault("MKL_NUM_THREADS","6")
os.environ.setdefault("HF_HOME","hf_cache")
sys.path.insert(0,"scripts")
import torch; torch.set_num_threads(6)
from benchmark.embeddings import TransformerEmbedder
SP=os.environ.get("REVISION_WORKDIR","revision_workdir")
EMB=f"{SP}/covabdab_emb"; os.makedirs(EMB, exist_ok=True)
model=sys.argv[1]
CAP=2000  # per class
d=pd.read_csv(f"{SP}/covabdab_neut.csv")
parts=[d[d.label==c].sample(min((d.label==c).sum(),CAP), random_state=42) for c in [0,1]]
w=pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
w.to_csv(f"{SP}/covabdab_work.csv", index=False)
class A: batch_size=16; pooling="mean"; local_files_only=False
emb=TransformerEmbedder(model, A())
for col,lvl in [("CDRH3","cdrh3"),("VH","vh")]:
    seqs=w[col].astype(str).tolist()
    stem=f"covabdab_{lvl}_{model}_mean"
    if os.path.exists(f"{EMB}/{stem}.npy"): print("skip",stem); continue
    import time; t=time.time()
    vecs=emb.encode(seqs).astype(np.float32)
    np.save(f"{EMB}/{stem}.npy", vecs)
    json.dump({"model":model,"level":lvl,"sequence_order":seqs}, open(f"{EMB}/{stem}_metadata.json","w"))
    print(f"wrote {stem} {vecs.shape} ({time.time()-t:.0f}s)", flush=True)
print(f"{model} DONE", flush=True)
