#!/usr/bin/env python3
"""Paired VDJdb embedding for TCR2vec (full) and CDR3vec: embed beta and alpha chains
separately (anchor-included input; TCR2vec on reconstructed full chain, CDR3vec on CDR3),
then concatenate [beta ; alpha]. Key = paired 'beta_cdr3:alpha_cdr3'. tcr2vec_env.
Writes tcr_level1_paired_{tcr2vec-fullP,cdr3vecP}_mean.npy."""
import sys, os, json, numpy as np, pandas as pd, torch
torch.set_num_threads(6)
from tcr2vec.model import TCR2vec
from tcr2vec.dataset import TCRLabeledDset
from torch.utils.data import DataLoader
from tcr2vec.utils import get_emb
from tcr2vec.cdr3_to_full_seq import to_full_seq
import tcr2vec
GENE=os.path.join(os.path.dirname(tcr2vec.__file__),"data","TCR_gene_segment_data")
W_FULL=os.environ.get("TCR2VEC_WEIGHTS","weights/tcr2vec_weights")+"/tcr2vec_full_dir/TCR2vec_120"
W_CDR3=os.environ.get("TCR2VEC_WEIGHTS","weights/tcr2vec_weights")+"/cdr3vec_dir/CDR3vec_120"
inp,outdir=sys.argv[1],sys.argv[2]
df=pd.read_csv(inp)
keys=df["key"].astype(str).tolist()
AA=set('ACDEFGHIKLMNPQRSTVWY')
def sani(x): return ''.join(c for c in str(x).upper() if c in AA) or 'A'
def anch(c):
    c=str(c).upper()
    if not c: return c
    if c[0]!='C': c='C'+c
    if c[-1] not in 'FW': c=c+'F'
    return c
def recon(v,j,c):
    try:
        if not isinstance(v,str) or not isinstance(j,str): return c
        o=to_full_seq(GENE,v,j,c); fs=str(o[0] if isinstance(o,tuple) else o)
        return fs if (fs and c in fs) else c
    except Exception: return c
# per-chain anchor-included inputs
bC=[anch(c) for c in df["CDR3B"]]; aC=[anch(c) for c in df["CDR3A"]]
bFull=[sani(recon(v,j,c)) for v,j,c in zip(df["VB"],df["JB"],bC)]
aFull=[sani(recon(v,j,c)) for v,j,c in zip(df["VA"],df["JA"],aC)]
bC=[sani(c) for c in bC]; aC=[sani(c) for c in aC]
def embed(W,seqs):
    m=TCR2vec(W); m.eval()
    d=TCRLabeledDset(list(seqs),only_tcr=True)
    return get_emb(m,DataLoader(d,batch_size=128,collate_fn=d.collate_fn,shuffle=False),detach=True).astype(np.float32)
for tag,W,bs,as_ in [("tcr2vec-full",W_FULL,bFull,aFull),("cdr3vec",W_CDR3,bC,aC)]:
    eb=embed(W,bs); ea=embed(W,as_)
    emb=np.concatenate([eb,ea],axis=1)   # [beta ; alpha]
    stem=f"tcr_level1_paired_{tag}_mean"
    np.save(os.path.join(outdir,stem+".npy"),emb)
    json.dump({"dataset":"tcr","level":"level1_paired","model":tag,"pooling":"mean",
               "n_sequences":len(keys),"embedding_dim":int(emb.shape[1]),"sequence_order":keys},
              open(os.path.join(outdir,stem+"_metadata.json"),"w"))
    print(f"paired {tag} {emb.shape} wrote {stem}",flush=True)
print("PAIRED_EMBED_DONE",flush=True)
