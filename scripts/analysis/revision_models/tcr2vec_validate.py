import torch, numpy as np, pandas as pd, os, time
t0=time.time()
from tcr2vec.model import TCR2vec
from tcr2vec.dataset import TCRLabeledDset
from torch.utils.data import DataLoader
from tcr2vec.utils import get_emb
import tcr2vec
pkg=os.path.dirname(tcr2vec.__file__)
W=os.environ.get("TCR2VEC_WEIGHTS","weights/tcr2vec_weights")+"/tcr2vec_full_dir/TCR2vec_120"
print("loading model...", flush=True)
m = TCR2vec(W); m.eval()
print("model loaded %.1fs"%(time.time()-t0), flush=True)
df = pd.read_csv(os.path.join(pkg,"data","sample.csv")).head(16)
d1 = TCRLabeledDset(list(df["full_seq"].values), only_tcr=True)
l1 = DataLoader(d1, batch_size=16, collate_fn=d1.collate_fn, shuffle=False)
e1 = get_emb(m, l1, detach=True)
print("full_seq emb", e1.shape, e1.dtype, "finite", bool(np.isfinite(e1).all()), "%.1fs"%(time.time()-t0), flush=True)
d2 = TCRLabeledDset(list(df["CDR3.beta"].values), only_tcr=True)
l2 = DataLoader(d2, batch_size=16, collate_fn=d2.collate_fn, shuffle=False)
e2 = get_emb(m, l2, detach=True)
print("cdr3 emb", e2.shape, flush=True)
from tcr2vec.cdr3_to_full_seq import to_full_seq
gd=os.path.join(pkg,"data","TCR_gene_segment_data")
fs=to_full_seq(gd,"TRBV12-3","TRBJ1-1*01","CASSPGTGGNEKLFF")
print("reconstruct:", str(fs)[:70],"len",len(str(fs)), flush=True)
print("VALIDATE_OK", flush=True)
