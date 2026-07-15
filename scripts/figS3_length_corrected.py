#!/usr/bin/env python3
"""Supp Fig S3 — CDR3-length-stratified BCR L1 retrieval.
Regenerated standalone (the original generator was lost in the S1-S8 renumbering).
Values reproduce the published figure; the only change is that the y-axis is extended
to 0.85 so the wide 13-15 aa error bars are no longer clipped at the top."""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
mpl.rcParams["font.sans-serif"]=["Arial","Helvetica","DejaVu Sans"]; mpl.rcParams["pdf.fonttype"]=42

bins=["10-12 aa","13-15 aa","$\\geq$16 aa"]
models=["BLOSUM62","ESM2-150M","ESM2-650M","ESM2-3B"]
cols=["#0173B2","#DE8F05","#029E73","#CC78BC"]      # seaborn colorblind (matches published)
vals={"BLOSUM62":[0.336,0.515,0.458],"ESM2-150M":[0.220,0.426,0.343],
      "ESM2-650M":[0.266,0.379,0.389],"ESM2-3B":[0.173,0.423,0.355]}
err ={"BLOSUM62":[0.048,0.073,0.044],"ESM2-150M":[0.052,0.072,0.050],
      "ESM2-650M":[0.046,0.084,0.043],"ESM2-3B":[0.043,0.088,0.030]}   # 95% CI half-widths (post-filter, 20-seed)
x=np.arange(3); bw=0.20
fig,ax=plt.subplots(figsize=(8.4,5.0))
for i,m in enumerate(models):
    ax.bar(x+(i-1.5)*bw,vals[m],bw,yerr=err[m],color=cols[i],edgecolor="black",lw=0.6,
           capsize=3,label=m)
ax.set_xticks(x); ax.set_xticklabels(bins,fontsize=11)
ax.set_xlabel("CDR3 length",fontsize=11)
ax.set_ylabel("Recall@1 (BCR CDR3)",fontsize=11)
ax.set_ylim(0,0.85)
ax.legend(loc="upper left",fontsize=9,frameon=False,ncol=2)
ax.grid(axis="y",ls="--",alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
out=Path("outputs/manuscript_submission/figures")
out.mkdir(parents=True, exist_ok=True); (out/"png").mkdir(parents=True, exist_ok=True)
plt.savefig(out/"figS3.pdf",bbox_inches="tight"); plt.savefig(out/"png"/"figS3.png",dpi=300,bbox_inches="tight")
print("Saved figS3 ->",out/"figS3.pdf")
