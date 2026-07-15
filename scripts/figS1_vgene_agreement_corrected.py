#!/usr/bin/env python3
"""Supp Fig S1 — V-gene-family agreement of top-1 retrievals, BCR L1 vs L4.
Regenerated standalone (the original generator was lost in the S1-S8 renumbering).
Values are the canonical 20-seed same-V-family agreement rates (all top-1 hits,
recomputed on the post-is_legit build_pilot_slice; BLOSUM62 L1 0.632, ESM2-150M L1 0.537,
BLOSUM62 L4 0.991, ESM2-150M L4 0.915; random same-V baseline ~0.34)."""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
mpl.rcParams["font.sans-serif"]=["Arial","Helvetica","DejaVu Sans"]; mpl.rcParams["pdf.fonttype"]=42

labels=["BLOSUM62\nCDR3","ESM2-150M\nCDR3","BLOSUM62\nfull domain","ESM2-150M\nfull domain"]
vals=[0.632,0.537,0.991,0.915]
err =[0.073,0.074,0.009,0.068]            # 20-seed mean/SD (canonical is_legit slice)
cols=["#0072B2","#0072B2","#E69F00","#E69F00"]   # L1 blue, L4 orange
x=np.arange(4)
fig,ax=plt.subplots(figsize=(7.2,4.6))
ax.bar(x,vals,0.66,yerr=err,color=cols,edgecolor="black",lw=0.7,capsize=4)
ax.axhline(0.34,ls=":",color="0.45",lw=1.4,label="random same-V baseline (0.34)")
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=10)
ax.set_ylabel("Top-1 same-V-gene-family rate",fontsize=11)
ax.set_ylim(0,1.08)
ax.legend(loc="upper left",fontsize=9,frameon=False)
ax.grid(axis="y",ls="--",alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
out=Path("outputs/manuscript_submission/figures")
out.mkdir(parents=True, exist_ok=True); (out/"png").mkdir(parents=True, exist_ok=True)
plt.savefig(out/"figS1.pdf",bbox_inches="tight"); plt.savefig(out/"png"/"figS1.png",dpi=300,bbox_inches="tight")
print("Saved figS1 ->",out/"figS1.pdf")
