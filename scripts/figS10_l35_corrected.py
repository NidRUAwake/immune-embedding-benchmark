#!/usr/bin/env python3
"""Supplementary Fig S10 — L3.5 framework ablation (corrected data, Okabe-Ito palette).
L3 -> L3.5 -> L4 trajectory for the two alignment methods and ESM2-150M. No title."""
import sys; sys.path.insert(0,"scripts")
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import figstyle; figstyle.apply()

xs=[0,1,2]; xlabels=["Paratope\n(CDR1+2+3)","Framework-CDR3\ncontrol","Full variable\ndomain"]
# (mean, lo, hi) corrected 20-seed nested bootstrap
series={
 "BLOSUM62":   ([0.539,0.517,0.522],[0.501,0.463,0.474],[0.579,0.565,0.572], "blosum62"),
 "Levenshtein":([0.549,0.509,0.507],[0.506,0.461,0.459],[0.588,0.555,0.557], "levenshtein"),
 "ESM2-150M":  ([0.445,0.474,0.515],[0.401,0.425,0.476],[0.489,0.526,0.552], "esm2-150m"),
}
fig,ax=plt.subplots(figsize=(6,4.3))
for name,(m,lo,hi,key) in series.items():
    ax.plot(xs,m,color=figstyle.METHOD[key],lw=2,ms=6,marker=figstyle.MARKER[key],
            linestyle=figstyle.LINESTYLE[key],label=name,zorder=3)
    ax.fill_between(xs,lo,hi,color=figstyle.METHOD[key],alpha=0.15,zorder=1)
ax.set_xticks(xs); ax.set_xticklabels(xlabels,fontsize=9.5)
ax.set_ylabel("Recall@1",fontsize=11); ax.set_ylim(0.36,0.62)
ax.grid(axis="y",ls="--",alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="lower right",fontsize=9.5,frameon=False)
plt.tight_layout()
out=Path("outputs/manuscript_submission/figures")
out.mkdir(parents=True, exist_ok=True); (out/"png").mkdir(parents=True, exist_ok=True)
plt.savefig(out/"figS8.pdf",bbox_inches="tight"); plt.savefig(out/"png"/"figS8.png",dpi=300,bbox_inches="tight")
print("Saved figS8 ->",out/"figS8.pdf")
