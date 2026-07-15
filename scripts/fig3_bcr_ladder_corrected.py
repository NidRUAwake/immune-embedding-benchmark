#!/usr/bin/env python3
"""Manuscript Fig 3 — BCR representation ladder (corrected tie_v2 data).
BLOSUM62 vs ESM2-150M across CDR3, VJ-/V-clonotype, paratope, full variable domain
with 95% nested-bootstrap CI bands. Reads corrected reports; writes to the manuscript figures dir."""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
mpl.rcParams["font.sans-serif"]=["Arial","Helvetica","DejaVu Sans"]
mpl.rcParams["pdf.fonttype"]=42

# Ordered benchmark ladder: CDR3, VJ-clonotype, V-clonotype, paratope, full variable domain.
# VJ-/V-clonotype are the clonotype-filter variants (shaded band); the others vary input content.
# (The length-matched framework-CDR3 control is supplementary-only: Supp Note S2.5, Table S3, Fig S8.)
xs=[0,0.6,1.2,2.0,2.8]
xlabels=["CDR3","VJ-clonotype","V-clonotype","Paratope\n(CDR1+2+3)","Full variable\ndomain"]

# (mean, lo, hi) per level — corrected 20-seed nested bootstrap; clonotype variants from l2_vjfilter run
blosum={0:(0.459,0.421,0.499),0.6:(0.433,0.395,0.474),1.2:(0.526,0.488,0.562),2.0:(0.539,0.501,0.579),2.8:(0.522,0.474,0.572)}
esm   ={0:(0.357,0.320,0.392),0.6:(0.397,0.360,0.435),1.2:(0.484,0.447,0.522),2.0:(0.445,0.401,0.489),2.8:(0.515,0.476,0.552)}

fig,ax=plt.subplots(figsize=(8.0,4.6))
ax.axvspan(0.3,1.5,color="#999999",alpha=0.10,zorder=0)  # clonotype-filter region (L2, L2.5)
def plot_series(data,color,label,marker,ls):
    x=list(data.keys()); m=[data[k][0] for k in x]
    lo=[data[k][1] for k in x]; hi=[data[k][2] for k in x]
    ax.plot(x,m,color=color,lw=2,ms=6,marker=marker,linestyle=ls,label=label,zorder=3)
    ax.fill_between(x,lo,hi,color=color,alpha=0.15,zorder=1)

plot_series(blosum,"#0072B2","BLOSUM62 (alignment)","o","-")
plot_series(esm,"#D55E00","ESM2-150M (PLM)","^","-.")

ax.set_xticks(xs); ax.set_xticklabels(xlabels,fontsize=8.5)
ax.set_ylabel("Recall@1",fontsize=11)
ax.set_ylim(0.28,0.62); ax.set_yticks(np.arange(0.30,0.61,0.05))
ax.grid(axis="y",ls="--",alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="upper left",fontsize=9.5,frameon=False)
plt.tight_layout()
out=Path("outputs/manuscript_submission/figures")
out.mkdir(parents=True, exist_ok=True); (out/"png").mkdir(parents=True, exist_ok=True)
plt.savefig(out/"fig3.pdf",bbox_inches="tight")
plt.savefig(out/"png"/"fig3.png",dpi=300,bbox_inches="tight")
print("Saved fig3 ->",out/"fig3.pdf")
