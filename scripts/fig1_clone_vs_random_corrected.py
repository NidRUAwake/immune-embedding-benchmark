#!/usr/bin/env python3
"""Manuscript Fig 1 (renumbered 2026-06: clone-aware figure is now Fig 1) -- random vs
clone-aware splitting inflation on cleaned BCR, 20-seed paired, for BOTH BLOSUM62 and
ESM2-150M, to show the leakage is method-general (not ESM2-specific). No title; numbers
in caption. Outputs to figures/fig1.pdf (+ png/fig1.png)."""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
mpl.rcParams["font.sans-serif"]=["Arial","Helvetica","DejaVu Sans"]; mpl.rcParams["pdf.fonttype"]=42

# POST-is_legit-filter clone-vs-random arms (layout: {strategy}/{model}/seed_N/).
# (NOT phase3_grand_slam_tie_v2/bcr_clone_vs_random, which is the stale pre-filter 05-30 tree.)
BASE=Path("outputs/phase3_grand_slam_postfilter/bcr_clone_vs_random")
SEEDS=[42+10*i for i in range(20)]
def r1(strategy,level,model):
    v=[]
    for s in SEEDS:
        fp=BASE/strategy/model/f"seed_{s}"/f"BCR_{level}_{model}_predictions.csv"
        if fp.exists(): v.append(pd.read_csv(fp)["exp_recall@1"].mean())
    return np.array(v)

# four groups: BLOSUM62 L1, ESM2-150M L1, BLOSUM62 L4, ESM2-150M L4
groups=[("blosum62","level1","BLOSUM62\nCDR3"),("esm2-150m","level1","ESM2-150M\nCDR3"),
        ("blosum62","level4","BLOSUM62\nfull domain"),("esm2-150m","level4","ESM2-150M\nfull domain")]
ca=[r1("clone-aware",lv,m) for m,lv,_ in groups]
rd=[r1("random",lv,m) for m,lv,_ in groups]
xl=[g[2] for g in groups]
x=np.arange(len(groups)); bw=0.38
fig,ax=plt.subplots(figsize=(6.6,4.2))
ax.bar(x-bw/2,[c.mean() for c in ca],bw,yerr=[c.std(ddof=1) for c in ca],capsize=3.5,
       color="#0072B2",edgecolor="black",lw=0.7,label="Clone-aware split")
ax.bar(x+bw/2,[r.mean() for r in rd],bw,yerr=[r.std(ddof=1) for r in rd],capsize=3.5,
       color="#E69F00",edgecolor="black",lw=0.7,label="Random split")
ax.set_xticks(x); ax.set_xticklabels(xl,fontsize=9.5)
ax.set_ylabel("Recall@1",fontsize=11); ax.set_ylim(0,0.85)
ax.legend(loc="upper right",fontsize=9.5,frameon=False)
ax.grid(axis="y",ls="--",alpha=0.3); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
out=Path("outputs/manuscript_submission/figures")
out.mkdir(parents=True, exist_ok=True); (out/"png").mkdir(parents=True, exist_ok=True)
plt.savefig(out/"fig1.pdf",bbox_inches="tight"); plt.savefig(out/"png"/"fig1.png",dpi=300,bbox_inches="tight")
print("Saved fig1 ->",out/"fig1.pdf")
for (m,lv,lab),c,r in zip(groups,ca,rd):
    print(f"  {m} {lv}: clone={c.mean():.3f} random={r.mean():.3f}  (+{(r.mean()-c.mean())*100:.1f} pp)")
