#!/usr/bin/env python3
"""Supp Fig S4 — clone-threshold sensitivity on the cleaned BCR set (20 seeds). Reads
threshold_sweep_tie_v2.csv. Absolute R@1 varies by up to ~8 pp across thresholds (lower at
stricter thresholds), but BLOSUM62 leads ESM2-150M at L1 at every threshold and the two are
comparable at L4."""
import sys; sys.path.insert(0,"scripts")
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from pathlib import Path
import figstyle; figstyle.apply()
d=pd.read_csv("outputs/reports/threshold_sweep_tie_v2.csv")
fig,ax=plt.subplots(figsize=(6.4,4.4))
# color = method, line style = level, marker shape distinct per series (filled L1 / open L4)
# so the legend unambiguously distinguishes all four series. SD shown as a light band.
series=[("blosum62","level1","BLOSUM62 CDR3","-","o",True),
        ("blosum62","level4","BLOSUM62 full domain","--","s",False),
        ("esm2-150m","level1","ESM2-150M CDR3","-","^",True),
        ("esm2-150m","level4","ESM2-150M full domain","--","D",False)]
for m,lv,lab,ls,mk,filled in series:
    sub=d[(d.model==m)&(d.level==lv)].sort_values("threshold")
    c=figstyle.METHOD[m]
    ax.fill_between(sub.threshold,sub.recall1-sub.sd,sub.recall1+sub.sd,color=c,alpha=0.12,lw=0)
    ax.plot(sub.threshold,sub.recall1,ls=ls,marker=mk,color=c,lw=1.9,ms=7,
            markerfacecolor=(c if filled else "white"),markeredgecolor=c,label=lab)
ax.set_xlabel("Clone-aware CDR3-similarity threshold",fontsize=10.5)
ax.set_ylabel("Recall@1",fontsize=11); ax.set_ylim(0.13,0.72)
ax.legend(fontsize=8.5,frameon=False,ncol=2,loc="lower center")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(axis="y",ls="--",alpha=0.3)
plt.tight_layout()
out=Path("outputs/manuscript_submission/figures")
out.mkdir(parents=True, exist_ok=True); (out/"png").mkdir(parents=True, exist_ok=True)
plt.savefig(out/"figS4.pdf",bbox_inches="tight"); plt.savefig(out/"png/figS4.png",dpi=300,bbox_inches="tight")
print("Saved figS4")
