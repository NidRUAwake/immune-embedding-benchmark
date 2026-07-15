#!/usr/bin/env python3
"""Supplementary Fig S2 (corrected) — pooling sensitivity (4 strategies).
On cleaned variable-domain inputs ESM2 CLS does NOT collapse (0.484 vs mean 0.507), and the two
region-restricted poolings (CDR-only 0.529, framework-only 0.489) both match mean within CI, so
the L4 result is pooling-robust. NOTE: the earlier "cdr_masked" bar was CDR-ONLY pooling
(averages CDR tokens), not framework-only — this version reports both, correctly labelled.
Values: canonical 20-seed cls_pooling_eval.py (bcr_singlechain_vh + is_legit + fixed split,
expected-R@1); see outputs/reports/pooling_sensitivity_4way_20seed.csv.
Writes directly to the manuscript figures dir."""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
mpl.rcParams["font.sans-serif"]=["Arial","Helvetica","DejaVu Sans"]
mpl.rcParams["pdf.fonttype"]=42

fig,axA=plt.subplots(1,1,figsize=(6.4,4.4))

# --- ESM2-150M BCR L4, four poolings (cleaned inputs, 20-seed canonical) ---
labels=["Mean","CLS","CDR-only\nmean","Framework-only\nmean"]
vals=[0.507,0.484,0.529,0.489]
err=[0.037,0.032,0.039,0.042]   # half-CI widths (1.96*SE, 20-seed canonical)
cols=["#0072B2","#E69F00","#CC79A7","#009E73"]
x=np.arange(4)
bars=axA.bar(x,vals,0.64,yerr=err,color=cols,edgecolor="black",lw=0.7,capsize=4)
base=axA.axhline(0.190,ls=":",color="0.4",lw=1.3,label="V-gene random baseline (0.190)")
axA.set_xticks(x); axA.set_xticklabels(labels,fontsize=9.5)
axA.set_ylabel("Recall@1",fontsize=11); axA.set_ylim(0,0.72)
# legend: only the baseline line (the four bars are already named on the x-axis)
axA.legend(handles=[base],loc="upper right",fontsize=9,frameon=False)
axA.grid(axis="y",ls="--",alpha=0.3)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

plt.tight_layout()
out=Path("outputs/manuscript_submission/figures")
out.mkdir(parents=True, exist_ok=True); (out/"png").mkdir(parents=True, exist_ok=True)
plt.savefig(out/"figS2.pdf",bbox_inches="tight")
plt.savefig(out/"png"/"figS2.png",dpi=300,bbox_inches="tight")
print("Saved figS2 ->",out/"figS2.pdf")
