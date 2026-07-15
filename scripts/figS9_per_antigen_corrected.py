#!/usr/bin/env python3
"""Supplementary Fig S9 (corrected) — BCR L4 per-antigen gap ESM2-150M minus BLOSUM62,
20-seed mean +- SE. Reads corrected tie_v2 predictions; writes to manuscript figures dir."""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams["font.sans-serif"]=["Arial","Helvetica","DejaVu Sans"]; mpl.rcParams["pdf.fonttype"]=42

# (label, delta, SE) — N_test/seed >= 10 antigens, from per-seed paired differences
data=[("HIV-1 envelope",0.212,0.049),
      ("SARS-CoV-2 spike",-0.095,0.030),
      ("HCV polyprotein",-0.013,0.057),
      ("Influenza HA",-0.072,0.025),
      ("Plasmodium CSP",-0.100,0.029)]
data=sorted(data,key=lambda t:t[1])
labels=[d[0] for d in data]; vals=[d[1] for d in data]; ses=[d[2] for d in data]
cols=["#D55E00" if v>0 else "#0072B2" for v in vals]
fig,ax=plt.subplots(figsize=(7,4))
y=np.arange(len(labels))
bars_pos=ax.barh(y,vals,xerr=ses,color=cols,edgecolor="black",lw=0.6,capsize=4)
ax.axvline(0,color="black",lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=10)
ax.set_xlabel("ESM2-150M $-$ BLOSUM62  (BCR full-domain Recall@1)",fontsize=10.5)
ax.set_xlim(-0.20,0.27)
# legend keys for bar direction (kept off the plotting area, no numeric text on bars)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#D55E00",label="ESM2-150M leads"),
                   Patch(facecolor="#0072B2",label="BLOSUM62 leads")],
          loc="lower right",fontsize=8.5,frameon=False)
ax.grid(axis="x",ls="--",alpha=0.3); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
from pathlib import Path
out=Path("outputs/manuscript_submission/figures")
out.mkdir(parents=True, exist_ok=True); (out/"png").mkdir(parents=True, exist_ok=True)
plt.savefig(out/"figS7.pdf",bbox_inches="tight"); plt.savefig(out/"png"/"figS7.png",dpi=300,bbox_inches="tight")
print("Saved figS7 ->",out/"figS7.pdf")
