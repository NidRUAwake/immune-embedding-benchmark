#!/usr/bin/env python3
"""V-gene germline-shortcut failure figure. (A) antigen x top-1-retrieved-antigen confusion
matrix (ESM2-150M, BCR L4, row-normalised, 5 seeds). (B) shared-V-gene-family rate among
correct vs wrong retrievals for both methods -- wrong retrievals are at least as V-gene-shared
as correct ones, i.e. errors are 'right germline, wrong antigen'. No titles; numbers in caption."""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
mpl.rcParams["font.sans-serif"]=["Arial","Helvetica","DejaVu Sans"]; mpl.rcParams["pdf.fonttype"]=42

cm=pd.read_csv("outputs/reports/vgene_confusion_matrix_esm2150m_l4.csv",index_col=0)
fig,(axA,axB)=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={"width_ratios":[1.25,1.0]})

# Panel A: confusion heatmap
im=axA.imshow(cm.values,cmap="Blues",vmin=0,vmax=1,aspect="auto")
axA.set_xticks(range(len(cm.columns))); axA.set_xticklabels(cm.columns,rotation=45,ha="right",fontsize=9)
axA.set_yticks(range(len(cm.index))); axA.set_yticklabels(cm.index,fontsize=9)
axA.set_xlabel("Top-1 retrieved antigen",fontsize=10); axA.set_ylabel("True antigen",fontsize=10)
for i in range(len(cm.index)):
    for j in range(len(cm.columns)):
        v=cm.values[i,j]
        if v>=0.01: axA.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8,
                             color="white" if v>0.5 else "0.2")
axA.set_title("A",loc="left",fontsize=12,weight="bold")
cbar=fig.colorbar(im,ax=axA,fraction=0.046,pad=0.04); cbar.set_label("fraction of true-antigen queries",fontsize=8)

# Panel B: shared-V-gene rate correct vs wrong, both methods
methods=["ESM2-150M","BLOSUM62"]
correct=[0.873,0.992]; wrong=[0.918,0.994]
x=np.arange(2); bw=0.36
axB.bar(x-bw/2,correct,bw,color="#7fb3d5",edgecolor="black",lw=0.6,label="Correct retrieval")
axB.bar(x+bw/2,wrong,bw,color="#e59866",edgecolor="black",lw=0.6,label="Wrong retrieval")
axB.set_xticks(x); axB.set_xticklabels(methods,fontsize=10)
# bars reach ~0.99; extend y to 1.25 so the upper-left legend clears the (tall) bars
axB.set_ylabel("Shared V-gene family rate",fontsize=10); axB.set_ylim(0,1.25)
axB.axhline(0.190,ls=":",color="0.4",lw=1.1,label="Random baseline (0.190)")  # random same-family rate
axB.legend(loc="upper left",fontsize=8.5,frameon=False)
axB.set_title("B",loc="left",fontsize=12,weight="bold")
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

plt.tight_layout()
out=Path("outputs/manuscript_submission/figures")
out.mkdir(parents=True, exist_ok=True); (out/"png").mkdir(parents=True, exist_ok=True)
plt.savefig(out/"figS11.pdf",bbox_inches="tight"); plt.savefig(out/"png"/"figS11.png",dpi=200,bbox_inches="tight")
print("Saved figS11 ->",out/"figS11.pdf")
