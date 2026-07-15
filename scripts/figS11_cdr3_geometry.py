#!/usr/bin/env python3
"""Supplementary Figure S11 - the CDR3 deficit is representational, not a cosine-geometry artifact.

BCR L1 Recall@1 for four zero-shot PLMs, raw vs PCA-whitened embeddings (whitening fit on
the reference set; the only significant isotropy fix). Even after whitening, every PLM stays
below the BLOSUM62 alignment baseline (dashed). All four are strongly anisotropic (0.95-0.99)
and the full-chain embeddings at parity are equally anisotropic, so anisotropy is not the cause
(Note S2.6). Numbers are 20-seed means from scratch/cdr3_mechanism_geometry.py + _supp.py.
Standards: Okabe-Ito palette (figstyle), no figure title / on-plot text, 300 dpi, editable fonts.
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
figstyle.apply()

ALIGN = figstyle.ALIGN          # BLOSUM reference (blue)
PLM = figstyle.PLM              # whitened (vermillion)
GREY = figstyle.OKABE["grey"]   # raw

models = ["ESM2-150M", "ESM2-650M", "ESM2-3B", "ESM-C-300m"]
raw = [0.353, 0.369, 0.344, 0.374]
whit = [0.384, 0.418, 0.369, 0.381]
x = np.arange(len(models)); w = 0.38

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.bar(x - w/2, raw, w, color=GREY, edgecolor="black", lw=0.6, label="raw embeddings")
ax.bar(x + w/2, whit, w, color=PLM, edgecolor="black", lw=0.6, label="PCA-whitened")
ax.axhline(0.459, ls="--", color=ALIGN, lw=1.4, label="BLOSUM62 (alignment)")
ax.axhline(0.10, ls=":", color="0.45", lw=1.1, label="random baseline")

ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9.5)
ax.set_ylabel("BCR CDR3 Recall@1", fontsize=10.5)
ax.set_ylim(0, 0.62)
ax.legend(loc="upper center", fontsize=8.5, frameon=False, ncol=2)
ax.grid(axis="y", ls="--", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()

out = Path("outputs/manuscript_submission/figures")
(out / "png").mkdir(parents=True, exist_ok=True)
plt.savefig(out / "figS11_cdr3_geometry.pdf", bbox_inches="tight")
plt.savefig(out / "png" / "figS11_cdr3_geometry.png", dpi=300, bbox_inches="tight")
print("Saved figS11_cdr3_geometry ->", out / "figS11_cdr3_geometry.pdf")
