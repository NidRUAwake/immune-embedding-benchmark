#!/usr/bin/env python3
"""Supplementary Figure S10 - why the zero-shot PLM underperforms on CDR3 (out-of-distribution).

Per-query BCR L1 expected-R@1 for ESM2-150M and BLOSUM62, binned by the CDR3's ESM2 masked-language
pseudo-perplexity (a per-sequence measure of how out-of-distribution / junctional it is). ESM2
retrieval falls monotonically with pseudo-perplexity while alignment does not, so the PLM's deficit
is concentrated on the most out-of-distribution CDR3s (Note S2.6). Data-driven from
outputs/reports/ood_cdr3_perquery.csv (20-seed pooled query instances; scratch/ood_cdr3_experiment.py).
Standards: no title, no on-plot numbers, Okabe-Ito palette, 300 dpi.
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
BLOSUM, ESM = "#0072B2", "#D55E00"

R = pd.read_csv("outputs/reports/ood_cdr3_perquery.csv")
q = R["ppl"].quantile([1/3, 2/3]).values
R["bin"] = np.where(R["ppl"] <= q[0], 0, np.where(R["ppl"] <= q[1], 1, 2))
labels = ["low\n(ESM2: ordinary)", "medium", "high\n(ESM2: surprising)"]
bl = [R[R.bin == i]["blosum"].mean() for i in range(3)]
es = [R[R.bin == i]["esm"].mean() for i in range(3)]

x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.bar(x - w/2, bl, w, color=BLOSUM, edgecolor="black", lw=0.6, label="BLOSUM62")
ax.bar(x + w/2, es, w, color=ESM, edgecolor="black", lw=0.6, label="ESM2-150M")
ax.axhline(0.10, ls=":", color="0.45", lw=1.2, label="random baseline (0.10)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_xlabel("CDR3 ESM2 pseudo-perplexity tertile", fontsize=10.5)
ax.set_ylabel("BCR CDR3 Recall@1", fontsize=11)
ax.set_ylim(0, 0.62)
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.grid(axis="y", ls="--", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
out = Path("outputs/manuscript_submission/figures")
(out / "png").mkdir(parents=True, exist_ok=True)
plt.savefig(out / "figS10.pdf", bbox_inches="tight")
plt.savefig(out / "png" / "figS10.png", dpi=300, bbox_inches="tight")
print("Saved figS10 ->", out / "figS10.pdf")
