#!/usr/bin/env python3
"""Supplementary Figure S9 - germline-reversion decomposition (forest plot).

Per-seed paired Delta R@1 (ESM2-150M, 20 seeds) from the germline-reversion test (Note S2.5):
  germline framework  = L3 -> germ-fw   (add germline FR1-3 to the all-CDR input)
  framework SHM        = germ-fw -> L4   (native vs germline framework; same length)
  CDR1/CDR2 SHM        = germ-cdr -> L4  (native vs germline CDR1/2)
Three panels: (A) BCR, (B) SAbDab, (C) framework-SHM effect split by antigen (BCR, leak-free).
Values: outputs/reports/germfw_decomposition_20seed.csv + the stratified / SAbDab / leak-free runs
(scratch/run_germfw_experiment.py, run_germfw_sabdab.py); see doc 370 sec 8.
Standards: no title and no floating on-plot text (dataset identity -> caption via A/B/C panel labels),
Okabe-Ito palette, 300 dpi, editable fonts.
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# Okabe-Ito
BLUE, ORANGE, GREY, VERM = "#0072B2", "#E69F00", "#999999", "#D55E00"

# each panel: (panel_label, [ (row_label, delta, lo, hi, color) ... ]); rows drawn top -> bottom
panels = [
    ("A", [
        ("Germline framework (paratope → germ-fw)", 0.069, 0.048, 0.090, BLUE),
        ("Framework SHM (germ-fw → full domain)",      0.039, 0.006, 0.073, ORANGE),
        ("CDR1/CDR2 SHM (germ-cdr → full domain)",     0.023, -0.008, 0.053, GREY),
    ]),
    ("B", [
        ("Germline framework (paratope → germ-fw)", 0.050, 0.005, 0.094, BLUE),
        ("Framework SHM (germ-fw → full domain)",     -0.070, -0.109, -0.032, ORANGE),
    ]),
    ("C", [
        ("HIV-1 envelope",  0.113, 0.033, 0.193, VERM),
        ("Other antigens", -0.012, -0.038, 0.014, GREY),
    ]),
]

XMIN, XMAX = -0.135, 0.27
nrows = [len(p[1]) for p in panels]
fig, axes = plt.subplots(len(panels), 1, figsize=(7.0, 5.6), sharex=True,
                         gridspec_kw={"height_ratios": nrows})

for ax, (plabel, rows) in zip(axes, panels):
    n = len(rows)
    for j, (lab, d, lo, hi, c) in enumerate(rows):       # j=0 is top row
        y = n - 1 - j
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=c, ecolor=c,
                    markersize=8, capsize=4, lw=2, zorder=3)
    ax.axvline(0, ls="--", color="0.4", lw=1.2, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(XMIN, XMAX)
    ax.grid(axis="x", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Paired change in Recall@1 (ΔR@1, 20 seeds)", fontsize=11)

handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=9, label="germline framework added"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=9, label="framework somatic mutation"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=VERM, markersize=9, label="HIV-1 framework SHM"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=9, label="no significant effect")]
axes[0].legend(handles=handles, loc="lower right", fontsize=8, frameon=False)

plt.tight_layout()
# panel labels at the figure's far left, outside the axes, vertically aligned to each panel top
for ax, (plabel, _r) in zip(axes, panels):
    fig.text(0.012, ax.get_position().y1, plabel, fontsize=13, weight="bold",
             va="bottom", ha="left")
out = Path("outputs/manuscript_submission/figures")
(out / "png").mkdir(parents=True, exist_ok=True)
plt.savefig(out / "figS9.pdf", bbox_inches="tight")
plt.savefig(out / "png" / "figS9.png", dpi=300, bbox_inches="tight")
print("Saved figS9 ->", out / "figS9.pdf")
