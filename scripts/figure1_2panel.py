"""Figure 1 — two-panel version (v1, structural-review draft).

Panel A: heavy-chain schematic with FW/CDR segments and L1/L3/L4 brackets
         (adapted from the graphical abstract panel A).
Panel B: BCR R@1 comparison at L1 (CDR3) vs L4 (full heavy chain) across
         BLOSUM62, Levenshtein, AbLang, AntiBERTy and ESM2-150M/650M/3B.

Output: outputs/manuscript_submission/figures/fig2.pdf (+ PNG sibling under
        figures/png/fig2.png for docx embedding).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from pathlib import Path

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# Colors
FW_COLOR = "#dcdcdc"
CDR_COLOR = "#E69F00"
CDR3_COLOR = "#D55E00"
L1_COLOR = "#0072B2"       # Okabe-Ito blue
L4_COLOR = "#E69F00"       # Okabe-Ito orange
REF_LINE_COLOR = "#0072B2"


fig = plt.figure(figsize=(8.0, 7.4))
gs = fig.add_gridspec(
    nrows=2, ncols=1,
    height_ratios=[1.0, 2.2],
    hspace=0.32,
    left=0.10, right=0.96, top=0.95, bottom=0.08,
)

# ---------------------------------------------------------------- Panel A
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(0, 100)
ax_a.set_ylim(0, 100)
ax_a.axis("off")
ax_a.set_title("A", loc="left", fontsize=12, weight="bold")

segments = [
    ("FW1", 0, 18, FW_COLOR),
    ("CDR1", 18, 28, CDR_COLOR),
    ("FW2", 28, 42, FW_COLOR),
    ("CDR2", 42, 52, CDR_COLOR),
    ("FW3", 52, 75, FW_COLOR),
    ("CDR3", 75, 86, CDR3_COLOR),
    ("FW4", 86, 100, FW_COLOR),
]

chain_y = 66
chain_h = 14
for name, x0, x1, color in segments:
    ax_a.add_patch(
        Rectangle((x0, chain_y), x1 - x0, chain_h,
                  facecolor=color, edgecolor="black", linewidth=0.7))
    ax_a.text((x0 + x1) / 2, chain_y + chain_h / 2,
              name, ha="center", va="center", fontsize=8.5)

ax_a.text(50, 88, "BCR / TCR variable region", ha="center",
          fontsize=10, style="italic")


def bracket(y, x0, x1, label):
    ax_a.plot([x0, x0], [y, y + 2.5], color="black", linewidth=1)
    ax_a.plot([x1, x1], [y, y + 2.5], color="black", linewidth=1)
    ax_a.plot([x0, x1], [y, y], color="black", linewidth=1)
    ax_a.text((x0 + x1) / 2, y - 6.5, label, ha="center", fontsize=9)


bracket(54, 75, 86, "CDR3")
bracket(40, 18, 86, "Paratope (CDR1+CDR2+CDR3)")
bracket(26, 0, 100, "Full variable domain")
# (L3.5 framework-only construction and the clone-aware split criterion are described
#  in the caption, not annotated on the schematic.)

# ---------------------------------------------------------------- Panel B
ax_b = fig.add_subplot(gs[1, 0])

methods = ["BLOSUM62", "Levenshtein", "AbLang", "AntiBERTy",
           "ESM2-150M", "ESM2-650M", "ESM2-3B"]
# Values from 20-seed nested-bootstrap CSV (cleaned 20-AA inputs, ANARCI CDRs, post-ANARCI
# re-embedded PLMs, expected-tie). AntiBERTy/AbLang are re-embedded via their pip packages
# (the original HF repos are gone), with correct per-residue tokenization (raising AntiBERTy
# L1 from 0.261 to 0.398); both are now reported at L1 AND L4. See discussions/366, 372.
r1_l1 = [0.459, 0.446, 0.390, 0.398, 0.357, 0.372, 0.350]
ci_l1_lo = [0.417, 0.407, 0.346, 0.354, 0.322, 0.333, 0.309]
ci_l1_hi = [0.498, 0.487, 0.438, 0.443, 0.393, 0.416, 0.393]

r1_l4 = [0.522, 0.508, 0.535, 0.551, 0.514, 0.541, 0.528]
ci_l4_lo = [0.472, 0.460, 0.485, 0.506, 0.470, 0.494, 0.479]
ci_l4_hi = [0.570, 0.559, 0.586, 0.593, 0.551, 0.584, 0.577]

x = np.arange(len(methods))
bw = 0.36

err_l1 = np.array([np.array(r1_l1) - np.array(ci_l1_lo),
                   np.array(ci_l1_hi) - np.array(r1_l1)])
err_l4 = np.array([np.array(r1_l4) - np.array(ci_l4_lo),
                   np.array(ci_l4_hi) - np.array(r1_l4)])
# AntiBERTy has no L4 (released weights could not embed the re-standardised inputs);
# nan height skips the bar, nan error deltas -> 0 to avoid matplotlib warnings.
err_l4 = np.nan_to_num(err_l4, nan=0.0)

ax_b.bar(x - bw/2, r1_l1, bw, yerr=err_l1,
         color=L1_COLOR, edgecolor="black", linewidth=0.7,
         capsize=3, label="CDR3")
ax_b.bar(x + bw/2, r1_l4, bw, yerr=err_l4,
         color=L4_COLOR, edgecolor="black", linewidth=0.7,
         capsize=3, label="Full variable domain")

ax_b.axhline(0.459, linestyle="--", color=REF_LINE_COLOR, linewidth=1.2,
             label="BLOSUM62 CDR3 reference (0.459)")

ax_b.set_xticks(x)
ax_b.set_xticklabels(methods, fontsize=9.5)
ax_b.set_ylabel("Recall@1", fontsize=11)
ax_b.set_ylim(0, 0.82)
ax_b.set_yticks(np.arange(0.0, 0.81, 0.1))
ax_b.grid(axis="y", linestyle="--", alpha=0.3)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.legend(loc="upper right", fontsize=9.5, frameon=False, ncol=1)
ax_b.set_title("B", loc="left", fontsize=12, weight="bold", pad=8)

# Save
out_dir_pdf = Path("outputs/manuscript_submission/figures")
out_dir_png = out_dir_pdf / "png"
out_dir_pdf.mkdir(parents=True, exist_ok=True)
out_dir_png.mkdir(parents=True, exist_ok=True)

plt.savefig(out_dir_pdf / "fig2.pdf", bbox_inches="tight")
plt.savefig(out_dir_png / "fig2.png", dpi=300, bbox_inches="tight")
print("Saved Fig 1 v1 (two-panel) to:")
print(f"  {out_dir_pdf / 'fig2.pdf'}")
print(f"  {out_dir_png / 'fig2.png'}")
