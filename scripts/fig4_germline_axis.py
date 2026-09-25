#!/usr/bin/env python3
"""Figure 4 (BiB) - PLM minus alignment gap across germline regimes.

Conceptual synthesis of the paper's spine, anchored to real BCR numbers: the
ESM2-150M minus BLOSUM62 Recall@1 difference across four settings grouped into
three germline regimes.

  Regime 1 - CDR-level input, no framework (same antibodies):
     L1 (CDR3 only)    0.357 - 0.459 = -0.10   alignment leads
     L3 (all CDRs)     0.445 - 0.538 = -0.09   alignment leads
  Regime 2 - germline framework added (same antibodies):
     L4 (full chain)   0.514 - 0.522 = -0.01   parity (TOST-equivalent)
  Regime 3 - far-from-germline antibodies (different, heavily mutated set), full chain:
     HIV-1 (L4)        0.661 - 0.449 = +0.21   PLM leads (leak-free +0.22)

This is deliberately CATEGORICAL, not a continuous axis: the first three bars vary
INPUT GRANULARITY for the same antibodies; the fourth changes the ANTIBODY SET
(HIV-1, heavily hypermutated) at the same full-chain granularity, so a divider
separates it. The gap stays negative through L1 and L3 and only closes once the
germline framework is added at L4 -- the visual signature of the germline shortcut.

Per the figure standard: NO on-plot text/numbers/annotations -- regime definitions,
values, the shortcut reading, the HIV "different antibody set" caveat and the
"illustrative; full evidence in Table 1/Sec 3.5" note all live in the caption.
Bars are colored by who leads (alignment blue / parity gray / PLM vermillion).
Okabe-Ito palette (figstyle), no figure title, 300 dpi, editable fonts.
"""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
figstyle.apply()

ALIGN = figstyle.ALIGN          # alignment leads (blue)
PLM = figstyle.PLM              # PLM leads (vermillion)
GREY = figstyle.OKABE["gray"]   # parity

labels = ["CDR3", "Paratope\n(all CDRs)", "Full variable\ndomain", "HIV-1\n(full domain)"]
delta = [-0.102, -0.093, -0.008, +0.212]
colors = [ALIGN, ALIGN, GREY, PLM]      # by who leads
xpos = [0, 1, 2, 3.3]                   # gap before HIV-1 = different antibody set

fig, ax = plt.subplots(figsize=(5.4, 4.2))
ax.axhline(0, color="black", lw=1.0, zorder=2)
ax.bar(xpos, delta, width=0.7, color=colors, edgecolor="black", linewidth=0.7, zorder=3)

# divider between the input-granularity group (same antibodies) and the HIV-1 subset
ax.axvline(2.65, color="0.55", ls=(0, (4, 3)), lw=1.0, zorder=1)

ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("ESM2-150M $-$ BLOSUM62  (Recall@1)", fontsize=10.5)
ax.set_xlim(-0.6, 3.95)
ax.set_ylim(-0.16, 0.26)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls="--", alpha=0.25)
plt.tight_layout()

out = Path("outputs/manuscript_submission/figures")
(out / "png").mkdir(parents=True, exist_ok=True)
plt.savefig(out / "fig4_germline_axis.pdf", bbox_inches="tight")
plt.savefig(out / "png" / "fig4_germline_axis.png", dpi=300, bbox_inches="tight")
print("Saved fig4_germline_axis ->", out / "fig4_germline_axis.pdf")
