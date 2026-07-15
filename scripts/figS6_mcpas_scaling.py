"""Supplementary Figure S6 (reframed) - McPAS-TCR ESM2 scaling.

Replaces the earlier sub-random / hyper-centralization figure. The corrected
post-filter 20-seed data show ESM2 L1 retrieval on McPAS rising modestly with scale
(0.272 -> 0.282 -> 0.305), staying above the 10-label random baseline (0.100)
and well below the BLOSUM62 reference (0.387).

Values are the post-filter canonical means (master_nested_ci_20seeds_postfilter.csv;
McPAS uses the top-10 epitope labels, parallel to the VDJdb 10-working-label set).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# McPAS-TCR L1 ESM2 scaling (nested mean, 95% CI) from canonical 20-seed CSV
scales = ["ESM2-150M", "ESM2-650M", "ESM2-3B"]
r1 = [0.272, 0.282, 0.305]
ci_lo = [0.254, 0.265, 0.284]
ci_hi = [0.290, 0.302, 0.324]

RANDOM_BASELINE = 0.100
BLOSUM_REF = 0.387

x = np.arange(len(scales))
err = np.array([[m - lo for m, lo in zip(r1, ci_lo)],
                [hi - m for m, hi in zip(r1, ci_hi)]])

fig, ax = plt.subplots(figsize=(6.0, 4.2))

ax.errorbar(x, r1, yerr=err, fmt="-o", color="#d6604d", linewidth=2.2,
            markersize=10, capsize=4, zorder=3, label="ESM2 (McPAS-TCR CDR3)")

ax.axhline(BLOSUM_REF, linestyle="--", color="#2166ac", linewidth=1.4,
           label=f"BLOSUM62 reference ({BLOSUM_REF:.3f})")
ax.axhline(RANDOM_BASELINE, linestyle=":", color="#666666", linewidth=1.4,
           label=f"10-label random baseline ({RANDOM_BASELINE:.3f})")

ax.set_xticks(x)
ax.set_xticklabels(scales, fontsize=10)
ax.set_ylabel("McPAS-TCR CDR3 Recall@1", fontsize=11)
ax.set_ylim(0.05, 0.58)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(loc="upper left", fontsize=9, frameon=False)

out_pdf = Path("outputs/manuscript_submission/figures/figS6.pdf")
out_png = Path("outputs/manuscript_submission/figures/png/figS6.png")
out_pdf.parent.mkdir(parents=True, exist_ok=True)
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Saved reframed Fig S6 to {out_pdf} and {out_png}")
