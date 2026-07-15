"""Graphical abstract for the clone-aware immune receptor retrieval benchmark.

Three-panel horizontal layout intended as the Bioinformatics graphical
abstract for draft 277. Panel A schematises the representation ladder against
the heavy-chain architecture; panel B shows the CDR3/paratope/full-domain trajectory using
manuscript-reported BCR R@1 values; panel C lists the take-home findings.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

BLOSUM_COLOR = "#1f77b4"
ESM_COLOR = "#d62728"
FW_COLOR = "#dcdcdc"
CDR_COLOR = "#f4a261"
CDR3_COLOR = "#e63946"

fig = plt.figure(figsize=(14, 5.2))
gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[1.05, 1.15, 1.00],
    wspace=0.32,
    left=0.04,
    right=0.98,
    top=0.86,
    bottom=0.08,
)

fig.suptitle(
    "Simple alignment matches or beats zero-shot PLM embeddings for immune "
    "receptor retrieval; full-length parity points to a germline shortcut",
    fontsize=12.5,
    weight="bold",
    y=0.97,
)


# ----------------------------------------------------------------------
# Panel A: heavy-chain schematic with representation brackets
# ----------------------------------------------------------------------

ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(0, 100)
ax_a.set_ylim(0, 100)
ax_a.axis("off")
ax_a.set_title("A  Receptor representations", loc="left", fontsize=11.5, weight="bold")

segments = [
    ("FW1", 0, 18, FW_COLOR),
    ("CDR1", 18, 28, CDR_COLOR),
    ("FW2", 28, 42, FW_COLOR),
    ("CDR2", 42, 52, CDR_COLOR),
    ("FW3", 52, 75, FW_COLOR),
    ("CDR3", 75, 86, CDR3_COLOR),
    ("FW4", 86, 100, FW_COLOR),
]

chain_y = 72
chain_h = 10
for name, x0, x1, color in segments:
    ax_a.add_patch(
        Rectangle(
            (x0, chain_y),
            x1 - x0,
            chain_h,
            facecolor=color,
            edgecolor="black",
            linewidth=0.7,
        )
    )
    ax_a.text(
        (x0 + x1) / 2,
        chain_y + chain_h / 2,
        name,
        ha="center",
        va="center",
        fontsize=7.2,
    )

ax_a.text(50, 88, "BCR / TCR variable region", ha="center", fontsize=9, style="italic")


def bracket(y, x0, x1, label):
    ax_a.plot([x0, x0], [y, y + 2], color="black", linewidth=1)
    ax_a.plot([x1, x1], [y, y + 2], color="black", linewidth=1)
    ax_a.plot([x0, x1], [y, y], color="black", linewidth=1)
    ax_a.text((x0 + x1) / 2, y - 4.5, label, ha="center", fontsize=8.5)


bracket(60, 75, 86, "CDR3")
bracket(48, 18, 86, "Paratope (CDR1+CDR2+CDR3)")
bracket(36, 0, 100, "Full variable domain")

ax_a.text(
    50,
    24,
    "(VJ-/V-clonotype: CDR3 ranking under a germline V+J / V gene filter; a procedure, not an input representation)",
    ha="center",
    fontsize=7.5,
    color="#777777",
    style="italic",
)
ax_a.text(50, 16, "Datasets", ha="center", fontsize=9.5, weight="bold")
ax_a.text(
    50,
    9,
    "IEDB BCR  ·  VDJdb TCR  ·  SAbDab  ·  McPAS-TCR",
    ha="center",
    fontsize=8.5,
)
ax_a.text(
    50,
    2,
    "Clone-aware split: CDR3 similarity ≥ 0.95 → connected components",
    ha="center",
    fontsize=7.5,
    color="#555555",
)

# ----------------------------------------------------------------------
# Panel B: representation trajectory (BCR)
# ----------------------------------------------------------------------

ax_b = fig.add_subplot(gs[0, 1])
xs = [0, 1, 2]
xlabels = ["CDR3", "Paratope\n(CDR1+2+3)", "Full variable\ndomain"]
blosum = [0.459, 0.538, 0.522]
esm2 = [0.357, 0.445, 0.514]

ax_b.plot(
    xs,
    blosum,
    "o-",
    color=BLOSUM_COLOR,
    linewidth=2.4,
    markersize=9,
    label="BLOSUM62",
    zorder=3,
)
ax_b.plot(
    xs,
    esm2,
    "s-",
    color=ESM_COLOR,
    linewidth=2.4,
    markersize=9,
    label="ESM2-150M",
    zorder=3,
)

# Per-point label placement; for the L4 convergence (markers nearly coincident),
# put BLOSUM62 to the left and ESM2-150M to the right to avoid overlap.
label_offsets = {
    "blosum": [(0, 0.024), (0, 0.024), (-0.12, -0.005)],   # L1 above, L3 above, L4 left
    "esm2":   [(0, -0.035), (0, -0.035), (0.12, -0.005)],  # L1 below, L3 below, L4 right
}
ha_blosum = ["center", "center", "right"]
ha_esm2   = ["center", "center", "left"]

for i, (x, y) in enumerate(zip(xs, blosum)):
    dx, dy = label_offsets["blosum"][i]
    ax_b.text(x + dx, y + dy, f"{y:.3f}", color=BLOSUM_COLOR,
              ha=ha_blosum[i], va="center", fontsize=8.5)

for i, (x, y) in enumerate(zip(xs, esm2)):
    dx, dy = label_offsets["esm2"][i]
    ax_b.text(x + dx, y + dy, f"{y:.3f}", color=ESM_COLOR,
              ha=ha_esm2[i], va="center", fontsize=8.5)

ax_b.set_xticks(xs)
ax_b.set_xticklabels(xlabels, fontsize=9.5)
ax_b.set_ylabel("BCR Recall@1", fontsize=10.5)
ax_b.set_ylim(0.15, 0.62)
ax_b.set_title(
    "B  Alignment leads on CDR3; parity at full length (BCR)",
    loc="left",
    fontsize=11.5,
    weight="bold",
)
ax_b.legend(loc="upper left", fontsize=9.5, frameon=False)
ax_b.grid(axis="y", alpha=0.25)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

ax_b.annotate(
    "Alignment\nadvantage",
    xy=(0, 0.459),
    xytext=(0.55, 0.40),
    fontsize=8.5,
    ha="center",
    color=BLOSUM_COLOR,
    arrowprops=dict(arrowstyle="->", color=BLOSUM_COLOR, linewidth=1),
)
ax_b.annotate(
    "PLM below alignment\nfrom CDR3 to paratope",
    xy=(1, 0.445),
    xytext=(1.45, 0.32),
    fontsize=8.5,
    ha="center",
    color=ESM_COLOR,
    arrowprops=dict(arrowstyle="->", color=ESM_COLOR, linewidth=1),
)
ax_b.annotate(
    "Convergence\n(germline-associated)",
    xy=(1.95, 0.55),
    xytext=(1.30, 0.595),
    fontsize=8.5,
    ha="center",
    color="black",
    arrowprops=dict(arrowstyle="->", color="black", linewidth=1),
)

# ----------------------------------------------------------------------
# Panel C: three-finding callouts
# ----------------------------------------------------------------------

ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_xlim(0, 100)
ax_c.set_ylim(0, 100)
ax_c.axis("off")
ax_c.set_title("C  Take-home findings", loc="left", fontsize=11.5, weight="bold")


def callout(y0, h, title, body, fill):
    ax_c.add_patch(
        FancyBboxPatch(
            (2, y0),
            96,
            h,
            boxstyle="round,pad=0.6",
            facecolor=fill,
            edgecolor="#444444",
            linewidth=0.9,
        )
    )
    # Title sits just below top edge; body uses va='top' for consistent line
    # spacing across three callouts. Linespacing 1.45 gives breathing room.
    ax_c.text(50, y0 + h - 4.5, title, ha="center", va="top",
              fontsize=9.5, weight="bold")
    ax_c.text(50, y0 + h - 11, body, ha="center", va="top",
              fontsize=8.7, linespacing=1.5)


callout(
    66,
    24,
    "CDR3-only retrieval",
    "BLOSUM62 > ESM2 across BCR, TCR,\nSAbDab and McPAS-TCR; ESM2 scaling\n(150M→650M→3B) does not close the gap",
    "#eaf2fb",
)

callout(
    36,
    24,
    "Full-length BCR convergence",
    "The PLM's full-length gain survives\nreverting the framework to germline:\na germline shortcut (working model)",
    "#fff1e0",
)

callout(
    6,
    24,
    "Clone-aware splits required",
    "Random splits inflate R@1 by\n15–28 percentage points relative to\nCDR3-similarity clone-aware splits",
    "#fde6e6",
)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

import os
_out = "outputs/manuscript_submission/figures"
os.makedirs(_out, exist_ok=True)
plt.savefig(os.path.join(_out, "Figure1_graphical_abstract.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(_out, "Figure1_graphical_abstract.png"), dpi=600, bbox_inches="tight")
print("Saved graphical abstract ->", os.path.join(_out, "Figure1_graphical_abstract.pdf"))

if __name__ == "__main__":
    plt.show()
