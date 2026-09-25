"""Shared figure style — colorblind-safe (Okabe-Ito) palette + Bioinformatics formatting.
Import in every figure script so colors and formatting stay consistent across the paper.

Usage:
    import figstyle; figstyle.apply()
    color = figstyle.METHOD["blosum62"]
"""
import matplotlib as mpl

# Okabe-Ito colorblind-safe palette
OKABE = {
    "black":    "#000000",
    "orange":   "#E69F00",
    "skyblue":  "#56B4E9",
    "green":    "#009E73",
    "yellow":   "#F0E442",
    "blue":     "#0072B2",
    "vermillion":"#D55E00",
    "purple":   "#CC79A7",
    "gray":     "#999999",
}

# Consistent method -> color (use everywhere)
METHOD = {
    "blosum62":   OKABE["blue"],
    "levenshtein":OKABE["skyblue"],
    "esm2-150m":  OKABE["vermillion"],
    "esm2-650m":  OKABE["green"],
    "esm2-3b":    OKABE["purple"],
    "ablang":     OKABE["orange"],
    "antiberty":  OKABE["gray"],
    "tcr-bert":   OKABE["purple"],
    "tcrdist":    OKABE["black"],
}
# Distinct markers + linestyles per method, so series are distinguishable WITHOUT color
# (Bioinformatics accessibility guidance: don't convey meaning by color alone).
MARKER = {"blosum62":"o","levenshtein":"s","esm2-150m":"^","esm2-650m":"D",
          "esm2-3b":"v","tcr-bert":"P","ablang":"X","tcrdist":"*"}
LINESTYLE = {"blosum62":"-","levenshtein":"--","esm2-150m":"-.","esm2-650m":(0,(1,1)),
             "esm2-3b":(0,(3,1,1,1)),"tcr-bert":(0,(5,1)),"ablang":"-","tcrdist":"-"}

# Granularity bars (L1 vs L4) and alignment-vs-PLM contrasts
LEVEL = {"L1": OKABE["blue"], "L4": OKABE["orange"]}
ALIGN = OKABE["blue"]      # alignment family
PLM   = OKABE["vermillion"] # PLM family
REF_LINE = OKABE["gray"]

def apply(dpi=300):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,   # editable text (TrueType) per Bioinformatics
        "ps.fonttype": 42,
        "savefig.dpi": dpi,
        "figure.dpi": 120,
        "axes.linewidth": 0.8,
    })
