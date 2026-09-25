#!/usr/bin/env python3
"""Supplementary Figure S5 - IEDB BCR CDR3 amino-acid composition vs UniProt background.

Per-residue frequency of the 20 standard amino acids in the IEDB BCR CDR3 benchmark
sequences (level-1 CDR3 builder over the human-filtered benchmark set) compared with the
Swiss-Prot / UniProt average composition. Descriptive control (no model, no embeddings): it
shows the V(D)J junctional composition bias (Y/D/A/G enriched; L/E/I depleted) and that this
does not introduce systematic bias into the retrieval evaluation.

Logic extracted verbatim from scratch/recompute_supp_figs.py (the original combined S1/S3/S5
generator); this standalone version drops the embedding-dependent S1/S3 parts so figS5 can be
regenerated without the embedding cache.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/ on path
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
import figstyle; figstyle.apply()
from benchmark.data import build_dataset_specs, load_table, filter_human_only


class A:
    include_bcr = True; include_tcr = False; include_sabdab = False
    bcr_file = "raw/iedb/bcr_singlechain_vh.tsv"; bcr_label_col = "Epitope_Source Molecule"
    human_only = True


spec = build_dataset_specs(A())[0]
df = load_table(spec); df = filter_human_only(df, spec)
seqs = spec.sequence_builders["level1"](df).dropna().astype(str)   # benchmark CDR3 set

AAs = list("ACDEFGHIKLMNPQRSTVWY")
c = Counter("".join(seqs)); tot = sum(c[a] for a in AAs)
freq = {a: c.get(a, 0) / tot for a in AAs}
# UniProt background (Swiss-Prot average composition)
uni = {"A": 0.0825, "R": 0.0553, "N": 0.0406, "D": 0.0546, "C": 0.0137, "Q": 0.0393, "E": 0.0672,
       "G": 0.0707, "H": 0.0227, "I": 0.0596, "L": 0.0966, "K": 0.0584, "M": 0.0242, "F": 0.0386,
       "P": 0.0470, "S": 0.0656, "T": 0.0534, "W": 0.0108, "Y": 0.0292, "V": 0.0687}

out = Path("outputs/manuscript_submission/figures"); (out / "png").mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7.5, 4)); x = np.arange(20); bw = 0.4
ax.bar(x - bw / 2, [freq[a] for a in AAs], bw, color=figstyle.OKABE["blue"],
       edgecolor="black", lw=0.4, label="IEDB BCR CDR3")
ax.bar(x + bw / 2, [uni[a] for a in AAs], bw, color=figstyle.OKABE["gray"],
       edgecolor="black", lw=0.4, label="UniProt background")
ax.set_xticks(x); ax.set_xticklabels(AAs, fontsize=9); ax.set_xlabel("Amino acid", fontsize=10.5)
ax.set_ylabel("Frequency", fontsize=11); ax.legend(fontsize=9, frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig(out / "figS5.pdf", bbox_inches="tight")
plt.savefig(out / "png" / "figS5.png", dpi=300, bbox_inches="tight")
plt.close()
kl = sum(freq[a] * np.log(freq[a] / uni[a]) for a in AAs if freq[a] > 0)
print(f"S5 done: KL(CDR3||UniProt)={kl:.3f}")
