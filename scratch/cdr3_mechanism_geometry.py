#!/usr/bin/env python3
"""Probing WHY zero-shot ESM2 cosine retrieval trails alignment on the BCR CDR3 (L1).

Two candidate mechanisms:
  (A) embedding-space GEOMETRY -- ESM2 CDR3 embeddings are anisotropic, so cosine
      similarity is dominated by a few shared directions and loses discriminative
      power. If so, isotropy restoration (centering, removing the top principal
      components ["all-but-the-top", Mu & Viswanath 2018], or PCA-whitening, all
      FIT ON THE REFERENCE SET ONLY) should recover retrieval toward alignment.
  (B) the representation genuinely lacks the discriminative CDR3 signal -- then no
      linear geometry fix recovers it (consistent with the underpowered linear probe).

We reuse the EXACT canonical pipeline (same slice, clone-aware split, offline ESM2-150M
embeddings, expected-R@1) so the raw baseline must reproduce the manuscript's ~0.357.
Transforms are fit on the reference (train) embeddings only and applied to both sides
(no leakage). 20 canonical seeds.
"""
import os, sys
os.environ["OFFLINE_EMBED_STRICT"] = "1"
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, str(Path("scripts").resolve()))

from benchmark.data import (build_dataset_specs, load_table, compute_shared_labels,
                            build_pilot_slice, init_label_config, filter_human_only)
from benchmark.split import clone_aware_split_fast
from benchmark.embeddings import OfflineEmbedder
from benchmark.evaluate import retrieval_metrics
import benchmark.run as run

# ---- replicate run.py main() setup with the canonical BCR L1 flags (m8 driver) ----
sys.argv = ["run.py", "--include-bcr", "--bcr-file", "raw/iedb/bcr_singlechain_vh.tsv",
            "--top-labels", "10", "--min-label-count", "14", "--levels", "level1",
            "--human-only", "--test-size", "0.2", "--clone-threshold", "0.95",
            "--offline-dir", "outputs/embeddings", "--models", "esm2-150m",
            "--output-dir", "/tmp/cdr3mech", "--random-state", "42"]
args = run.parse_args()
cfg = yaml.safe_load(open(args.config))            # configs/standard_eval.yaml
if "evaluation" in cfg:
    for k, v in cfg["evaluation"].items(): setattr(args, k, v)
if "scale_up" in cfg:
    args.top_labels = cfg["scale_up"].get("bcr_top_labels", args.top_labels)
    args.min_label_count = cfg["scale_up"].get("bcr_min_label_count", args.min_label_count)
init_label_config(args.label_config)

bcr = [s for s in build_dataset_specs(args) if s.name == "BCR"][0]
df0 = filter_human_only(load_table(bcr), bcr)
forced = compute_shared_labels(df0, bcr, args.top_labels, args.min_label_count, args.bcr_label_contains)
print("forced labels (%d):" % len(forced), forced)

emb = OfflineEmbedder("outputs/embeddings/bcr_level1_esm2-150m_mean.npy")

# ---- transforms (FIT on reference d_emb, applied to both) ----
def fit_apply(d, q, kind, param=None):
    d = d.astype(np.float64); q = q.astype(np.float64)
    if kind == "raw":
        return d, q
    mu = d.mean(0)
    dc, qc = d - mu, q - mu
    if kind == "center":
        return dc, qc
    if kind == "standardize":
        sd = d.std(0) + 1e-8
        return dc / sd, qc / sd
    U, S, Vt = np.linalg.svd(dc, full_matrices=False)   # Vt rows = principal directions
    if kind == "abtt":                                   # remove top-k PCs (all-but-the-top)
        Vk = Vt[:param]
        return dc - (dc @ Vk.T) @ Vk, qc - (qc @ Vk.T) @ Vk
    if kind == "whiten":                                 # PCA-whiten in top-r subspace
        r = min(param, len(S) - 1)
        Vr, Sr = Vt[:r], S[:r] + 1e-8
        return (dc @ Vr.T) / Sr, (qc @ Vr.T) / Sr
    raise ValueError(kind)

def r1(d, q, dl, ql, qt):
    m = retrieval_metrics(q_emb=q, d_emb=d, q_labels=ql, d_labels=dl, q_antigen_types=qt,
                          method="esm2-150m", distance_metric="cosine", exclude_self=False)
    return m["recall@1"]

def anisotropy(X):                                       # mean pairwise cosine of a sample
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n = min(len(Xn), 400)
    idx = np.arange(len(Xn))[:n]
    C = Xn[idx] @ Xn[idx].T
    iu = np.triu_indices(n, 1)
    return float(C[iu].mean())

TRANSFORMS = [("raw", None), ("center", None), ("standardize", None),
              ("abtt", 1), ("abtt", 5), ("abtt", 10), ("abtt", 20),
              ("whiten", 64), ("whiten", 128)]

seeds = [42 + i * 10 for i in range(20)]
res = {f"{k}{'' if p is None else p}": [] for k, p in TRANSFORMS}
aniso_raw, aniso_cen, top1var = [], [], []
slice_sizes = []

for seed in seeds:
    sl = build_pilot_slice(df0, bcr, "level1", args.max_per_label_bcr, seed, forced, args.bcr_label_contains)
    df = sl.dataframe
    cdr3_col = "split_sequence" if "split_sequence" in df.columns else sl.sequence_col
    tr, te = clone_aware_split_fast(df, cdr3_col=cdr3_col, label_col="label", test_size=0.2,
                                    clone_similarity_threshold=0.95, random_state=seed,
                                    min_clones_per_label=2, show_progress=False)
    seqcol = sl.sequence_col
    d_seq, q_seq = df[seqcol].values[tr], df[seqcol].values[te]
    dl, ql = df["label"].values[tr], df["label"].values[te]
    qt = df["antigen_type"].values[te]
    d_emb, q_emb = emb.encode(d_seq), emb.encode(q_seq)
    slice_sizes.append((len(tr), len(te)))
    # diagnostics
    aniso_raw.append(anisotropy(d_emb))
    aniso_cen.append(anisotropy(d_emb - d_emb.mean(0)))
    s = np.linalg.svd(d_emb.astype(np.float64) - d_emb.mean(0), compute_uv=False)
    top1var.append(float((s[0]**2) / (s**2).sum()))
    for k, p in TRANSFORMS:
        dT, qT = fit_apply(d_emb, q_emb, k, p)
        res[f"{k}{'' if p is None else p}"].append(r1(dT, qT, dl, ql, qt))

print("\nslice sizes (train,test) per seed (first 3):", slice_sizes[:3])
print("anisotropy raw  (mean pairwise cosine): %.3f" % np.mean(aniso_raw))
print("anisotropy centered                   : %.3f" % np.mean(aniso_cen))
print("top-1 PC variance fraction            : %.3f" % np.mean(top1var))
print("\n=== BCR L1 ESM2-150M expected-R@1 by embedding transform (20-seed mean +/- SD) ===")
print("(raw must reproduce the canonical ~0.357; BLOSUM62 L1 reference = 0.459)")
for k, p in TRANSFORMS:
    key = f"{k}{'' if p is None else p}"
    v = np.array(res[key])
    print("  %-14s %.3f  +/- %.3f" % (key, v.mean(), v.std()))

# paired (same-seed) differences vs raw -- is any geometry fix a real, non-noise gain?
raw = np.array(res["raw"])
print("\n=== paired difference vs raw (20 seeds): mean +/- SD, and gap remaining to BLOSUM 0.459 ===")
for key in ["center", "whiten64", "whiten128", "abtt10"]:
    d = np.array(res[key]) - raw
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
    print("  %-12s d=%+.3f +/- %.3f  (t=%+.1f, %d/%d seeds up)  gap_to_BLOSUM=%.3f"
          % (key, d.mean(), d.std(ddof=1), t, (d > 0).sum(), len(d), 0.459 - np.array(res[key]).mean()))
