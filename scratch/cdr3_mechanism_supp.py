#!/usr/bin/env python3
"""Supplementary robustness for the CDR3-mechanism (geometry vs representation) result.

Three checks, all reusing the canonical pipeline (raw baselines must reproduce the
manuscript values), transforms fit on the reference set only:
  A. Cross-model (L1): ESM2-150M/650M/3B and ESM-C-300m -- is the
     "anisotropic but whitening barely helps" pattern PLM-general, not 150M-specific?
  B. L4 control (ESM2-150M full chain, where PLM reaches parity with alignment):
     is L4 also anisotropic? If yes, anisotropy cannot be what causes the L1 deficit.
  C. Metric (ESM2-150M L1): euclidean vs cosine, raw and whitened -- is the deficit
     cosine-specific (a metric artifact) or general (representational)?
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

sys.argv = ["run.py", "--include-bcr", "--bcr-file", "raw/iedb/bcr_singlechain_vh.tsv",
            "--top-labels", "10", "--min-label-count", "14", "--levels", "level1",
            "--human-only", "--test-size", "0.2", "--clone-threshold", "0.95",
            "--offline-dir", "outputs/embeddings", "--models", "esm2-150m",
            "--output-dir", "/tmp/cdr3mech", "--random-state", "42"]
args = run.parse_args()
cfg = yaml.safe_load(open(args.config))
if "evaluation" in cfg:
    for k, v in cfg["evaluation"].items(): setattr(args, k, v)
if "scale_up" in cfg:
    args.top_labels = cfg["scale_up"].get("bcr_top_labels", args.top_labels)
    args.min_label_count = cfg["scale_up"].get("bcr_min_label_count", args.min_label_count)
init_label_config(args.label_config)
bcr = [s for s in build_dataset_specs(args) if s.name == "BCR"][0]
df0 = filter_human_only(load_table(bcr), bcr)
forced = compute_shared_labels(df0, bcr, args.top_labels, args.min_label_count, args.bcr_label_contains)
seeds = [42 + i * 10 for i in range(20)]

def whiten(d, q, r=128):                       # PCA-whiten in top-r subspace, fit on d
    d = d.astype(np.float64); q = q.astype(np.float64)
    mu = d.mean(0); dc, qc = d - mu, q - mu
    U, S, Vt = np.linalg.svd(dc, full_matrices=False)
    r = min(r, len(S) - 1); Vr, Sr = Vt[:r], S[:r] + 1e-8
    return (dc @ Vr.T) / Sr, (qc @ Vr.T) / Sr

def aniso(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n = min(len(Xn), 400); C = Xn[:n] @ Xn[:n].T
    return float(C[np.triu_indices(n, 1)].mean())

def r1(d, q, dl, ql, qt, metric="cosine"):
    return retrieval_metrics(q_emb=q, d_emb=d, q_labels=ql, d_labels=dl, q_antigen_types=qt,
                             method="x", distance_metric=metric, exclude_self=False)["recall@1"]

def splits_for(level):                         # per-seed clone-aware splits at a level
    out = []
    for seed in seeds:
        sl = build_pilot_slice(df0, bcr, level, args.max_per_label_bcr, seed, forced, args.bcr_label_contains)
        df = sl.dataframe
        cc = "split_sequence" if "split_sequence" in df.columns else sl.sequence_col
        tr, te = clone_aware_split_fast(df, cdr3_col=cc, label_col="label", test_size=0.2,
                                        clone_similarity_threshold=0.95, random_state=seed,
                                        min_clones_per_label=2, show_progress=False)
        sc = sl.sequence_col
        out.append((df[sc].values[tr], df[sc].values[te],
                    df["label"].values[tr], df["label"].values[te], df["antigen_type"].values[te]))
    return out

def evaluate(level, model_file, metric="cosine", do_whiten=True):
    sp = splits_for(level)
    e = OfflineEmbedder(f"outputs/embeddings/{model_file}")
    raw, wh, an = [], [], []
    for (d_seq, q_seq, dl, ql, qt) in sp:
        d, q = e.encode(d_seq), e.encode(q_seq)
        an.append(aniso(d))
        raw.append(r1(d, q, dl, ql, qt, metric))
        if do_whiten:
            dw, qw = whiten(d, q)
            wh.append(r1(dw, qw, dl, ql, qt, metric))
    raw, an = np.array(raw), np.array(an)
    s = "  raw=%.3f" % raw.mean()
    if do_whiten:
        wh = np.array(wh); dd = wh - raw
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)) + 1e-12)
        s += "  whiten=%.3f (dW=%+.3f t=%.1f)" % (wh.mean(), dd.mean(), t)
    s += "  anisotropy=%.3f" % an.mean()
    return s

print("=== A. Cross-model, BCR L1, cosine (raw must reproduce Table 1; BLOSUM62 L1=0.459) ===")
for m in ["esm2-150m", "esm2-650m", "esm2-3b", "esmc-300m"]:
    print("  %-10s %s" % (m, evaluate("level1", f"bcr_level1_{m}_mean.npy")))

print("\n=== B. L4 control, BCR L4, cosine (raw must reproduce ~parity; ESM2-150M L4=0.514) ===")
for m in ["esm2-150m", "esm2-650m", "esm2-3b"]:
    print("  %-10s %s" % (m, evaluate("level4", f"bcr_level4_{m}_mean.npy")))

print("\n=== C. Metric: ESM2-150M L1, euclidean vs cosine ===")
print("  cosine     %s" % evaluate("level1", "bcr_level1_esm2-150m_mean.npy", "cosine"))
print("  euclidean  %s" % evaluate("level1", "bcr_level1_esm2-150m_mean.npy", "euclidean"))
