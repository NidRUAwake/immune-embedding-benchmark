#!/usr/bin/env python3
"""Per-antigen BCR R@1 (S7=L1, S8=L4) stratified directly from the CANONICAL tree.

Rationale: the canonical grand_slam_tie_v2 tree already dumps, per query, the
order-independent expected R@1 (`exp_recall@1`) alongside the query's `true_label`.
Stratifying these per-query records by antigen reproduces the per-antigen breakdown
WITHOUT re-running retrieval, so it is guaranteed consistent with Table 1 / Table S1
(same input filter, same split, same metric). The previous driver
(analyze_per_antigen_r1.py) read the OLD input + used single-pick ranks==0; this
replaces it.

Per-seed mean R@1 within each antigen, then mean +/- SD across the 20 seeds (matches
the S8 caption "mean +/- SD across 20 seeds").
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TREE = Path("outputs/phase3_filtered_tie_v2/bcr")  # canonical POST-is_legit-filter tree
# (NOT phase3_grand_slam_tie_v2/bcr, which is the stale pre-filter tree dated 05-30)
SEEDS = [42 + 10 * i for i in range(20)]
METHODS = ["levenshtein", "blosum62", "esm2-150m", "esm2-650m", "esm2-3b"]
LEVELS = {"S7_L1": "level1", "S8_L4": "level4"}


def load_per_query(method: str, level: str, seed: int) -> pd.DataFrame | None:
    f = TREE / method / f"seed_{seed}" / f"BCR_{level}_{method}_predictions.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df[["true_label", "exp_recall@1"]].copy()


def aggregate(level: str) -> pd.DataFrame:
    # per (method, antigen): list of per-seed means + per-seed n
    rows = []
    for method in METHODS:
        # collect per-seed per-antigen mean
        per_seed = {}  # antigen -> list of (seed_mean, n)
        for seed in SEEDS:
            df = load_per_query(method, level, seed)
            if df is None:
                continue
            g = df.groupby("true_label")["exp_recall@1"]
            means = g.mean()
            counts = g.size()
            for ag in means.index:
                per_seed.setdefault(ag, []).append((float(means[ag]), int(counts[ag])))
        for ag, vals in per_seed.items():
            seed_means = np.array([v[0] for v in vals])
            ns = np.array([v[1] for v in vals])
            rows.append({
                "method": method,
                "antigen": ag,
                "n_seeds": len(vals),
                "n_test_per_seed": float(ns.mean()),
                "r1_mean": float(seed_means.mean()),
                "r1_sd": float(seed_means.std(ddof=1)) if len(vals) > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def main():
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, level in LEVELS.items():
        df = aggregate(level)
        df = df.sort_values(["antigen", "method"]).reset_index(drop=True)
        out = out_dir / f"per_antigen_{tag}_canonical.csv"
        df.to_csv(out, index=False)
        print(f"\n===== {tag} ({level}) -> {out} =====")
        # pivot for readability: antigen x method (r1_mean)
        piv = df.pivot(index="antigen", columns="method", values="r1_mean")
        piv = piv.reindex(columns=[m for m in METHODS if m in piv.columns])
        # attach mean n_test/seed (from esm2-150m or any)
        nser = df[df["method"] == "blosum62"].set_index("antigen")["n_test_per_seed"]
        piv.insert(0, "n_test/seed", nser)
        with pd.option_context("display.width", 200, "display.max_columns", 20,
                               "display.float_format", lambda x: f"{x:.3f}"):
            print(piv.sort_values("n_test/seed", ascending=False))
        # global micro-average (pool all queries, all seeds) per method
        print("--- global micro-average (per-seed-mean-then-avg over ALL antigens) ---")
        gm = df.groupby("method").apply(
            lambda s: np.average(s["r1_mean"], weights=s["n_test_per_seed"])
        )
        for m in METHODS:
            if m in gm.index:
                print(f"  {m:12s} {gm[m]:.3f}")


if __name__ == "__main__":
    main()
