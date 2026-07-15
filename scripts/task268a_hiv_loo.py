#!/usr/bin/env python
"""
Task 268A: HIV-1 Leave-One-Out Leakage Quantification

Follows analyze_per_antigen_r1.py protocol exactly (build_bcr_frame,
compute_shared_labels top_n=10/min_count=14, per-label cap 150,
min_clones_per_label=2). Adds leakage detection on HIV-1 test CDR3s.
"""
import sys
import json
from pathlib import Path

# analyze_per_antigen_r1 lives in scripts/analysis; add both dirs to path
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "analysis"))

import numpy as np
import pandas as pd
import parasail
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize

from benchmark.data import (
    build_dataset_specs, load_table, filter_human_only,
    init_label_config, compute_shared_labels, normalize_labels,
    build_pilot_slice,
)
from benchmark.split import clone_aware_split_fast
from benchmark.embeddings import OfflineEmbedder
from benchmark.evaluate import retrieval_metrics, compute_similarity_matrix
from run_v34_experiments import MockArgs  # build_bcr_frame superseded by build_pilot_slice

# Known HIV-1 bNAb HCDR3 reference sequences -- CANONICAL set, identical to the
# run_task256_s.py audit (24 bNAbs across CD4bs / V1V2 / V3-glycan / MPER / outer-domain
# classes). The previous 4-entry list here was a truncated, partly-fabricated stub that
# matched NO real CDR3 (VRC01-class HCDR3 is TRGKNCDYNWDFEH-like, not GGYSSS...).
_BNAB_REFERENCES = {
    "VRC01": ["TRGKNCDYNWDFEH", "TRGKNCDYNWDFE", "TRGKNCDYNWDFEHP", "GKNCDYNWDFEH"],
    "VRC03": ["TRGKYCTARDYYNWDFEH", "TRGKYCTARDYYNWDFE", "RGKYCTARDYYNWDFEH"],
    "VRC06": ["ARDYYNWDFEH"],
    "VRC13": ["ARDRSGYDDWLDY"],
    "VRC16": ["ASGKYCTARDYYNWDFEH"],
    "VRC18": ["ARDYYDFGGPLYYGMDV"],
    "VRC27": ["ASGYTDFGGELYYGMDV"],
    "3BNC117": ["ARDRSGYDDWLDY", "TRDRSGYDDWLDY"],
    "NIH45-46": ["TRGKTYCTARDYYNWDFEH", "TRGKTYCTARDYYNWDFE"],
    "CH31": ["ARDYYDFGGXVYGPNDY"],
    "CH103": ["ARDRYDFWSGYPPYYYYMDV", "AREGRQYGDYGGNYYGMDV"],
    "PG9": ["AREGVTGYYDFWSGYPPYYYYMDV", "AREGVTGYYDFWSGYPPYYYYMD", "REGVTGYYDFWSGYPPYYYYMDV"],
    "PG16": ["AREGVTGYYDFYSGYPPYYYYMDV", "AREGVTGYYDFYSGYPPYYYYMD"],
    "PGT145": ["ARRGQRIYGIVADFDF"],
    "CH01": ["AREHGTTGWGWLGKPGAFDI"],
    "CAP256-VRC26": ["ARREWYTGWGWLGPNDY"],
    "PGT121": ["AREGNYYGMDV", "AREGNYYGMD"],
    "PGT128": ["ARGGDFSGPDAFDI"],
    "10-1074": ["ARHRHGPSSWYPDAFDI"],
    "2F5": ["ARESLRRGGYFDY"],
    "4E10": ["AREGWGWLGKPGAFDI"],
    "10E8": ["ARGRGWGWLGKPGAFDI"],
    "2G12": ["ARDRGPFRGPNWYFDV", "ARDRGPHRYPNWYFDV"],
}
KNOWN_BNAB_CDR3S = [s for refs in _BNAB_REFERENCES.values() for s in refs]

SEEDS    = [42 + i * 10 for i in range(20)]
EMB_PATH = "outputs/embeddings/bcr_level4_esm2-150m_mean.npy"
HIV_LABEL = "hiv-1 envelope"
OUT_DIR  = Path("outputs/task268a_results")


def levenshtein_identity(a: str, b: str) -> float:
    a, b = str(a).strip(), str(b).strip()
    if not a or not b:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        curr = [i + 1]
        for j, c2 in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return 1.0 - prev[-1] / max(len(a), len(b))


def is_leaked(cdr3: str, threshold: float = 0.80) -> bool:
    if pd.isna(cdr3) or not str(cdr3).strip():
        return False
    return any(levenshtein_identity(str(cdr3), ref) >= threshold for ref in KNOWN_BNAB_CDR3S)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    init_label_config("label_aliases.json")
    mock_args = MockArgs()
    spec      = build_dataset_specs(mock_args)[0]
    df_raw    = load_table(spec)
    df_raw    = filter_human_only(df_raw, spec)

    embedder       = OfflineEmbedder(EMB_PATH)
    forced_labels  = compute_shared_labels(df_raw, spec, 10, 14, None)
    print(f"[*] Forced label set ({len(forced_labels)}): {forced_labels}")

    # Collect all leaked CDR3s in the dataset for reporting
    all_hiv_cdr3s = normalize_labels(df_raw[spec.label_col])
    hiv_rows = df_raw[all_hiv_cdr3s == HIV_LABEL]
    cdr3_col = next(c for c in ["Chain 1_CDR3 ANARCI", "Chain 1_CDR3 Curated"] if c in df_raw.columns)
    unique_hiv_cdr3s = hiv_rows[cdr3_col].dropna().unique()
    leaked_unique = [c for c in unique_hiv_cdr3s if is_leaked(c)]
    print(f"[*] Total unique HIV-1 CDR3s in dataset: {len(unique_hiv_cdr3s)}")
    print(f"[*] Leaked CDR3s detected: {leaked_unique}")

    if leaked_unique:
        rows = []
        for q in leaked_unique:
            for ref in KNOWN_BNAB_CDR3S:
                sim = levenshtein_identity(q, ref)
                if sim >= 0.80:
                    rows.append({"query_cdr3": q, "ref_cdr3": ref, "similarity": sim})
        pd.DataFrame(rows).to_csv(OUT_DIR / "leaked_cdr3s.csv", index=False)

    results = []
    for seed in SEEDS:
        # Use the CANONICAL slice builder (identical frame + row order to the grand_slam tree)
        # so the clone-aware split partition matches Table 1 / Table S8 exactly; the only
        # manipulation in this analysis is the leaked-sequence removal.
        slice_ = build_pilot_slice(df_raw, spec, "level4", 150, seed, forced_labels)
        frame = slice_.dataframe.rename(columns={slice_.sequence_col: "sequence"})

        train_idx, test_idx = clone_aware_split_fast(
            frame, cdr3_col="split_sequence", label_col="label",
            test_size=0.2, clone_similarity_threshold=0.95,
            random_state=seed, min_clones_per_label=2, show_progress=False
        )

        train_df = frame.iloc[train_idx].reset_index(drop=True)
        test_df  = frame.iloc[test_idx].reset_index(drop=True)

        # HIV-1 in test
        hiv_mask = test_df["label"] == HIV_LABEL
        df_hiv_full = test_df[hiv_mask].reset_index(drop=True)
        n_full = len(df_hiv_full)

        if n_full == 0:
            print(f"  seed={seed}: no HIV-1 in test, skipping")
            continue

        # Leakage check (CDR3 = split_sequence in frame)
        df_hiv_full = df_hiv_full.copy()
        df_hiv_full["leaked"] = df_hiv_full["split_sequence"].apply(is_leaked)
        n_leaked = int(df_hiv_full["leaked"].sum())
        df_hiv_clean = df_hiv_full[~df_hiv_full["leaked"]].reset_index(drop=True)
        n_clean = len(df_hiv_clean)

        if n_clean == 0:
            print(f"  seed={seed}: all HIV-1 test sequences leaked, skipping")
            continue

        # Embeddings (normalize once)
        d_seqs   = train_df["sequence"].tolist()
        d_lbls   = train_df["label"].values
        d_emb    = normalize(embedder.encode(d_seqs).astype(np.float32))

        # Both methods scored with the CANONICAL order-independent expected-R@1 (retrieval_metrics),
        # identical to the grand_slam tree / Table 1 -- NOT single-pick argmax. The HIV subset is
        # ranked against the full training pool (all antigens).
        def cosine_r1(q_df):
            q_emb = normalize(embedder.encode(q_df["sequence"].tolist()).astype(np.float32))
            sim   = q_emb @ d_emb.T
            m = retrieval_metrics(
                q_emb=q_emb, d_emb=d_emb,
                q_labels=q_df["label"].values, d_labels=d_lbls,
                q_antigen_types=q_df["antigen_type"].values,
                method="esm2-150m", precomputed_sim=sim,
            )
            return float(m["recall@1"])

        def blosum_r1(q_df):
            q_seqs = q_df["sequence"].tolist()
            def row(q): return np.array([
                parasail.sw_striped_16(q, d, 10, 1, parasail.blosum62).score / max(len(q), len(d))
                for d in d_seqs
            ])
            sim = np.vstack(Parallel(n_jobs=-1)(delayed(row)(q) for q in q_seqs))
            m = retrieval_metrics(
                q_emb=None, d_emb=None,
                q_labels=q_df["label"].values, d_labels=d_lbls,
                q_antigen_types=q_df["antigen_type"].values,
                method="blosum62", precomputed_sim=sim,
            )
            return float(m["recall@1"])

        r1_esm_full  = cosine_r1(df_hiv_full)
        r1_esm_clean = cosine_r1(df_hiv_clean)
        r1_bl_full   = blosum_r1(df_hiv_full)
        r1_bl_clean  = blosum_r1(df_hiv_clean)

        results.append({
            "seed":           seed,
            "n_test_full":    n_full,
            "n_test_clean":   n_clean,
            "n_leaked":       n_leaked,
            "esm2_r1_full":   r1_esm_full,
            "esm2_r1_clean":  r1_esm_clean,
            "blosum62_r1_full":  r1_bl_full,
            "blosum62_r1_clean": r1_bl_clean,
        })
        print(f"  seed={seed}: n_full={n_full}, n_leaked={n_leaked}, "
              f"esm2_full={r1_esm_full:.3f}, esm2_clean={r1_esm_clean:.3f}, "
              f"blosum62_full={r1_bl_full:.3f}, blosum62_clean={r1_bl_clean:.3f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_DIR / "hiv_loo_per_seed.csv", index=False)

    cols = ["n_test_full", "n_test_clean", "n_leaked",
            "esm2_r1_full", "esm2_r1_clean", "blosum62_r1_full", "blosum62_r1_clean"]
    summary = pd.DataFrame({
        "metric": cols,
        "mean": [df_results[c].mean() for c in cols],
        "std":  [df_results[c].std()  for c in cols],
    })
    summary.to_csv(OUT_DIR / "hiv_loo_summary.csv", index=False)

    print(f"\n[+] Results saved to {OUT_DIR}/")
    print(summary.to_string(index=False))
    print(f"\nESM2 memorisation contribution: "
          f"{df_results['esm2_r1_full'].mean() - df_results['esm2_r1_clean'].mean():.4f}")
    print(f"ESM2 advantage (clean set, pp): "
          f"{(df_results['esm2_r1_clean'].mean() - df_results['blosum62_r1_clean'].mean()) * 100:.2f}")


if __name__ == "__main__":
    main()
