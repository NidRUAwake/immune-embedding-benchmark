#!/usr/bin/env python3
"""Decompose the PLM's L3->L4 framework gain into germline-framework vs framework-SHM.

Three BCR inputs on identical clone-aware splits (20 seeds, canonical top-10 labels, expected-R@1):
  L3      = native CDR1|CDR2|CDR3 (no framework)
  germfw  = germline FR1-3 + native CDR1/2/3 + native FW4  (framework reverted to germline)
  L4      = native full VH

Decomposition (per method): L3 -> germfw measures the GERMLINE framework's contribution;
germfw -> L4 measures the FRAMEWORK-SHM contribution.
  germline shortcut  => for the PLM, L3->germfw ~ L3->L4 and germfw->L4 ~ 0
  framework-SHM signal => germfw->L4 > 0 (framework mutations carry real signal)

ESM2-150M embeddings computed on CPU (cached); BLOSUM62 via parasail. Run OFFLINE_EMBED_STRICT=0.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU, deterministic
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
from scipy import stats

from benchmark.data import (build_dataset_specs, load_table, filter_human_only,
                            normalize_labels, is_legit_sequence, init_label_config,
                            compute_shared_labels, clean_sequence)
from benchmark.split import clone_aware_split_fast
from benchmark.embeddings import get_embedder
from benchmark.evaluate import retrieval_metrics

SEEDS = [42 + 10 * i for i in range(int(os.environ.get("NSEEDS", "20")))]
MAX_PER_LABEL = 150


class M:
    include_bcr = True; include_tcr = False; include_sabdab = False
    bcr_file = "raw/iedb/bcr_singlechain_vh.tsv"; bcr_label_col = "Epitope_Source Molecule"
    human_only = True; top_labels = 10; min_label_count = 14; max_per_label_bcr = MAX_PER_LABEL
    shared_labels = True; batch_size = 32; local_files_only = False
    offline_dir = None; pooling = "mean"


def main():
    init_label_config("label_aliases.json")
    m = M(); spec = build_dataset_specs(m)[0]
    df = load_table(spec); df = filter_human_only(df, spec)
    labels = normalize_labels(df[spec.label_col])
    l1 = spec.sequence_builders["level1"](df)     # CDR3 (clone key)
    l3 = spec.sequence_builders["level3"](df)      # CDR1|CDR2|CDR3
    l4 = spec.sequence_builders["level4"](df)      # full VH
    gf = df["Chain 1_germfw"].map(clean_sequence) if "Chain 1_germfw" in df else pd.Series([None]*len(df))
    gc = df["Chain 1_germcdr"].map(clean_sequence) if "Chain 1_germcdr" in df else pd.Series([None]*len(df))

    work = pd.DataFrame({"label": labels, "l1": l1, "l3": l3, "l4": l4, "gf": gf, "gc": gc,
                         "antigen_type": labels.map(lambda x: "protein")}).dropna()
    # keep rows legit at ALL four input levels so the decomposition is paired per query
    for c in ("l3", "l4", "gf", "gc"):
        work = work[work[c].map(is_legit_sequence)]
    keep = compute_shared_labels(df, spec, 10, 14)
    work = work[work["label"].isin(keep)].reset_index(drop=True)
    if os.environ.get("LEAKFREE"):
        try:
            flagged = set(pd.read_csv("outputs/task256_s_hiv_leakage_audit/hiv_leakage_hits.csv")
                          ["query_cdr3"].astype(str))
            n0 = len(work); work = work[~work["l1"].isin(flagged)].reset_index(drop=True)
            print(f"LEAKFREE: dropped {n0 - len(work)} flagged-bNAb CDR3 rows")
        except Exception as e:
            print("leak file err", e)
    print(f"working set: {len(work)} seqs, {work['label'].nunique()} labels, "
          f"germ-fw coverage retained (all 3 levels legit)")

    emb = get_embedder("esm2-150m", m)
    bl = get_embedder("blosum62", m)
    LEVELS = {"L3": "l3", "germfw": "gf", "germcdr": "gc", "L4": "l4"}

    # Precompute ESM2-150M embeddings ONCE for every unique sequence (cache), reused across seeds.
    uniq = sorted(set(work["l3"]) | set(work["gf"]) | set(work["gc"]) | set(work["l4"]))
    print(f"embedding {len(uniq)} unique sequences with ESM2-150M (CPU, one pass)...")
    mat = emb.encode(uniq)
    e2 = {s_: mat[i] for i, s_ in enumerate(uniq)}

    per = {(lv, mo): [] for lv in LEVELS for mo in ("blosum62", "esm2-150m")}
    # stratify the framework-SHM gain (germfw->L4, ESM2) by antigen: HIV-1 envelope vs the rest
    strat_per = {(st, lv): [] for st in ("HIV", "nonHIV") for lv in ("germfw", "L4")}
    n_hiv = []
    for s in SEEDS:
        # one clone-aware split per seed on the L1 CDR3 key, reused for all three inputs
        cap = (work.groupby("label", group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), MAX_PER_LABEL), random_state=s))
                   .sample(frac=1.0, random_state=s).reset_index(drop=True))
        tr, te = clone_aware_split_fast(cap, cdr3_col="l1", label_col="label", test_size=0.2,
                                        clone_similarity_threshold=0.95, random_state=s,
                                        min_clones_per_label=2, show_progress=False)
        trd = cap.iloc[tr]; ted = cap.iloc[te]
        tl = ted["label"].to_numpy(); dl = trd["label"].to_numpy(); ta = ted["antigen_type"].to_numpy()
        for lv, col in LEVELS.items():
            qs = ted[col].tolist(); ds = trd[col].tolist()
            for mo in ("blosum62", "esm2-150m"):
                if mo == "esm2-150m":
                    qe = np.array([e2[x] for x in qs]); de = np.array([e2[x] for x in ds])
                else:
                    qe = bl.encode(qs); de = bl.encode(ds)
                mm = retrieval_metrics(q_emb=qe, d_emb=de,
                                       q_labels=tl, d_labels=dl, q_antigen_types=ta, method=mo)
                per[(lv, mo)].append(mm["recall@1"])
        # --- HIV vs non-HIV stratification of germfw/L4 (ESM2), queries vs full DB ---
        hiv = np.array([("hiv" in str(x).lower()) or ("envelope" in str(x).lower()) for x in tl])
        n_hiv.append(int(hiv.sum()))
        for st, mask in (("HIV", hiv), ("nonHIV", ~hiv)):
            for lv in ("germfw", "L4"):
                if mask.sum() < 3:
                    strat_per[(st, lv)].append(np.nan); continue
                col = LEVELS[lv]
                qe = np.array([e2[x] for x in np.array(ted[col].tolist())[mask]])
                de = np.array([e2[x] for x in trd[col].tolist()])
                mm = retrieval_metrics(q_emb=qe, d_emb=de, q_labels=tl[mask], d_labels=dl,
                                       q_antigen_types=ta[mask], method="esm2-150m")
                strat_per[(st, lv)].append(mm["recall@1"])
        print(f"seed {s}: " + " | ".join(
            f"{lv} {mo.split('-')[0][:4]} {per[(lv,mo)][-1]:.3f}"
            for lv in LEVELS for mo in ("blosum62", "esm2-150m")))

    def arr(lv, mo): return np.array(per[(lv, mo)])
    rows = []
    print("\n=== means (R@1) ===")
    for mo in ("blosum62", "esm2-150m"):
        print(f"  {mo:10s}  L3 {arr('L3',mo).mean():.3f}  germfw {arr('germfw',mo).mean():.3f}  L4 {arr('L4',mo).mean():.3f}")
        rows += [dict(model=mo, level=lv, mean=round(arr(lv,mo).mean(),4),
                      sd=round(arr(lv,mo).std(ddof=1),4)) for lv in LEVELS]

    def paired(a, b):
        d = a - b; n = len(d); se = d.std(ddof=1)/np.sqrt(n)
        t, p = stats.ttest_rel(a, b)
        return d.mean(), d.mean()-1.96*se, d.mean()+1.96*se, p
    print("\n=== decomposition (per-seed paired) ===")
    for mo in ("blosum62", "esm2-150m"):
        for a, b, name in [("germfw","L3","germline-FW gain (L3->germfw)"),
                           ("L4","germfw","framework-SHM gain (germfw->L4)"),
                           ("L4","L3","total framework gain (L3->L4)"),
                           ("L4","germcdr","CDR1/2-SHM gain (germcdr->L4)")]:
            m_,lo,hi,p = paired(arr(a,mo), arr(b,mo))
            print(f"  {mo:10s} {name:32s} {m_:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3g}")

    print(f"\n=== framework-SHM gain (germfw->L4, ESM2) by antigen  (mean HIV queries/seed = {np.mean(n_hiv):.0f}) ===")
    for st in ("HIV", "nonHIV"):
        gf_ = np.array(strat_per[(st, "germfw")], float); l4_ = np.array(strat_per[(st, "L4")], float)
        d = l4_ - gf_; d = d[~np.isnan(d)]
        if len(d) >= 3:
            se = d.std(ddof=1)/np.sqrt(len(d)); _, p = stats.ttest_1samp(d, 0)
            print(f"  {st:7s}: germfw {np.nanmean(gf_):.3f} -> L4 {np.nanmean(l4_):.3f}  "
                  f"framework-SHM gain {d.mean():+.4f} [{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}] p={p:.3g} (n={len(d)})")

    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv("outputs/reports/germfw_decomposition_20seed.csv", index=False)
    print("\nSaved outputs/reports/germfw_decomposition_20seed.csv")


if __name__ == "__main__":
    main()
