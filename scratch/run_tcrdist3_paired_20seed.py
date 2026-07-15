#!/usr/bin/env python3
"""
Real tcrdist3 (TCRrep) paired alpha+beta TCRdist baseline, 20 seeds.

Motivation
----------
The "tcrdist" method in scripts/benchmark/evaluate.py is NOT the TCRdist3 algorithm;
it is a BLOSUM62 CDR-weighted surrogate. On concatenated paired chains (level4_paired)
it has no CDR1|CDR2|CDR3 structure, so it falls back to plain BLOSUM62 -> byte-identical
to BLOSUM62. This script computes the *real* TCRdist (Dash et al. 2017) via
tcrdist3.TCRrep, on the SAME clone-aware split as
outputs/phase3_grand_slam_vdjdb_paired_v2 (BLOSUM62 / Levenshtein / ESM2), so the
paired-chain comparison is fair and uses a genuinely independent distance.

Method
------
For each seed we reproduce build_pilot_slice + clone_aware_split_fast EXACTLY (verified:
seed 42 -> n_train=817, matching the committed blosum62 level4_paired), while carrying
the V/J/CDR3 columns that tcrdist3 needs. We then build a TCRrep on (train+test) and
take the combined alpha+beta distance (pw_alpha + pw_beta) for the paired metric and
pw_beta alone for the single-chain metric. Similarity = -distance; ranking and metrics
reuse benchmark.evaluate.retrieval_metrics (same code path / tie handling as paired_v2).

Outputs
-------
- outputs/phase3_grand_slam_vdjdb_paired_v2/tcr/tcrdist3/seed_{S}/pilot_results.csv
  (level4_paired = paired alpha+beta; level4 = beta-only single, for the Table S4 pair)
- outputs/reports/tcrdist3_paired_v2_nested_ci.csv  (nested-bootstrap CI, 20 seeds)
"""
import sys, warnings, time, re
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd

from benchmark.data import (build_dataset_specs, load_table, filter_human_only,
                            init_label_config, normalize_labels, assign_antigen_type)
from benchmark.split import clone_aware_split_fast
from benchmark.evaluate import retrieval_metrics
from analysis.nested_bootstrap import nested_bootstrap_metrics
from tcrdist.repertoire import TCRrep

# --- config: must match run_vdjdb_paired_v2.py / run.py defaults -------------
SEEDS = [42 + 10 * i for i in range(20)]            # 42..232
MAX_PER_LABEL = 100                                  # --max-per-label-tcr default
TEST_SIZE = 0.2
CLONE_THRESHOLD = 0.95
MIN_CLONES = 2
OUT_TREE = Path("outputs/phase3_grand_slam_tie_v2/tcr_paired/tcrdist3")
REPORT = Path("outputs/reports/tcr_paired_ladder_tcrdist3_nested_ci.csv")

with open("scratch/vdjdb_paired_external_task/manifest.json") as f:
    import json
    FIXED_LABELS = [l.lower() for l in json.load(f).get("labels", [])]
assert len(FIXED_LABELS) == 10, FIXED_LABELS


def build_slice_with_genes(df, spec, level, seed):
    """Faithful reproduction of build_pilot_slice that also carries tcrdist3 columns.
    Gene columns are NOT in the dropna subset, so the kept/sampled rows are identical
    to the canonical slice (verified against paired_v2 n_train)."""
    work = pd.DataFrame(index=df.index)
    work["label"] = normalize_labels(df[spec.label_col])
    work["sequence"] = spec.sequence_builders[level](df)
    work["split_sequence"] = spec.sequence_builders["level1"](df)
    work["antigen_type"] = assign_antigen_type(work["label"])
    for new, old in [("cdr3_a_aa", "cdr3.alpha"), ("v_a_gene", "v.alpha"), ("j_a_gene", "j.alpha"),
                     ("cdr3_b_aa", "cdr3.beta"), ("v_b_gene", "v.beta"), ("j_b_gene", "j.beta")]:
        work[new] = df[old]
    work = work.dropna(subset=["label", "sequence", "split_sequence"])
    work = work[work["label"].isin(FIXED_LABELS)]
    parts = []
    for lab in work["label"].unique():
        ld = work[work["label"] == lab].copy()
        if len(ld) > MAX_PER_LABEL:
            ld = ld.sample(MAX_PER_LABEL, random_state=seed)
        parts.append(ld)
    pilot = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pilot


def build_gene_cleaner(valid):
    """Map a VDJdb gene id onto tcrdist3's reference gene set (~0.1% need rescue).
    Handles whitespace, unicode dashes, ambiguous ';' lists, Arden 'S' nomenclature,
    leading zeros, and finally a same-family fallback. Returns None if unmappable."""
    valid = set(valid)
    valid_sorted = sorted(valid)

    def clean(g):
        g = str(g).strip().replace(" ", "").replace("–", "-").replace("_", "")
        g = g.split(";")[0]                       # ambiguous list -> first
        if "*" not in g:
            g = g + "*01"
        if g in valid:
            return g
        base = g.split("*")[0]
        if base + "*01" in valid:
            return base + "*01"
        base2 = re.sub(r'(?<=[VJ])0(\d)', r'\1', base)   # TRBV06-5 -> TRBV6-5
        base2 = re.sub(r'S\d+$', '', base2)              # drop Arden S-suffix
        for cand in (base2 + "*01", base2 + "*02"):
            if cand in valid:
                return cand
        fam = re.match(r'(TR[AB][VJ]\d+)', base2)         # same-family fallback
        if fam:
            cands = [v for v in valid_sorted if v.startswith(fam.group(1))]
            if cands:
                return cands[0]
        return None
    return clean


def tcr_cells(sub, clean):
    """Return (cell_df, labels, antigen_types) for rows whose 4 genes all map; drop the rest."""
    recs, labels, types = [], [], []
    for _, r in sub.iterrows():
        va, ja = clean(r["v_a_gene"]), clean(r["j_a_gene"])
        vb, jb = clean(r["v_b_gene"]), clean(r["j_b_gene"])
        if None in (va, ja, vb, jb):
            continue
        recs.append({"cdr3_a_aa": str(r["cdr3_a_aa"]), "v_a_gene": va, "j_a_gene": ja,
                     "cdr3_b_aa": str(r["cdr3_b_aa"]), "v_b_gene": vb, "j_b_gene": jb, "count": 1})
        labels.append(r["label"]); types.append(r["antigen_type"])
    return pd.DataFrame(recs), np.array(labels), np.array(types)


def main():
    init_label_config("label_aliases.json")

    class A:
        include_bcr = False; include_tcr = True; include_sabdab = False
        tcr_file = "raw/vdjdb/vdjdb_full.txt"; tcr_label_col = "antigen.epitope"
        bcr_file = "raw/iedb/bcr_full_single_header_anarci.tsv"; bcr_label_col = "Epitope_Source Molecule"
        sabdab_file = "raw/sabdab/sabdab_paired_clean_human_full.csv"; sabdab_label_col = "label"
        human_only = True; local_files_only = False; offline_dir = None

    spec = build_dataset_specs(A())[0]
    df = load_table(spec); df = filter_human_only(df, spec)

    # tcrdist3 reference gene set -> cleaner
    _probe = TCRrep(cell_df=pd.DataFrame({"cdr3_a_aa": ["CAVRDSNYQLIW"], "v_a_gene": ["TRAV3*01"],
                    "j_a_gene": ["TRAJ33*01"], "cdr3_b_aa": ["CASSLAPGATNEKLFF"], "v_b_gene": ["TRBV9*01"],
                    "j_b_gene": ["TRBJ1-1*01"], "count": [1]}), organism="human",
                    chains=["alpha", "beta"], compute_distances=False, deduplicate=False)
    clean = build_gene_cleaner(_probe.all_genes["human"].keys())

    paired_seed_data, single_seed_data = [], []
    tot_in = tot_kept = 0
    print("=" * 80)
    print("Real tcrdist3 (TCRrep) paired baseline — 20 seeds")
    print("=" * 80)

    for seed in SEEDS:
        pilot = build_slice_with_genes(df, spec, "level4_paired", seed)
        tr_idx, te_idx = clone_aware_split_fast(
            pilot, cdr3_col="split_sequence", label_col="label",
            test_size=TEST_SIZE, clone_similarity_threshold=CLONE_THRESHOLD,
            random_state=seed, min_clones_per_label=MIN_CLONES, show_progress=False)
        train, test = pilot.iloc[tr_idx], pilot.iloc[te_idx]
        n_train_full, n_test_full = len(train), len(test)

        train_cells, d_lbl, _ = tcr_cells(train, clean)
        test_cells, q_lbl, q_typ = tcr_cells(test, clean)
        n_train, n_test = len(train_cells), len(test_cells)
        tot_in += n_train_full + n_test_full
        tot_kept += n_train + n_test

        cell = pd.concat([train_cells, test_cells], ignore_index=True)
        t0 = time.time()
        tr = TCRrep(cell_df=cell, organism="human", chains=["alpha", "beta"],
                    compute_distances=True, deduplicate=False)
        assert tr.clone_df.shape[0] == len(cell), \
            f"seed {seed}: tcrdist3 dropped rows ({tr.clone_df.shape[0]} != {len(cell)})"

        pw_a, pw_b = tr.pw_alpha, tr.pw_beta
        # test (rows n_train:) x train (cols :n_train); similarity = -distance
        sim_paired = -(pw_a + pw_b)[n_train:, :n_train].astype(float)
        sim_single = -(pw_b)[n_train:, :n_train].astype(float)

        # We need predictions CSV for each level to let nested_bootstrap.py load them
        import json
        out_dir = OUT_TREE / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        m_p = retrieval_metrics(None, None, q_lbl, d_lbl, q_typ, "tcrdist3",
                                precomputed_sim=sim_paired, return_ranks=True, return_predictions=True)
        m_s = retrieval_metrics(None, None, q_lbl, d_lbl, q_typ, "tcrdist3",
                                precomputed_sim=sim_single, return_ranks=True, return_predictions=True)

        for m, store in ((m_p, paired_seed_data), (m_s, single_seed_data)):
            r = np.asarray(m["ranks"])
            store.append({"hits1": (r == 0), "hits5": (r < 5), "rr": 1.0 / (r + 1)})

        # Export predictions files for the paired ladder
        paired_levels = ["level1_paired", "level2_paired", "level3_paired", "level4_paired"]
        for lvl in paired_levels:
            pred_csv = out_dir / f"TCR_{lvl}_tcrdist3_predictions.csv"
            pd.DataFrame({
                "true_label": q_lbl,
                "rank": m_p["ranks"],
                "exp_recall@1": m_p["pq_recall@1"],
                "exp_recall@5": m_p["pq_recall@5"],
                "exp_mrr": m_p["pq_mrr"],
                "tie_size": m_p["pq_tie_size"]
            }).to_csv(pred_csv, index=False)
            
            pred_json = out_dir / f"TCR_{lvl}_tcrdist3_predictions.json"
            with open(pred_json, "w") as f:
                json.dump({
                    "query_labels": q_lbl.tolist(),
                    "predictions": m_p["predictions"]
                }, f)

        # Export single-chain predictions file
        single_csv = out_dir / f"TCR_level4_tcrdist3_predictions.csv"
        pd.DataFrame({
            "true_label": q_lbl,
            "rank": m_s["ranks"],
            "exp_recall@1": m_s["pq_recall@1"],
            "exp_recall@5": m_s["pq_recall@5"],
            "exp_mrr": m_s["pq_mrr"],
            "tie_size": m_s["pq_tie_size"]
        }).to_csv(single_csv, index=False)
        
        single_json = out_dir / f"TCR_level4_tcrdist3_predictions.json"
        with open(single_json, "w") as f:
            json.dump({
                "query_labels": q_lbl.tolist(),
                "predictions": m_s["predictions"]
            }, f)

        # write per-seed pilot_results.csv containing 5 rows (four paired levels + single level4)
        results_rows = []
        for lvl in paired_levels:
            results_rows.append({
                "dataset": "tcr", "level": lvl, "model": "tcrdist3",
                "n_train": n_train, "n_test": n_test,
                "recall@1": m_p["recall@1"], "recall@5": m_p["recall@5"], "mrr": m_p["mrr"]
            })
        results_rows.append({
            "dataset": "tcr", "level": "level4", "model": "tcrdist3",
            "n_train": n_train, "n_test": n_test,
            "recall@1": m_s["recall@1"], "recall@5": m_s["recall@5"], "mrr": m_s["mrr"]
        })
        pd.DataFrame(results_rows).to_csv(out_dir / "pilot_results.csv", index=False)

        print(f"  seed {seed:3d}: n_train={n_train}/{n_train_full} n_test={n_test}/{n_test_full} "
              f"| paired R@1={m_p['recall@1']:.4f}  single(beta) R@1={m_s['recall@1']:.4f}  ({time.time()-t0:.1f}s)")

    print(f"\ngene coverage: {tot_kept}/{tot_in} rows retained ({100*tot_kept/tot_in:.2f}%)")
    print("\n--- nested bootstrap (1000 reps) ---")
    rows = []
    for level, data in (("level4_paired", paired_seed_data), ("level4", single_seed_data)):
        st = nested_bootstrap_metrics(data, n_outer=1000)
        for metric in ("recall@1", "recall@5", "mrr"):
            rows.append({"dataset": "tcr", "model": "tcrdist3", "level": level, "metric": metric,
                         "nested_mean": st[metric]["mean"], "nested_lower": st[metric]["lower"],
                         "nested_upper": st[metric]["upper"]})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
