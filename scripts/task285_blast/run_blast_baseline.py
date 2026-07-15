#!/usr/bin/env python3
import argparse
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.data import build_dataset_specs, load_table, filter_human_only, init_label_config, build_pilot_slice, compute_shared_labels
from benchmark.split import clone_aware_split_fast
from benchmark.evaluate import retrieval_metrics
from analysis.nested_bootstrap import nested_bootstrap_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bcr", "tcr", "sabdab", "mcpas"], required=True)
    parser.add_argument("--level", default="level1")
    parser.add_argument("--seeds", type=str, required=True, help="Comma separated list of seeds")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sanity", required=True)
    args = parser.parse_args()

    init_label_config("label_aliases.json")

    class MockArgs:
        include_bcr = args.dataset == "bcr"
        include_tcr = args.dataset in ["tcr", "mcpas"]
        include_sabdab = args.dataset == "sabdab"
        
        bcr_file = "raw/iedb/bcr_singlechain_vh.tsv"
        bcr_label_col = "Epitope_Source Molecule"
        
        if args.dataset == "mcpas":
            tcr_file = "outputs/intermediate/mcpas_standardized.tsv"
        else:
            tcr_file = "raw/vdjdb/vdjdb_full.txt"
        tcr_label_col = "antigen.epitope"
        
        sabdab_file = "raw/sabdab/sabdab_paired_clean_human_full.csv"
        sabdab_label_col = "label"
        
        human_only = True
        local_files_only = False
        offline_dir = None

    mock_args = MockArgs()
    spec = build_dataset_specs(mock_args)[0]
    df_raw = load_table(spec)
    df_raw = filter_human_only(df_raw, spec)
    
    if args.dataset == "bcr":
        top_labels = 10
        min_label_count = 14
        max_per_label = 150
    elif args.dataset == "tcr":
        top_labels = 10
        min_label_count = 50
        max_per_label = 100
    elif args.dataset == "sabdab":
        top_labels = 5
        min_label_count = 25
        max_per_label = 100
    elif args.dataset == "mcpas":
        top_labels = 10       # canonical McPAS is top-10 (doc 350); was stale top-5
        min_label_count = 30
        max_per_label = 100   # canonical --max-per-label-tcr 100; was stale 200
        
    forced_labels = compute_shared_labels(df_raw, spec, top_labels=top_labels, min_label_count=min_label_count)
    print(f"-> Shared labels (n={len(forced_labels)}): {forced_labels}")
    
    seeds = [int(s) for s in args.seeds.split(",")]
    
    seed_data = {"blast": []}
    
    sanity_records = []
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        slice_ = build_pilot_slice(df_raw, spec, args.level, max_per_label, seed, forced_labels)
        if slice_ is None:
            print(f"Skipped seed {seed} (Insufficient Data)")
            continue
            
        df = slice_.dataframe
        seqs = df[slice_.sequence_col].values
        
        train_idx, test_idx = clone_aware_split_fast(
            df, cdr3_col="split_sequence", label_col="label", 
            test_size=0.2, clone_similarity_threshold=0.95,
            random_state=seed, min_clones_per_label=2,
            show_progress=False
        )
        
        train_seq, test_seq = seqs[train_idx], seqs[test_idx]
        train_lbl, test_lbl = df["label"].values[train_idx], df["label"].values[test_idx]
        train_type, test_type = df["antigen_type"].values[train_idx], df["antigen_type"].values[test_idx]
        
        print(f"  Dataset size: n_train={len(train_seq)}, n_test={len(test_seq)}")
        
        metrics = retrieval_metrics(
            q_emb=test_seq, d_emb=train_seq,
            q_labels=test_lbl, d_labels=train_lbl,
            q_antigen_types=test_type,
            method="blast",
            return_ranks=True
        )

        # Aggregate on per-query EXPECTED credit (order-independent), matching the rest of the
        # pipeline, instead of the old best-rank single-pick (ranks==0 / ranks<5 / 1/(rank+1)).
        seed_data["blast"].append({
            'hits1': np.asarray(metrics["pq_recall@1"], dtype=float),
            'hits5': np.asarray(metrics["pq_recall@5"], dtype=float),
            'rr': np.asarray(metrics["pq_mrr"], dtype=float),
        })
        print(f"  BLAST -> R@1: {metrics['recall@1']:.3f} | R@5: {metrics['recall@5']:.3f} | Time: {metrics['query_time']:.2f}s | DB Size: {metrics['db_size_bytes']} bytes")
        
        sanity_records.append({
            "dataset": args.dataset,
            "level": args.level,
            "seed": seed,
            "model": "blast",
            "recall@1": metrics["recall@1"],
            "recall@5": metrics["recall@5"],
            "mrr": metrics["mrr"],
            "n_train": len(train_seq),
            "n_test": len(test_seq),
            "query_time_sec": metrics["query_time"],
            "db_size_bytes": metrics["db_size_bytes"]
        })
            
    print("\n--- Running Double Bootstrap (1000 resamples) ---")
    np.random.seed(42)  # reproducible aggregate (see nested_bootstrap --bootstrap-seed)
    out_records = []
    stats = nested_bootstrap_metrics(seed_data["blast"], n_outer=1000)
    for metric_name, m_key in [("recall@1", "recall@1"), ("recall@5", "recall@5"), ("mrr", "mrr")]:
        out_records.append({
            "dataset": args.dataset,
            "model": "blast",
            "level": args.level,
            "metric": metric_name,
            "nested_mean": stats[m_key]["mean"],
            "nested_lower": stats[m_key]["lower"],
            "nested_upper": stats[m_key]["upper"]
        })
            
    out_df = pd.DataFrame(out_records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # append to nested CI
    if out_path.exists():
        out_df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        out_df.to_csv(out_path, index=False)
    
    sanity_df = pd.DataFrame(sanity_records)
    sanity_path = Path(args.sanity)
    if sanity_path.exists():
        sanity_df.to_csv(sanity_path, mode='a', header=False, index=False)
    else:
        sanity_df.to_csv(sanity_path, index=False)
    
    print(f"\nSuccessfully generated nested bootstrap results at {out_path}")
    print(out_df)

if __name__ == "__main__":
    main()
