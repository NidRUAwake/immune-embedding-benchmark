#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

if __package__ in (None, ""):
    # Allow direct execution: `python scripts/benchmark/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.data import (
    build_dataset_specs,
    load_table,
    compute_shared_labels,
    build_pilot_slice,
    init_label_config,
    filter_human_only,
)
from benchmark.split import clone_aware_split_fast, zero_shot_antigen_split
from benchmark.embeddings import get_embedder
from benchmark.evaluate import retrieval_metrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular BCR/TCR Embedding Benchmark.")
    parser.add_argument("--config", default="configs/standard_eval.yaml", help="Path to YAML config file.")
    parser.add_argument("--bcr-file", default="raw/iedb/bcr_singlechain_vh.tsv")  # canonical post-ANARCI VH
    parser.add_argument("--tcr-file", default="raw/vdjdb/vdjdb_full.txt")
    parser.add_argument("--sabdab-file", default="raw/sabdab/sabdab_paired_clean_human_full.csv")
    parser.add_argument("--output-dir", default="outputs/pilot_modular")
    parser.add_argument("--models", nargs="+", default=["onehot", "levenshtein", "blosum62", "atchley"])
    parser.add_argument("--include-bcr", action="store_true")
    parser.add_argument("--include-tcr", action="store_true")
    parser.add_argument("--include-sabdab", action="store_true")
    parser.add_argument("--human-only", action="store_true", help="Filter to human receptor sequences only.")
    parser.add_argument("--inspect-only", action="store_true")
    
    # Label Handling
    parser.add_argument("--top-labels", type=int, default=5)
    parser.add_argument("--min-label-count", type=int, default=25)
    parser.add_argument("--label-config", default="label_aliases.json")
    parser.add_argument("--max-per-label-bcr", type=int, default=150)
    parser.add_argument("--max-per-label-tcr", type=int, default=100)
    parser.add_argument("--max-per-label-sabdab", type=int, default=100)
    parser.add_argument("--bcr-label-col", default="Epitope_Source Molecule")
    parser.add_argument("--tcr-label-col", default="antigen.epitope")
    parser.add_argument("--sabdab-label-col", default="label")
    parser.add_argument("--bcr-label-contains", default=None)
    parser.add_argument("--tcr-label-contains", default=None)
    parser.add_argument("--sabdab-label-contains", default=None)
    parser.add_argument("--tcr-level4-col", default=None)
    parser.add_argument("--forced-labels", type=str, default=None, help="Comma-separated list of labels to force (override shared label computation).")
    parser.add_argument("--no-shared-labels", action="store_false", dest="shared_labels")
    
    # Splitting & Eval
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-strategy", choices=["clone-aware", "zero-shot", "random"], default="clone-aware")
    parser.add_argument("--clone-threshold", type=float, default=0.95)
    parser.add_argument("--min-clones-per-label", type=int, default=2)
    parser.add_argument("--exclude-non-protein", action="store_true")
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--distance-metric", choices=["cosine", "euclidean"], default="cosine")
    
    # Embedding params
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline-dir", help="Directory containing pre-computed .npy embeddings.")
    # Levels constraint
    parser.add_argument("--levels", nargs="+", default=None, help="Specific levels to run.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config if exists
    if Path(args.config).exists():
        print(f"Loading standard configuration from {args.config}...")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # Override defaults if not explicitly set on CLI?
            # Actually, let's just update args with config values as the new baseline
            if "evaluation" in config:
                for k, v in config["evaluation"].items():
                    setattr(args, k, v)
            if "scale_up" in config:
                if "bcr_top_labels" in config["scale_up"]:
                    args.top_labels = config["scale_up"]["bcr_top_labels"]
                if "bcr_min_label_count" in config["scale_up"]:
                    args.min_label_count = config["scale_up"]["bcr_min_label_count"]
    
    use_bcr, use_tcr, use_sabdab = args.include_bcr, args.include_tcr, args.include_sabdab
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    init_label_config(args.label_config)
    
    use_bcr = args.include_bcr or not (args.include_bcr or args.include_tcr or args.include_sabdab)
    use_tcr = args.include_tcr or not (args.include_bcr or args.include_tcr or args.include_sabdab)
    use_sabdab = args.include_sabdab or not (args.include_bcr or args.include_tcr or args.include_sabdab)
    slices, results = [], []

    for spec in build_dataset_specs(args):
        if (spec.name == "BCR" and not use_bcr) or (spec.name == "TCR" and not use_tcr) or (spec.name == "SAbDab" and not use_sabdab): 
            continue
            
        df = load_table(spec)
        if args.human_only:
            before = len(df)
            df = filter_human_only(df, spec)
            print(f"-> {spec.name} human-only filter: {before} -> {len(df)} rows")
        
        if spec.name == "BCR": label_filter = args.bcr_label_contains
        elif spec.name == "TCR": label_filter = args.tcr_label_contains
        else: label_filter = args.sabdab_label_contains
        
        forced_labels = None
        if args.forced_labels:
            # Use provided labels directly (Task 308: fixed canonical label set)
            forced_labels = [label.lower() for label in args.forced_labels.split(",")]
            print(f"-> {spec.name} Forced Labels: {forced_labels}")
        elif args.shared_labels:
            forced_labels = compute_shared_labels(df, spec, args.top_labels, args.min_label_count, label_filter)
            print(f"-> {spec.name} Shared Labels: {forced_labels}")
            
        for level in spec.sequence_builders:
            if getattr(args, "levels", None) and level not in args.levels:
                continue
            if spec.name == "BCR": max_per_label = args.max_per_label_bcr
            elif spec.name == "TCR": max_per_label = args.max_per_label_tcr
            else: max_per_label = args.max_per_label_sabdab

            slice_ = build_pilot_slice(
                df, spec, level, 
                max_per_label, 
                args.random_state, forced_labels, label_filter
            )
            if slice_: 
                slices.append(slice_)
            else: 
                print(f"Skipped {spec.name} {level} (Insufficient Data)")

    if args.inspect_only:
        return

    embedder_cache = {}
    for s in slices:
        df = s.dataframe
        cdr3_col = "split_sequence" if "split_sequence" in df.columns else s.sequence_col

        vj_filter = None
        if s.level in ("level2", "level2_paired"):
            vj_filter = "vj"
        elif s.level in ("level2.5", "level2.5_paired"):
            vj_filter = "v"

        is_paired_clono = s.level in ("level2_paired", "level2.5_paired")
        if is_paired_clono:
            # Clonotype variants rank by the paired CDR3 (the L1_paired embedding); the V/J
            # gene filter (both chains) is applied at retrieval time. Reuse L1_paired embeddings.
            embed_level = "level1_paired"
            embed_seq_col = s.sequence_col
        elif vj_filter:
            embed_level = "level1"
            embed_seq_col = "split_sequence"
        else:
            embed_level = s.level
            embed_seq_col = s.sequence_col

        print(f"\n--- Splitting {s.dataset} {s.level} (Strategy: {args.split_strategy}) ---")
        if args.split_strategy == "zero-shot":
            train_idx, test_idx = zero_shot_antigen_split(
                df, label_col="label", test_size=args.test_size,
                random_state=args.random_state, min_samples_per_label=args.min_clones_per_label
            )
        elif args.split_strategy == "random":
            # For a fair comparison, we should still apply the min_clones_per_label filter
            # even if we are not doing clone-aware splitting.
            counts = df["label"].value_counts()
            labels_to_keep = counts[counts >= args.min_clones_per_label].index
            df_filtered = df[df["label"].isin(labels_to_keep)].copy()
            
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(df_filtered))
            
            # Ensure each class has at least 2 samples for stratification
            stratify_labels = df_filtered["label"]
            if (df_filtered["label"].value_counts() < 2).any():
                print("  Warning: Some labels have < 2 samples, skipping stratification.")
                stratify_labels = None
                
            train_sub_idx, test_sub_idx = train_test_split(
                indices, test_size=args.test_size, random_state=args.random_state,
                stratify=stratify_labels
            )
            # Map back to original dataframe indices
            train_idx = df_filtered.index[train_sub_idx].values
            test_idx = df_filtered.index[test_sub_idx].values
        else:
            train_idx, test_idx = clone_aware_split_fast(
                df, cdr3_col=cdr3_col, label_col="label", 
                test_size=args.test_size, clone_similarity_threshold=args.clone_threshold,
                random_state=args.random_state, min_clones_per_label=args.min_clones_per_label,
                show_progress=False
            )

        train_seq, test_seq = df[embed_seq_col].values[train_idx], df[embed_seq_col].values[test_idx]
        train_lbl, test_lbl = df["label"].values[train_idx], df["label"].values[test_idx]
        train_type, test_type = df["antigen_type"].values[train_idx], df["antigen_type"].values[test_idx]
        train_v = df["v_gene"].values[train_idx] if "v_gene" in df.columns else None
        train_j = df["j_gene"].values[train_idx] if "j_gene" in df.columns else None
        test_v = df["v_gene"].values[test_idx] if "v_gene" in df.columns else None
        test_j = df["j_gene"].values[test_idx] if "j_gene" in df.columns else None

        for model_name in args.models:
            print(f"Evaluating {model_name} ...")
            
            # Check for offline override
            actual_model = model_name
            if getattr(args, "offline_dir", None):
                # Try to find a matching .npy file in the offline dir
                # Pattern generated by generate_embeddings.py: {dataset.lower()}_{level}_{model}_{pooling}.npy
                offline_pattern = f"{s.dataset.lower()}_{embed_level}_{model_name}_{args.pooling}.npy"
                offline_path = Path(args.offline_dir) / offline_pattern
                if offline_path.exists():
                    print(f"  Using offline embeddings from {offline_path}")
                    actual_model = str(offline_path)
                else:
                    # Special Case: AbLang doesn't support Level 4 (Architectural Limit)
                    if model_name == "ablang" and s.level.startswith("level4"):
                        print(f"  Skipping {model_name} for {s.level} (Known architectural limit: 160aa)")
                        continue
                    
                    # NEW: Skip large models if offline embedding is missing to avoid CPU overload
                    if model_name in ["esm2-650m", "esm2-3b", "tcr-bert"]:
                        print(f"  Skipping {model_name} for {s.level} (Offline embedding missing and online is too slow)")
                        continue
                    
                    print(f"  Warning: Offline embedding not found at {offline_path}. Falling back to online model.")
            
            # Cache embedders
            if actual_model not in embedder_cache:
                embedder_cache[actual_model] = get_embedder(actual_model, args)
            embedder = embedder_cache[actual_model]
            
            # Encode
            q_emb = embedder.encode(test_seq)
            d_emb = embedder.encode(train_seq)
            
            # Retrieval
            import json
            
            label_overlap = set(train_lbl) & set(test_lbl)
            if args.split_strategy == "zero-shot" and len(label_overlap) == 0:
                # Retrieval@k needs positive labels in DB; strict zero-shot antigen split violates this.
                metrics = {"recall@1": np.nan, "recall@5": np.nan, "recall@10": np.nan, "mrr": np.nan}
                print("  Warning: zero-shot split has disjoint labels; retrieval metrics are undefined (set to NaN).")
            else:
                metrics = retrieval_metrics(
                    q_emb=q_emb, d_emb=d_emb,
                    q_labels=test_lbl, d_labels=train_lbl,
                    q_antigen_types=test_type,
                    method=model_name,
                    exclude_non_protein=args.exclude_non_protein,
                    exclude_self=False,
                    return_predictions=args.export_predictions,
                    return_ranks=args.export_predictions,
                    distance_metric=args.distance_metric,
                    vj_filter=vj_filter,
                    q_v_gene=test_v,
                    q_j_gene=test_j,
                    d_v_gene=train_v,
                    d_j_gene=train_j,
                )
                
            if args.export_predictions and "ranks" in metrics:
                out_path = Path(args.output_dir) / f"{s.dataset}_{s.level}_{model_name}_predictions.csv"
                # Use query_labels from metrics because they might be filtered (e.g. protein-only)
                q_labels = metrics.get("query_labels", test_lbl)
                pred_df = pd.DataFrame({
                    "true_label": q_labels,
                    "rank": metrics["ranks"],                          # best-case rank (legacy column)
                    "exp_recall@1": metrics.get("pq_recall@1"),        # expected R@1 per query (c_top/t_top)
                    "exp_recall@5": metrics.get("pq_recall@5"),        # expected R@5 per query (exact)
                    "exp_mrr":      metrics.get("pq_mrr"),             # expected MRR per query (exact)
                    "tie_size":     metrics.get("pq_tie_size"),        # |top tie group|
                })
                pred_df.to_csv(out_path, index=False)
                print(f"  Saved predictions to {out_path}")

            if args.export_predictions and "predictions" in metrics:
                pred_path = Path(args.output_dir) / f"{s.dataset}_{s.level}_{model_name}_predictions.json"
                with open(pred_path, "w") as f:
                    json.dump({
                        "query_labels": metrics.pop("query_labels"),
                        "predictions": metrics.pop("predictions")
                    }, f)
            
            results.append({
                "dataset": s.dataset,
                "level": s.level,
                "model": model_name,
                "exclude_non_protein": args.exclude_non_protein,
                "n_train": len(train_seq),
                "n_test": len(test_seq),
                **metrics
            })
            if np.isnan(metrics["recall@1"]):
                print("  R@1: NaN | R@5: NaN")
            else:
                print(f"  R@1: {metrics['recall@1']:.3f} | R@5: {metrics['recall@5']:.3f}")

    pd.DataFrame(results).to_csv(f"{args.output_dir}/pilot_results.csv", index=False)
    print(f"\nSaved to {args.output_dir}/pilot_results.csv")

if __name__ == "__main__":
    main()
