#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Force joblib Parallel to run sequentially
import joblib
original_parallel = joblib.Parallel
class ForceSerialParallel(original_parallel):
    def __init__(self, *args, **kwargs):
        if "n_jobs" in kwargs:
            kwargs["n_jobs"] = 1
        super().__init__(*args, **kwargs)
joblib.Parallel = ForceSerialParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.data import build_dataset_specs, load_table, filter_human_only, init_label_config, build_pilot_slice, clean_sequence, compute_shared_labels
from benchmark.split import clone_aware_split_fast
from benchmark.embeddings import get_embedder
from benchmark.evaluate import retrieval_metrics
from analysis.nested_bootstrap import nested_bootstrap_metrics

def make_l35_padded_sequences(df):
    results_padded = []
    results_fw4 = []
    for idx, row in df.iterrows():
        l4_seq = row.get("Chain 1_Protein Sequence")
        if not isinstance(l4_seq, str):
            l4_seq = row.get("Chain 1_Full Sequence")
        if not isinstance(l4_seq, str) or len(l4_seq) == 0:
            results_padded.append(None)
            results_fw4.append(None)
            continue
        
        cdr1_start = row.get("Chain 1_CDR1 Start Calculated")
        cdr1_end = row.get("Chain 1_CDR1 End Calculated")
        cdr2_start = row.get("Chain 1_CDR2 Start Calculated")
        cdr2_end = row.get("Chain 1_CDR2 End Calculated")
        cdr3_start = row.get("Chain 1_CDR3 Start Calculated")
        cdr3_end = row.get("Chain 1_CDR3 End Calculated")
        
        cdr3_seq = row.get("Chain 1_CDR3 ANARCI")
        if not isinstance(cdr3_seq, str): cdr3_seq = row.get("Chain 1_CDR3 Curated")
        if not isinstance(cdr3_seq, str): cdr3_seq = row.get("Chain 1_CDR3 Calculated")
        
        l4_seq = clean_sequence(l4_seq)
        cdr3_seq = clean_sequence(cdr3_seq)
        
        if not l4_seq or not cdr3_seq:
            results_padded.append(None)
            results_fw4.append(None)
            continue
            
        if not pd.isna(cdr1_start) and not pd.isna(cdr1_end) and not pd.isna(cdr2_start) and not pd.isna(cdr2_end) and not pd.isna(cdr3_start) and not pd.isna(cdr3_end):
            try:
                c1_s = int(float(cdr1_start)) - 1
                c1_e = int(float(cdr1_end))
                c2_s = int(float(cdr2_start)) - 1
                c2_e = int(float(cdr2_end))
                c3_s = int(float(cdr3_start)) - 1
                c3_e = int(float(cdr3_end))
                
                if 0 <= c1_s < c1_e < c2_s < c2_e < c3_s < c3_e <= len(l4_seq):
                    fr1 = l4_seq[0 : c1_s]
                    cdr1_len = c1_e - c1_s
                    fr2 = l4_seq[c1_e : c2_s]
                    cdr2_len = c2_e - c2_s
                    fr3 = l4_seq[c2_e : c3_s]
                    fw4 = l4_seq[c3_e : ]
                    
                    seq_padded = fr1 + ("G" * cdr1_len) + fr2 + ("G" * cdr2_len) + fr3 + cdr3_seq + fw4
                    seq_fw4 = fr1 + fr2 + fr3 + cdr3_seq + fw4
                    
                    results_padded.append(seq_padded)
                    results_fw4.append(seq_fw4)
                else:
                    results_padded.append(None)
                    results_fw4.append(None)
            except Exception:
                results_padded.append(None)
                results_fw4.append(None)
        else:
            results_padded.append(None)
            results_fw4.append(None)
            
    return pd.Series(results_padded, index=df.index).map(clean_sequence), pd.Series(results_fw4, index=df.index).map(clean_sequence)

def main():
    init_label_config("label_aliases.json")
    
    class MockArgs:
        include_bcr = True
        include_tcr = False
        include_sabdab = False
        bcr_file = "raw/iedb/bcr_singlechain_vh.tsv"
        bcr_label_col = "Epitope_Source Molecule"
        human_only = True
        top_labels = 10
        min_label_count = 14
        max_per_label_bcr = 150
        shared_labels = True
        batch_size = 32
        local_files_only = False
        offline_dir = None
        pooling = "mean"

    mock_args = MockArgs()
    spec = build_dataset_specs(mock_args)[0]
    df_raw = load_table(spec)
    df_raw = filter_human_only(df_raw, spec)
    
    # Bug 1 Fix: Explicit label filtering
    forced_labels = compute_shared_labels(df_raw, spec, top_labels=10, min_label_count=14)
    print(f"-> BCR shared labels (n={len(forced_labels)}): {forced_labels}")
    
    padded_series, fw4_series = make_l35_padded_sequences(df_raw)
    
    seeds = [42 + i * 10 for i in range(20)]
    methods = ["esm2-150m", "blosum62"]
    variants = ["level3p5_padded", "level3p5_fw4"] # Bug 4 Fix: ASCII-safe names
    
    out_records = []
    sample_padded = []
    sample_fw4 = []
    
    print("Loading embedder...")
    embedder = get_embedder("esm2-150m", mock_args)
    emb_cache = {}
    
    for variant in variants:
        print(f"\n=== Running variant {variant} ===")
        if variant == "level3p5_padded":
            spec.sequence_builders[variant] = lambda df, s=padded_series: s.reindex(df.index) # Bug 2 Fix: reindex
        else:
            spec.sequence_builders[variant] = lambda df, s=fw4_series: s.reindex(df.index)
            
        seed_data = {m: [] for m in methods}
        
        for seed in seeds:
            print(f"\n--- Running Seed {seed} ---")
            slice_ = build_pilot_slice(df_raw, spec, variant, mock_args.max_per_label_bcr, seed, forced_labels)
            if slice_ is None:
                continue
                
            df = slice_.dataframe
            seqs = df[slice_.sequence_col].values
            
            if seed == 42 and variant == "level3p5_padded":
                for seq in seqs[:20]:
                    sample_padded.append(seq)
            elif seed == 42 and variant == "level3p5_fw4":
                for seq in seqs[:20]:
                    sample_fw4.append(seq)
            
            train_idx, test_idx = clone_aware_split_fast(
                df, cdr3_col="split_sequence", label_col="label", 
                test_size=0.2, clone_similarity_threshold=0.95,
                random_state=seed, min_clones_per_label=2,
                show_progress=False
            )
            
            train_seq, test_seq = seqs[train_idx], seqs[test_idx]
            train_lbl, test_lbl = df["label"].values[train_idx], df["label"].values[test_idx]
            train_type, test_type = df["antigen_type"].values[train_idx], df["antigen_type"].values[test_idx]
            
            for method in methods:
                if method == "blosum62":
                    with joblib.parallel_backend("loky", n_jobs=10):
                        metrics = retrieval_metrics(
                            q_emb=test_seq, d_emb=train_seq,
                            q_labels=test_lbl, d_labels=train_lbl,
                            q_antigen_types=test_type,
                            method=method,
                            return_ranks=True
                        )
                else:
                    unique_seqs = list(set(test_seq).union(set(train_seq)))
                    seqs_to_compute = [s for s in unique_seqs if s not in emb_cache]
                    
                    if seqs_to_compute:
                        print(f"  Computing embeddings for {len(seqs_to_compute)} new sequences with {method}...")
                        new_emb = embedder.encode(seqs_to_compute)
                        for s, e in zip(seqs_to_compute, new_emb):
                            emb_cache[s] = e
                            
                    q_emb = np.array([emb_cache[s] for s in test_seq])
                    d_emb = np.array([emb_cache[s] for s in train_seq])
                    
                    metrics = retrieval_metrics(
                        q_emb=q_emb, d_emb=d_emb,
                        q_labels=test_lbl, d_labels=train_lbl,
                        q_antigen_types=test_type,
                        method=method,
                        return_ranks=True
                    )
                    
                seed_data[method].append({
                    'hits1': np.array(metrics["pq_recall@1"]),
                    'hits5': np.array(metrics["pq_recall@5"]),
                    'rr': np.array(metrics["pq_mrr"])
                })
                
        print(f"\n--- Bootstrap for {variant} ---")
        for method in methods:
            stats = nested_bootstrap_metrics(seed_data[method], n_outer=1000)
            for metric_name, m_key in [("recall@1", "recall@1"), ("recall@5", "recall@5"), ("mrr", "mrr")]:
                out_records.append({
                    "dataset": "bcr",
                    "model": method,
                    "level": variant,
                    "metric": metric_name,
                    "nested_mean": stats[m_key]["mean"],
                    "nested_lower": stats[m_key]["lower"],
                    "nested_upper": stats[m_key]["upper"]
                })
                
    out_df = pd.DataFrame(out_records)
    out_dir = Path("outputs/task282_l35_padded")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "bcr_l35_padded_20seeds.csv", index=False)
    
    samples_df = pd.DataFrame({
        "variant": ["level3p5_padded"] * len(sample_padded) + ["level3p5_fw4"] * len(sample_fw4),
        "sequence": sample_padded + sample_fw4,
        "length": [len(s) if s else 0 for s in sample_padded] + [len(s) if s else 0 for s in sample_fw4]
    })
    samples_df.to_csv(out_dir / "bcr_l35_padded_sequences_sample.csv", index=False)

if __name__ == "__main__":
    main()
