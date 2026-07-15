#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_rel
from joblib import Parallel, delayed

# Add scripts/ directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.data import (
    build_dataset_specs,
    load_table,
    filter_human_only,
    init_label_config,
    normalize_labels,
    assign_antigen_type,
    compute_shared_labels,
    clean_sequence,
    is_legit_sequence
)
from benchmark.split import clone_aware_split_fast
from benchmark.embeddings import get_embedder, TransformerEmbedder, OfflineEmbedder
from benchmark.evaluate import retrieval_metrics, compute_similarity_matrix, blosum_score_fast

# Setup threads
torch.set_num_threads(32)
device = "cuda" if torch.cuda.is_available() else "cpu"

class MockArgs:
    include_bcr = True
    include_tcr = False
    include_sabdab = False
    bcr_file = "raw/iedb/bcr_singlechain_vh.tsv"  # canonical input (post-ANARCI single-chain VH)
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

def get_v_family(v_gene: object) -> str | None:
    if pd.isna(v_gene):
        return None
    text = str(v_gene).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text.split("-")[0].split("*")[0]

def build_bcr_frame(spec, df_raw, level: str, seed: int) -> pd.DataFrame:
    labels = normalize_labels(df_raw[spec.label_col])
    level4_sequences = spec.sequence_builders["level4"](df_raw)
    level1_sequences = spec.sequence_builders["level1"](df_raw)
    v_gene = df_raw["Chain 1_Curated V Gene"].fillna(df_raw["Chain 1_Calculated V Gene"])
    v_family = v_gene.map(get_v_family)

    work = pd.DataFrame(
        {
            "label": labels,
            "sequence": level4_sequences.map(clean_sequence) if level == "level4" else level1_sequences.map(clean_sequence),
            "split_sequence": level1_sequences.map(clean_sequence),
            "v_gene": v_gene,
            "v_family": v_family,
            "antigen_type": assign_antigen_type(labels),
        }
    ).dropna(subset=["label", "sequence", "split_sequence"])

    # Canonical strict 20-AA legitimacy filter (matches build_pilot_slice), applied BEFORE the
    # per-label cap sampling so the retained pool aligns bit-for-bit with the canonical tree.
    work = work[work["sequence"].map(is_legit_sequence)].copy()

    forced_labels = compute_shared_labels(df_raw, spec, 10, 14, None)
    if forced_labels:
        work = work[work["label"].isin(forced_labels)].copy()

    sampled_parts = []
    for label in work["label"].unique():
        label_df = work[work["label"] == label].copy()
        if len(label_df) > 150:
            label_df = label_df.sample(150, random_state=seed)
        sampled_parts.append(label_df)

    frame = (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return frame

def compute_shannon_entropy(series: pd.Series) -> float:
    counts = series.value_counts()
    probs = counts / counts.sum()
    return -float(np.sum(probs * np.log2(probs)))

def main():
    init_label_config("label_aliases.json")
    mock_args = MockArgs()
    spec = build_dataset_specs(mock_args)[0]
    df_raw = load_table(spec)
    df_raw = filter_human_only(df_raw, spec)

    # Pre-collect unique sequences across all 20 seeds
    seeds = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232]
    unique_seqs = set()
    seed_frames = {}
    print("Loading seed frames...")
    for seed in seeds:
        frame = build_bcr_frame(spec, df_raw, "level4", seed)
        seed_frames[seed] = frame
        unique_seqs.update(frame["sequence"].tolist())
        
    unique_seqs_list = sorted(list(unique_seqs))
    print(f"Total unique Level 4 sequences across all seeds: {len(unique_seqs_list)}")
    
    # Load default ESM2-150M embedder (using default mean pooling)
    npy_path = "outputs/embeddings/bcr_level4_esm2-150m_mean.npy"
    if os.path.exists(npy_path):
        print(f"Loading offline embeddings from {npy_path}...")
        embedder = OfflineEmbedder(npy_path)
    else:
        print("Falling back to online ESM2-150M...")
        embedder = get_embedder("esm2-150m", mock_args)
    
    embs_mean = embedder.encode(unique_seqs_list)
    mean_dict = {s: emb for s, emb in zip(unique_seqs_list, embs_mean)}

    # =========================================================================
    # Task A: L4 Strict Split Sensitivity Sweep
    # =========================================================================
    print("\n=== Task A: L4 Strict Split Sensitivity Sweep ===")
    thresholds = [None, 0.80, 0.75, 0.70]
    results_a = []

    # Precompute pairwise blosum similarities for all unique seqs to avoid redos
    print("Pre-computing unique pairwise BLOSUM62 similarities...")
    blosum_embedder = get_embedder("blosum62", mock_args)
    embs_blosum = blosum_embedder.encode(unique_seqs_list)
    blosum_dict = {s: emb for s, emb in zip(unique_seqs_list, embs_blosum)}

    # Sweep thresholds
    for thresh in thresholds:
        r1_bl_seeds = []
        r1_esm_seeds = []
        n_test_seeds = []
        
        print(f"Evaluating threshold: {thresh}...")
        for seed in seeds:
            frame = seed_frames[seed]
            train_idx, test_idx = clone_aware_split_fast(
                frame, cdr3_col="split_sequence", label_col="label", 
                test_size=0.2, clone_similarity_threshold=0.95,
                random_state=seed, min_clones_per_label=2,
                show_progress=False
            )
            
            train_df = frame.iloc[train_idx].reset_index(drop=True)
            test_df = frame.iloc[test_idx].reset_index(drop=True)
            
            # Apply strict split check if threshold is defined
            if thresh is not None:
                # Compute maximum blosum similarity of each test sequence to all train sequences
                keep_indices = []
                for idx, row in test_df.iterrows():
                    q_seq = row["sequence"]
                    max_sim = max([blosum_score_fast(q_seq, d_seq) for d_seq in train_df["sequence"]])
                    if max_sim <= thresh:
                        keep_indices.append(idx)
                
                filtered_test_df = test_df.iloc[keep_indices].reset_index(drop=True)
            else:
                filtered_test_df = test_df
                
            n_test = len(filtered_test_df)
            n_test_seeds.append(n_test)
            
            if n_test == 0:
                r1_bl_seeds.append(np.nan)
                r1_esm_seeds.append(np.nan)
                continue
                
            # Evaluate BLOSUM62
            q_emb_bl = np.array(filtered_test_df["sequence"].tolist(), dtype=object)
            d_emb_bl = np.array(train_df["sequence"].tolist(), dtype=object)
            metrics_bl = retrieval_metrics(
                q_emb=q_emb_bl, d_emb=d_emb_bl,
                q_labels=filtered_test_df["label"].values, d_labels=train_df["label"].values,
                q_antigen_types=filtered_test_df["antigen_type"].values,
                method="blosum62"
            )
            r1_bl_seeds.append(metrics_bl["recall@1"])
            
            # Evaluate ESM2-150M
            q_emb_mean = np.array([mean_dict[s] for s in filtered_test_df["sequence"]])
            d_emb_mean = np.array([mean_dict[s] for s in train_df["sequence"]])
            metrics_mean = retrieval_metrics(
                q_emb=q_emb_mean, d_emb=d_emb_mean,
                q_labels=filtered_test_df["label"].values, d_labels=train_df["label"].values,
                q_antigen_types=filtered_test_df["antigen_type"].values,
                method="esm2-150m"
            )
            r1_esm_seeds.append(metrics_mean["recall@1"])
            
        # Compile stats across seeds
        avg_n_test = float(np.mean(n_test_seeds))
        mean_bl = float(np.nanmean(r1_bl_seeds))
        mean_esm = float(np.nanmean(r1_esm_seeds))
        
        # Hedges' g calculation
        if avg_n_test >= 50:
            sd_bl = np.std(r1_bl_seeds, ddof=1)
            sd_esm = np.std(r1_esm_seeds, ddof=1)
            pooled_sd = np.sqrt(((len(seeds) - 1) * (sd_bl**2) + (len(seeds) - 1) * (sd_esm**2)) / (2 * len(seeds) - 2))
            if pooled_sd > 0:
                cohens_d = (mean_esm - mean_bl) / pooled_sd
                j = 1.0 - (3.0 / (4 * (2 * len(seeds)) - 9))
                hedges_g = cohens_d * j
            else:
                hedges_g = 0.0
        else:
            hedges_g = np.nan
            
        results_a.append({
            "L4 threshold": "无额外过滤" if thresh is None else f"≤ {thresh:.2f}",
            "N_test (avg/seed)": avg_n_test,
            "BLOSUM62 R@1": mean_bl,
            "ESM2 R@1": mean_esm,
            "Hedges' g": hedges_g
        })
        
    df_a = pd.DataFrame(results_a)
    out_a_path = Path("outputs/reports/bcr_l4_strict_split_sensitivity.csv")
    out_a_path.parent.mkdir(parents=True, exist_ok=True)
    df_a.to_csv(out_a_path, index=False)
    print(f"Saved Strict Split Sensitivity Results to {out_a_path}")
    print(df_a.to_string(index=False))

    # =========================================================================
    # Task B: Same V-gene Rate Per Antigen Breakdown
    # =========================================================================
    print("\n=== Task B: Same V-gene Rate Per Antigen Breakdown ===")
    
    # Track top-1 correct hits and their V-gene alignment across all 20 seeds
    antigen_correct_hits = {} # {label: [is_same_v_family]}
    
    for seed in seeds:
        frame = seed_frames[seed]
        train_idx, test_idx = clone_aware_split_fast(
            frame, cdr3_col="split_sequence", label_col="label", 
            test_size=0.2, clone_similarity_threshold=0.95,
            random_state=seed, min_clones_per_label=2,
            show_progress=False
        )
        
        train_df = frame.iloc[train_idx].reset_index(drop=True)
        test_df = frame.iloc[test_idx].reset_index(drop=True)
        
        q_emb = np.array([mean_dict[s] for s in test_df["sequence"]])
        d_emb = np.array([mean_dict[s] for s in train_df["sequence"]])
        
        sim = compute_similarity_matrix(q_emb, d_emb, "esm2-150m")
        
        for i, row in test_df.iterrows():
            q_label = row["label"]
            q_v_fam = row["v_family"]
            
            top_idx = int(np.argmax(sim[i]))
            top_label = train_df.loc[top_idx, "label"]
            top_v_fam = train_df.loc[top_idx, "v_family"]
            
            if top_label == q_label:
                # Correct hit!
                if q_label not in antigen_correct_hits:
                    antigen_correct_hits[q_label] = []
                is_same = (q_v_fam is not None and top_v_fam is not None and q_v_fam == top_v_fam)
                antigen_correct_hits[q_label].append(is_same)

    # Compute raw counts and diversity from raw human-filtered dataset
    raw_labels = normalize_labels(df_raw[spec.label_col])
    raw_v_gene = df_raw["Chain 1_Curated V Gene"].fillna(df_raw["Chain 1_Calculated V Gene"])
    raw_v_family = raw_v_gene.map(get_v_family)
    
    raw_df = pd.DataFrame({"label": raw_labels, "v_family": raw_v_family}).dropna()
    forced_labels = compute_shared_labels(df_raw, spec, 10, 14, None)
    if forced_labels:
        raw_df = raw_df[raw_df["label"].isin(forced_labels)].copy()
        
    results_b = []
    for label in forced_labels:
        label_raw = raw_df[raw_df["label"] == label]
        n_seqs = len(label_raw)
        entropy = compute_shannon_entropy(label_raw["v_family"])
        
        hits = antigen_correct_hits.get(label, [])
        if len(hits) > 0:
            same_v_rate = float(np.mean(hits))
        else:
            same_v_rate = np.nan
            
        results_b.append({
            "抗原": label,
            "N_seqs": n_seqs,
            "V-gene 多样性 (Shannon H)": entropy,
            "Same-V 命中率": same_v_rate
        })
        
    df_b = pd.DataFrame(results_b).sort_values("Same-V 命中率", ascending=False).reset_index(drop=True)
    
    # Format Same-V 命中率 as percentage string in markdown but keep float in CSV
    df_b_display = df_b.copy()
    df_b_display["Same-V 命中率"] = df_b_display["Same-V 命中率"].map(lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "N/A")
    df_b_display["V-gene 多样性 (Shannon H)"] = df_b_display["V-gene 多样性 (Shannon H)"].round(4)
    
    out_b_path = Path("outputs/reports/bcr_l4_antigen_vgene_breakdown.csv")
    df_b.to_csv(out_b_path, index=False)
    print(f"Saved V-gene Antigen Breakdown to {out_b_path}")
    print(df_b_display.to_string(index=False))

    # =========================================================================
    # Task C: VDJdb Cosine SD Diagnostics
    # =========================================================================
    print("\n=== Task C: VDJdb Cosine SD Diagnostics ===")
    
    # Load VDJdb TCR Level 1 sequences
    mock_args_tcr = MockArgs()
    mock_args_tcr.include_bcr = False
    mock_args_tcr.include_tcr = True
    spec_tcr = build_dataset_specs(mock_args_tcr)[0]
    df_raw_tcr = load_table(spec_tcr)
    df_raw_tcr = filter_human_only(df_raw_tcr, spec_tcr)
    
    tcr_l1_seqs = spec_tcr.sequence_builders["level1"](df_raw_tcr).map(clean_sequence).dropna().unique().tolist()
    print(f"Total unique VDJdb TCR Level 1 sequences: {len(tcr_l1_seqs)}")
    
    # Randomly sample n = 500 sequences
    np.random.seed(42)
    sampled_tcr = sorted(list(np.random.choice(tcr_l1_seqs, size=500, replace=False)))
    
    models_tcr = [
        ("esm2-150m", "outputs/embeddings/tcr_level1_esm2-150m_mean.npy"),
        ("esm2-650m", "outputs/embeddings/tcr_level1_esm2-650m_mean.npy"),
        ("esm2-3b", "outputs/embeddings/tcr_level1_esm2-3b_mean.npy")
    ]
    
    results_c = []
    
    for name, path in models_tcr:
        print(f"Loading precomputed VDJdb embeddings for {name} from {path}...")
        embedder_tcr = OfflineEmbedder(path)
        embs_tcr = embedder_tcr.encode(sampled_tcr)
        
        sim_matrix = cosine_similarity(embs_tcr)
        upper_tri = sim_matrix[np.triu_indices(500, k=1)]
        
        mean_val = float(upper_tri.mean())
        std_val = float(upper_tri.std())
        
        results_c.append({
            "model": name,
            "cosine_mean": mean_val,
            "cosine_std": std_val
        })
        
    df_c = pd.DataFrame(results_c)
    out_c_path = Path("outputs/diagnostics/vdjdb_l1_cosine_centralization.csv")
    df_c.to_csv(out_c_path, index=False)
    print(f"Saved VDJdb Cosine Centralization to {out_c_path}")
    print(df_c.to_string(index=False))

    print("\n=== All Tasks completed successfully! ===")

if __name__ == "__main__":
    main()
