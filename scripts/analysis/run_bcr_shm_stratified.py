import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from anarci import anarci

# Add scripts directory to path to import benchmark modules (portable: relative to this file)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.data import (
    build_dataset_specs,
    load_table,
    compute_shared_labels,
    init_label_config,
    filter_human_only,
    is_legit_sequence,
    normalize_labels,
    assign_antigen_type,
)
from benchmark.split import clone_aware_split_fast

class Args:
    include_bcr = True
    bcr_file = "raw/iedb/bcr_singlechain_vh.tsv"   # canonical input (was old bcr_full_single_header_anarci)
    top_labels = 10
    min_label_count = 14
    random_state = 42
    human_only = True
    test_size = 0.2
    clone_threshold = 0.95
    min_clones_per_label = 2
    bcr_label_col = "Epitope_Source Molecule"
    max_per_label_bcr = 150
    label_config = "label_aliases.json"
    shared_labels = True

args = Args()
init_label_config(args.label_config)

# ============================================================================
# 1. Parse germline V-genes from IGV.fasta
# ============================================================================
print("Parsing germline V-genes from imgt/IGV.fasta...")
v_genes = {}
for record in SeqIO.parse("imgt/IGV.fasta", "fasta"):
    parts = record.description.split("|")
    if len(parts) > 1:
        gene_name = parts[1].strip()
        v_genes[gene_name] = str(record.seq).upper().replace(".", "")

print(f"Loaded {len(v_genes)} germline sequences.")

# ============================================================================
# 2. Load BCR Table and filter human-only
# ============================================================================
print("Loading BCR table...")
specs = build_dataset_specs(args)
spec = specs[0]
df = load_table(spec)
df = filter_human_only(df, spec)
print(f"Loaded {len(df)} human-only BCR rows.")

# ============================================================================
# 3. Replicate slice to find all eligible sequences
# ============================================================================
forced_labels = compute_shared_labels(df, spec, args.top_labels, args.min_label_count, None)
print("Top-10 antigens:", forced_labels)

# Reconstruct the pilot_df creation to preserve the original index mapping
def replicate_bcr_pilot_df(df, spec, args, seed):
    labels = normalize_labels(df[spec.label_col])
    sequences = spec.sequence_builders["level4"](df)
    split_sequences = spec.sequence_builders["level1"](df)
    
    work = pd.DataFrame({
        "label": labels, 
        "sequence": sequences,
        "split_sequence": split_sequences,
        "antigen_type": assign_antigen_type(labels),
        "original_index": df.index
    }).dropna(subset=["label", "sequence", "split_sequence"])

    # Match build_pilot_slice: strict 20-AA legitimacy filter on the level sequence, so this
    # replicated pool/split aligns with the filtered canonical tree we read predictions from.
    work = work[work["sequence"].map(is_legit_sequence)]

    if forced_labels is not None:
        work = work[work["label"].isin(forced_labels)]
    
    sampled_parts = []
    for label in work["label"].unique():
        label_df = work[work["label"] == label].copy()
        if len(label_df) > args.max_per_label_bcr:
            label_df = label_df.sample(args.max_per_label_bcr, random_state=seed)
        sampled_parts.append(label_df)

    pilot_df = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sequence_col = "bcr_level4_sequence"
    pilot_df = pilot_df.rename(columns={"sequence": sequence_col})
    return pilot_df, sequence_col

# We gather all original_indices that appear in any seed's pilot slice
all_slice_indices = set()
seeds = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232]
for s in seeds:
    pdf, _ = replicate_bcr_pilot_df(df, spec, args, s)
    all_slice_indices.update(pdf["original_index"].tolist())

print(f"Total unique sequences in pilot slices across 20 seeds: {len(all_slice_indices)}")

# ============================================================================
# 4. Map each slice sequence to germline and run ANARCI
# ============================================================================
# Map each original index to germline info
mapped_germlines = {}  # original_index -> (mapped_gene_name, germline_seq)
query_seqs = {}       # original_index -> query_seq
v_gene_strs = {}      # original_index -> raw V gene string from tsv

for idx in all_slice_indices:
    row = df.loc[idx]
    q_seq = spec.sequence_builders["level4"](pd.DataFrame([row])).iloc[0]
    v_gene_str = row["Chain 1_Calculated V Gene"]
    
    mapped_gene = None
    g_seq = None
    if not pd.isna(v_gene_str):
        sub_genes = [g.strip() for g in str(v_gene_str).split(",") if g.strip()]
        # Try exact match
        for sg in sub_genes:
            if sg in v_genes:
                mapped_gene = sg
                g_seq = v_genes[sg]
                break
        # Try fallback
        if g_seq is None:
            for sg in sub_genes:
                base = sg.split("*")[0] if "*" in sg else sg
                # Find any allele in v_genes
                candidates = [k for k in v_genes if k.startswith(base + "*")]
                if candidates:
                    mapped_gene = candidates[0]
                    g_seq = v_genes[candidates[0]]
                    break
                    
    mapped_germlines[idx] = (mapped_gene, g_seq)
    query_seqs[idx] = q_seq
    v_gene_strs[idx] = v_gene_str

# Run ANARCI in batch for query sequences and germlines
unique_queries = list(set(query_seqs.values()))
unique_germlines = list(set([g[1] for g in mapped_germlines.values() if g[1] is not None]))

print(f"Running ANARCI on {len(unique_queries)} unique query sequences...")
def run_anarci_batch(seq_list, batch_size=200):
    res = {}
    tuples = [(str(i), seq) for i, seq in enumerate(seq_list)]
    for idx in range(0, len(tuples), batch_size):
        batch = tuples[idx : idx + batch_size]
        numbered, alignment_details, hit_tables = anarci(batch, scheme='imgt')
        for i, (name, seq) in enumerate(batch):
            num_data = numbered[i]
            if num_data is not None and len(num_data) > 0:
                res[seq] = num_data[0][0]  # list of ((pos, ins), aa)
            else:
                res[seq] = None
    return res

query_num = run_anarci_batch(unique_queries)
print(f"Running ANARCI on {len(unique_germlines)} unique germline sequences...")
germline_num = run_anarci_batch(unique_germlines)

# Compute SHM density for each index
shm_densities = {}  # original_index -> shm_density
print("Computing SHM densities...")
unmapped_count = 0
anarci_fail_count = 0

for idx in all_slice_indices:
    q_seq = query_seqs[idx]
    mapped_gene, g_seq = mapped_germlines[idx]
    
    if g_seq is None:
        unmapped_count += 1
        shm_densities[idx] = 0.0
        continue
        
    q_n = query_num.get(q_seq)
    g_n = germline_num.get(g_seq)
    
    if q_n is None or g_n is None:
        anarci_fail_count += 1
        shm_densities[idx] = 0.0
        continue
        
    v_q = {pos: aa for pos, aa in q_n if pos[0] <= 104 and aa != '-'}
    v_g = {pos: aa for pos, aa in g_n if pos[0] <= 104 and aa != '-'}
    
    if len(v_g) == 0:
        shm_densities[idx] = 0.0
        continue
        
    mismatches = 0
    all_positions = set(v_q.keys()) | set(v_g.keys())
    for pos in all_positions:
        aa_q = v_q.get(pos, '-')
        aa_g = v_g.get(pos, '-')
        if aa_q == '-' and aa_g == '-':
            continue
        if aa_q != aa_g:
            mismatches += 1
            
    shm_densities[idx] = mismatches / len(v_g)

print(f"SHM density computation complete. Unmapped: {unmapped_count}, ANARCI fail: {anarci_fail_count}")

# Print sample SHM densities
sample_indices = list(all_slice_indices)[:10]
for idx in sample_indices:
    print(f"Index: {idx} | V-gene: {v_gene_strs[idx]} | Mapped: {mapped_germlines[idx][0]} | SHM Density: {shm_densities[idx]:.4f}")

# ============================================================================
# 5. Perform Stratified Retrieval Analysis across 20 seeds
# ============================================================================
print("\nStarting 20-seed stratified retrieval analysis...")
seed_results = []  # list of dicts: {seed, bin_name, blosum_r1, esm_r1, count}

for s in seeds:
    # 1. Replicate pilot_df
    pilot_df, sequence_col = replicate_bcr_pilot_df(df, spec, args, s)
    
    # 2. Perform clone-aware split
    train_idx, test_idx = clone_aware_split_fast(
        pilot_df, cdr3_col="split_sequence", label_col="label", 
        test_size=args.test_size, clone_similarity_threshold=args.clone_threshold,
        random_state=s, min_clones_per_label=args.min_clones_per_label,
        show_progress=False
    )
    
    # 3. Load actual predictions
    blosum_pred_path = f"outputs/phase3_filtered_tie_v2/bcr/blosum62/seed_{s}/BCR_level4_blosum62_predictions.csv"
    esm_pred_path = f"outputs/phase3_filtered_tie_v2/bcr/esm2-150m/seed_{s}/BCR_level4_esm2-150m_predictions.csv"
    
    if not os.path.exists(blosum_pred_path) or not os.path.exists(esm_pred_path):
        print(f"Warning: Prediction files missing for seed {s}. Skipping.")
        continue
        
    blosum_pred = pd.read_csv(blosum_pred_path)
    esm_pred = pd.read_csv(esm_pred_path)
    
    assert len(blosum_pred) == len(test_idx), f"Length mismatch blosum seed {s}"
    assert len(esm_pred) == len(test_idx), f"Length mismatch esm seed {s}"
    
    # 4. Map each test index to original index and retrieve SHM density
    test_shm = []
    test_original_idx = pilot_df.loc[test_idx, "original_index"].values
    for o_idx in test_original_idx:
        test_shm.append(shm_densities[o_idx])
        
    # Combine predictions and SHM
    test_results = pd.DataFrame({
        "original_index": test_original_idx,
        "shm": test_shm,
        "blosum_correct": blosum_pred["exp_recall@1"].values,   # expected-R@1 (was single-pick rank==0)
        "esm_correct": esm_pred["exp_recall@1"].values
    })
    
    # Binning
    # Low: < 0.05
    # Medium: 0.05 <= shm < 0.15
    # High: shm >= 0.15
    bins = {
        "Low (<5%)": test_results[test_results["shm"] < 0.05],
        "Medium (5-15%)": test_results[(test_results["shm"] >= 0.05) & (test_results["shm"] < 0.15)],
        "High (>=15%)": test_results[test_results["shm"] >= 0.15]
    }
    
    for bin_name, bin_df in bins.items():
        count = len(bin_df)
        if count > 0:
            blosum_r1 = bin_df["blosum_correct"].mean()
            esm_r1 = bin_df["esm_correct"].mean()
        else:
            blosum_r1 = np.nan
            esm_r1 = np.nan
            
        seed_results.append({
            "seed": s,
            "bin": bin_name,
            "blosum_r1": blosum_r1,
            "esm_r1": esm_r1,
            "count": count
        })

# Aggregate results
results_df = pd.DataFrame(seed_results)

# Group by bin and compute mean and SD
summary_rows = []
for bin_name in ["Low (<5%)", "Medium (5-15%)", "High (>=15%)"]:
    bin_data = results_df[results_df["bin"] == bin_name]
    
    # Compute mean and SD across seeds
    blosum_mean = bin_data["blosum_r1"].mean()
    blosum_sd = bin_data["blosum_r1"].std()
    esm_mean = bin_data["esm_r1"].mean()
    esm_sd = bin_data["esm_r1"].std()
    avg_count = bin_data["count"].mean()
    
    # Compute delta per seed and average delta
    delta_series = bin_data["esm_r1"] - bin_data["blosum_r1"]
    delta_mean = delta_series.mean()
    delta_sd = delta_series.std()
    
    summary_rows.append({
        "SHM Bin": bin_name,
        "N_seqs_avg": avg_count,
        "BLOSUM62 R@1 Mean": blosum_mean,
        "BLOSUM62 R@1 SD": blosum_sd,
        "ESM2-150M R@1 Mean": esm_mean,
        "ESM2-150M R@1 SD": esm_sd,
        "Delta Mean": delta_mean,
        "Delta SD": delta_sd
    })

summary_df = pd.DataFrame(summary_rows)

print("\nSHM-Stratified Summary:")
print(summary_df.to_string(index=False))

# ============================================================================
# 6. Save results to output path
# ============================================================================
output_path = Path("outputs/task249_k_shm_stratified/bcr_l4_shm_stratified_r1.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(output_path, index=False)
print(f"\nSaved results to {output_path}")
