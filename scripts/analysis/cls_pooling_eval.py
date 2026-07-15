#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import torch
from typing import Iterable
import pickle

# Add scripts/ directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.data import build_dataset_specs, load_table, filter_human_only, init_label_config, build_pilot_slice, compute_shared_labels
from benchmark.split import clone_aware_split_fast
from benchmark.embeddings import get_embedder, TransformerEmbedder
from benchmark.evaluate import retrieval_metrics
from anarci import anarci

CDR_MASK_CACHE = {}

CACHE_FILE = Path("scratch/esm2_150m_hidden_cache.pkl")
HIDDEN_STATE_CACHE = {}

def load_cache():
    global HIDDEN_STATE_CACHE
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                HIDDEN_STATE_CACHE = pickle.load(f)
            print(f"Loaded {len(HIDDEN_STATE_CACHE)} cached sequence hidden states from {CACHE_FILE}")
        except Exception as e:
            print(f"Failed to load cache: {e}")

def save_cache():
    if not CACHE_FILE.parent.exists():
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(HIDDEN_STATE_CACHE, f)
        print(f"Saved {len(HIDDEN_STATE_CACHE)} cached sequence hidden states to {CACHE_FILE}")
    except Exception as e:
        print(f"Failed to save cache: {e}")

def get_cdr_mask(sequence: str) -> np.ndarray:
    if sequence in CDR_MASK_CACHE:
        return CDR_MASK_CACHE[sequence]
        
    mask = np.zeros(len(sequence), dtype=float)
    try:
        results = anarci([("0", sequence)], scheme='imgt')
        numbered, alignment_details, hit_tables = results
        
        if numbered[0] is not None and len(numbered[0]) > 0:
            domain_residues = numbered[0][0][0]
            align_details = alignment_details[0][0]
            
            if align_details is not None:
                query_start = align_details['query_start']
                seq_idx = query_start
                for (pos, ins), res in domain_residues:
                    if res == '-':
                        continue
                    
                    if seq_idx < len(sequence) and sequence[seq_idx] == res:
                        is_cdr = (27 <= pos <= 38) or (56 <= pos <= 65) or (105 <= pos <= 117)
                        if is_cdr:
                            mask[seq_idx] = 1.0
                        seq_idx += 1
    except Exception as e:
        print(f"ANARCI parsing failed for sequence: {sequence}. Error: {e}")
        
    CDR_MASK_CACHE[sequence] = mask
    return mask

FRAMEWORK_MASK_CACHE = {}

def get_framework_mask(sequence: str) -> np.ndarray:
    """In-domain FRAMEWORK mask: 1.0 on variable-domain positions ANARCI numbers that are
    NOT in a CDR (IMGT CDR1 27-38, CDR2 56-65, CDR3 105-117). Positions outside the numbered
    domain stay 0, so this is framework-ONLY (not 'every non-CDR token'). Mirror of
    get_cdr_mask with the CDR test negated."""
    if sequence in FRAMEWORK_MASK_CACHE:
        return FRAMEWORK_MASK_CACHE[sequence]

    mask = np.zeros(len(sequence), dtype=float)
    try:
        results = anarci([("0", sequence)], scheme='imgt')
        numbered, alignment_details, hit_tables = results

        if numbered[0] is not None and len(numbered[0]) > 0:
            domain_residues = numbered[0][0][0]
            align_details = alignment_details[0][0]

            if align_details is not None:
                query_start = align_details['query_start']
                seq_idx = query_start
                for (pos, ins), res in domain_residues:
                    if res == '-':
                        continue

                    if seq_idx < len(sequence) and sequence[seq_idx] == res:
                        is_cdr = (27 <= pos <= 38) or (56 <= pos <= 65) or (105 <= pos <= 117)
                        if not is_cdr:
                            mask[seq_idx] = 1.0
                        seq_idx += 1
    except Exception as e:
        print(f"ANARCI parsing failed for sequence: {sequence}. Error: {e}")

    FRAMEWORK_MASK_CACHE[sequence] = mask
    return mask

CURRENT_POOLING = "mean"

def custom_encode(self, sequences: Iterable[str]) -> np.ndarray:
    sequences = list(sequences)
    
    # Check for missing sequences
    uncached = [s for s in sequences if s not in HIDDEN_STATE_CACHE]
    if uncached:
        print(f"Encoding {len(uncached)} new sequences with ESM2-150M on CPU...")
        for start in range(0, len(uncached), self.batch_size):
            batch_seqs = uncached[start:start + self.batch_size]
            inputs = self.tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state.cpu().numpy()
                att_mask = inputs["attention_mask"].cpu().numpy()
                
            for i, seq in enumerate(batch_seqs):
                seq_len = int(att_mask[i].sum())
                # Slice out the non-padded tokens
                seq_hidden = hidden[i:i+1, :seq_len, :]
                seq_mask = att_mask[i:i+1, :seq_len]
                HIDDEN_STATE_CACHE[seq] = (seq_hidden, seq_mask)
                
    # Now all sequences are in cache. Construct the pooled representations.
    pooled_list = []
    for seq in sequences:
        hidden_np, att_mask_np = HIDDEN_STATE_CACHE[seq]
        if CURRENT_POOLING == "cls":
            pooled_val = hidden_np[0, 0, :]
        elif CURRENT_POOLING == "cdr_masked":
            length = int(att_mask_np.sum())
            mask = get_cdr_mask(seq)
            seq_len = length - 2
            
            res_hidden = hidden_np[0, 1:length-1, :]
            
            if len(mask) == seq_len and mask.sum() > 0:
                mask_tensor = mask[:, np.newaxis]
                pooled_val = (res_hidden * mask_tensor).sum(axis=0) / mask_tensor.sum()
            else:
                pooled_val = res_hidden.mean(axis=0)
        elif CURRENT_POOLING == "framework_masked":
            length = int(att_mask_np.sum())
            mask = get_framework_mask(seq)
            seq_len = length - 2

            res_hidden = hidden_np[0, 1:length-1, :]

            if len(mask) == seq_len and mask.sum() > 0:
                mask_tensor = mask[:, np.newaxis]
                pooled_val = (res_hidden * mask_tensor).sum(axis=0) / mask_tensor.sum()
            else:
                pooled_val = res_hidden.mean(axis=0)
        else:  # mean pooling
            pooled_val = hidden_np[0].mean(axis=0)
            
        pooled_list.append(pooled_val)
        
    return np.vstack(pooled_list)

TransformerEmbedder.encode = custom_encode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bcr-file", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="comma-separated seeds; overrides --seed for multi-seed aggregation")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    load_cache()
    init_label_config("label_aliases.json")
    
    # Mock args
    class MockArgs:
        include_bcr = True
        include_tcr = False
        include_sabdab = False
        bcr_file = args.bcr_file
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
    
    # Ensure consistent 10 shared labels
    forced_labels = compute_shared_labels(df_raw, spec, top_labels=10, min_label_count=14)
    print(f"-> BCR shared labels (n={len(forced_labels)}): {forced_labels}")
    
    embedder = get_embedder("esm2-150m", mock_args)
    results = []

    global CURRENT_POOLING
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]

    for seed in seeds:
      # level4 poolings: mean, cls, cdr_masked (=CDR-ONLY: averages CDR1/2/3 residues),
      # framework_masked (=FRAMEWORK-ONLY: averages variable-domain non-CDR residues). level1: mean, cls.
      for level in ["level4", "level1"]:
        print(f"\n=== seed {seed} | Slicing and splitting for level {level} ===")
        slice_ = build_pilot_slice(df_raw, spec, level, mock_args.max_per_label_bcr, seed, forced_labels)
        if slice_ is None:
            print(f"Skipping {level} - insufficient data")
            continue

        df = slice_.dataframe
        train_idx, test_idx = clone_aware_split_fast(
            df, cdr3_col="split_sequence", label_col="label",
            test_size=0.2, clone_similarity_threshold=0.95,
            random_state=seed, min_clones_per_label=2,
            show_progress=False
        )

        train_seq = df[slice_.sequence_col].values[train_idx]
        test_seq = df[slice_.sequence_col].values[test_idx]
        train_lbl = df["label"].values[train_idx]
        test_lbl = df["label"].values[test_idx]
        train_type = df["antigen_type"].values[train_idx]
        test_type = df["antigen_type"].values[test_idx]

        poolings_to_run = ["mean", "cls"]
        if level == "level4":
            poolings_to_run += ["cdr_masked", "framework_masked"]

        for pooling in poolings_to_run:
            print(f"Encoding seed {seed} {level} with {pooling} pooling...")
            CURRENT_POOLING = pooling

            q_emb = embedder.encode(test_seq)
            d_emb = embedder.encode(train_seq)

            metrics = retrieval_metrics(
                q_emb=q_emb, d_emb=d_emb,
                q_labels=test_lbl, d_labels=train_lbl,
                q_antigen_types=test_type,
                method="esm2-150m"
            )

            results.append({
                "seed": seed,
                "level": level,
                "pooling": pooling,
                "recall@1": metrics["recall@1"],
                "recall@5": metrics["recall@5"],
                "mrr": metrics["mrr"]
            })
            print(f"  seed {seed} {level} {pooling} -> R@1: {metrics['recall@1']:.3f} | R@5: {metrics['recall@5']:.3f}")

    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    res_df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")
    save_cache()

if __name__ == "__main__":
    main()
