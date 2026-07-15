import numpy as np
import pandas as pd
from typing import List, Tuple

try:
    from rapidfuzz.distance import Levenshtein as RapidLevenshtein
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False
    print("Warning: rapidfuzz not available, using slower fallback")

# ============================================================================
# Optimized Clone Clustering
# ============================================================================

def levenshtein_distance_fast(s1: str, s2: str) -> int:
    """Compute Levenshtein distance (optimized with rapidfuzz if available)."""
    if HAVE_RAPIDFUZZ:
        return RapidLevenshtein.distance(s1, s2)
    else:
        return _levenshtein_fallback(s1, s2)

def _levenshtein_fallback(s1: str, s2: str) -> int:
    """Fallback Levenshtein implementation."""
    if len(s1) < len(s2):
        return _levenshtein_fallback(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def cdr3_similarity_fast(s1: str, s2: str) -> float:
    """Compute CDR3 sequence similarity (optimized)."""
    if not s1 or not s2:
        return 0.0
    dist = levenshtein_distance_fast(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0

def fast_clone_clustering(
    sequences: List[str], 
    threshold: float = 0.95,
    show_progress: bool = False
) -> np.ndarray:
    """Fast greedy clustering using rapidfuzz with length-filtering and early-exit."""
    if len(sequences) == 0:
        return np.array([])
    
    from collections import defaultdict
    
    cluster_ids = np.zeros(len(sequences), dtype=int)
    # Map: length -> list of (cluster_id, rep_seq)
    reps_by_len = defaultdict(list)
    
    # First sequence initializes cluster 0
    first_seq = sequences[0]
    cluster_ids[0] = 0
    reps_by_len[len(first_seq)].append((0, first_seq))
    next_cluster_id = 1
    
    iterator = enumerate(sequences[1:], start=1)
    
    for i, seq in iterator:
        q_len = len(seq)
        max_len_diff = int(round((1.0 - threshold) * q_len))
        
        # Greedy FIRST-match (not best-match): the sequence joins the first existing
        # representative within `threshold`, scanning length buckets low->high and, within a
        # bucket, in insertion order. Deterministic given the input sequence order (which is
        # fixed by the upstream dataframe row order), but it is an order-dependent heuristic,
        # not a globally optimal clustering.
        matched_cluster = -1
        matched_similarity = -1.0

        # Only check representatives within allowed length difference
        for check_len in range(q_len - max_len_diff, q_len + max_len_diff + 1):
            if check_len not in reps_by_len:
                continue

            for cluster_id, rep_seq in reps_by_len[check_len]:
                sim = cdr3_similarity_fast(seq, rep_seq)
                if sim >= threshold:
                    # Early exit: found a cluster within threshold
                    matched_similarity = sim
                    matched_cluster = cluster_id
                    break

            if matched_cluster != -1:
                break

        if matched_cluster != -1 and matched_similarity >= threshold:
            cluster_ids[i] = matched_cluster
        else:
            # Create a new cluster
            cluster_ids[i] = next_cluster_id
            reps_by_len[q_len].append((next_cluster_id, seq))
            next_cluster_id += 1
            
    return cluster_ids

# ============================================================================
# Clone-aware Splitting
# ============================================================================

def clone_aware_split_fast(
    df: pd.DataFrame,
    cdr3_col: str,
    label_col: str,
    test_size: float = 0.3,
    clone_similarity_threshold: float = 0.95,
    random_state: int = 42,
    min_clones_per_label: int = 2,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast clone-aware split to prevent data leakage.

    Returns POSITIONAL indices into `df` (use `df.iloc[idx]`). CONTRACT: pass a `df` with a
    0..n-1 RangeIndex. After dropping labels with too few clones the function remaps via
    `orig_pos = df.index.to_numpy()`, so the returned indices are the positions of surviving
    rows in the ORIGINAL df; `df.iloc[...]` is correct only if that index is positional.
    `build_pilot_slice` always returns a reset_index df, so all canonical callers are safe.
    """
    np.random.seed(random_state)
    
    df = df.copy()
    df['clone_id'] = -1
    
    clone_offset = 0
    label_clone_stats = []
    
    for label in df[label_col].unique():
        label_mask = df[label_col] == label
        label_df = df[label_mask]
        
        cdr3_sequences = label_df[cdr3_col].tolist()
        
        clone_ids = fast_clone_clustering(
            cdr3_sequences, 
            threshold=clone_similarity_threshold,
            show_progress=show_progress
        )
        
        clone_ids += clone_offset
        df.loc[label_mask, 'clone_id'] = clone_ids
        
        n_clones = len(np.unique(clone_ids))
        clone_offset += n_clones
        
        label_clone_stats.append({
            'label': label,
            'n_clones': n_clones,
        })
    
    labels_to_keep = [
        stat['label'] for stat in label_clone_stats 
        if stat['n_clones'] >= min_clones_per_label
    ]
    
    # **CRITICAL FIX**: Reset index to get iloc positions instead of index values
    # Keep the kept rows' ORIGINAL positions (the input df has a RangeIndex from
    # build_pilot_slice, so index == positional index). When a label is dropped above
    # (n_clones < min_clones_per_label), reset_index would renumber the survivors and the
    # returned indices would no longer line up with the caller's df[col].values[idx]
    # (positional) access -> silent train/test misassignment. We therefore work internally on
    # the reset frame but remap the returned indices back to these original positions.
    df = df[df[label_col].isin(labels_to_keep)].copy()
    orig_pos = df.index.to_numpy()
    df = df.reset_index(drop=True)

    train_indices = []
    test_indices = []
    
    for label in labels_to_keep:
        label_mask = df[label_col] == label
        label_df = df[label_mask]
        
        unique_clones = label_df['clone_id'].unique()
        n_clones = len(unique_clones)
        n_test_clones = max(1, int(n_clones * test_size))
        
        test_clones = np.random.choice(unique_clones, n_test_clones, replace=False)
        
        for clone_id in unique_clones:
            clone_indices = label_df[label_df['clone_id'] == clone_id].index.tolist()
            if clone_id in test_clones:
                test_indices.extend(clone_indices)
            else:
                train_indices.extend(clone_indices)
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # Verification
    train_clones = set(df.loc[train_indices, 'clone_id'])
    test_clones = set(df.loc[test_indices, 'clone_id'])
    overlap = train_clones & test_clones
    
    if overlap:
        raise ValueError(f"Clone leakage detected! {len(overlap)} clones in both sets")

    # Remap reset-frame positions back to the caller's original positions (see note above).
    return orig_pos[train_indices], orig_pos[test_indices]

def zero_shot_antigen_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.3,
    random_state: int = 42,
    min_samples_per_label: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data such that entire antigen classes are held out for zero-shot testing.
    The test set contains labels that DO NOT exist in the train set.
    """
    np.random.seed(random_state)
    
    # Only consider labels with enough samples
    label_counts = df[label_col].value_counts()
    valid_labels = label_counts[label_counts >= min_samples_per_label].index.tolist()
    
    n_test_labels = max(1, int(len(valid_labels) * test_size))
    test_labels = set(np.random.choice(valid_labels, n_test_labels, replace=False))
    
    valid_mask = df[label_col].isin(valid_labels)
    test_mask = valid_mask & df[label_col].isin(test_labels)
    train_mask = valid_mask & ~test_mask
    
    return df.index[train_mask].values, df.index[test_mask].values
