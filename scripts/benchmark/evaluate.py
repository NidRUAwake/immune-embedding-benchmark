import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple, Optional
from joblib import Parallel, delayed

try:
    import parasail
    HAVE_PARASAIL = True
except ImportError:
    HAVE_PARASAIL = False
    warnings.warn(
        "parasail is NOT installed: BLOSUM62 alignment will fall back to an ungapped "
        "positional scorer that produces DIFFERENT scores and rankings than the canonical "
        "Smith-Waterman path. Results will NOT reproduce the published numbers. "
        "Install parasail (see environment.yml) before running the benchmark.",
        RuntimeWarning,
    )

# Copy BLOSUM62 from data.py or keep it here for fallback
# Actually, since BLOSUM62 is quite large, I'll put it here.
BLOSUM62 = {
    ('A', 'A'): 4, ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2, ('A', 'C'): 0,
    ('A', 'Q'): -1, ('A', 'E'): -1, ('A', 'G'): 0, ('A', 'H'): -2, ('A', 'I'): -1,
    ('A', 'L'): -1, ('A', 'K'): -1, ('A', 'M'): -1, ('A', 'F'): -2, ('A', 'P'): -1,
    ('A', 'S'): 1, ('A', 'T'): 0, ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0,
    ('R', 'R'): 5, ('R', 'N'): 0, ('R', 'D'): -2, ('R', 'C'): -3, ('R', 'Q'): 1,
    ('R', 'E'): 0, ('R', 'G'): -2, ('R', 'H'): 0, ('R', 'I'): -3, ('R', 'L'): -2,
    ('R', 'K'): 2, ('R', 'M'): -1, ('R', 'F'): -3, ('R', 'P'): -2, ('R', 'S'): -1,
    ('R', 'T'): -1, ('R', 'W'): -3, ('R', 'Y'): -2, ('R', 'V'): -3,
    ('N', 'N'): 6, ('N', 'D'): 1, ('N', 'C'): -3, ('N', 'Q'): 0, ('N', 'E'): 0,
    ('N', 'G'): 0, ('N', 'H'): 1, ('N', 'I'): -3, ('N', 'L'): -3, ('N', 'K'): 0,
    ('N', 'M'): -2, ('N', 'F'): -3, ('N', 'P'): -2, ('N', 'S'): 1, ('N', 'T'): 0,
    ('N', 'W'): -4, ('N', 'Y'): -2, ('N', 'V'): -3,
    ('D', 'D'): 6, ('D', 'C'): -3, ('D', 'Q'): 0, ('D', 'E'): 2, ('D', 'G'): -1,
    ('D', 'H'): -1, ('D', 'I'): -3, ('D', 'L'): -4, ('D', 'K'): -1, ('D', 'M'): -3,
    ('D', 'F'): -3, ('D', 'P'): -1, ('D', 'S'): 0, ('D', 'T'): -1, ('D', 'W'): -4,
    ('D', 'Y'): -3, ('D', 'V'): -3,
    ('C', 'C'): 9, ('C', 'Q'): -3, ('C', 'E'): -4, ('C', 'G'): -3, ('C', 'H'): -3,
    ('C', 'I'): -1, ('C', 'L'): -1, ('C', 'K'): -3, ('C', 'M'): -1, ('C', 'F'): -2,
    ('C', 'P'): -3, ('C', 'S'): -1, ('C', 'T'): -1, ('C', 'W'): -2, ('C', 'Y'): -2,
    ('C', 'V'): -1,
    ('Q', 'Q'): 5, ('Q', 'E'): 2, ('Q', 'G'): -2, ('Q', 'H'): 0, ('Q', 'I'): -3,
    ('Q', 'L'): -2, ('Q', 'K'): 1, ('Q', 'M'): 0, ('Q', 'F'): -3, ('Q', 'P'): -1,
    ('Q', 'S'): 0, ('Q', 'T'): -1, ('Q', 'W'): -2, ('Q', 'Y'): -1, ('Q', 'V'): -2,
    ('E', 'E'): 5, ('E', 'G'): -2, ('E', 'H'): 0, ('E', 'I'): -3, ('E', 'L'): -3,
    ('E', 'K'): 1, ('E', 'M'): -2, ('E', 'F'): -3, ('E', 'P'): -1, ('E', 'S'): 0,
    ('E', 'T'): -1, ('E', 'W'): -3, ('E', 'Y'): -2, ('E', 'V'): -2,
    ('G', 'G'): 6, ('G', 'H'): -2, ('G', 'I'): -4, ('G', 'L'): -4, ('G', 'K'): -2,
    ('G', 'M'): -3, ('G', 'F'): -3, ('G', 'P'): -2, ('G', 'S'): 0, ('G', 'T'): -2,
    ('G', 'W'): -2, ('G', 'Y'): -3, ('G', 'V'): -3,
    ('H', 'H'): 8, ('H', 'I'): -3, ('H', 'L'): -3, ('H', 'K'): -1, ('H', 'M'): -2,
    ('H', 'F'): -1, ('H', 'P'): -2, ('H', 'S'): -1, ('H', 'T'): -2, ('H', 'W'): -2,
    ('H', 'Y'): 2, ('H', 'V'): -3,
    ('I', 'I'): 4, ('I', 'L'): 2, ('I', 'K'): -3, ('I', 'M'): 1, ('I', 'F'): 0,
    ('I', 'P'): -3, ('I', 'S'): -2, ('I', 'T'): -1, ('I', 'W'): -3, ('I', 'Y'): -1,
    ('I', 'V'): 3,
    ('L', 'L'): 4, ('L', 'K'): -2, ('L', 'M'): 2, ('L', 'F'): 0, ('L', 'P'): -3,
    ('L', 'S'): -2, ('L', 'T'): -1, ('L', 'W'): -2, ('L', 'Y'): -1, ('L', 'V'): 1,
    ('K', 'K'): 5, ('K', 'M'): -1, ('K', 'F'): -3, ('K', 'P'): -1, ('K', 'S'): 0,
    ('K', 'T'): -1, ('K', 'W'): -3, ('K', 'Y'): -2, ('K', 'V'): -2,
    ('M', 'M'): 5, ('M', 'F'): 0, ('M', 'P'): -2, ('M', 'S'): -1, ('M', 'T'): -1,
    ('M', 'W'): -1, ('M', 'Y'): -1, ('M', 'V'): 1,
    ('F', 'F'): 6, ('F', 'P'): -4, ('F', 'S'): -2, ('F', 'T'): -2, ('F', 'W'): 1,
    ('F', 'Y'): 3, ('F', 'V'): -1,
    ('P', 'P'): 7, ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'W'): -4, ('P', 'Y'): -3,
    ('P', 'V'): -2,
    ('S', 'S'): 4, ('S', 'T'): 1, ('S', 'W'): -3, ('S', 'Y'): -2, ('S', 'V'): -2,
    ('T', 'T'): 5, ('T', 'W'): -2, ('T', 'Y'): -2, ('T', 'V'): 0,
    ('W', 'W'): 11, ('W', 'Y'): 2, ('W', 'V'): -3,
    ('Y', 'Y'): 7, ('Y', 'V'): -1,
    ('V', 'V'): 4,
}

def blosum_score_fast(seq1: str, seq2: str) -> float:
    if HAVE_PARASAIL:
        result = parasail.sw_striped_16(seq1, seq2, 10, 1, parasail.blosum62)
        max_len = max(len(seq1), len(seq2))
        return result.score / max_len if max_len > 0 else 0
    # parasail missing: the ungapped fallback below gives DIFFERENT scores/rankings than the
    # canonical Smith-Waterman path -> fail hard so it can never silently produce wrong paper
    # numbers. Opt in with ALLOW_BLOSUM_FALLBACK=1 only for non-canonical/exploratory use.
    if os.environ.get("ALLOW_BLOSUM_FALLBACK") != "1":
        raise RuntimeError(
            "parasail not installed but BLOSUM62 scoring requested. The ungapped fallback is NOT "
            "the canonical Smith-Waterman alignment and would give wrong numbers. Install parasail "
            "(environment.yml), or set ALLOW_BLOSUM_FALLBACK=1 to override (non-canonical).")
    min_len, score = min(len(seq1), len(seq2)), 0.0
    for i in range(min_len):
        key = (seq1[i], seq2[i]) if (seq1[i], seq2[i]) in BLOSUM62 else (seq2[i], seq1[i])
        score += BLOSUM62.get(key, -4)
    score -= abs(len(seq1) - len(seq2)) * 2
    return score / max(len(seq1), len(seq2)) if max(len(seq1), len(seq2)) > 0 else 0

def compute_similarity_matrix(query_emb: np.ndarray, db_emb: np.ndarray, method: str, distance_metric: str = "cosine") -> np.ndarray:
    if query_emb.dtype.kind in ('U', 'S', 'O'):
        # It's an alignment method using raw strings (Unicode, String, or Object)
        from benchmark.split import levenshtein_distance_fast
        
        def compute_row(q, db_emb, method):
            row = np.zeros(len(db_emb), dtype=np.float32)
            for j, d in enumerate(db_emb):
                if method == "levenshtein":
                    dist, max_len = levenshtein_distance_fast(q, d), max(len(q), len(d))
                    row[j] = 1.0 - (dist / max_len) if max_len > 0 else 1.0
                elif method in ("tcrdist", "bcrdist"):
                    # Expecting "CDR1|CDR2|CDR3" format for Level 3
                    q_parts = str(q).split('|')
                    d_parts = str(d).split('|')
                    if len(q_parts) == 3 and len(d_parts) == 3:
                        # Weights: CDR1=1, CDR2=1, CDR3=3
                        s1 = blosum_score_fast(q_parts[0], d_parts[0]) if q_parts[0] and d_parts[0] else 0
                        s2 = blosum_score_fast(q_parts[1], d_parts[1]) if q_parts[1] and d_parts[1] else 0
                        s3 = blosum_score_fast(q_parts[2], d_parts[2]) if q_parts[2] and d_parts[2] else 0
                        row[j] = (s1 + s2 + 3 * s3) / 5.0
                    else:
                        # Fallback to simple blosum if format is unexpected
                        row[j] = blosum_score_fast(str(q), str(d))
                else:
                    row[j] = blosum_score_fast(str(q), str(d))
            return row

        # Parallelize across cores (override with ALIGN_N_JOBS to avoid oversubscription
        # when several run.py invocations run concurrently). Results are assembled in input
        # order regardless of n_jobs, so the output is deterministic.
        n_jobs = int(os.environ.get("ALIGN_N_JOBS", "10"))
        print(f"Parallelizing {method} computation across {n_jobs} cores...")
        rows = Parallel(n_jobs=n_jobs)(delayed(compute_row)(q, db_emb, method) for q in query_emb)
        return np.vstack(rows)
    
    
    # Otherwise it's a numeric embedding method
    if distance_metric == "euclidean":
        from sklearn.metrics.pairwise import euclidean_distances
        # Negative distance so that higher is more similar
        return -euclidean_distances(query_emb, db_emb)
    
    return cosine_similarity(query_emb, db_emb)

def retrieval_metrics(
    q_emb: np.ndarray, 
    d_emb: np.ndarray, 
    q_labels: np.ndarray, 
    d_labels: np.ndarray, 
    q_antigen_types: np.ndarray,
    method: str, 
    k_values: Tuple[int, ...] = (1, 5, 10),
    exclude_non_protein: bool = False,
    exclude_self: bool = False,
    return_predictions: bool = False,
    return_ranks: bool = False,
    precomputed_sim: Optional[np.ndarray] = None,
    distance_metric: str = "cosine",
    vj_filter: Optional[str] = None,
    q_v_gene: Optional[np.ndarray] = None,
    q_j_gene: Optional[np.ndarray] = None,
    d_v_gene: Optional[np.ndarray] = None,
    d_j_gene: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate retrieval metrics, supporting precomputed similarity for late-fusion."""
    
    # BLAST is now routed through the SAME order-independent expected-R@1 machinery as every
    # other method: blast_retrieval_scores returns a bitscore matrix (unreported references get
    # a sentinel below all bitscores so they collapse into one bottom tie-group), which we feed
    # in as the similarity matrix. This removes the previous order-dependent best-rank tie-break.
    blast_meta = None
    if method == "blast":
        from task285_blast.blast_retrieval import blast_retrieval_scores
        sim, query_time, db_size_bytes = blast_retrieval_scores(
            queries=q_emb,
            references=d_emb,
            blast_path=os.environ.get("BLAST_BIN", "blastp"),
        )
        blast_meta = {"query_time": query_time, "db_size_bytes": db_size_bytes}
    elif precomputed_sim is not None:
        sim = precomputed_sim
    else:
        sim = compute_similarity_matrix(q_emb, d_emb, method, distance_metric=distance_metric)
    
    # Filter query antigens if needed
    if exclude_non_protein:
        protein_mask = q_antigen_types == "protein"
        if not protein_mask.any():
            # If all are non-protein, return 0s
            return {**{f"recall@{k}": 0.0 for k in k_values}, "mrr": 0.0}
            
        sim = sim[protein_mask]
        q_labels = q_labels[protein_mask]

    # --- Expected R@1/R@5/MRR under uniform-random tie resolution (order-independent) ---
    # See discussions/311 + 312. R@1 = c/t for the top tie-group (exact); R@k and MRR
    # use the closed-form expectation over the random within-group permutation of the
    # *first* tie-group containing a same-label candidate. PLM cosine scores effectively
    # never tie, so PLM R@1 is unchanged vs. the previous single-pick implementation.
    recalls = {k: [] for k in k_values}              # per-query expected recall@k
    mrr = []                                         # per-query expected MRR
    opt = []                                         # per-query optimistic R@1
    pess = []                                        # per-query pessimistic R@1
    tie_sizes = []                                   # |top tie group| (diagnostic)
    predictions = []
    ranks = []                                       # best-case rank of any same-label

    def _expected_one(sim_scores, is_same, k_values):
        """Exact expected metrics under uniform random tie-resolution."""
        n = len(sim_scores)
        # top tie group -> exact R@1 expected / optimistic / pessimistic
        mx = sim_scores.max()
        top_mask = (sim_scores == mx)
        t_top = int(top_mask.sum())
        c_top = int(is_same[top_mask].sum())
        r1_exp = c_top / t_top
        r1_opt = 1.0 if c_top >= 1 else 0.0
        r1_pess = 1.0 if c_top == t_top else 0.0
        # find g* = first tie group (desc score) containing a same-label
        order = np.argsort(-sim_scores, kind="stable")
        s_sorted = sim_scores[order]
        same_sorted = is_same[order]
        off, i, gstar = 0, 0, None
        while i < n:
            j = i
            while j < n and s_sorted[j] == s_sorted[i]:
                j += 1
            s_g = j - i
            m_g = int(same_sorted[i:j].sum())
            if m_g > 0:
                gstar = (off, s_g, m_g); break
            off += s_g; i = j
        if gstar is None:
            # no same-label anywhere
            return r1_exp, r1_opt, r1_pess, {k: 0.0 for k in k_values}, 0.0, t_top, n
        off, s, m = gstar
        # P(first same-label within g* at position >= t) = prod_{a=0..t-1} (s-m-a)/(s-a)
        # incremental computation
        def P_ge_table():
            arr = np.empty(s - m + 2)
            arr[0] = 1.0
            for t in range(1, s - m + 1):
                arr[t] = arr[t - 1] * (s - m - (t - 1)) / (s - (t - 1))
            arr[s - m + 1] = 0.0
            return arr
        P_ge = P_ge_table()
        # E[recall@k] = 1 - P(p >= k-off) capped by group bounds
        rec = {}
        for k in k_values:
            q = k - off
            if q <= 0:
                rec[k] = 0.0
            elif q > (s - m):
                rec[k] = 1.0
            else:
                rec[k] = 1.0 - float(P_ge[q])
        # E[MRR] = sum_p (P_ge[p] - P_ge[p+1]) / (off + p + 1)
        mrr_val = 0.0
        for p in range(0, s - m + 1):
            Pp = float(P_ge[p] - P_ge[p + 1])
            if Pp > 0:
                mrr_val += Pp / (off + p + 1)
        # best-case rank (deterministic minimum over same-label positions)
        best_rank = off
        return r1_exp, r1_opt, r1_pess, rec, mrr_val, t_top, best_rank

    if vj_filter not in {None, "v", "vj"}:
        raise ValueError(f"Unknown vj_filter: {vj_filter}")
    if vj_filter is not None and (q_v_gene is None or d_v_gene is None):
        raise ValueError("vj_filter requires v_gene arrays")
    if vj_filter == "vj" and (q_j_gene is None or d_j_gene is None):
        raise ValueError("vj_filter='vj' requires j_gene arrays")

    def _append_fail():
        # query has no usable V/J gene, or no candidate survives the V/J filter -> recall fails
        for k in k_values:
            recalls[k].append(0.0)
        mrr.append(0.0); opt.append(0.0); pess.append(0.0)
        tie_sizes.append(0); ranks.append(len(d_labels))
        if return_predictions:
            predictions.append([])

    for i, label in enumerate(q_labels):
        sim_scores = sim[i].copy()
        if exclude_self:
            sim_scores[i] = -np.inf
        if sim_scores.shape[0] != d_labels.shape[0]:
            raise IndexError(f"Dimension mismatch: sim_scores {sim_scores.shape} vs d_labels {d_labels.shape}")
        is_same = (d_labels == label)
        d_labels_local = d_labels
        if vj_filter is not None:
            # Two-stage clonotype retrieval: RESTRICT the candidate pool to same-gene
            # candidates, then rank by CDR3 similarity. Filtered-out candidates are removed
            # from the pool entirely (not just down-weighted), so they cannot earn R@5/MRR
            # credit. A query with no usable gene, or with no surviving candidate, fails.
            qv = q_v_gene[i] if q_v_gene is not None else None
            qj = q_j_gene[i] if q_j_gene is not None else None
            if vj_filter == "v":
                valid = qv is not None
                mask = (d_v_gene == qv) if valid else None
            else:
                valid = (qv is not None) and (qj is not None)
                mask = ((d_v_gene == qv) & (d_j_gene == qj)) if valid else None
            if (not valid) or (not np.any(mask)):
                _append_fail()
                continue
            sim_scores = sim_scores[mask]
            is_same = is_same[mask]
            d_labels_local = d_labels[mask]
        r1_e, r1_o, r1_p, rec_k, mrr_v, t_top, best_rank = _expected_one(sim_scores, is_same, k_values)
        for k in k_values:
            recalls[k].append(r1_e if k == 1 else rec_k[k])
        mrr.append(mrr_v); opt.append(r1_o); pess.append(r1_p); tie_sizes.append(t_top); ranks.append(best_rank)
        if return_predictions:
            # for traceability; order is determined by stable argsort (display only)
            order_desc = np.argsort(-sim_scores, kind="stable")
            predictions.append(d_labels_local[order_desc][:max(k_values)].tolist())

    res = {**{f"recall@{k}": float(np.mean(recalls[k])) for k in k_values},
           "mrr": float(np.mean(mrr)),
           "recall@1_optimistic": float(np.mean(opt)),
           "recall@1_pessimistic": float(np.mean(pess)),
           "tie_size_mean": float(np.mean(tie_sizes))}
    # per-query arrays so the downstream nested bootstrap can resample on expected credit
    res["pq_recall@1"] = recalls[1] if 1 in k_values else None
    res["pq_recall@5"] = recalls[5] if 5 in k_values else None
    res["pq_mrr"] = mrr
    res["pq_tie_size"] = tie_sizes
    if return_predictions:
        res["predictions"] = predictions
        res["query_labels"] = q_labels.tolist()
    if return_ranks:
        res["ranks"] = ranks
    if blast_meta is not None:
        res.update(blast_meta)
    return res
