import os
import tempfile
import subprocess
import time
from collections import defaultdict

import numpy as np


def _run_blastp(queries, references, blast_path="blastp"):
    """Run makeblastdb + blastp and return (best_ref_scores_per_query, query_time, db_size_bytes).

    best_ref_scores_per_query[i] is a dict {ref_index: best_bitscore} for query i.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Write references to FASTA
        ref_fasta = os.path.join(tmpdir, "db.fasta")
        with open(ref_fasta, "w") as f:
            for i, seq in enumerate(references):
                f.write(f">ref|{i}\n{seq}\n")

        # 2. Make BLAST DB
        db_out = os.path.join(tmpdir, "db")
        subprocess.run(
            ["makeblastdb", "-in", ref_fasta, "-dbtype", "prot", "-out", db_out],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        # Calculate size for sanity
        db_size_bytes = 0
        for f in os.listdir(tmpdir):
            if f.startswith("db."):
                db_size_bytes += os.path.getsize(os.path.join(tmpdir, f))

        # 3. Process queries
        query_fasta = os.path.join(tmpdir, "queries.fasta")
        with open(query_fasta, "w") as f:
            for i, seq in enumerate(queries):
                f.write(f">query|{i}\n{seq}\n")

        out_tsv = os.path.join(tmpdir, "results.tsv")

        # 4. Run BLAST
        n_threads = max(1, os.cpu_count() - 1)
        start_time = time.time()
        subprocess.run(
            [
                blast_path,
                "-query", query_fasta,
                "-db", db_out,
                "-out", out_tsv,
                "-outfmt", "6 qseqid sseqid bitscore",
                "-task", "blastp-short",
                "-num_threads", str(n_threads),
                "-max_target_seqs", "25",
                "-evalue", "1000",
                "-word_size", "2",
                "-comp_based_stats", "F",
            ],
            check=True
        )
        query_time = time.time() - start_time

        # 5. Parse output -> best bitscore per (query, reference)
        query_hits = defaultdict(list)
        if os.path.exists(out_tsv):
            with open(out_tsv, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        q_id = int(parts[0].split("|")[1])
                        s_id = int(parts[1].split("|")[1])
                        score = float(parts[2])
                        query_hits[q_id].append((s_id, score))

        best_ref_scores_per_query = []
        for i in range(len(queries)):
            best_ref_scores = {}
            for s_id, score in query_hits.get(i, []):
                if s_id not in best_ref_scores or score > best_ref_scores[s_id]:
                    best_ref_scores[s_id] = score
            best_ref_scores_per_query.append(best_ref_scores)

    return best_ref_scores_per_query, query_time, db_size_bytes


def blast_retrieval_scores(queries, references, blast_path="blastp", missing_score=-1e9):
    """Run blastp and return (sim_matrix, query_time, db_size_bytes).

    sim_matrix is [n_queries x n_references] of best per-reference bitscores. References that
    blastp did not report for a given query are set to `missing_score` so they sort strictly
    last and collapse into a single bottom tie-group. This lets the caller apply the SAME
    order-independent expected-R@1 tie resolution (c_top/t_top over the top tie-group) used for
    every other method, instead of an order-dependent best-rank tie-break.
    """
    n_q, n_r = len(queries), len(references)
    best_ref_scores_per_query, query_time, db_size_bytes = _run_blastp(
        queries, references, blast_path=blast_path)
    sim = np.full((n_q, n_r), float(missing_score), dtype=np.float64)
    for i, best_ref_scores in enumerate(best_ref_scores_per_query):
        for ref_idx, score in best_ref_scores.items():
            sim[i, ref_idx] = score
    return sim, query_time, db_size_bytes


def blast_retrieval_ranks(queries, references, q_labels, d_labels, blast_path="blastp"):
    """Legacy best-case-rank retrieval (order-dependent tie-break). Retained for reference;
    the canonical evaluator now uses blast_retrieval_scores + expected-R@1 instead.
    Returns a list of 0-indexed best ranks.
    """
    best_ref_scores_per_query, query_time, db_size_bytes = _run_blastp(
        queries, references, blast_path=blast_path)
    ranks = []
    for i in range(len(queries)):
        sorted_refs = sorted(best_ref_scores_per_query[i].items(), key=lambda x: x[1], reverse=True)
        true_label = q_labels[i]
        rank = len(d_labels)
        for pos, (ref_idx, score) in enumerate(sorted_refs):
            if d_labels[ref_idx] == true_label:
                rank = pos
                break
        ranks.append(rank)
    return ranks, query_time, db_size_bytes
