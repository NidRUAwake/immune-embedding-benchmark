#!/usr/bin/env python3
"""Full 20-seed L2 (V+J gene filter) and L2.5 (V-only filter) retrieval rerun, all datasets.
Uses the fixed two-stage clonotype retrieval (gene-level filter + CDR3 ranking, L1 embeddings
reused offline -> no GPU). Writes to a SEPARATE tree so canonical L1/L3/L4 results are untouched.
Per-dataset configs mirror the canonical grand_slam runs.
"""
import sys, subprocess
from pathlib import Path

OUT = Path("outputs/phase3_l2_vjfilter")
SEEDS = [42 + i * 10 for i in range(20)]   # 42..232
COMMON = ["--test-size", "0.2", "--clone-threshold", "0.95", "--export-predictions",
          "--offline-dir", "outputs/embeddings", "--levels", "level2", "level2.5"]

SCENARIOS = [
    {"name": "bcr",
     "flags": ["--include-bcr", "--bcr-file", "raw/iedb/bcr_singlechain_vh.tsv",
               "--top-labels", "10", "--min-label-count", "14", "--human-only"],
     "models": ["levenshtein", "blosum62", "esm2-150m", "esm2-650m", "esm2-3b", "antiberty", "ablang"]},
    {"name": "tcr",
     "flags": ["--include-tcr", "--tcr-file", "raw/vdjdb/vdjdb_full.txt",
               "--top-labels", "15", "--min-label-count", "50", "--human-only"],
     "models": ["levenshtein", "blosum62", "esm2-150m", "esm2-650m", "esm2-3b", "tcr-bert"]},
    {"name": "mcpas",
     "flags": ["--include-tcr", "--tcr-file", "outputs/intermediate/mcpas_standardized.tsv",
               "--top-labels", "5", "--min-label-count", "30", "--max-per-label-tcr", "100"],
     "models": ["levenshtein", "blosum62", "esm2-150m", "esm2-650m", "esm2-3b", "tcr-bert"]},
    # SAbDab ships no V/J calls; we infer them with ANARCI (human, IMGT germline) via
    # scratch/annotate_sabdab_vj.py -> heavy_v_gene/heavy_j_gene, making L2/L2.5 assessable.
    {"name": "sabdab",
     "flags": ["--include-sabdab", "--sabdab-file", "outputs/intermediate/sabdab_vj_annotated.csv",
               "--human-only"],
     "models": ["levenshtein", "blosum62", "esm2-150m", "esm2-650m", "esm2-3b", "antiberty", "ablang"]},
]

py = sys.executable
for s in SCENARIOS:
    for seed in SEEDS:
        od = OUT / s["name"] / f"seed_{seed}"
        if (od / "pilot_results.csv").exists():
            print(f"skip {s['name']} seed{seed} (done)", flush=True)
            continue
        od.mkdir(parents=True, exist_ok=True)
        cmd = [py, "scripts/benchmark/run.py", *s["flags"], "--models", *s["models"],
               "--random-state", str(seed), *COMMON, "--output-dir", str(od)]
        print(f"RUN {s['name']} seed{seed}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ERR {s['name']} seed{seed}:\n{r.stderr[-1200:]}", flush=True)
print("ALLDONE", flush=True)
