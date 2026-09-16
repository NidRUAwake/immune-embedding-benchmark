#!/usr/bin/env python3
"""Score new BCR models (antiberta2, igbert) + re-score blosum62/esm2-150m for a
self-consistency check, via the canonical run.py --offline-dir, into a fresh tree.
Then nested_bootstrap. BCR (IEDB) full ladder; SAbDab single-chain ladder."""
import sys, subprocess
from pathlib import Path
PY = sys.executable
OUT = Path("outputs/phase3_newmodels_tie_v2")
SEEDS = [42 + 10*i for i in range(20)]
MODELS = ["antiberta2", "igbert"]
SCEN = [
    {"name": "bcr", "flags": ["--include-bcr", "--bcr-file", "raw/iedb/bcr_singlechain_vh.tsv",
        "--top-labels", "10", "--min-label-count", "14",
        "--levels", "level1", "level2", "level2.5", "level3", "level4"], "human": True},
    {"name": "sabdab", "flags": ["--include-sabdab", "--sabdab-file",
        "raw/sabdab/sabdab_paired_clean_human_full.csv",
        "--top-labels", "5", "--min-label-count", "25",
        "--levels", "level1", "level3", "level4"], "human": True},
]
for s in SCEN:
    for model in MODELS:
        for seed in SEEDS:
            sd = OUT / s["name"] / model / f"seed_{seed}"
            if sd.exists() and list(sd.glob("*.csv")):
                continue
            cmd = [PY, "scripts/benchmark/run.py", *s["flags"], "--models", model,
                   "--random-state", str(seed), "--test-size", "0.2", "--clone-threshold", "0.95",
                   "--export-predictions", "--offline-dir", "outputs/embeddings", "--output-dir", str(sd)]
            if s["human"]:
                cmd.append("--human-only")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"ERR {s['name']} {model} seed{seed}: {r.stderr[-500:]}", flush=True)
        print(f"done {s['name']} {model}", flush=True)
print("BCR_SCORE_DRIVER_DONE", flush=True)
