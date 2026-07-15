#!/usr/bin/env python3
"""20-seed VDJdb paired clonotype benchmark (L2_paired = V+J both chains; L2.5_paired = V both chains).
Two-stage retrieval: filter reference candidates to those sharing the query's paired germline
genes, then rank by the paired CDR3 (reuses L1_paired offline embeddings). tcrdist3 is
level-independent and not part of the clonotype variants. Mirrors run_vdjdb_paired_v2.py flags.
Output: outputs/phase3_paired_clonotype/tcr/<model>/seed_<s>/pilot_results.csv
"""
import os, sys, json, subprocess
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

with open("scratch/vdjdb_paired_external_task/manifest.json") as f:
    FIXED_LABELS = json.load(f)["labels"]

SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 232]
MODELS = ["levenshtein", "blosum62", "esm2-150m", "esm2-650m", "esm2-3b"]
OUT = Path("outputs/phase3_paired_clonotype/tcr")
labels_str = ",".join(FIXED_LABELS)

total = len(SEEDS) * len(MODELS); done = 0; failed = []
for si, seed in enumerate(SEEDS, 1):
    for model in MODELS:
        odir = OUT / model / f"seed_{seed}"
        odir.mkdir(parents=True, exist_ok=True)
        rf = odir / "pilot_results.csv"
        if rf.exists() and rf.stat().st_size > 300:
            done += 1; print(f"  [{si:2}/20] seed {seed} {model:12} cached"); continue
        cmd = [sys.executable, "scripts/benchmark/run.py", "--include-tcr",
               "--tcr-file", "raw/vdjdb/vdjdb_full.txt", "--models", model,
               "--random-state", str(seed), "--output-dir", str(odir),
               "--top-labels", "10", "--min-label-count", "14", "--forced-labels", labels_str,
               "--human-only", "--test-size", "0.2", "--split-strategy", "clone-aware",
               "--clone-threshold", "0.95", "--export-predictions",
               "--levels", "level2_paired", "level2.5_paired"]
        if model.startswith("esm2-"):
            cmd += ["--offline-dir", "outputs/embeddings"]
        print(f"  [{si:2}/20] seed {seed} {model:12}", end="", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        if r.returncode == 0 and rf.exists() and rf.stat().st_size > 300:
            print(" OK"); done += 1
        else:
            print(f" FAIL ({r.returncode})"); failed.append(f"{seed}/{model}")
            if r.stderr: print("   ", r.stderr.strip()[-300:])

print(f"\nDone {done}/{total}; failed {len(failed)}: {failed[:8]}")
