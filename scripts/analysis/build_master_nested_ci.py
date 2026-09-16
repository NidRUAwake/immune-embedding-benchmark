#!/usr/bin/env python3
"""Build the CANONICAL master nested-CI table (the source for Table 1 / Table S1 main grid).

CRITICAL: this is a TWO-TREE aggregation, because the strict-20AA `is_legit` filter only
changes BCR/McPAS (TCR/SAbDab delete 0 records, so the filter is a no-op there):
  - BCR + McPAS  -> post-filter tree  outputs/phase3_filtered_tie_v2/
  - VDJdb + SAbDab -> outputs/phase3_grand_slam_tie_v2/  (filter is a no-op)
Running `nested_bootstrap.py` with its DEFAULT `--base-dir outputs/phase3_grand_slam_tie_v2`
would aggregate BCR/McPAS from the STALE pre-filter tree (BCR L1 esm2 0.373 instead of 0.357)
-> does NOT match the manuscript. This wrapper does the correct two-tree assembly.

Output: outputs/reports/master_nested_ci_20seeds_postfilter.csv (matches manuscript Table 1/S1).
Reproducible: seeded (`--bootstrap-seed 42`). Run with OFFLINE_EMBED_STRICT=1 upstream."""
from __future__ import annotations
import subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
NB = ROOT / "scripts/analysis/nested_bootstrap.py"
TMP_POST = ROOT / "outputs/reports/_tmp_master_bcr_mcpas_post.csv"
TMP_TS = ROOT / "outputs/reports/_tmp_master_tcr_sabdab.csv"
OUT = ROOT / "outputs/reports/master_nested_ci_20seeds_postfilter.csv"

def run(base, datasets, out):
    subprocess.run([PY, str(NB), "--base-dir", base, "--datasets", datasets,
                    "--bootstrap-seed", "42", "--output-nested", str(out),
                    "--output-comparison", f"{tempfile.gettempdir()}/_nb_cmp.csv"], check=True, cwd=ROOT)

def main():
    run("outputs/phase3_filtered_tie_v2", "bcr,mcpas", TMP_POST)       # post-filter BCR/McPAS
    run("outputs/phase3_grand_slam_tie_v2", "tcr,sabdab", TMP_TS)      # VDJdb/SAbDab (filter no-op)
    post = pd.read_csv(TMP_POST); ts = pd.read_csv(TMP_TS)
    master = pd.concat([post[post.dataset.isin(["bcr", "mcpas"])],
                        ts[ts.dataset.isin(["tcr", "sabdab"])]], ignore_index=True)
    master.to_csv(OUT, index=False)
    TMP_POST.unlink(missing_ok=True); TMP_TS.unlink(missing_ok=True)
    print(f"Saved canonical master -> {OUT}  ({len(master)} rows)")

if __name__ == "__main__":
    main()
