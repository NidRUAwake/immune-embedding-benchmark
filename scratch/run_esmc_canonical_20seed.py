#!/usr/bin/env python3
"""Canonical ESM-C-300m retrieval (mean pooling) on the 4 databases x {L1,L4}, 20 seeds,
using the re-extracted 100%-coverage offline embeddings (doc 351). Output tree mirrors
grand_slam so nested_bootstrap can aggregate it. CPU only."""
import subprocess, sys
from pathlib import Path
PY=sys.executable
SEEDS=[42+10*i for i in range(20)]
OUT="outputs/esmc_canonical"   # mean pooling tree
COMMON=["--models","esmc-300m","--pooling","mean","--test-size","0.2","--clone-threshold","0.95",
        "--export-predictions","--offline-dir","outputs/embeddings"]
DS=[
 ("bcr",   ["--include-bcr","--bcr-file","raw/iedb/bcr_singlechain_vh.tsv","--top-labels","10","--min-label-count","14","--human-only"]),
 ("tcr",   ["--include-tcr","--tcr-file","raw/vdjdb/vdjdb_full.txt","--top-labels","15","--min-label-count","50","--human-only"]),
 ("sabdab",["--include-sabdab","--sabdab-file","outputs/intermediate/sabdab_vj_annotated_paired.csv","--top-labels","5","--min-label-count","25","--human-only"]),
 ("mcpas", ["--include-tcr","--tcr-file","outputs/intermediate/mcpas_standardized.tsv","--top-labels","10","--min-label-count","30"]),
]
bad=[]
for ds,flags in DS:
    for seed in SEEDS:
        d=Path(OUT)/ds/"esmc-300m"/f"seed_{seed}"
        if d.exists() and list(d.glob("*_predictions.csv")):
            continue
        d.mkdir(parents=True,exist_ok=True)
        cmd=[PY,"scripts/benchmark/run.py",*flags,"--levels","level1","level4",
             "--random-state",str(seed),*COMMON,"--output-dir",str(d)]
        r=subprocess.run(cmd,capture_output=True,text=True)
        # enforce 100% offline hit
        if "NOT found" in (r.stderr+r.stdout):
            bad.append((ds,seed,"offline<100%"))
        if r.returncode!=0:
            bad.append((ds,seed,r.stderr[-300:]))
    print(f"[{ds}] done")
print("BAD:",bad if bad else "none")
