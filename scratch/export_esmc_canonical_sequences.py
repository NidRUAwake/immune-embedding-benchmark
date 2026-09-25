#!/usr/bin/env python3
"""Export the EXACT canonical-slice sequences (union over all 20 seeds, deduped) for
ESM-C re-extraction, so the returned embeddings give 100% offline-hit coverage on the
canonical benchmark. One CSV per (dataset, level): seq_idx,sequence.

Datasets/levels mirror the main table: BCR, VDJdb-TCR, SAbDab, McPAS x {L1, L4}.
"""
import sys; sys.path.insert(0, 'scripts')
import json
from types import SimpleNamespace
from pathlib import Path
import pandas as pd
from benchmark.data import build_dataset_specs, build_pilot_slice, compute_shared_labels
from benchmark.run import load_table  # reuse exact loader
from benchmark.data import filter_human_only

SEEDS = [42 + 10 * i for i in range(20)]
OUT = Path("scratch/external_task_esmc_canonical"); OUT.mkdir(parents=True, exist_ok=True)

# (key, args-namespace, human_only, top, min, cap, levels, expected_anchor)
CFG = [
    ("bcr",    dict(include_bcr=True,  bcr_file="raw/iedb/bcr_singlechain_vh.tsv"),                 True,  10, 14, 150, ("level1","level4"), "~556/120"),
    ("tcr",    dict(include_tcr=True,  tcr_file="raw/vdjdb/vdjdb_full.txt"),                        True,  15, 50, 100, ("level1","level4"), "~801/199"),
    ("sabdab", dict(include_sabdab=True, sabdab_file="outputs/intermediate/sabdab_vj_annotated_paired.csv"), True, 5, 25, 100, ("level1","level4"), "~251/59"),
    ("mcpas",  dict(include_tcr=True,  tcr_file="outputs/intermediate/mcpas_standardized.tsv"),     False, 10, 30, 100, ("level1","level4"), "~801/199"),
]

def cap_for(spec, top_cap_bcr, top_cap_tcr, top_cap_sab):
    return {"BCR": top_cap_bcr, "TCR": top_cap_tcr}.get(spec.name, top_cap_sab)

manifest = {}
for key, akw, human, top, mincount, cap, levels, anchor in CFG:
    base = dict(include_bcr=False, include_tcr=False, include_sabdab=False)
    base.update(akw)
    args = SimpleNamespace(**base)
    spec = build_dataset_specs(args)[0]
    df = load_table(spec)
    if human:
        df = filter_human_only(df, spec)
    fl = compute_shared_labels(df, spec, top_labels=top, min_label_count=mincount)
    print(f"\n[{key}] spec={spec.name} labels={len(fl)} : {sorted(fl)}")
    for level in levels:
        seqs = set()
        n_tr = n_te = None
        for seed in SEEDS:
            sl = build_pilot_slice(df, spec, level, max_per_label=cap, random_state=seed, forced_labels=fl)
            if sl is None:
                continue
            col = sl.dataframe[sl.sequence_col].dropna().astype(str)
            col = col[col.str.len() > 0]
            seqs |= set(col.tolist())
            # anchor from level1 split sizes (train/test) — approximate via label split share
        uniq = sorted(seqs)
        out_csv = OUT / f"{key}_{level}_sequences.csv"
        pd.DataFrame({"seq_idx": range(len(uniq)), "sequence": uniq}).to_csv(out_csv, index=False)
        manifest[f"{key}_{level}"] = {"n_unique_sequences": len(uniq),
                                      "labels": len(fl),
                                      "anchor_train_test": anchor,
                                      "csv": out_csv.name}
        print(f"  {key} {level}: {len(uniq)} unique sequences -> {out_csv.name}")

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("\nmanifest:", json.dumps(manifest, indent=2))
