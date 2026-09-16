#!/usr/bin/env python3
"""Extract the VDJdb-paired working set: per receptor, beta (V,CDR3,J), alpha (V,CDR3,J),
and the pipeline's paired level1 key. Union across 20 seeds (top-10/min-14 paired config).
Writes paired input CSV for TCR2vec/CDR3vec paired embedding."""
import sys, os; sys.path.insert(0,"scripts")
import pandas as pd, numpy as np
from benchmark.data import (build_dataset_specs, load_table, filter_human_only, compute_shared_labels,
    normalize_labels, init_label_config, strip_tcr_cdr3_anchors, clean_sequence)
init_label_config("label_aliases.json")
SEEDS=[42+10*i for i in range(20)]; OUT=sys.argv[1]
class Args:
    include_bcr=False; include_sabdab=False; include_tcr=True
    tcr_label_col="antigen.epitope"; human_only=True; local_files_only=False; offline_dir=None
    bcr_file="raw/iedb/bcr_singlechain_vh.tsv"; bcr_label_col="x"
    sabdab_file="x"; sabdab_label_col="label"; tcr_file="raw/vdjdb/vdjdb_full.txt"
spec=build_dataset_specs(Args())[0]; df=load_table(spec); df=filter_human_only(df,spec)
top,minc,maxpl=10,14,100
forced=compute_shared_labels(df,spec,top,minc)
lab=normalize_labels(df[spec.label_col])
keyb=spec.sequence_builders["level1_paired"](df)   # "beta_cdr3:alpha_cdr3" stripped/cleaned
w=pd.DataFrame({"label":lab,"key":keyb})
for c in ["cdr3.beta","v.beta","j.beta","cdr3.alpha","v.alpha","j.alpha"]:
    w[c]=df[c] if c in df else None
# require both chains present (paired)
w=w.dropna(subset=["label","key","cdr3.beta","v.beta","j.beta","cdr3.alpha","v.alpha","j.alpha"])
w=w[w["label"].isin(forced)]
w=w[w["key"].str.contains(":", na=False)]   # both chains in key
keep=[]
for seed in SEEDS:
    for l in w["label"].unique():
        ld=w[w["label"]==l]
        keep.append(ld.sample(min(len(ld),maxpl),random_state=seed))
u=pd.concat(keep).drop_duplicates(subset=["key"])
# per-chain stripped CDR3 for keying model-input; keep raw V/J for reconstruction
u["CDR3B"]=u["cdr3.beta"].map(strip_tcr_cdr3_anchors).map(clean_sequence)
u["CDR3A"]=u["cdr3.alpha"].map(strip_tcr_cdr3_anchors).map(clean_sequence)
out=u[["key","CDR3B","v.beta","j.beta","CDR3A","v.alpha","j.alpha"]].rename(
    columns={"v.beta":"VB","j.beta":"JB","v.alpha":"VA","j.alpha":"JA"})
out.to_csv(OUT,index=False)
print(f"VDJdb-paired working set: {len(out)} unique paired receptors -> {OUT}")
print("Vbeta ok %.2f Valpha ok %.2f"%(out['VB'].notna().mean(), out['VA'].notna().mean()))
