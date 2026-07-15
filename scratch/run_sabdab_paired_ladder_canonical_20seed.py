#!/usr/bin/env python3
"""
SAbDab paired-chain ladder (L1/L2/L2.5/L3) on the CANONICAL slice, 20 seeds — corrected.

Replaces scratch/run_sabdab_paired_ladder_20seed.py, which read the FULL annotated file
(1019 rows, 39% dup CDR3, SARS-CoV-2 = 69%, no per-label cap) and so produced inflated,
non-comparable numbers. Here the slice comes from build_pilot_slice with the canonical
SAbDab args (top-5, min-25, max-per-label 100), so it is byte-identical to the SAbDab
single-chain canonical run (verified: 251/59 at seed 42, plain-Lev L1 = 0.5669).

ESM2 paired embeddings are looked up by sequence string from the full-bank .npy
(level1_paired / level3_paired); the canonical-filtered sequences are a subset of the
full bank, so lookup is exact.
"""
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/analysis")
import numpy as np, pandas as pd
from benchmark.data import (build_dataset_specs, load_table, filter_human_only,
                            init_label_config, build_pilot_slice, compute_shared_labels)
from benchmark.split import clone_aware_split_fast
from benchmark.evaluate import retrieval_metrics
from nested_bootstrap import nested_bootstrap_metrics
import rapidfuzz, parasail

SEEDS = [42 + 10*i for i in range(20)]
init_label_config("label_aliases.json")
ANNOT = "outputs/intermediate/sabdab_vj_annotated_paired.csv"
EMB = "outputs/embeddings"
TASKDIR = "scratch/sabdab_paired_external_task"

class A:
    include_bcr=False; include_tcr=False; include_sabdab=True
    sabdab_file=ANNOT; sabdab_label_col="label"
    bcr_file="x"; bcr_label_col="x"; tcr_file="x"; tcr_label_col="x"
    human_only=True; local_files_only=False; offline_dir=None

spec=[s for s in build_dataset_specs(A()) if s.name=="SAbDab"][0]
df=filter_human_only(load_table(spec), spec)
FL=compute_shared_labels(df, spec, 5, 25, None)
print(f"SAbDab labels={FL}, rows={len(df)}")

# ── ESM2 banks: seq-string -> embedding (from full-bank unique-seq CSV + npy) ──
def load_bank(level, model):
    csv=f"{TASKDIR}/sabdab_{level}_fullbank_unique_sequences.csv"
    npy=f"{EMB}/sabdab_{level}_{model}_mean.npy"
    seqs=pd.read_csv(csv)["sequence"].tolist(); arr=np.load(npy)
    assert len(seqs)==arr.shape[0], f"{level}/{model}: {len(seqs)} seq vs {arr.shape[0]} emb"
    return {s:arr[i] for i,s in enumerate(seqs)}
BANK={m:{"level1_paired":load_bank("level1_paired",m),"level3_paired":load_bank("level3_paired",m)}
      for m in ["esm2-150m","esm2-650m","esm2-3b"]}

def align_mat(qs, rs, method):
    if method=="lev":
        return np.array([[-rapidfuzz.distance.Levenshtein.normalized_distance(str(q),str(r)) for r in rs] for q in qs])
    return np.array([[parasail.sw_striped_16(str(q),str(r),10,1,parasail.blosum62).score/max(len(str(q)),len(str(r)),1) for r in rs] for q in qs])

LEVELS=["level1_paired","level2_paired","level2.5_paired","level3_paired"]
METHODS=["levenshtein","blosum62","esm2-150m","esm2-650m","esm2-3b"]
# seed_data[level][method] = list of per-seed dicts {hits1,hits5,rr} for query-weighted nested CI
seed_data={lv:{mth:[] for mth in METHODS} for lv in LEVELS}

def eval_one(sl, vj, mth, emb_lv):
    pdf=sl.dataframe; sc=sl.sequence_col
    tr,te=clone_aware_split_fast(pdf, cdr3_col="split_sequence", label_col="label",
        test_size=0.2, clone_similarity_threshold=0.95, random_state=seed,
        min_clones_per_label=2, show_progress=False)
    trd,ted=pdf.iloc[tr], pdf.iloc[te]
    kw=dict(q_labels=ted["label"].values, d_labels=trd["label"].values,
            q_antigen_types=np.array(["protein"]*len(ted)), method="cosine",
            vj_filter=vj, q_v_gene=ted["v_gene"].values, q_j_gene=ted["j_gene"].values,
            d_v_gene=trd["v_gene"].values, d_j_gene=trd["j_gene"].values)
    if mth in ("levenshtein","blosum62"):
        sim=align_mat(ted[sc].values, trd[sc].values, "lev" if mth=="levenshtein" else "blo")
    else:
        bank=BANK[mth][emb_lv]; dim=next(iter(bank.values())).shape
        qe=np.array([bank.get(s, np.zeros(dim)) for s in ted[sc].values])
        de=np.array([bank.get(s, np.zeros(dim)) for s in trd[sc].values])
        qn=qe/(np.linalg.norm(qe,axis=1,keepdims=True)+1e-9); dn=de/(np.linalg.norm(de,axis=1,keepdims=True)+1e-9)
        sim=qn@dn.T
    m=retrieval_metrics(q_emb=None,d_emb=None,precomputed_sim=sim,**kw)
    return len(te), {"hits1":np.array(m["pq_recall@1"]),"hits5":np.array(m["pq_recall@5"]),"rr":np.array(m["pq_mrr"])}

# ── VALIDATION: SAbDab single-chain L1 BLOSUM via same nested agg must ≈ canonical 0.362 ──
val=[]
for seed in SEEDS:
    sl=build_pilot_slice(df, spec, "level1", 100, seed, FL, None)
    _,sd=eval_one(sl,None,"blosum62","level1_paired"); val.append(sd)
vstat=nested_bootstrap_metrics(val,1000)["recall@1"]
print(f"\n[VALIDATION] SAbDab single L1 BLOSUM nested = {vstat['mean']:.3f} (canonical query-weighted = 0.362)")

for level in LEVELS:
    emb_lv = "level3_paired" if level=="level3_paired" else "level1_paired"
    vj = "vj" if level=="level2_paired" else ("v" if level=="level2.5_paired" else None)
    for seed in SEEDS:
        sl=build_pilot_slice(df, spec, level, 100, seed, FL, None)
        if seed==42 and level=="level1_paired":
            tr,te=clone_aware_split_fast(sl.dataframe, cdr3_col="split_sequence", label_col="label",
                test_size=0.2, clone_similarity_threshold=0.95, random_state=42, min_clones_per_label=2, show_progress=False)
            assert len(te)==59 and len(tr)==251, f"slice mismatch {len(tr)}/{len(te)}"
        for mth in METHODS:
            _,sd=eval_one(sl,vj,mth,emb_lv); seed_data[level][mth].append(sd)

rows=[]
print("\n===== SAbDab paired ladder (canonical, query-weighted nested 20-seed [95% CI]) =====")
for level in LEVELS:
    print(f"\n{level}:")
    for mth in METHODS:
        st=nested_bootstrap_metrics(seed_data[level][mth],1000)["recall@1"]
        rows.append({"level":level,"method":mth,"mean":st["mean"],"lo":st["lower"],"hi":st["upper"]})
        print(f"  {mth:12s} {st['mean']:.3f} [{st['lower']:.3f},{st['upper']:.3f}]")
pd.DataFrame(rows).to_csv("outputs/reports/sabdab_paired_ladder_canonical_20seed.csv", index=False)
print("\nSaved → outputs/reports/sabdab_paired_ladder_canonical_20seed.csv")
