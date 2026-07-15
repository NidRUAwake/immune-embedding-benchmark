#!/usr/bin/env python3
"""CANONICAL reproduction harness (post-`is_legit`-filter, expected-R@1).

Self-contained: reproduces the BCR alignment-method numbers (Levenshtein, BLOSUM62) at L1 and L4
end-to-end from the canonical input, using the SAME canonical pipeline as the manuscript
(`build_pilot_slice` -> strict-20AA `is_legit` filter -> `clone_aware_split_fast` ->
order-independent expected-R@1 `retrieval_metrics`). Alignment methods need NO precomputed
embeddings (parasail only), so this validates the core pipeline + the headline
"alignment beats PLM at CDR3" numbers without the large embedding cache.

Checks (PASS/FAIL):
  1. clone-aware split is deterministic and has ZERO within-label clone leakage (the paper's
     central methodological claim).
  2. 20-seed per-seed-mean expected-R@1 for Lev/BLOSUM at BCR L1 & L4 matches the committed
     canonical values within tolerance.

NOT covered here (needs the embedding cache + raw data): PLM (ESM2/AntiBERTy/AbLang) rows,
the nested-bootstrap CIs, and the full Table 1. For those, obtain embeddings + run
`scripts/analysis/build_master_nested_ci.py` (two-tree) with OFFLINE_EMBED_STRICT=1.
See discussions/368_external_code_review_guide.md.

Usage:  python reproduction/run_reproduction_canonical.py [--seeds N]
"""
from __future__ import annotations
import os, sys, argparse, csv
from pathlib import Path
os.environ.setdefault("OFFLINE_EMBED_STRICT", "1")          # fail hard on any offline cache miss
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from benchmark.data import (build_dataset_specs, load_table, filter_human_only,
    init_label_config, compute_shared_labels, build_pilot_slice)
from benchmark.split import clone_aware_split_fast
from benchmark.embeddings import get_embedder
from benchmark.evaluate import retrieval_metrics

# canonical input: prefer the repo's raw/, else the bundled copy in reproduction/data/
BCR = REPO / "raw/iedb/bcr_singlechain_vh.tsv"
if not BCR.exists():
    BCR = Path(__file__).resolve().parent / "data" / "bcr_singlechain_vh.tsv"
EXPECTED = Path(__file__).resolve().parent / "expected" / "expected_bcr_canonical.csv"
TOL = 0.003   # per-seed-mean is deterministic; tolerance covers only float noise

class M:
    include_bcr=True; include_tcr=False; include_sabdab=False
    bcr_file=str(BCR); bcr_label_col="Epitope_Source Molecule"
    human_only=True; top_labels=10; min_label_count=14; max_per_label_bcr=150
    shared_labels=True; batch_size=32; local_files_only=False; offline_dir=None; pooling="mean"

def nlev(a,b):
    if len(a)<len(b): a,b=b,a
    prev=list(range(len(b)+1))
    for i,c1 in enumerate(a):
        cur=[i+1]
        for j,c2 in enumerate(b): cur.append(min(prev[j+1]+1,cur[j]+1,prev[j]+(c1!=c2)))
        prev=cur
    return 1-prev[-1]/max(len(a),len(b))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="number of canonical seeds (default 20)")
    args=ap.parse_args()
    if not BCR.exists():
        sys.exit(f"FAIL: canonical input not found. Put bcr_singlechain_vh.tsv at {BCR} "
                 f"(see REPRODUCIBILITY.md Data Availability / download_scripts).")
    # Resolve label_aliases.json relative to the repo/package root, NOT the CWD: running from
    # inside reproduction/ would otherwise silently skip label aliasing (load_label_config returns
    # empty on a missing path) and produce a wrong slice + wrong numbers. Fail hard instead.
    alias_path = REPO / "label_aliases.json"
    if not alias_path.exists():
        sys.exit(f"FAIL: {alias_path} not found — label aliasing would be silently skipped, "
                 f"yielding wrong numbers. Run from the package root or restore label_aliases.json.")
    init_label_config(str(alias_path))
    spec=build_dataset_specs(M())[0]; df=load_table(spec); df=filter_human_only(df,spec)
    fl=compute_shared_labels(df,spec,10,14,None)
    SEEDS=[42+10*i for i in range(args.seeds)]
    embs={m:get_embedder(m,M()) for m in ["levenshtein","blosum62"]}

    # ---- Check 1: split determinism + zero within-label clone leak (seed 42, L1) ----
    sl=build_pilot_slice(df,spec,"level1",150,42,fl); fr=sl.dataframe; sc=sl.sequence_col
    tr,te=clone_aware_split_fast(fr,cdr3_col="split_sequence",label_col="label",test_size=0.2,
        clone_similarity_threshold=0.95,random_state=42,min_clones_per_label=2,show_progress=False)
    tr2,te2=clone_aware_split_fast(fr,cdr3_col="split_sequence",label_col="label",test_size=0.2,
        clone_similarity_threshold=0.95,random_state=42,min_clones_per_label=2,show_progress=False)
    det = np.array_equal(tr,tr2) and np.array_equal(te,te2)
    trd=fr.iloc[tr]; ted=fr.iloc[te]
    within_leak=sum(1 for _,q in ted.iterrows()
                    if any(nlev(q.split_sequence,d)>=0.95
                           for d in trd[trd.label==q.label]['split_sequence'].values))
    print(f"[Check 1] split deterministic={det}; within-label clone leak={within_leak}/{len(ted)} "
          f"(expect True / 0)")

    # ---- Check 2: 20-seed per-seed-mean expected-R@1 vs committed canonical ----
    exp={}
    for r in csv.DictReader(l for l in open(EXPECTED) if not l.startswith("#")):
        exp[(r["model"],r["level"])]=float(r["recall1_perseed_mean"])
    # The committed expected values are 20-seed means; only assert numeric equality at the
    # canonical 20 seeds. With fewer seeds this is an informational smoke run (the per-seed-mean
    # legitimately differs from the 20-seed mean), so Check 2 reports but does not fail.
    strict = (len(SEEDS) == 20)
    print(f"\n[Check 2] {len(SEEDS)}-seed per-seed-mean expected-R@1 (canonical pipeline)"
          + ("" if strict else "  [informational — strict check needs --seeds 20]") + ":")
    print(f"  {'model':12s}{'level':8s}{'computed':>9s}{'expected(20s)':>14s}{'Δ':>8s}  status")
    fails=0
    for m in ["levenshtein","blosum62"]:
        for lv in ["level1","level4"]:
            per=[]
            for s in SEEDS:
                slc=build_pilot_slice(df,spec,lv,150,s,fl); f=slc.dataframe; c=slc.sequence_col
                a,b=clone_aware_split_fast(f,cdr3_col="split_sequence",label_col="label",test_size=0.2,
                    clone_similarity_threshold=0.95,random_state=s,min_clones_per_label=2,show_progress=False)
                qd=f.iloc[b]; dd=f.iloc[a]
                mm=retrieval_metrics(q_emb=embs[m].encode(qd[c].tolist()),d_emb=embs[m].encode(dd[c].tolist()),
                    q_labels=qd["label"].values,d_labels=dd["label"].values,
                    q_antigen_types=qd["antigen_type"].values,method=m)
                per.append(mm["recall@1"])
            got=float(np.mean(per)); want=exp.get((m,lv),float("nan"))
            ok = abs(got-want)<=TOL
            status = ("OK" if ok else "FAIL") if strict else f"{got-want:+.3f} (n={len(SEEDS)})"
            if strict and not ok: fails+=1
            print(f"  {m:12s}{lv:8s}{got:9.4f}{want:14.4f}{got-want:+8.4f}  {status}")
    ok_all = det and within_leak==0 and (fails==0 if strict else True)
    if ok_all:
        print("\nPASS — pipeline deterministic, zero within-label clone leak"
              + (", alignment R@1 reproduced (20-seed)." if strict
                 else f". (Ran {len(SEEDS)} seeds; re-run with --seeds 20 for the strict numeric check.)"))
    else:
        print(f"\nFAIL — det={det} leak={within_leak} metric_fails={fails}")
    sys.exit(0 if ok_all else 1)

if __name__=="__main__":
    main()
