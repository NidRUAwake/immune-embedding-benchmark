#!/usr/bin/env python3
"""
BCRdist on the CANONICAL benchmark slice (20 seeds) — corrected re-run.

Fixes the earlier scratch/run_bcrdist_20seed.py, which read raw IEDB/SAbDab files
WITHOUT build_pilot_slice (no alias-merge, no per-label cap, no dedup) and therefore
produced inflated, non-comparable numbers (plain Levenshtein hit 0.727 on that set).

Here the slice comes from the REAL build_pilot_slice at level3 / level3_paired, so the
CDR1/CDR2/CDR3 loops are carried THROUGH the canonical filtering and the train/test
membership is byte-identical to the BLOSUM62/ESM2 runs (post-is_legit-filter data:
BCR seed-42 slice 538/135, ~546/128 averaged over 20 seeds; SAbDab 251/59;
plain-Lev L1 reproduces canonical CDR3 R@1, ~0.440/0.421).

Kernel: the real pwseqdist nb_tcrdist fails to unbox under this env's Numba, so we use a
faithful re-implementation of the tcrdist single-CDR distance (mid-gap alignment, ntrim/
ctrim trim, tcrdist substitution costs, CDR3 weight 3). BCRdist is an ADAPTATION of the
TCRdist kernel to antibody CDRs, not a canonical established tool.

For honesty we compute BLOSUM62-L3 and Levenshtein-L3 on the IDENTICAL slices, so the
comparison is on the same queries (not against separately-reported nested means).
"""
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/analysis")
import numpy as np, pandas as pd
from pathlib import Path
from benchmark.data import (build_dataset_specs, load_table, filter_human_only,
                            init_label_config, build_pilot_slice, compute_shared_labels)
from benchmark.split import clone_aware_split_fast
from benchmark.evaluate import retrieval_metrics
import rapidfuzz, parasail

SEEDS = [42 + 10*i for i in range(20)]
init_label_config("label_aliases.json")

# ── faithful tcrdist single-CDR kernel ───────────────────────────────────
_SUBS = {('S','A'):3,('A','S'):3,('S','T'):3,('T','S'):3,('F','Y'):3,('Y','F'):3,
  ('K','R'):2,('R','K'):2,('I','L'):3,('L','I'):3,('I','V'):3,('V','I'):3,('L','V'):3,('V','L'):3,
  ('M','L'):3,('L','M'):3,('M','I'):3,('I','M'):3,('D','E'):2,('E','D'):2,('N','D'):3,('D','N'):3,
  ('Q','K'):2,('K','Q'):2,('Q','E'):2,('E','Q'):2,('N','S'):3,('S','N'):3,('Q','H'):3,('H','Q'):3,
  ('H','N'):3,('N','H'):3,('G','A'):3,('A','G'):3,('W','F'):3,('F','W'):3,('T','A'):3,('A','T'):3}

def _cdr_dist(a, b, gap=4, ntrim=3, ctrim=2):
    """tcrdist single-CDR: mid-gap align shorter to longer, trim ends, sum sub/gap costs."""
    a, b = str(a or ''), str(b or '')
    if a == b: return 0
    if len(a) == 0 or len(b) == 0: return gap * max(len(a), len(b))
    if len(a) > len(b): a, b = b, a            # a = shorter
    ng = len(b) - len(a)
    gp = len(a) // 2                            # insert gaps mid (tcrdist fixed_gappos≈middle)
    a = a[:gp] + '.'*ng + a[gp:]
    lo = ntrim; hi = len(a) - ctrim
    if hi <= lo: lo, hi = 0, len(a)
    d = 0
    for x, y in zip(a[lo:hi], b[lo:hi]):
        if x == y: continue
        d += gap if (x == '.' or y == '.') else _SUBS.get((x, y), 4)
    return d

def bcrdist_sim(q_cdrs, r_cdrs, paired):
    """q_cdrs/r_cdrs: list of dicts with cdr1,cdr2,cdr3 (+_l for paired). Returns -distance."""
    keys = [("cdr1",1,0,0),("cdr2",1,0,0),("cdr3",3,3,2)]
    if paired: keys += [("cdr1_l",1,0,0),("cdr2_l",1,0,0),("cdr3_l",3,3,2)]
    D = np.zeros((len(q_cdrs), len(r_cdrs)))
    for k, w, nt, ct in keys:
        for i, q in enumerate(q_cdrs):
            for j, r in enumerate(r_cdrs):
                D[i, j] += w * _cdr_dist(q.get(k, ''), r.get(k, ''), ntrim=nt, ctrim=ct)
    return -D

def _split_l3(seq, paired):
    """level3 sequence 'cdr1|cdr2|cdr3' (paired: 'h1|h2|h3:l1|l2|l3') -> dict."""
    s = str(seq or '')
    if paired:
        h, _, l = s.partition(':')
        hp = (h.split('|') + ['','',''])[:3]; lp = (l.split('|') + ['','',''])[:3]
        return {"cdr1":hp[0],"cdr2":hp[1],"cdr3":hp[2],"cdr1_l":lp[0],"cdr2_l":lp[1],"cdr3_l":lp[2]}
    p = (s.split('|') + ['','',''])[:3]
    return {"cdr1":p[0],"cdr2":p[1],"cdr3":p[2]}

def _align_sim(q, r, method):
    if method == "lev":
        return -rapidfuzz.distance.Levenshtein.normalized_distance(str(q), str(r))
    res = parasail.sw_striped_16(str(q), str(r), 10, 1, parasail.blosum62)
    return res.score / max(len(str(q)), len(str(r)), 1)

def _mat(qs, rs, method):
    return np.array([[_align_sim(q, r, method) for r in rs] for q in qs])

def run(name, spec, df, level, paired, want_ntr=None, want_nte=None):
    bcr_r, blo_r, lev_r = [], [], []
    for seed in SEEDS:
        sl = build_pilot_slice(df, spec, level, MAXCAP[name], seed, FL[name], None)
        pdf = sl.dataframe; sc = sl.sequence_col
        tr, te = clone_aware_split_fast(pdf, cdr3_col="split_sequence", label_col="label",
            test_size=0.2, clone_similarity_threshold=0.95, random_state=seed,
            min_clones_per_label=2, show_progress=False)
        trd, ted = pdf.iloc[tr], pdf.iloc[te]
        if seed == 42 and want_ntr:
            if len(trd) != want_ntr or len(te) != want_nte:
                print(f"  [slice@42] {name}: {len(trd)}/{len(te)}  (script default {want_ntr}/{want_nte})")
        qd = [_split_l3(s, paired) for s in ted[sc].values]
        rd = [_split_l3(s, paired) for s in trd[sc].values]
        kw = dict(q_labels=ted["label"].values, d_labels=trd["label"].values,
                  q_antigen_types=np.array(["protein"]*len(ted)), method="cosine")
        bcr_r.append(retrieval_metrics(q_emb=None,d_emb=None,precomputed_sim=bcrdist_sim(qd,rd,paired),**kw)["recall@1"])
        # apples-to-apples baselines on the SAME L3 (all-CDR) sequence string
        blo_r.append(retrieval_metrics(q_emb=None,d_emb=None,precomputed_sim=_mat(ted[sc].values,trd[sc].values,"blo"),**kw)["recall@1"])
        lev_r.append(retrieval_metrics(q_emb=None,d_emb=None,precomputed_sim=_mat(ted[sc].values,trd[sc].values,"lev"),**kw)["recall@1"])
    def ci(v):
        v=np.array(v); b=np.array([np.mean(np.random.choice(v,len(v),replace=True)) for _ in range(1000)])
        return np.mean(v), np.percentile(b,2.5), np.percentile(b,97.5)
    bm,bl,bu=ci(bcr_r); om,ol,ou=ci(blo_r); lm,ll,lu=ci(lev_r)
    print(f"\n=== {name} ({level}) ===")
    print(f"  BCRdist    : {bm:.3f} [{bl:.3f},{bu:.3f}]")
    print(f"  BLOSUM62-L3: {om:.3f} [{ol:.3f},{ou:.3f}]   (canonical-comparable baseline, same queries)")
    print(f"  Lev-L3     : {lm:.3f} [{ll:.3f},{lu:.3f}]")
    print(f"  BCRdist - BLOSUM62-L3 = {bm-om:+.3f}")
    return {"dataset":name,"level":level,"bcrdist_mean":bm,"bcrdist_lo":bl,"bcrdist_hi":bu,
            "blosumL3_mean":om,"levL3_mean":lm,"delta_vs_blosumL3":bm-om,"n_seeds":len(SEEDS)}

# ── datasets ──────────────────────────────────────────────────────────────
class A:
    include_bcr=True; include_tcr=False; include_sabdab=True
    bcr_file="raw/iedb/bcr_singlechain_vh.tsv"; bcr_label_col="Epitope_Source Molecule"
    sabdab_file="outputs/intermediate/sabdab_vj_annotated_paired.csv"; sabdab_label_col="label"
    tcr_file="x"; tcr_label_col="x"; human_only=True; local_files_only=False; offline_dir=None

specs={s.name:s for s in build_dataset_specs(A())}
dfs={}; FL={}; MAXCAP={"BCR":150,"SAbDab":100,"SAbDab_paired":100}
for nm,(topL,minC) in {"BCR":(10,14),"SAbDab":(5,25)}.items():
    spec=specs[nm]; d=filter_human_only(load_table(spec),spec); dfs[nm]=d
    FL[nm]=compute_shared_labels(d,spec,topL,minC,None)
FL["SAbDab_paired"]=FL["SAbDab"]

rows=[]
rows.append(run("BCR", specs["BCR"], dfs["BCR"], "level3", False, 538, 135))
rows.append(run("SAbDab", specs["SAbDab"], dfs["SAbDab"], "level3", False, 251, 59))
# SAbDab paired BCRdist (heavy+light) — level3_paired carries all 6 CDRs
FL_key="SAbDab";
r=run("SAbDab_paired", specs["SAbDab"], dfs["SAbDab"], "level3_paired", True, 251, 59)
rows.append(r)

# CDR3-level (L1) plain-Levenshtein sanity on the SAME canonical slices (note verification figure)
for nm in ["BCR", "SAbDab"]:
    r1 = []
    for seed in SEEDS:
        sl = build_pilot_slice(dfs[nm], specs[nm], "level1", MAXCAP[nm], seed, FL[nm], None)
        pdf = sl.dataframe; sc = sl.sequence_col
        tr, te = clone_aware_split_fast(pdf, cdr3_col="split_sequence", label_col="label",
            test_size=0.2, clone_similarity_threshold=0.95, random_state=seed,
            min_clones_per_label=2, show_progress=False)
        trd, ted = pdf.iloc[tr], pdf.iloc[te]
        kw = dict(q_labels=ted["label"].values, d_labels=trd["label"].values,
                  q_antigen_types=np.array(["protein"]*len(ted)), method="cosine")
        r1.append(retrieval_metrics(q_emb=None, d_emb=None,
                  precomputed_sim=_mat(ted[sc].values, trd[sc].values, "lev"), **kw)["recall@1"])
    print(f"  Lev-L1 sanity {nm}: {np.mean(r1):.4f}")

pd.DataFrame(rows).to_csv("outputs/reports/bcrdist_canonical_20seed.csv", index=False)
print("\nSaved → outputs/reports/bcrdist_canonical_20seed.csv")
