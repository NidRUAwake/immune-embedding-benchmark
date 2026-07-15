#!/usr/bin/env python3
"""Experiment (2): real-TCR germline-shortcut signature.
On REAL VDJdb TCR CDR3 (L1) retrieval (real CDR3, real deposited TRBV), measure how often the
top-1 retrieved candidate shares the query's TRBV gene -- among ALL top-1 and among CORRECT top-1 --
versus the random expectation. This mirrors the BCR same-V analysis (Table S10) on native TCR data,
so the germline-shortcut signature does not depend on the germline-reconstructed TCR L4.
"""
import os, sys
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
# portable: scripts/ dir relative to this file (scratch/ -> repo -> scripts)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import numpy as np, pandas as pd
from benchmark.data import build_dataset_specs, load_table, filter_human_only, compute_shared_labels, init_label_config, build_pilot_slice
from benchmark.split import clone_aware_split_fast
from benchmark.evaluate import compute_similarity_matrix
from benchmark.embeddings import OfflineEmbedder

class Args:
    include_tcr=True; tcr_file="raw/vdjdb/vdjdb_full.txt"
    include_bcr=False; include_sabdab=False
    top_labels=15; min_label_count=50; human_only=True
    test_size=0.2; clone_threshold=0.95; min_clones_per_label=2
    max_per_label_tcr=100; label_config="label_aliases.json"; shared_labels=True
args=Args(); init_label_config(args.label_config)

specs=build_dataset_specs(args); spec=[s for s in specs if s.name=="TCR"][0]
df=filter_human_only(load_table(spec), spec)
import json
forced=json.load(open("scratch/vdjdb_paired_external_task/manifest.json"))["labels"]  # canonical 10
print("labels:", forced)

esm=OfflineEmbedder("outputs/embeddings/tcr_level1_esm2-150m_mean.npy")
fam=lambda g: g.split("-")[0] if isinstance(g,str) else None   # TRBV7-2 -> TRBV7 (family)
SEEDS=[42,52,62,72,82,92,102,112,122,132,142,152,162,172,182,192,202,212,222,232]

def topk_stats(sim, q_v, d_v, q_lab, d_lab):
    """top-1 retrieval; return (R@1, sameV_all, sameV_correct, n_correct) using gene-level V."""
    top=np.argmax(sim, axis=1)
    correct=(d_lab[top]==q_lab)
    qv=np.array([fam(x) for x in q_v], dtype=object); dv=np.array([fam(x) for x in d_v], dtype=object)
    valid=np.array([ (a is not None) and (b is not None) for a,b in zip(qv, dv[top]) ])
    sameV=np.array([ a==b for a,b in zip(qv, dv[top]) ]) & valid
    r1=correct.mean()
    sameV_all=sameV[valid].mean() if valid.any() else np.nan
    cm=correct & valid
    sameV_corr=sameV[cm].mean() if cm.any() else np.nan
    return r1, sameV_all, sameV_corr, int(correct.sum())

rows=[]
for s in SEEDS:
    sl=build_pilot_slice(df, spec, "level1", args.max_per_label_tcr, s, forced)
    if sl is None: print(f"seed {s}: no slice"); continue
    d=sl.dataframe.reset_index(drop=True)
    seqcol=sl.sequence_col
    tr,te=clone_aware_split_fast(d, cdr3_col="split_sequence", label_col="label",
            test_size=args.test_size, clone_similarity_threshold=args.clone_threshold,
            random_state=s, min_clones_per_label=args.min_clones_per_label, show_progress=False)
    trd, ted = d.loc[tr], d.loc[te]
    q_lab=ted["label"].values; d_lab=trd["label"].values
    q_v=ted["v_gene"].values; d_v=trd["v_gene"].values
    # random same-family expectation: per query, fraction of train pool sharing query's V family
    dvf=np.array([fam(x) for x in d_v], dtype=object)
    rand=[]
    for qf in [fam(x) for x in q_v]:
        if qf is None: continue
        rand.append(np.mean(dvf==qf))
    rand_sameV=np.mean(rand) if rand else np.nan
    # BLOSUM62 (alignment on CDR3 strings)
    simB=compute_similarity_matrix(ted[seqcol].values.astype(object), trd[seqcol].values.astype(object), "blosum62")
    rB=topk_stats(simB, q_v, d_v, q_lab, d_lab)
    # ESM2-150M (cosine on cached L1 embeddings)
    qe=esm.encode(ted[seqcol].values); de=esm.encode(trd[seqcol].values)
    simE=compute_similarity_matrix(qe, de, "esm2-150m")
    rE=topk_stats(simE, q_v, d_v, q_lab, d_lab)
    rows.append(dict(seed=s, n_test=len(te), rand_sameV=rand_sameV,
                     bl_r1=rB[0], bl_sameV_all=rB[1], bl_sameV_corr=rB[2],
                     esm_r1=rE[0], esm_sameV_all=rE[1], esm_sameV_corr=rE[2]))
    print(f"seed {s}: nte={len(te)} rand_sameV={rand_sameV:.3f} | BL R@1={rB[0]:.3f} sameV(corr)={rB[2]:.3f} | ESM R@1={rE[0]:.3f} sameV(corr)={rE[2]:.3f}")

R=pd.DataFrame(rows); R.to_csv("outputs/reports/tcr_germline_signature.csv", index=False)
print("\n=== 20-seed means (TCR L1, V-family level) ===")
for c in ["rand_sameV","bl_r1","bl_sameV_all","bl_sameV_corr","esm_r1","esm_sameV_all","esm_sameV_corr"]:
    print(f"  {c:16s}: {R[c].mean():.3f} ± {R[c].std():.3f}")
print("\nsaved -> outputs/reports/tcr_germline_signature.csv")
