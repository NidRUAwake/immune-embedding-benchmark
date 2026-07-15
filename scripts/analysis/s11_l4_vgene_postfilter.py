#!/usr/bin/env python3
"""S11 L4 V-gene block, POST-FILTER, canonical expected-R@1.
Computes, on the V-assignable subset of BCR L4 test queries (20 seeds):
  - per-method R@1 (Lev/BLOSUM/ESM2-150M), full candidate pool;
  - two-stage oracle (filter candidates to same V-gene family, rank by BLOSUM62);
  - V-gene-only analytical baseline (uniform random among same-V-family candidates).
Uses build_pilot_slice (is_legit filter) + retrieval_metrics (order-independent expected R@1)
+ vj_filter='v' for the oracle, so everything matches the manuscript's metric.
V-gene family = subgroup (split on '-'/'*'), matching the same-V rates already in Table S11."""
from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from benchmark.data import (build_dataset_specs, load_table, filter_human_only,
    init_label_config, compute_shared_labels, build_pilot_slice)
from benchmark.split import clone_aware_split_fast
from benchmark.embeddings import get_embedder, OfflineEmbedder
from benchmark.evaluate import retrieval_metrics

SEEDS = [42 + 10*i for i in range(20)]

def vfam(g):
    if not isinstance(g, str) or not g.strip() or g.lower() in {"nan","none"}: return None
    return g.split("-")[0].split("*")[0]

def main():
    init_label_config("label_aliases.json")
    class M:
        include_bcr=True; include_tcr=False; include_sabdab=False
        bcr_file="raw/iedb/bcr_singlechain_vh.tsv"; bcr_label_col="Epitope_Source Molecule"
        human_only=True; top_labels=10; min_label_count=14; max_per_label_bcr=150
        shared_labels=True; batch_size=32; local_files_only=False; offline_dir="outputs/embeddings"; pooling="mean"
    m=M(); spec=build_dataset_specs(m)[0]; df=load_table(spec); df=filter_human_only(df,spec)
    fl=compute_shared_labels(df,spec,10,14,None)
    esm=OfflineEmbedder("outputs/embeddings/bcr_level4_esm2-150m_mean.npy")
    bl=get_embedder("blosum62",m); lev=get_embedder("levenshtein",m)
    rows={k:[] for k in ["levenshtein","blosum62","esm2-150m","oracle","vbaseline"]}
    for seed in SEEDS:
        sl=build_pilot_slice(df,spec,"level4",150,seed,fl); fr=sl.dataframe; sc=sl.sequence_col
        tr,te=clone_aware_split_fast(fr,cdr3_col="split_sequence",label_col="label",test_size=0.2,
            clone_similarity_threshold=0.95,random_state=seed,min_clones_per_label=2,show_progress=False)
        trd=fr.iloc[tr].reset_index(drop=True); ted=fr.iloc[te].reset_index(drop=True)
        tv=ted["v_gene"].map(vfam).to_numpy(); dv=trd["v_gene"].map(vfam).to_numpy()
        qok=np.array([v is not None for v in tv])
        if qok.sum()==0: continue
        tlab=ted["label"].to_numpy(); dlab=trd["label"].to_numpy(); tatg=ted["antigen_type"].to_numpy()
        qseq=ted[sc].to_numpy(); dseq=trd[sc].to_numpy()
        # encode
        enc={"esm2-150m":(esm.encode(qseq.tolist()),esm.encode(dseq.tolist())),
             "blosum62":(bl.encode(qseq.tolist()),bl.encode(dseq.tolist())),
             "levenshtein":(lev.encode(qseq.tolist()),lev.encode(dseq.tolist()))}
        idx=np.where(qok)[0]
        # per-method R@1 on V-assignable test, full pool
        for meth,(qe,de) in enc.items():
            mm=retrieval_metrics(q_emb=qe[idx],d_emb=de,q_labels=tlab[idx],d_labels=dlab,
                                 q_antigen_types=tatg[idx],method=meth)
            rows[meth].append(mm["recall@1"])
        # oracle: same-V filter + BLOSUM rank
        qe,de=enc["blosum62"]
        mo=retrieval_metrics(q_emb=qe[idx],d_emb=de,q_labels=tlab[idx],d_labels=dlab,
                             q_antigen_types=tatg[idx],method="blosum62",vj_filter="v",
                             q_v_gene=tv[idx],d_v_gene=dv)
        rows["oracle"].append(mo["recall@1"])
        # V-only baseline: expected R@1 = mean over V-assignable queries of c_sameV/n_sameV
        bvals=[]
        for i in idx:
            same=(dv==tv[i])
            n=int(same.sum())
            if n==0: bvals.append(0.0); continue
            c=int((dlab[same]==tlab[i]).sum())
            bvals.append(c/n)
        rows["vbaseline"].append(float(np.mean(bvals)))
    print("=== S11 L4 V-assignable subset, post-filter, expected-R@1 (20-seed mean [95% CI]) ===")
    for k in ["levenshtein","blosum62","esm2-150m","oracle","vbaseline"]:
        a=np.array(rows[k]); ci=1.96*a.std(ddof=1)/np.sqrt(len(a))
        print(f"  {k:12s} {a.mean():.3f} [{a.mean()-ci:.3f}, {a.mean()+ci:.3f}]  (n={len(a)})")

if __name__=="__main__":
    main()
