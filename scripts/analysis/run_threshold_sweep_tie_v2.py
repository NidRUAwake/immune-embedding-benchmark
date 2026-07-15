#!/usr/bin/env python3
"""S6 clone-threshold sensitivity (BCR), CANONICAL expected-R@1, post-filter.
Reproduces outputs/reports/threshold_sweep_tie_v2.csv (the source for Supplementary Table S6
and Supplementary Figure S4). For each clone-similarity threshold the BCR L1/L4 slice is built
via build_pilot_slice (is_legit filter) and split with clone_aware_split_fast at that threshold;
R@1 is the order-independent expected-R@1 (retrieval_metrics), NOT single-pick ranks==0.
Per-seed mean +/- SD across the 20 canonical seeds. (Earlier this CSV was produced inline; this
committed script closes that reproducibility gap. Run with OFFLINE_EMBED_STRICT=1.)"""
from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from benchmark.data import (build_dataset_specs, load_table, filter_human_only,
    init_label_config, compute_shared_labels, build_pilot_slice)
from benchmark.split import clone_aware_split_fast
from benchmark.embeddings import get_embedder, OfflineEmbedder
from benchmark.evaluate import retrieval_metrics

SEEDS=[42+10*i for i in range(20)]
THRESHOLDS=[0.85,0.90,0.93,0.95,0.97,0.99]
MODELS=["blosum62","esm2-150m"]
LEVELS=["level1","level4"]

def main():
    init_label_config("label_aliases.json")
    class M:
        include_bcr=True; include_tcr=False; include_sabdab=False
        bcr_file="raw/iedb/bcr_singlechain_vh.tsv"; bcr_label_col="Epitope_Source Molecule"
        human_only=True; top_labels=10; min_label_count=14; max_per_label_bcr=150
        shared_labels=True; batch_size=32; local_files_only=False; offline_dir="outputs/embeddings"; pooling="mean"
    m=M(); spec=build_dataset_specs(m)[0]; df=load_table(spec); df=filter_human_only(df,spec)
    fl=compute_shared_labels(df,spec,10,14,None)
    emb={"blosum62":get_embedder("blosum62",m),
         ("esm2-150m","level1"):OfflineEmbedder("outputs/embeddings/bcr_level1_esm2-150m_mean.npy"),
         ("esm2-150m","level4"):OfflineEmbedder("outputs/embeddings/bcr_level4_esm2-150m_mean.npy")}
    rows=[]
    for thr in THRESHOLDS:
        for lv in LEVELS:
            per={mo:[] for mo in MODELS}
            for s in SEEDS:
                sl=build_pilot_slice(df,spec,lv,150,s,fl); fr=sl.dataframe; sc=sl.sequence_col
                tr,te=clone_aware_split_fast(fr,cdr3_col="split_sequence",label_col="label",test_size=0.2,
                    clone_similarity_threshold=thr,random_state=s,min_clones_per_label=2,show_progress=False)
                trd=fr.iloc[tr].reset_index(drop=True); ted=fr.iloc[te].reset_index(drop=True)
                tl=ted["label"].to_numpy(); dl=trd["label"].to_numpy(); ta=ted["antigen_type"].to_numpy()
                qs=ted[sc].to_numpy(); ds=trd[sc].to_numpy()
                for mo in MODELS:
                    e=emb["blosum62"] if mo=="blosum62" else emb[(mo,lv)]
                    mm=retrieval_metrics(q_emb=e.encode(qs.tolist()),d_emb=e.encode(ds.tolist()),
                                         q_labels=tl,d_labels=dl,q_antigen_types=ta,method=mo)
                    per[mo].append(mm["recall@1"])
            for mo in MODELS:
                a=np.array(per[mo])
                rows.append(dict(threshold=thr,model=mo,level=lv,recall1=round(a.mean(),4),
                                 sd=round(a.std(ddof=1),4),n_seeds=len(a)))
            print(f"thr {thr} {lv}: "+", ".join(f"{mo} {np.mean(per[mo]):.3f}" for mo in MODELS))
    out=Path("outputs/reports/threshold_sweep_tie_v2.csv")
    pd.DataFrame(rows).to_csv(out,index=False)
    print(f"Saved {out}")

if __name__=="__main__":
    main()
