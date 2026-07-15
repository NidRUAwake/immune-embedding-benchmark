#!/usr/bin/env python3
"""figS3 data, POST-FILTER, canonical expected-R@1: BCR L1 Recall@1 stratified by CDR3 length
(10-12 / 13-15 / >=16 aa) for BLOSUM62, ESM2-150M, ESM2-650M, ESM2-3B.
build_pilot_slice (is_legit) + clone split + retrieval_metrics on each length-bin query subset
against the full train pool. 20 seeds, mean +/- 1.96 SE per (method, bin)."""
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
BINS=[("<=12",lambda L:L<=12),("13-15",lambda L:(L>=13)&(L<=15)),(">=16",lambda L:L>=16)]
MODELS=["blosum62","esm2-150m","esm2-650m","esm2-3b"]

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
         "esm2-150m":OfflineEmbedder("outputs/embeddings/bcr_level1_esm2-150m_mean.npy"),
         "esm2-650m":OfflineEmbedder("outputs/embeddings/bcr_level1_esm2-650m_mean.npy"),
         "esm2-3b":OfflineEmbedder("outputs/embeddings/bcr_level1_esm2-3b_mean.npy")}
    acc={(mo,b):[] for mo in MODELS for b,_ in BINS}
    ncnt={b:[] for b,_ in BINS}
    for seed in SEEDS:
        sl=build_pilot_slice(df,spec,"level1",150,seed,fl); fr=sl.dataframe; sc=sl.sequence_col
        tr,te=clone_aware_split_fast(fr,cdr3_col="split_sequence",label_col="label",test_size=0.2,
            clone_similarity_threshold=0.95,random_state=seed,min_clones_per_label=2,show_progress=False)
        trd=fr.iloc[tr].reset_index(drop=True); ted=fr.iloc[te].reset_index(drop=True)
        qseq=ted[sc].to_numpy(); dseq=trd[sc].to_numpy()
        L=np.array([len(s) for s in qseq])
        tlab=ted["label"].to_numpy(); dlab=trd["label"].to_numpy(); tatg=ted["antigen_type"].to_numpy()
        E={mo:(emb[mo].encode(qseq.tolist()),emb[mo].encode(dseq.tolist())) for mo in MODELS}
        for b,cond in BINS:
            idx=np.where(cond(L))[0]
            if len(idx)==0: continue
            ncnt[b].append(len(idx))
            for mo in MODELS:
                qe,de=E[mo]
                mm=retrieval_metrics(q_emb=qe[idx],d_emb=de,q_labels=tlab[idx],d_labels=dlab,
                                     q_antigen_types=tatg[idx],method=mo)
                acc[(mo,b)].append(mm["recall@1"])
    print("=== figS3 post-filter: BCR L1 R@1 by CDR3 length (20-seed mean [95% CI]) ===")
    print(f"avg n/seed per bin: "+", ".join(f"{b}={np.mean(ncnt[b]):.0f}" for b,_ in BINS))
    for mo in MODELS:
        cells=[]
        for b,_ in BINS:
            a=np.array(acc[(mo,b)]); ci=1.96*a.std(ddof=1)/np.sqrt(len(a))
            cells.append(f"{a.mean():.3f}+-{ci:.3f}")
        print(f"  {mo:11s} "+" | ".join(cells))

if __name__=="__main__":
    main()
