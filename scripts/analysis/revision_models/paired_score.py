#!/usr/bin/env python3
"""Score paired VDJdb retrieval: TCR2vec(full) and CDR3vec paired (concatenated beta+alpha),
vs alignment, ESM2-150M, SCEPTR paired. Same clone-aware protocol as the paired ladder.
Offline paired embeddings keyed by the paired 'beta_cdr3:alpha_cdr3' key. Main env."""
import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,"scripts")
from benchmark.data import (build_dataset_specs, load_table, filter_human_only, compute_shared_labels,
    normalize_labels, assign_antigen_type, init_label_config, clean_sequence)
from benchmark.split import clone_aware_split_fast
from benchmark.evaluate import retrieval_metrics, compute_similarity_matrix
SEEDS=[42+10*i for i in range(20)]; EMB="outputs/embeddings"
class Args:
    include_bcr=False; include_sabdab=False; include_tcr=True
    tcr_label_col="antigen.epitope"; human_only=True; local_files_only=False; offline_dir=None
    bcr_file="x"; bcr_label_col="x"; sabdab_file="x"; sabdab_label_col="label"; tcr_file="raw/vdjdb/vdjdb_full.txt"
def load_offline(path):
    data=np.load(path); meta=json.load(open(path.replace(".npy","_metadata.json")))
    idx={s:i for i,s in enumerate(meta["sequence_order"])}
    cl={clean_sequence(str(s)):i for i,s in enumerate(meta["sequence_order"])}
    def enc(seqs):
        out=np.zeros((len(seqs),data.shape[1]),data.dtype); f=0
        for i,s in enumerate(seqs):
            j=idx.get(s); j=cl.get(clean_sequence(str(s))) if j is None else j
            if j is not None: out[i]=data[j]; f+=1
        return out, f/len(seqs)
    return enc
init_label_config("label_aliases.json")
spec=build_dataset_specs(Args())[0]; df=load_table(spec); df=filter_human_only(df,spec)
forced=compute_shared_labels(df,spec,10,14)
lab=normalize_labels(df[spec.label_col]); key=spec.sequence_builders["level1_paired"](df)
sb=spec.sequence_builders["level1"](df)   # beta cdr3 (stripped) for clone split
w=pd.DataFrame({"label":lab,"key":key,"betacdr3":sb})
w=w.dropna(subset=["label","key","betacdr3"]); w=w[w["label"].isin(forced)]
w=w[w["key"].str.contains(":",na=False)]; w["antigen_type"]=assign_antigen_type(w["label"])
paired_models=["tcr2vec-full","cdr3vec","esm2-150m","sceptr"]
encs={m:load_offline(f"{EMB}/tcr_level1_paired_{m}_mean.npy") for m in paired_models}
# restrict to the subset covered by the reconstruction-dependent TCR2vec paired embedding
# (records with complete V/J on both chains), so every method is scored on the SAME subset.
_covkeys=set(json.load(open(f"{EMB}/tcr_level1_paired_tcr2vec-full_mean_metadata.json"))["sequence_order"])
_before=len(w); w=w[w["key"].isin(_covkeys)]
print(f"restricted to gene-complete paired subset: {len(w)}/{_before} working records")
rows=[]
for seed in SEEDS:
    parts=[w[w["label"]==l].sample(min(len(w[w["label"]==l]),100),random_state=seed) for l in w["label"].unique()]
    sl=pd.concat(parts,ignore_index=True).sample(frac=1.0,random_state=seed).reset_index(drop=True)
    tri,tei=clone_aware_split_fast(sl,cdr3_col="betacdr3",label_col="label",test_size=0.2,
             clone_similarity_threshold=0.95,random_state=seed,min_clones_per_label=2,show_progress=False)
    tr,te=sl.iloc[tri],sl.iloc[tei]
    qk,dk=te["key"].values,tr["key"].values; ql,dl=te["label"].values,tr["label"].values; qt=te["antigen_type"].values
    r={"seed":seed,"n_test":len(te)}
    for meth in ["blosum62","levenshtein"]:
        sim=compute_similarity_matrix(qk,dk,meth)   # align on the paired 'beta:alpha' string
        r[meth]=retrieval_metrics(None,None,ql,dl,qt,meth,precomputed_sim=sim)["recall@1"]
    for m in paired_models:
        qe,_=encs[m](qk); de,cov=encs[m](dk)
        sim=compute_similarity_matrix(qe,de,m,"cosine")
        r[m]=retrieval_metrics(None,None,ql,dl,qt,m,precomputed_sim=sim)["recall@1"]; r[m+"_cov"]=round(cov,3)
    rows.append(r)
R=pd.DataFrame(rows)
print(f"\n===== VDJdb PAIRED (20-seed clone-aware) n_test~{round(R.n_test.mean())} =====")
for m in ["blosum62","levenshtein","esm2-150m","sceptr","tcr2vec-full","cdr3vec"]:
    cov=f" cov={R[m+'_cov'].mean():.2f}" if m+"_cov" in R else ""
    print(f"  {m:14s} {R[m].mean():.3f}{cov}")
R.to_csv("outputs/reports/tcr2vec_paired_vdjdb_20seed.csv",index=False)
print("PAIRED_SCORE_DONE")
