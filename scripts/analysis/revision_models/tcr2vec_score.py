#!/usr/bin/env python3
"""Score TCR2vec (full-length + CDR3-only input) vs alignment/ESM2 on the canonical
clone-aware TCR retrieval task, at CDR3 and under the V-gene clonotype filter.
Germline-shortcut test: does TCR2vec-full's CDR3-level edge over TCR2vec-cdr3 (and over
alignment) shrink once retrieval is restricted to same-V candidates?
Runs in the MAIN env. Reads offline .npy from outputs/embeddings.
"""
import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from benchmark.data import (build_dataset_specs, load_table, filter_human_only, compute_shared_labels,
                            normalize_labels, assign_antigen_type, init_label_config,
                            strip_tcr_cdr3_anchors, clean_sequence)
from benchmark.split import clone_aware_split_fast
from benchmark.evaluate import retrieval_metrics, compute_similarity_matrix
SEEDS = [42 + 10*i for i in range(20)]; TEST=0.2; CT=0.95; MINCL=2
EMB="outputs/embeddings"

class Args:
    include_bcr=False; include_sabdab=False; include_tcr=True
    tcr_label_col="antigen.epitope"; local_files_only=False; offline_dir=None
    bcr_file="raw/iedb/bcr_singlechain_vh.tsv"; bcr_label_col="x"
    sabdab_file="raw/sabdab/sabdab_paired_clean_human_full.csv"; sabdab_label_col="label"
    def __init__(s,f,h): s.tcr_file=f; s.human_only=h

def load_offline(path):
    data=np.load(path); meta=json.load(open(path.replace(".npy","_metadata.json")))
    idx={}
    for i,sq in enumerate(meta["sequence_order"]):
        for k in (sq, clean_sequence(str(sq)), clean_sequence(strip_tcr_cdr3_anchors(str(sq)))):
            if k and k not in idx: idx[k]=i
    def enc(seqs):
        out=np.zeros((len(seqs),data.shape[1]),data.dtype); f=0
        for i,sq in enumerate(seqs):
            j=idx.get(sq)
            if j is None: j=idx.get(clean_sequence(str(sq)))
            if j is None: j=idx.get(clean_sequence(strip_tcr_cdr3_anchors(str(sq))))
            if j is not None: out[i]=data[j]; f+=1
        return out, f/len(seqs)
    return enc

def gene(x):
    if not isinstance(x,str): return None
    return x.split("*")[0].strip()

def slice_df(df, spec, maxpl, seed, forced):
    lab=normalize_labels(df[spec.label_col]); seq=spec.sequence_builders["level1"](df)
    w=pd.DataFrame({"label":lab,"cdr3":seq})
    for c in ["v.beta","j.beta"]: w[c]=df[c] if c in df else None
    w=w.dropna(subset=["label","cdr3"]); w=w[w["label"].isin(forced)]
    w["antigen_type"]=assign_antigen_type(w["label"])
    parts=[]
    for l in w["label"].unique():
        ld=w[w["label"]==l]
        parts.append(ld.sample(min(len(ld),maxpl),random_state=seed))
    return pd.concat(parts,ignore_index=True).sample(frac=1.0,random_state=seed).reset_index(drop=True)

def run(name, tcr_file, human, top, minc, maxpl):
    spec=build_dataset_specs(Args(tcr_file,human))[0]; df=load_table(spec)
    if human: df=filter_human_only(df,spec)
    forced=compute_shared_labels(df,spec,top,minc)
    encs={m:load_offline(f"{EMB}/{name.lower()}_level1_{m}_mean.npy")
          for m in ["esm2-150m","tcr2vec-full","tcr2vec-cdr3","cdr3vec","tcr2vec-small","cdr3vec-small","tcr2vec-tcrdb"]}
    rows=[]
    for seed in SEEDS:
        sl=slice_df(df,spec,maxpl,seed,forced)
        tri,tei=clone_aware_split_fast(sl,cdr3_col="cdr3",label_col="label",test_size=TEST,
                 clone_similarity_threshold=CT,random_state=seed,min_clones_per_label=MINCL,show_progress=False)
        tr,te=sl.iloc[tri],sl.iloc[tei]
        q,d=te["cdr3"].values,tr["cdr3"].values
        ql,dl=te["label"].values,tr["label"].values; qt=te["antigen_type"].values
        qv=np.array([gene(x) for x in te["v.beta"].values]); dv=np.array([gene(x) for x in tr["v.beta"].values])
        r={"seed":seed,"n_test":len(te)}
        # alignment
        for meth in ["blosum62","levenshtein"]:
            sim=compute_similarity_matrix(q,d,meth)
            r[f"{meth}|cdr3"]=retrieval_metrics(None,None,ql,dl,qt,meth,precomputed_sim=sim)["recall@1"]
            r[f"{meth}|V"]=retrieval_metrics(None,None,ql,dl,qt,meth,precomputed_sim=sim,
                            vj_filter="v",q_v_gene=qv,d_v_gene=dv)["recall@1"]
        # embedding methods
        for m in ["esm2-150m","tcr2vec-full","tcr2vec-cdr3","cdr3vec","tcr2vec-small","cdr3vec-small","tcr2vec-tcrdb"]:
            qe,_=encs[m](q); de,cov=encs[m](d)
            sim=compute_similarity_matrix(qe,de,m,"cosine")
            r[f"{m}|cdr3"]=retrieval_metrics(None,None,ql,dl,qt,m,precomputed_sim=sim)["recall@1"]
            r[f"{m}|V"]=retrieval_metrics(None,None,ql,dl,qt,m,precomputed_sim=sim,
                          vj_filter="v",q_v_gene=qv,d_v_gene=dv)["recall@1"]
            r[f"{m}|cov"]=round(cov,3)
        rows.append(r)
    R=pd.DataFrame(rows)
    print(f"\n===== {name} (20-seed clone-aware) n_test~{round(R.n_test.mean())} =====")
    print(f"{'method':16s} {'CDR3':>8s} {'V-filter':>9s}")
    for m in ["blosum62","levenshtein","esm2-150m","tcr2vec-full","tcr2vec-cdr3","cdr3vec","tcr2vec-small","cdr3vec-small","tcr2vec-tcrdb"]:
        cov = f" cov={R[m+'|cov'].mean():.2f}" if m+"|cov" in R else ""
        print(f"{m:16s} {R[m+'|cdr3'].mean():8.3f} {R[m+'|V'].mean():9.3f}{cov}")
    R.to_csv(f"outputs/reports/tcr2vec_{name.lower()}_20seed.csv",index=False)
    return R

if __name__=="__main__":
    init_label_config("label_aliases.json")
    os.makedirs("outputs/reports",exist_ok=True)
    run("McPAS","outputs/intermediate/mcpas_standardized.tsv",False,10,30,100)
    run("TCR","raw/vdjdb/vdjdb_full.txt",True,15,50,100)
    print("\nTCR2VEC_SCORE_DONE")
