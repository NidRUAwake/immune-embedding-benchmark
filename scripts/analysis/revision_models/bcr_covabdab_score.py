#!/usr/bin/env python3
"""Zero-shot SARS-CoV-2 neutralization discrimination on CoV-AbDab (BCR-native task).
Clone-aware split on CDRH3 (95%), nearest-neighbor label agreement (R@1) + kNN AUROC,
for alignment (BLOSUM62, Levenshtein) and PLM (ESM2-150m, AntiBERTa2, IgBert),
at CDRH3 and full VH. Mirrors the main-paper clone-aware / expected-R@1 protocol."""
import os, sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,"scripts")
from benchmark.split import clone_aware_split_fast
from benchmark.evaluate import retrieval_metrics, compute_similarity_matrix
from sklearn.metrics import roc_auc_score
SP=os.environ.get("REVISION_WORKDIR","revision_workdir")
EMB=f"{SP}/covabdab_emb"; SEEDS=[42+10*i for i in range(20)]
w=pd.read_csv(f"{SP}/covabdab_work.csv")
w["atype"]="protein"
def load(model,lvl):
    d=np.load(f"{EMB}/covabdab_{lvl}_{model}_mean.npy")
    so=json.load(open(f"{EMB}/covabdab_{lvl}_{model}_mean_metadata.json"))["sequence_order"]
    idx={s:i for i,s in enumerate(so)}
    return lambda seqs: np.vstack([d[idx[s]] for s in seqs])
PLM=["esm2-150m","antiberta2","igbert"]
encs={(m,lvl):load(m,lvl) for m in PLM for lvl in ["cdrh3","vh"]}
def auroc(sim,ql,dl,k=5):
    # score each query by neutralizer-fraction among its top-k refs
    sc=[]
    for i in range(sim.shape[0]):
        top=np.argsort(-sim[i])[:k]; sc.append(dl[top].mean())
    try: return roc_auc_score(ql,sc)
    except Exception: return np.nan
rows=[]
for seed in SEEDS:
    tri,tei=clone_aware_split_fast(w,cdr3_col="CDRH3",label_col="label",test_size=0.2,
             clone_similarity_threshold=0.95,random_state=seed,min_clones_per_label=2,show_progress=False)
    tr,te=w.iloc[tri],w.iloc[tei]
    ql,dl=te["label"].values,tr["label"].values; qt=te["atype"].values
    r={"seed":seed,"n_test":len(te),"pos":round(ql.mean(),2)}
    for lvl,col in [("cdrh3","CDRH3"),("vh","VH")]:
        q,d=te[col].values,tr[col].values
        for meth in ["blosum62","levenshtein"]:
            sim=compute_similarity_matrix(q,d,meth)
            r[f"{meth}|{lvl}|r1"]=retrieval_metrics(None,None,ql,dl,qt,meth,precomputed_sim=sim)["recall@1"]
            r[f"{meth}|{lvl}|auc"]=auroc(sim,ql,dl)
        for m in PLM:
            qe,de=encs[(m,lvl)](q),encs[(m,lvl)](d)
            sim=compute_similarity_matrix(qe,de,m,"cosine")
            r[f"{m}|{lvl}|r1"]=retrieval_metrics(None,None,ql,dl,qt,m,precomputed_sim=sim)["recall@1"]
            r[f"{m}|{lvl}|auc"]=auroc(sim,ql,dl)
    rows.append(r)
R=pd.DataFrame(rows)
print(f"\n===== CoV-AbDab SARS-CoV-2 neutralization (20-seed clone-aware, n_test~{round(R.n_test.mean())}, pos~{R.pos.mean():.2f}) =====")
print(f"{'method':14s} {'CDRH3 R@1':>10s} {'CDRH3 AUC':>10s} {'VH R@1':>8s} {'VH AUC':>8s}")
for m in ["blosum62","levenshtein","esm2-150m","antiberta2","igbert"]:
    print(f"{m:14s} {R[m+'|cdrh3|r1'].mean():10.3f} {R[m+'|cdrh3|auc'].mean():10.3f} {R[m+'|vh|r1'].mean():8.3f} {R[m+'|vh|auc'].mean():8.3f}")
R.to_csv("outputs/reports/covabdab_neutralization_20seed.csv",index=False)
print("COVABDAB_SCORE_DONE")
