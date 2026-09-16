#!/usr/bin/env python3
"""Build a zero-shot BCR-native task from CoV-AbDab: SARS-CoV-2 neutralization
discrimination (neutralizer vs binder-only) among antibodies that bind SARS-CoV-2.
Outputs a table with full VH, CDRH3, Heavy V gene, and a binary label.
Clean, real negatives (binder-but-non-neutralizing), avoiding constructed-negative bias."""
import os
import pandas as pd, re, sys
CSV=os.environ.get("COVABDAB_CSV","data/covabdab.csv")
OUT=sys.argv[1]
d=pd.read_csv(CSV, low_memory=False)
d=d[d["Ab or Nb"].astype(str).str.startswith("Ab")]                 # antibodies only
AA=set("ACDEFGHIKLMNPQRSTVWY")
def clean(x):
    x=re.sub(r"[^A-Za-z]","",str(x).upper()); return x if x and set(x)<=AA else None
d["VH"]=d["VHorVHH"].map(clean)
d["CDRH3"]=d["CDRH3"].map(clean)
d["Vgene"]=d["Heavy V Gene"].astype(str).str.split("(").str[0].str.strip().str.split("*").str[0]
binds=d["Binds to"].astype(str).str.contains("SARS-CoV2", na=False)
neut =d["Neutralising Vs"].astype(str).str.contains("SARS-CoV2", na=False)
d=d[binds & d["VH"].notna() & d["CDRH3"].notna()]
d["label"]=neut[d.index].astype(int)                                # 1=neutralizer, 0=binder-only
d["len"]=d["VH"].str.len()
d=d[(d["VH"].str.len().between(90,160)) & (d["CDRH3"].str.len().between(5,35))]
# drop exact-VH duplicates
d=d.drop_duplicates(subset=["VH"])
keep=d[["Name","VH","CDRH3","Vgene","label"]].reset_index(drop=True)
keep.to_csv(OUT, index=False)
print(f"CoV-AbDab SARS-CoV-2 neutralization task: {len(keep)} unique antibodies")
print(f"  neutralizers {keep.label.sum()} ({keep.label.mean():.2f}), binder-only {(1-keep.label).sum()}")
print(f"  V genes: {keep.Vgene.nunique()} distinct; top: {list(keep.Vgene.value_counts().head(3).index)}")
