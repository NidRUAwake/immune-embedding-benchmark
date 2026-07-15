#!/usr/bin/env python3
"""Validate the germline-reversion constructions BEFORE trusting the decomposition results.
Checks (BCR + SAbDab):
  1. CDR3 preserved in germ-fw and germ-cdr (native antigen-determining loop untouched).
  2. germ-fw framework actually reverted to germline (differs from native where SHM exists,
     equals native where unmutated); germ-cdr CDR1/2 actually replaced by germline.
  3. Lengths sane (~100-135 aa); germ-fw length ~ native length.
  4. NOT a collapse artifact: within a V-gene, germ-fw sequences stay diverse (driven by native CDRs).
  5. ANARCI a sample of germ-fw -> parses as a valid VH whose V-gene family matches the assignment
     and whose CDR3 equals the native CDR3.
"""
import os, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratch")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from build_germline_framework_bcr import load_ighv_framework, pick_allele
from anarci import anarci

AA = set("ACDEFGHIKLMNPQRSTVWY")
lut = load_ighv_framework()


def clean(s): return re.sub(r"[^A-Z]", "", str(s).upper())


def anarci_check(seqs_alleles):
    """ANARCI a few germ-fw seqs; return (parsed_ok, vgene_family_match, cdr3_preserved)."""
    out = []
    formatted = [(str(i), s) for i, (s, _, _) in enumerate(seqs_alleles)]
    numbered, details, _ = anarci(formatted, scheme="imgt", ncpu=4)
    for k, (s, al, natc3) in enumerate(seqs_alleles):
        ok = numbered[k] is not None and len(numbered[k]) > 0
        fammatch = c3ok = False
        if ok:
            dom = numbered[k][0][0]
            c3 = "".join(r for (pos, ins), r in dom if 105 <= pos <= 117 and r != "-")
            c3ok = (c3 == natc3)
            try:
                vg = details[k][0].get("germlines", {}).get("v_gene", [["", ""]])
                agene = vg[0][1] if isinstance(vg[0], (list, tuple)) else str(vg[0])
                fammatch = al.split("*")[0][:5] in str(agene)
            except Exception:
                fammatch = False
        out.append((ok, fammatch, c3ok))
    return out


def leakage_check(label, l1, l3, gf, l4, seed=42):
    """Does the clone-aware split (on L1 CDR3) leave germ-fw exact duplicates across train/test?
    If germ-fw collapse created cross-split identical sequences, retrieval would be trivially
    inflated. Compare germ-fw against L3/L4 (same split)."""
    from benchmark.split import clone_aware_split_fast
    cap = pd.DataFrame({"label": label, "l1": l1, "l3": l3, "gf": gf, "l4": l4}).dropna()
    cap = (cap.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), 150), random_state=seed))
              .sample(frac=1.0, random_state=seed).reset_index(drop=True))
    tr, te = clone_aware_split_fast(cap, cdr3_col="l1", label_col="label", test_size=0.2,
                                    clone_similarity_threshold=0.95, random_state=seed, min_clones_per_label=2)
    trd, ted = cap.iloc[tr], cap.iloc[te]
    print(f"  cross-split exact-duplicate queries (test seq also in train, same level), n_test={len(ted)}:")
    for col, nm in (("l1", "L1"), ("l3", "L3"), ("gf", "germ-fw"), ("l4", "L4")):
        dup = ted[col].isin(set(trd[col])).mean()
        print(f"     {nm:8s}: {100*dup:.1f}% of test queries have an identical {nm} in train")


def check_bcr():
    print("=== BCR ===")
    df = pd.read_csv("raw/iedb/bcr_singlechain_vh.tsv", sep="\t", dtype=str)
    df = df[(df["Chain 1_germfw"].fillna("") != "")].copy()
    nat = df["Chain 1_Protein Sequence"].map(clean)
    gf = df["Chain 1_germfw"].map(clean)
    gc = df["Chain 1_germcdr"].fillna("").map(clean)
    c3 = df["Chain 1_CDR3_anarci"].fillna(df.get("Chain 1_CDR3 ANARCI")).map(clean) \
        if "Chain 1_CDR3_anarci" in df else df["Chain 1_CDR3 ANARCI"].map(clean)
    print(f"n with germ-fw: {len(df)}")
    print(f"  CDR3 preserved in germ-fw: {100*np.mean([c in g for c,g in zip(c3,gf)]):.1f}%")
    print(f"  germ-fw == native (unmutated FR): {100*np.mean(gf.values==nat.values):.1f}%  (rest are SHM-reverted)")
    print(f"  len germ-fw mean {gf.map(len).mean():.0f} (native {nat.map(len).mean():.0f}); "
          f"clean-AA: {100*np.mean([set(g)<=AA for g in gf]):.1f}%")
    gcok = gc[gc != ""]
    print(f"  CDR3 preserved in germ-cdr: {100*np.mean([c in g for c,g in zip(c3[gc!=''],gcok)]):.1f}%")
    # collapse check within V-gene
    vg = df["Chain 1_Vgene_anarci"].map(lambda x: str(x).split("*")[0])
    tab = pd.DataFrame({"vg": vg.values, "gf": gf.values})
    grp = tab.groupby("vg")["gf"].agg(["count", "nunique"])
    grp = grp[grp["count"] >= 5]
    print(f"  collapse check (V-genes with >=5 seqs): mean unique/total = "
          f"{(grp['nunique']/grp['count']).mean():.2f}  (1.0 = no collapse)")
    # ANARCI sample
    samp = df.sample(min(12, len(df)), random_state=1)
    sa = [(clean(r["Chain 1_germfw"]), pick_allele(r["Chain 1_Vgene_anarci"], lut) or "",
           clean(r.get("Chain 1_CDR3_anarci") or r.get("Chain 1_CDR3 ANARCI"))) for _, r in samp.iterrows()]
    chk = anarci_check(sa)
    print(f"  ANARCI sample (n={len(chk)}): parsed {sum(o for o,_,_ in chk)}/{len(chk)}, "
          f"V-fam match {sum(f for _,f,_ in chk)}/{len(chk)}, CDR3 preserved {sum(c for _,_,c in chk)}/{len(chk)}")
    # CRITICAL: does germ-fw collapse create cross-split exact-match leakage?
    c1col = df["Chain 1_CDR1_anarci"].fillna("").map(clean)
    c2col = df["Chain 1_CDR2_anarci"].fillna("").map(clean)
    l3 = (c1col + c2col + c3).values
    leakage_check(df["Epitope_Source Molecule"].values, c3.values, l3, gf.values, nat.values)


def check_sabdab():
    print("\n=== SAbDab ===")
    f = "outputs/intermediate/sabdab_vj_annotated_paired.csv"
    df = pd.read_csv(f)
    rows = []
    for _, r in df.iterrows():
        nat = clean(r["sequence_heavy"]); c1 = clean(r["cdr1_heavy"]); c2 = clean(r["cdr2_heavy"]); c3 = clean(r["cdr3_heavy"])
        al = pick_allele(r.get("heavy_v_gene"), lut)
        if not (nat and c1 and c2 and c3 and al): continue
        fw1, fw2, fw3, gc1, gc2 = lut[al]
        i1 = nat.find(c1); rest = nat[i1+len(c1):] if i1 >= 0 else ""
        i2 = rest.find(c2); rest2 = rest[i2+len(c2):] if i2 >= 0 else ""
        i3 = rest2.find(c3); ok = i1 >= 0 and i2 >= 0 and i3 >= 0
        fw4 = rest2[i3+len(c3):] if ok else ""
        gf = fw1+c1+fw2+c2+fw3+c3+fw4
        if set(gf) <= AA and len(gf) >= 90:
            rows.append((nat, gf, c3, al, ok))
    print(f"n with germ-fw: {len(rows)}")
    print(f"  CDR3 preserved: {100*np.mean([c in g for _,g,c,_,_ in rows]):.1f}%")
    print(f"  germ-fw==native: {100*np.mean([n==g for n,g,_,_,_ in rows]):.1f}%")
    print(f"  len germ-fw mean {np.mean([len(g) for _,g,_,_,_ in rows]):.0f} "
          f"(native {np.mean([len(n) for n,_,_,_,_ in rows]):.0f})")
    sa = [(g, al, c) for _, g, c, al, _ in rows[:12]]
    chk = anarci_check(sa)
    print(f"  ANARCI sample (n={len(chk)}): parsed {sum(o for o,_,_ in chk)}/{len(chk)}, "
          f"V-fam match {sum(f for _,f,_ in chk)}/{len(chk)}, CDR3 preserved {sum(c for _,_,c in chk)}/{len(chk)}")


if __name__ == "__main__":
    check_bcr()
    try:
        check_sabdab()
    except Exception as e:
        print("SAbDab check note:", e)
