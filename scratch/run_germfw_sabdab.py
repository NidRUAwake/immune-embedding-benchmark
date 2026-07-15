#!/usr/bin/env python3
"""Cross-database replication of the germline-reversion decomposition on SAbDab (native antibody DB).

Mirrors scratch/run_germfw_experiment.py but on SAbDab heavy chains. SAbDab's top labels include
'Envelope Glycoprotein Gp160' (HIV-1 envelope), so we can test BOTH the rule (germline-framework
shortcut on non-HIV) and the exception (framework-SHM signal on HIV) on a second native database.

germ-fw = germline FR1-3 (assigned IGHV allele) + native CDR1/2/3 + native FW4.
germ-cdr = native FR + germline CDR1/2 + native CDR3 + native FW4.
"""
from __future__ import annotations
import os, sys, re
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scratch"))
import numpy as np, pandas as pd
from scipy import stats
from benchmark.data import (build_dataset_specs, load_table, filter_human_only, normalize_labels,
                            is_legit_sequence, init_label_config, clean_sequence)
from benchmark.split import clone_aware_split_fast
from benchmark.embeddings import get_embedder
from benchmark.evaluate import retrieval_metrics
from build_germline_framework_bcr import load_ighv_framework, pick_allele

SEEDS = [42 + 10 * i for i in range(int(os.environ.get("NSEEDS", "20")))]
CAP = 100
AA = set("ACDEFGHIKLMNPQRSTVWY")


class M:
    include_bcr = False; include_tcr = False; include_sabdab = True
    sabdab_file = "raw/sabdab/sabdab_paired_clean_human_full.csv"; sabdab_label_col = "label"
    human_only = True; batch_size = 32; local_files_only = False; offline_dir = None; pooling = "mean"


def variants(row, lut):
    nat = re.sub(r"[^A-Z]", "", str(row["sequence_heavy"]).upper())
    c1 = re.sub(r"[^A-Z]", "", str(row["cdr1_heavy"]).upper())
    c2 = re.sub(r"[^A-Z]", "", str(row["cdr2_heavy"]).upper())
    c3 = re.sub(r"[^A-Z]", "", str(row["cdr3_heavy"]).upper())
    al = pick_allele(row.get("heavy_v_gene"), lut)
    if not (nat and c1 and c2 and c3 and al):
        return None, None
    fw1, fw2, fw3, gc1, gc2 = lut[al]
    i1 = nat.find(c1); rest = nat[i1+len(c1):] if i1 >= 0 else ""
    i2 = rest.find(c2); rest2 = rest[i2+len(c2):] if i2 >= 0 else ""
    i3 = rest2.find(c3); ok = (i1 >= 0 and i2 >= 0 and i3 >= 0)
    nfw1, nfw2, nfw3 = (nat[:i1], rest[:i2], rest2[:i3]) if ok else ("", "", "")
    fw4 = rest2[i3+len(c3):] if ok else (nat[nat.rfind(c3)+len(c3):] if nat.rfind(c3) >= 0 else "")
    gf = fw1 + c1 + fw2 + c2 + fw3 + c3 + fw4
    gc = (nfw1 + gc1 + nfw2 + gc2 + nfw3 + c3 + fw4) if ok else ""
    gf = gf if (set(gf) <= AA and len(gf) >= 90) else None
    gc = gc if (ok and set(gc) <= AA and len(gc) >= 90) else None
    return gf, gc


def main():
    init_label_config("label_aliases.json")
    lut = load_ighv_framework()
    m = M(); spec = build_dataset_specs(m)[0]
    df = load_table(spec); df = filter_human_only(df, spec).reset_index(drop=True)
    lab = normalize_labels(df[spec.label_col])
    l1 = spec.sequence_builders["level1"](df); l3 = spec.sequence_builders["level3"](df)
    l4 = spec.sequence_builders["level4"](df)
    gf, gc = [], []
    for _, row in df.iterrows():
        a, b = variants(row, lut); gf.append(a); gc.append(b)
    work = pd.DataFrame({"label": lab, "l1": l1, "l3": l3, "l4": l4,
                         "gf": pd.Series(gf, dtype="object").map(clean_sequence),
                         "gc": pd.Series(gc, dtype="object").map(clean_sequence),
                         "antigen_type": "protein"}).dropna()
    for c in ("l3", "l4", "gf", "gc"):
        work = work[work[c].map(is_legit_sequence)]
    keep = work["label"].value_counts().head(5).index.tolist()
    work = work[work["label"].isin(keep)].reset_index(drop=True)
    hiv_lab = [l for l in keep if any(k in str(l).lower() for k in ("gp160", "envelope", "hiv"))]
    print(f"SAbDab working set: {len(work)} seqs, labels={list(keep)}")
    print(f"HIV labels: {hiv_lab}")

    emb = get_embedder("esm2-150m", m); bl = get_embedder("blosum62", m)
    LEVELS = {"L3": "l3", "germfw": "gf", "germcdr": "gc", "L4": "l4"}
    uniq = sorted(set(work["l3"]) | set(work["gf"]) | set(work["gc"]) | set(work["l4"]))
    print(f"embedding {len(uniq)} unique sequences (ESM2-150M, CPU)...")
    mat = emb.encode(uniq); e2 = {s_: mat[i] for i, s_ in enumerate(uniq)}

    per = {(lv, mo): [] for lv in LEVELS for mo in ("blosum62", "esm2-150m")}
    strat = {(st, lv): [] for st in ("HIV", "nonHIV") for lv in ("germfw", "L4")}
    for s in SEEDS:
        cap = (work.groupby("label", group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), CAP), random_state=s))
                   .sample(frac=1.0, random_state=s).reset_index(drop=True))
        tr, te = clone_aware_split_fast(cap, cdr3_col="l1", label_col="label", test_size=0.2,
                                        clone_similarity_threshold=0.95, random_state=s,
                                        min_clones_per_label=2, show_progress=False)
        trd = cap.iloc[tr]; ted = cap.iloc[te]
        tl = ted["label"].to_numpy(); dl = trd["label"].to_numpy(); ta = ted["antigen_type"].to_numpy()
        for lv, col in LEVELS.items():
            qs = ted[col].tolist(); ds = trd[col].tolist()
            for mo in ("blosum62", "esm2-150m"):
                qe, de = (np.array([e2[x] for x in qs]), np.array([e2[x] for x in ds])) if mo == "esm2-150m" \
                    else (bl.encode(qs), bl.encode(ds))
                per[(lv, mo)].append(retrieval_metrics(q_emb=qe, d_emb=de, q_labels=tl, d_labels=dl,
                                                       q_antigen_types=ta, method=mo)["recall@1"])
        if hiv_lab:
            hiv = np.array([str(x) in hiv_lab for x in tl])
            for st, mask in (("HIV", hiv), ("nonHIV", ~hiv)):
                for lv in ("germfw", "L4"):
                    if mask.sum() < 3: strat[(st, lv)].append(np.nan); continue
                    col = LEVELS[lv]
                    qe = np.array([e2[x] for x in np.array(ted[col].tolist())[mask]])
                    de = np.array([e2[x] for x in trd[col].tolist()])
                    strat[(st, lv)].append(retrieval_metrics(q_emb=qe, d_emb=de, q_labels=tl[mask],
                        d_labels=dl, q_antigen_types=ta[mask], method="esm2-150m")["recall@1"])

    def arr(lv, mo): return np.array(per[(lv, mo)])
    print("\n=== SAbDab means (R@1) ===")
    for mo in ("blosum62", "esm2-150m"):
        print(f"  {mo:10s} L3 {arr('L3',mo).mean():.3f} germfw {arr('germfw',mo).mean():.3f} "
              f"germcdr {arr('germcdr',mo).mean():.3f} L4 {arr('L4',mo).mean():.3f}")
    def paired(a, b):
        d = a-b; se = d.std(ddof=1)/np.sqrt(len(d)); _, p = stats.ttest_rel(a, b)
        return d.mean(), d.mean()-1.96*se, d.mean()+1.96*se, p
    print("\n=== SAbDab decomposition (per-seed paired) ===")
    for mo in ("blosum62", "esm2-150m"):
        for a, b, nm in [("germfw","L3","germline-FW gain (L3->germfw)"),
                         ("L4","germfw","framework-SHM gain (germfw->L4)"),
                         ("L4","germcdr","CDR1/2-SHM gain (germcdr->L4)")]:
            mn, lo, hi, p = paired(arr(a, mo), arr(b, mo))
            print(f"  {mo:10s} {nm:32s} {mn:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3g}")
    if hiv_lab:
        print("\n=== SAbDab framework-SHM gain by antigen (ESM2) ===")
        for st in ("HIV", "nonHIV"):
            g = np.array(strat[(st,"L4")], float) - np.array(strat[(st,"germfw")], float); g = g[~np.isnan(g)]
            if len(g) >= 3:
                se = g.std(ddof=1)/np.sqrt(len(g)); _, p = stats.ttest_1samp(g, 0)
                print(f"  {st:7s}: germfw {np.nanmean(strat[(st,'germfw')]):.3f} -> L4 "
                      f"{np.nanmean(strat[(st,'L4')]):.3f}  gain {g.mean():+.4f} "
                      f"[{g.mean()-1.96*se:+.4f},{g.mean()+1.96*se:+.4f}] p={p:.3g} (n={len(g)})")


if __name__ == "__main__":
    main()
