#!/usr/bin/env python3
"""Germline-subtracted embedding residual: where does the somatic-framework signal live?

For BCR L4 we have native sequences and germ-fw sequences (germline framework + native CDRs).
The residual  r = embed(native L4) - embed(germ-fw)  isolates the embedding shift caused by the
framework's SOMATIC mutations (everything else is shared). We then retrieve on r alone (cosine)
and break R@1 down by antigen. If the residual retrieves HIV-1 envelope well and nothing else, the
PLM's somatic-framework signal is localized to the most hypermutated antibodies -- a sharper version
of the germ-fw->L4 HIV result (Note S2.5). 20-seed clone-aware splits; ESM2-150M (CPU).
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np, pandas as pd
from benchmark.data import (build_dataset_specs, load_table, filter_human_only, normalize_labels,
                            is_legit_sequence, init_label_config, compute_shared_labels, clean_sequence)
from benchmark.split import clone_aware_split_fast
from benchmark.embeddings import get_embedder

SEEDS = [42 + 10 * i for i in range(int(os.environ.get("NSEEDS", "20")))]
CAP = 150
_cache = {}


def embed(emb, seqs):
    todo = [s for s in set(seqs) if s not in _cache]
    for i in range(0, len(todo), 64):
        b = todo[i:i+64]
        for s, v in zip(b, emb.encode(b)):
            _cache[s] = v
    return np.array([_cache[s] for s in seqs])


def cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def exp_r1(S, qlab, dlab):
    out = []
    for i in range(S.shape[0]):
        mx = S[i].max(); grp = np.isclose(S[i], mx)
        out.append(float((dlab[grp] == qlab[i]).mean()))
    return np.array(out)


def main():
    init_label_config("label_aliases.json")
    class M:
        include_bcr = True; include_tcr = include_sabdab = False
        bcr_file = "raw/iedb/bcr_singlechain_vh.tsv"; bcr_label_col = "Epitope_Source Molecule"
        human_only = True; top_labels = 10; min_label_count = 14; max_per_label_bcr = 150
        shared_labels = True; batch_size = 32; local_files_only = False; offline_dir = None; pooling = "mean"
    m = M(); spec = build_dataset_specs(m)[0]; df = load_table(spec); df = filter_human_only(df, spec)
    lab = normalize_labels(df[spec.label_col])
    l1 = spec.sequence_builders["level1"](df); l4 = spec.sequence_builders["level4"](df)
    gf = df["Chain 1_germfw"].map(clean_sequence) if "Chain 1_germfw" in df else pd.Series([None]*len(df))
    work = pd.DataFrame({"label": lab, "l1": l1, "l4": l4, "gf": gf}).dropna()
    work = work[work["l4"].map(is_legit_sequence) & work["gf"].map(is_legit_sequence)]
    keep = compute_shared_labels(df, spec, 10, 14)
    work = work[work["label"].isin(keep)].reset_index(drop=True)
    hivset = [l for l in keep if any(k in str(l).lower() for k in ("hiv", "envelope"))]
    print(f"working set {len(work)} seqs; HIV labels = {hivset}")

    emb = get_embedder("esm2-150m", m)
    per = {("residual", st): [] for st in ("HIV", "nonHIV", "all")}
    per_native = {st: [] for st in ("HIV", "nonHIV", "all")}
    for s in SEEDS:
        cap = (work.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), CAP), random_state=s))
                   .sample(frac=1.0, random_state=s).reset_index(drop=True))
        tr, te = clone_aware_split_fast(cap, cdr3_col="l1", label_col="label", test_size=0.2,
                                        clone_similarity_threshold=0.95, random_state=s, min_clones_per_label=2)
        trd, ted = cap.iloc[tr], cap.iloc[te]
        ql = ted["label"].to_numpy(); dl = trd["label"].to_numpy()
        q_res = embed(emb, list(ted["l4"])) - embed(emb, list(ted["gf"]))
        d_res = embed(emb, list(trd["l4"])) - embed(emb, list(trd["gf"]))
        r_res = exp_r1(cos(q_res, d_res), ql, dl)
        r_nat = exp_r1(cos(embed(emb, list(ted["l4"])), embed(emb, list(trd["l4"]))), ql, dl)
        hiv = np.array([str(x) in hivset for x in ql])
        for st, mask in (("HIV", hiv), ("nonHIV", ~hiv), ("all", np.ones(len(ql), bool))):
            if mask.sum() >= 3:
                per[("residual", st)].append(r_res[mask].mean()); per_native[st].append(r_nat[mask].mean())
        print(f"seed {s}: residual all {r_res.mean():.3f} | residual HIV {r_res[hiv].mean() if hiv.sum() else float('nan'):.3f}")

    # random baseline = 1/n_labels (=0.1 for 10 labels) for HIV-vs-rest interpretation
    print("\n=== residual-only retrieval R@1 (germline-subtracted), by antigen ===")
    for st in ("HIV", "nonHIV", "all"):
        a = np.array(per[("residual", st)]); n = np.array(per_native[st])
        print(f"  {st:7s}: residual {a.mean():.3f} [{a.mean()-1.96*a.std(ddof=1)/np.sqrt(len(a)):.3f},"
              f"{a.mean()+1.96*a.std(ddof=1)/np.sqrt(len(a)):.3f}]   (native-L4 ref {n.mean():.3f})")
    print("random baseline (10 labels) = 0.100")
    print("\nReading: if residual HIV >> residual nonHIV (~random), the somatic-framework signal is HIV-localized.")


if __name__ == "__main__":
    main()
