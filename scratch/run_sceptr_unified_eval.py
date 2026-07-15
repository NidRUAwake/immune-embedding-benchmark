#!/usr/bin/env python3
"""SCEPTR vs alignment vs TCRdist3 head-to-head on a COMMON gene-mappable query set (manuscript Sec 3.5).

Reproduces the Sec 3.5 / Supplementary-Methods numbers. All five methods (BLOSUM62, Levenshtein,
ESM2-150M, SCEPTR, TCRdist3) are scored on the IDENTICAL per-seed clone-aware split AND restricted
to the common subset of test/reference rows whose V/J genes tcrdist3 can map (tcrdist3 silently
drops gene-unmappable records). 20 seeds, expected Recall@1.

Runs in the MAIN environment (embedding_benchmark_v1): it imports the canonical benchmark library
(scripts/benchmark) and tcrdist3 (TCRrep), and reads SCEPTR / ESM2-150M embeddings from the offline
.npy caches. SCEPTR embeddings are produced separately by scratch/run_sceptr_embed.py in the
isolated SCEPTR environment (see that script + CODE_GUIDE.md).

Offline embedding caches expected (Zenodo data package / outputs/embeddings/):
  mcpas_level1_sceptr_mean.npy, tcr_level1_sceptr_mean.npy, tcr_level1_paired_sceptr_mean.npy
  + matching esm2-150m caches.
Raw inputs: outputs/intermediate/mcpas_standardized.tsv, raw/vdjdb/vdjdb_full.txt.
Run from the package/repo root. Writes per-config CSVs to outputs/reports/sceptr_unified/.
"""
import sys, os, warnings, re, json, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark.data import (build_dataset_specs, load_table, filter_human_only, compute_shared_labels,
                            normalize_labels, assign_antigen_type, normalize_gene_name, init_label_config,
                            strip_tcr_cdr3_anchors, clean_sequence)
from benchmark.split import clone_aware_split_fast
from benchmark.evaluate import retrieval_metrics, compute_similarity_matrix
from tcrdist.repertoire import TCRrep

SEEDS = [42 + 10 * i for i in range(20)]; TEST_SIZE = 0.2; CT = 0.95; MINCL = 2
OUTDIR = Path("outputs/reports/sceptr_unified")

class Args:
    include_bcr = False; include_tcr = True; include_sabdab = False
    tcr_label_col = "antigen.epitope"; local_files_only = False; offline_dir = None
    bcr_file = "raw/iedb/bcr_singlechain_vh.tsv"; bcr_label_col = "x"
    sabdab_file = "raw/sabdab/sabdab_paired_clean_human_full.csv"; sabdab_label_col = "label"
    def __init__(s, tcr_file, human): s.tcr_file = tcr_file; s.human_only = human

def slice_with_genes(df, spec, level, max_per_label, seed, forced):
    labels = normalize_labels(df[spec.label_col])
    seq = spec.sequence_builders[level](df)
    ssq = spec.sequence_builders["level1"](df)
    w = pd.DataFrame({"label": labels, "sequence": seq, "split_sequence": ssq,
                      "antigen_type": assign_antigen_type(labels)})
    for c in ["cdr3.beta", "v.beta", "j.beta", "cdr3.alpha", "v.alpha", "j.alpha"]:
        w[c] = df[c] if c in df else None
    w = w.dropna(subset=["label", "sequence", "split_sequence"])
    w = w[w["label"].isin(forced)]
    parts = []
    for lab in w["label"].unique():
        ld = w[w["label"] == lab]
        if len(ld) > max_per_label:
            ld = ld.sample(max_per_label, random_state=seed)
        parts.append(ld)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

def load_offline(path):
    data = np.load(path); meta = json.load(open(path.replace('.npy', '_metadata.json')))
    so = meta['sequence_order']
    idx = {}
    for i, sq in enumerate(so):
        for k in (sq, clean_sequence(str(sq)), clean_sequence(strip_tcr_cdr3_anchors(str(sq)))):
            if k and k not in idx:
                idx[k] = i
    def enc(seqs):
        out = np.zeros((len(seqs), data.shape[1]), data.dtype); f = 0
        for i, sq in enumerate(seqs):
            j = idx.get(sq)
            if j is None:
                j = idx.get(clean_sequence(str(sq)))
            if j is not None:
                out[i] = data[j]; f += 1
        return out, f / len(seqs)
    return enc

GENE_CACHE = {}
def get_cleaner():
    if 'c' not in GENE_CACHE:
        probe = TCRrep(cell_df=pd.DataFrame({"cdr3_a_aa": ["CAVRDSNYQLIW"], "v_a_gene": ["TRAV3*01"], "j_a_gene": ["TRAJ33*01"],
                       "cdr3_b_aa": ["CASSLAPGATNEKLFF"], "v_b_gene": ["TRBV9*01"], "j_b_gene": ["TRBJ1-1*01"], "count": [1]}),
                       organism="human", chains=["alpha", "beta"], compute_distances=False, deduplicate=False)
        valid = set(probe.all_genes["human"].keys()); vs = sorted(valid)
        def clean(g):
            g = str(g).strip().replace(" ", "").replace("–", "-").replace("_", "").split(";")[0]
            if "*" not in g: g += "*01"
            if g in valid: return g
            base = g.split("*")[0]
            if base + "*01" in valid: return base + "*01"
            b2 = re.sub(r'(?<=[VJ])0(\d)', r'\1', base); b2 = re.sub(r'S\d+$', '', b2)
            for c in (b2 + "*01", b2 + "*02"):
                if c in valid: return c
            fam = re.match(r'(TR[AB][VJ]\d+)', b2)
            if fam:
                cs = [v for v in vs if v.startswith(fam.group(1))]
                if cs: return cs[0]
            return None
        GENE_CACHE['c'] = clean
    return GENE_CACHE['c']

def _valid_mask(sub, paired):
    clean = get_cleaner(); m = []
    for _, r in sub.iterrows():
        vb, jb = clean(r["v.beta"]), clean(r["j.beta"])
        ok = (vb is not None and jb is not None and pd.notna(r["cdr3.beta"]))
        if paired:
            va, ja = clean(r["v.alpha"]), clean(r["j.alpha"])
            ok = ok and (va is not None and ja is not None and pd.notna(r["cdr3.alpha"]))
        m.append(ok)
    return np.array(m)

def run_config(name, tcr_file, human, top, minc, maxpl, level, paired, sceptr_npy, esm_npy):
    spec = build_dataset_specs(Args(tcr_file, human))[0]
    df = load_table(spec)
    if human: df = filter_human_only(df, spec)
    forced = compute_shared_labels(df, spec, top, minc)
    enc_s = load_offline(sceptr_npy); enc_e = load_offline(esm_npy)
    rows = []
    for seed in SEEDS:
        sl = slice_with_genes(df, spec, level, maxpl, seed, forced)
        tri, tei = clone_aware_split_fast(sl, cdr3_col="split_sequence", label_col="label",
                   test_size=TEST_SIZE, clone_similarity_threshold=CT, random_state=seed,
                   min_clones_per_label=MINCL, show_progress=False)
        tr, te = sl.iloc[tri], sl.iloc[tei]
        # COMMON valid subset = where tcrdist genes map (most restrictive); all methods scored on it
        qk = _valid_mask(te, paired); tk = _valid_mask(tr, paired)
        trv, tev = tr[tk], te[qk]
        q, d = tev["sequence"].values, trv["sequence"].values
        ql, dl = tev["label"].values, trv["label"].values
        qt = tev["antigen_type"].values
        res = {"seed": seed, "n_train": len(trv), "n_test": len(tev), "kept": f"{qk.sum()}/{len(qk)}"}
        for meth in ["blosum62", "levenshtein"]:
            sim = compute_similarity_matrix(q, d, meth)
            res[meth] = retrieval_metrics(None, None, ql, dl, qt, meth, precomputed_sim=sim)["recall@1"]
        for meth, enc in [("sceptr", enc_s), ("esm2-150m", enc_e)]:
            qe, fq = enc(q); de, fd = enc(d)
            sim = compute_similarity_matrix(qe, de, meth, "cosine")
            res[meth] = retrieval_metrics(None, None, ql, dl, qt, meth, precomputed_sim=sim)["recall@1"]
            res[meth + "_cov"] = round(min(fq, fd), 3)
        # tcrdist on the SAME subset
        clean = get_cleaner()
        def cells(sub):
            recs = []
            for _, r in sub.iterrows():
                vb, jb = clean(r["v.beta"]), clean(r["j.beta"])
                if paired:
                    va, ja = clean(r["v.alpha"]), clean(r["j.alpha"])
                    recs.append({"cdr3_a_aa": str(r["cdr3.alpha"]), "v_a_gene": va, "j_a_gene": ja,
                                 "cdr3_b_aa": str(r["cdr3.beta"]), "v_b_gene": vb, "j_b_gene": jb, "count": 1})
                else:
                    recs.append({"cdr3_b_aa": str(r["cdr3.beta"]), "v_b_gene": vb, "j_b_gene": jb, "count": 1})
            return pd.DataFrame(recs)
        chains = ["alpha", "beta"] if paired else ["beta"]
        cell = pd.concat([cells(trv), cells(tev)], ignore_index=True)
        trep = TCRrep(cell_df=cell, organism="human", chains=chains, compute_distances=True, deduplicate=False)
        nt = len(trv); pw = (trep.pw_alpha + trep.pw_beta) if paired else trep.pw_beta
        sim = -(pw[nt:, :nt].astype(float))
        res["tcrdist3"] = retrieval_metrics(None, None, ql, dl, qt, "tcrdist3", precomputed_sim=sim)["recall@1"]
        rows.append(res)
    return pd.DataFrame(rows)

def run_all():
    init_label_config("label_aliases.json")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfgs = [
     ("McPAS-CDR3", "outputs/intermediate/mcpas_standardized.tsv", False, 10, 30, 100, "level1", False,
      "outputs/embeddings/mcpas_level1_sceptr_mean.npy", "outputs/embeddings/mcpas_level1_esm2-150m_mean.npy"),
     ("VDJdb-CDR3", "raw/vdjdb/vdjdb_full.txt", True, 15, 50, 100, "level1", False,
      "outputs/embeddings/tcr_level1_sceptr_mean.npy", "outputs/embeddings/tcr_level1_esm2-150m_mean.npy"),
     ("VDJdb-paired-CDR3", "raw/vdjdb/vdjdb_full.txt", True, 10, 14, 100, "level1_paired", True,
      "outputs/embeddings/tcr_level1_paired_sceptr_mean.npy", "outputs/embeddings/tcr_level1_paired_esm2-150m_mean.npy"),
    ]
    meth = ["tcrdist3", "sceptr", "blosum62", "levenshtein", "esm2-150m"]
    for nm, *a in cfgs:
        r = run_config(nm, *a)
        means = r[meth].mean(); stds = r[meth].std()
        print(f"\n===== {nm} (20-seed, common TCRdist-valid subset) =====")
        print(f"  kept e.g. {r['kept'].iloc[0]}  n_test~{round(r.n_test.mean())} n_train~{round(r.n_train.mean())}")
        for m in meth:
            print(f"  {m:14s} {means[m]:.4f} +/- {stds[m]:.4f}")
        r.to_csv(OUTDIR / f"{nm}.csv", index=False)
    print("\nALL DONE")

if __name__ == "__main__":
    run_all()
