#!/usr/bin/env python3
"""Does the zero-shot PLM's CDR3 deficit grow with how out-of-distribution the CDR3 is?

For BCR L1 (CDR3), per query we compute:
  - ESM2-150M pseudo-perplexity (PPL): mask each residue, P(true | rest); PPL = exp(mean -logP).
    Higher PPL = the masked-LM finds the CDR3 more surprising / less natural-protein-like (OOD).
  - per-query expected-R@1 for ESM2-150M (cosine) and BLOSUM62 (parasail SW), and the
    deficit = ESM2 R@1 - BLOSUM62 R@1 (negative = PLM worse; controls for query difficulty).
Hypothesis (germline axis): the PLM's deficit is largest on the most OOD (high-PPL, junctional)
CDR3s -> corr(PPL, deficit) < 0. Embeddings + PPL both from one EsmForMaskedLM load (hidden states
== canonical mean-pool; logits == PPL). 20-seed clone-aware splits; instances pooled across seeds.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np, pandas as pd, torch
from scipy import stats
from transformers import AutoTokenizer, EsmForMaskedLM
from benchmark.data import (build_dataset_specs, load_table, filter_human_only, normalize_labels,
                            is_legit_sequence, init_label_config, compute_shared_labels, build_pilot_slice)
from benchmark.split import clone_aware_split_fast
from benchmark.evaluate import compute_similarity_matrix

SEEDS = [42 + 10 * i for i in range(int(os.environ.get("NSEEDS", "20")))]
MODEL = "facebook/esm2_t30_150M_UR50D"
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(MODEL)
mlm = EsmForMaskedLM.from_pretrained(MODEL).eval()

_emb_cache, _ppl_cache = {}, {}


def embed(seqs):
    out, todo = {}, [s for s in set(seqs) if s not in _emb_cache]
    for i in range(0, len(todo), 64):
        b = todo[i:i+64]
        enc = tok(b, return_tensors="pt", padding=True)
        h = mlm(**enc, output_hidden_states=True).hidden_states[-1]
        m = enc["attention_mask"].unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        for s, v in zip(b, pooled.numpy()):
            _emb_cache[s] = v
    return np.array([_emb_cache[s] for s in seqs])


def ppl(seq):
    if seq in _ppl_cache:
        return _ppl_cache[seq]
    enc = tok(seq, return_tensors="pt")
    ids = enc["input_ids"][0]
    pos = list(range(1, len(ids) - 1))  # exclude BOS/EOS
    batch = enc["input_ids"].repeat(len(pos), 1).clone()
    for k, p in enumerate(pos):
        batch[k, p] = tok.mask_token_id
    am = enc["attention_mask"].repeat(len(pos), 1)
    logits = mlm(input_ids=batch, attention_mask=am).logits
    lp = []
    for k, p in enumerate(pos):
        logp = torch.log_softmax(logits[k, p], -1)[ids[p]].item()
        lp.append(logp)
    val = float(np.exp(-np.mean(lp)))
    _ppl_cache[seq] = val
    return val


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
    fl = compute_shared_labels(df, spec, 10, 14)
    rows = []
    for s in SEEDS:
        sl = build_pilot_slice(df, spec, "level1", 150, s, fl); fr = sl.dataframe; sc = sl.sequence_col
        tr, te = clone_aware_split_fast(fr, cdr3_col="split_sequence", label_col="label", test_size=0.2,
                                        clone_similarity_threshold=0.95, random_state=s, min_clones_per_label=2)
        trd = fr.iloc[tr]; ted = fr.iloc[te]
        qs = ted[sc].to_numpy(); ds = trd[sc].to_numpy()
        ql = ted["label"].to_numpy(); dl = trd["label"].to_numpy()
        qe, de = embed(list(qs)), embed(list(ds))
        qn = qe / (np.linalg.norm(qe, axis=1, keepdims=True) + 1e-9)
        dn = de / (np.linalg.norm(de, axis=1, keepdims=True) + 1e-9)
        esm_r1 = exp_r1(qn @ dn.T, ql, dl)
        bl_r1 = exp_r1(compute_similarity_matrix(qs.astype(object), ds.astype(object), "blosum62"), ql, dl)
        for k in range(len(qs)):
            rows.append((qs[k], ql[k], esm_r1[k], bl_r1[k]))
        print(f"seed {s}: nq={len(qs)} esm {esm_r1.mean():.3f} blosum {bl_r1.mean():.3f}")
    R = pd.DataFrame(rows, columns=["seq", "label", "esm", "blosum"])
    R["ppl"] = R["seq"].map(ppl)
    R["deficit"] = R["esm"] - R["blosum"]
    R.to_csv("outputs/reports/ood_cdr3_perquery.csv", index=False)

    print(f"\nn query-instances = {len(R)}, unique CDR3 = {R['seq'].nunique()}")
    rho, p = stats.spearmanr(R["ppl"], R["deficit"])
    rp, pp = stats.pearsonr(R["ppl"], R["deficit"])
    print(f"corr(PPL, ESM2-BLOSUM deficit): Spearman {rho:+.3f} (p={p:.2g}); Pearson {rp:+.3f} (p={pp:.2g})")
    print("(negative = PLM deficit grows with OOD-ness, supporting the OOD hypothesis)")
    q = R["ppl"].quantile([1/3, 2/3]).values
    R["bin"] = np.where(R["ppl"] <= q[0], "low-OOD", np.where(R["ppl"] <= q[1], "mid", "high-OOD"))
    print("\nby PPL tertile:")
    for b in ["low-OOD", "mid", "high-OOD"]:
        d = R[R["bin"] == b]
        print(f"  {b:8s} (n={len(d)}): PPL {d['ppl'].mean():.2f}  ESM2 {d['esm'].mean():.3f}  "
              f"BLOSUM {d['blosum'].mean():.3f}  deficit {d['deficit'].mean():+.3f}")
    print("\nSaved outputs/reports/ood_cdr3_perquery.csv")


if __name__ == "__main__":
    main()
