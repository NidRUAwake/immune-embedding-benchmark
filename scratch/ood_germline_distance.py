#!/usr/bin/env python3
"""Model-independent OOD axis for the BCR CDR3 deficit (mock-review 374, pt 1a).

The §3.2 OOD result used ESM2's OWN pseudo-perplexity to score how out-of-distribution
each CDR3 is, which is model-internal. Here we build a BIOLOGICAL, model-independent
"germline-distance" axis and ask whether the PLM's per-query (ESM2 - BLOSUM) deficit
grows with it, exactly as it did with pseudo-perplexity.

germline-distance(CDR3) = min over germline V-J recombined templates T of
    normalized Levenshtein(CDR3, T),
where T = (germline IGHV CDR3 contribution) + (germline IGHJ CDR3 contribution),
i.e. the minimal V-J direct join with NO D segment and NO N/P insertions. A CDR3 far
from every such template carries junctional (D + N) and/or somatically mutated content
-> more non-germline / out-of-distribution. Uses ONLY the CDR3 string + germline
reference (imgt/IGV.fasta, imgt/IGJ.fasta); never touches ESM2.

Validation prints come BEFORE any correlation, to catch parsing bugs.
"""
import re
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from rapidfuzz.distance import Levenshtein as RFLev
from rapidfuzz.process import cdist as rf_cdist

ROOT = Path(__file__).resolve().parent.parent
OOD = ROOT / "outputs/reports/ood_cdr3_perquery.csv"
IGV = ROOT / "imgt/IGV.fasta"
IGJ = ROOT / "imgt/IGJ.fasta"
IGD = ROOT / "imgt/IGD.fasta"
AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(p):
    recs, name, seq = [], None, []
    for line in open(p):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name is not None:
                recs.append((name, "".join(seq)))
            name, seq = line[1:], []
        else:
            seq.append(line)
    if name is not None:
        recs.append((name, "".join(seq)))
    return recs


def lev(a, b):
    """Plain Levenshtein distance (short strings)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[lb]


# ---- germline V CDR3 contribution: residues after the conserved 2nd-Cys (Cys104) ----
vcontribs = {}
for name, seq in read_fasta(IGV):
    if "Homo sapiens" not in name:
        continue
    s = seq.replace(".", "").upper()
    ci = s.rfind("C")               # 2nd-Cys is the last C in a V-REGION germline
    if ci == -1:
        continue
    vc = s[ci + 1:]
    if 0 < len(vc) <= 5 and set(vc) <= AA:   # V contributes ~1-4 residues (e.g. AR, AK, TR)
        vcontribs[vc] = vcontribs.get(vc, 0) + 1

# ---- germline J CDR3 contribution: residues before the conserved Trp (W of WG[QKR]G) ----
jcontribs = {}
wmotif = re.compile(r"W(G[A-Z]G)")
for name, seq in read_fasta(IGJ):
    if "Homo sapiens" not in name:
        continue
    s = seq.replace(".", "").upper()
    m = wmotif.search(s)
    if not m:
        continue
    jc = s[:m.start()]
    if 1 <= len(jc) <= 12 and set(jc) <= AA:
        jcontribs[jc] = jcontribs.get(jc, 0) + 1

# ---- germline D CDR3 contribution: IMGT D-REGION AA (frame 1 only in IGD.fasta; a known limitation) ----
dcontribs_all, dcontribs_F = {}, {}
for name, seq in read_fasta(IGD):
    if "Homo sapiens" not in name:
        continue
    s = seq.replace(".", "").upper()
    if len(s) < 3 or not (set(s) <= AA):
        continue
    dcontribs_all[s] = dcontribs_all.get(s, 0) + 1
    if "|F|" in name:          # functional D only
        dcontribs_F[s] = dcontribs_F.get(s, 0) + 1

print("=== VALIDATION ===")
print("distinct human V CDR3-contribs:", len(vcontribs))
print("  top:", sorted(vcontribs.items(), key=lambda x: -x[1])[:10])
print("distinct human J CDR3-contribs:", len(jcontribs))
print("  all:", sorted(jcontribs.items(), key=lambda x: -x[1]))
print("distinct D AA (all/functional):", len(dcontribs_all), "/", len(dcontribs_F),
      " (frame-1 only; D-assignment-free: we min over ALL of them, never pick one)")

VC = sorted(vcontribs)
JC = sorted(jcontribs)
DC_all = sorted(dcontribs_all)
DC_F = sorted(dcontribs_F)
templates = sorted({v + j for v in VC for j in JC})        # V-J only (no D)
print("germline V-J templates:", len(templates), "e.g.", templates[:6])


def build_vdj(DC):
    """V + (any D) + J templates, plus the no-D V-J join (handles D-less / heavily-trimmed CDR3s)."""
    t = set(templates)
    for v in VC:
        for d in DC:
            for j in JC:
                t.add(v + d + j)
    return sorted(t)


def gdist_to(queries, tmpls):
    """min normalized Levenshtein from each query to any template (rapidfuzz, vectorized)."""
    tl = np.array([len(t) for t in tmpls])
    ql = np.array([len(q) for q in queries])
    M = rf_cdist(queries, tmpls, scorer=RFLev.distance).astype(float)  # (nq, nt) raw edits
    denom = np.maximum(ql[:, None], tl[None, :])
    return (M / denom).min(axis=1)

# ---- per-unique-CDR3 germline-distance ----
df = pd.read_csv(OOD)
df["deficit"] = df["esm"] - df["blosum"]
g = (df.groupby("seq")
       .agg(esm=("esm", "mean"), blosum=("blosum", "mean"),
            deficit=("deficit", "mean"), ppl=("ppl", "mean"), n=("esm", "size"))
       .reset_index())
g["L"] = g["seq"].str.len()

queries = g["seq"].tolist()
tmpl_vdj_all = build_vdj(DC_all)
tmpl_vdj_F = build_vdj(DC_F)
print("germline V-D-J templates: all-D=%d  functional-D=%d" % (len(tmpl_vdj_all), len(tmpl_vdj_F)))
g["gdist"] = gdist_to(queries, templates)        # V-J only (no D) -- prior axis
g["gdist_vdj"] = gdist_to(queries, tmpl_vdj_all) # V-(any D)-J, assignment-free
g["gdist_vdjF"] = gdist_to(queries, tmpl_vdj_F)  # functional-D only (robustness)

# nearest template sanity for a few (V-D-J, the assignment-free axis we report)
print("\n=== nearest germline V-D-J template sanity (5 short + 5 long CDR3s) ===")
gsort = g.sort_values("L")
for _, r in pd.concat([gsort.head(5), gsort.tail(5)]).iterrows():
    cd = r["seq"]
    nt = min(tmpl_vdj_all, key=lambda t: RFLev.distance(cd, t) / max(len(cd), len(t)))
    print(f"  L={int(r['L']):2d} gdist_vj={r['gdist']:.3f} gdist_vdj={r['gdist_vdj']:.3f}  {cd:30s} -> {nt}")

print("\n=== distributions ===")
print(g[["L", "ppl", "gdist", "gdist_vdj", "gdist_vdjF", "deficit"]].describe().round(3).to_string())

def sp(a, b):
    rho, p = spearmanr(g[a], g[b]); return rho, p
from scipy.stats import rankdata
def partial(y, x, z):  # partial Spearman of y~x controlling z (rank-residual)
    ry, rx, rz = rankdata(g[y]), rankdata(g[x]), rankdata(g[z])
    by = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    bx = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    return pearsonr(by, bx)

print("\n=== correlations over %d unique CDR3s ===" % len(g))
print("  -- the model-independent germline-distance axes vs the deficit --")
for x in ["gdist", "gdist_vdj", "gdist_vdjF"]:
    rho, p = sp(x, "deficit"); pr = partial("deficit", x, "L")
    print(f"  {x:11s} vs deficit  rho={rho:+.3f} p={p:.2e}   partial|length rho={pr[0]:+.3f} p={pr[1]:.2e}")
print("  -- reference: the model-internal axis --")
rho, p = sp("ppl", "deficit"); print(f"  {'ppl':11s} vs deficit  rho={rho:+.3f} p={p:.2e}")
print("  -- length confound + axis agreement --")
for x in ["gdist", "gdist_vdj", "gdist_vdjF", "ppl"]:
    rho, p = sp(x, "L"); print(f"  {x:11s} vs length   rho={rho:+.3f} p={p:.2e}")
rho, p = sp("deficit", "L"); print(f"  {'deficit':11s} vs length   rho={rho:+.3f} p={p:.2e}")
rho, p = sp("gdist_vdj", "ppl"); print(f"  gdist_vdj   vs ppl      rho={rho:+.3f} p={p:.2e}  (do the two OOD axes agree?)")

# tertiles of the assignment-free V-D-J axis (parallel to the ppl tertile table)
g["bin"] = pd.qcut(g["gdist_vdj"], 3, labels=["low(germline-like)", "medium", "high(junctional)"])
print("\n=== by germline V-D-J distance tertile ===")
print(g.groupby("bin", observed=True)[["gdist_vdj", "esm", "blosum", "deficit", "L", "ppl"]].mean().round(3).to_string())

out = ROOT / "outputs/reports/ood_germline_distance_perCDR3.csv"
g.to_csv(out, index=False)
print("\nsaved ->", out)
