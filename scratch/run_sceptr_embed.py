#!/usr/bin/env python3
"""Generate SCEPTR (Nagano et al., 2025) embeddings for a TCR table -> offline .npy + metadata.

SCEPTR's published package cannot co-install with the main `embedding_benchmark_v1` environment
(incompatible torch / numpy / CUDA pins), so this script is run in an ISOLATED environment whose
ONLY job is to produce the embedding cache:

    python -m venv /tmp/sceptr_env && /tmp/sceptr_env/bin/pip install sceptr tidytcells pandas numpy
    /tmp/sceptr_env/bin/python scratch/run_sceptr_embed.py <input.csv> <out.npy> <out_meta.json> <dataset>

Input CSV columns (any missing are created empty): TRAV, CDR3A, TRAJ, TRBV, CDR3B, TRBJ, key.
`key` is the per-row identifier written to metadata['sequence_order'] so the downstream offline
scorer (scratch/run_sceptr_unified_eval.py) can align embeddings to slice rows by sequence string.

Gene names are standardized with tidytcells (subgroup-only calls such as TRBV4 are rescued to a
representative gene; unresolvable V/J are imputed from the per-chain mode). We use the paired
`sceptr.variant.default()` model for both databases: VDJdb mixes beta-chain and alpha-only level-1
records and only the default variant embeds all of them in one 64-d space; on McPAS the default and
beta-specific variants agree to within 0.001 R@1, so the choice is immaterial.

The resulting .npy files (mcpas_level1_sceptr_mean.npy, tcr_level1_sceptr_mean.npy,
tcr_level1_paired_sceptr_mean.npy) are then scored in the MAIN environment by the same retrieval
pipeline as every other method. See CODE_GUIDE.md (SCEPTR) and Supplementary Methods.
"""
import sys, pandas as pd, numpy as np, json, re, sceptr, tidytcells as tt
inp_csv, out_npy, out_meta, dataset = sys.argv[1:5]
df = pd.read_csv(inp_csv)
for c in ['TRAV', 'CDR3A', 'TRAJ', 'TRBV', 'CDR3B', 'TRBJ']:
    if c not in df:
        df[c] = None
AA = set('ACDEFGHIKLMNPQRSTVWY')

def std_gene(x, pref):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x)
    r = tt.tr.standardize(s, enforce_functional=True, precision='gene', suppress_warnings=True)
    if r and r.startswith(pref):
        return r
    m = re.match(r'^(TR[AB][VJ]\d+)$', s)
    if m:
        for n in range(1, 13):
            c = tt.tr.standardize(f'{m.group(1)}-{n}', enforce_functional=True, precision='gene', suppress_warnings=True)
            if c and c.startswith(pref):
                return c
    r2 = tt.tr.standardize(re.sub(r'[*:/].*$', '', s), enforce_functional=True, precision='gene', suppress_warnings=True)
    return r2 if (r2 and r2.startswith(pref)) else None

def clean_cdr3(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    core = ''.join(c for c in str(s).upper() if c in AA)
    if not core:
        return None
    if core[0] != 'C':
        core = 'C' + core
    if core[-1] not in 'FW':
        core = core + 'F'
    return tt.junction.standardize(core, suppress_warnings=True) or core

for vg, jg, cg, pv, pj in [('TRBV', 'TRBJ', 'CDR3B', 'TRBV', 'TRBJ'), ('TRAV', 'TRAJ', 'CDR3A', 'TRAV', 'TRAJ')]:
    df[vg] = df[vg].map(lambda x: std_gene(x, pv))
    df[jg] = df[jg].map(lambda x: std_gene(x, pj))
    df[cg] = df[cg].map(clean_cdr3)
inp = df[['TRAV', 'CDR3A', 'TRAJ', 'TRBV', 'CDR3B', 'TRBJ']].copy()

def impute(mask, col):
    m = inp[col].dropna().mode()
    if len(m):
        inp.loc[mask & inp[col].isna(), col] = m[0]

bm = inp['CDR3B'].notna()
am = inp['CDR3A'].notna()
impute(bm, 'TRBV'); impute(bm, 'TRBJ'); impute(am, 'TRAV'); impute(am, 'TRAJ')
# rows with no complete chain -> placeholder beta (kept only so row order matches `key`)
nochain = ~bm & ~am
inp.loc[nochain, 'CDR3B'] = 'CASSLGSSYEQYF'
impute(nochain, 'TRBV'); impute(nochain, 'TRBJ')
emb = sceptr.variant.default().calc_vector_representations(inp)
np.save(out_npy, emb.astype(np.float32))
json.dump({'dataset': dataset, 'level': 'level1', 'model': 'sceptr', 'resolved_model': 'default',
           'pooling': 'mean', 'n_sequences': len(df), 'embedding_dim': int(emb.shape[1]),
           'sequence_order': df['key'].tolist()}, open(out_meta, 'w'))
print(dataset, 'emb', emb.shape, 'nochain', int(nochain.sum()))
