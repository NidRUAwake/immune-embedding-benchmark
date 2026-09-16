# Clone-aware immune-receptor embedding benchmark — code

Benchmarking **protein language models against sequence alignment** for
antigen-specificity retrieval of B- and T-cell receptors, under **clone-aware**
data splitting. This repository holds only the **canonical scripts that produce
the manuscript numbers** — begin at [`CODE_GUIDE.md`](CODE_GUIDE.md).

## Quick start

```bash
conda env create -f environment.yml && conda activate embedding_benchmark_v1
export OFFLINE_EMBED_STRICT=1

python reproduction/run_reproduction_canonical.py             # self-contained PASS check (no embeddings needed)
python reproduction/run_reproduction_canonical.py --seeds 20  # strict 20-seed numeric check
```

Run every command **from the repository root** (drivers add `scripts/` to the path relative to the working directory).

## Repository map

| Path | Contents |
|---|---|
| [**`CODE_GUIDE.md`**](CODE_GUIDE.md) | **Start here** — canonical artifact → script → data-tree map + reviewer checklist |
| `reproduction/` | Self-contained reproduction harness (verified PASS) |
| `scripts/benchmark/` | Core library — clone-aware `split.py`, `data.py`, `embeddings.py`, `evaluate.py` (expected-R@1), `run.py` |
| `scripts/analysis/`, `scripts/task*`, `scratch/` | Canonical drivers, one per manuscript artifact |
| `scripts/analysis/revision_models/` | Revision models — TCR2vec/CDR3vec, AntiBERTa2/IgBert, CoV-AbDab task ([details](scripts/analysis/revision_models/README.md)) |
| `scripts/fig*.py` | Figure generators |
| `download_scripts/` | Fetch raw databases (IEDB / VDJdb / SAbDab / McPAS) |
| `results/Table_S1_Full_CI.csv` | Released master results table |
| `REPRODUCIBILITY.md` | Environment, per-number reproduction, data availability |

## Environments

Everything runs in the **main** conda env, except two *embedding* steps whose
dependencies conflict with it and run in isolated envs. **All scoring uses the main env** —
so you can skip the isolated envs entirely if you use the released embedding `.npy` files.

| Environment | File | Used for |
|---|---|---|
| **Main** (`embedding_benchmark_v1`) | `environment.yml` / `.lock.yml` | Everything: data prep, alignment, ESM2, tcrdist3, **all scoring** |
| SCEPTR embedding | `environment_sceptr.txt` | Only `scratch/run_sceptr_embed.py` (pins torch 2.12 / numpy 2.x) |
| TCR2vec embedding | `environment.revision.yml` | Only `scripts/analysis/revision_models/*_embed.py` (pins torch 1.13 + tape-proteins) |

## Data (not bundled)

Raw inputs and the precomputed embedding cache are obtained separately (size / redistribution):

- **Raw databases** — `download_scripts/` (+ preprocessing)
- **Embeddings** — the Zenodo data package (see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md))

Always keep `OFFLINE_EMBED_STRICT=1` set, so a missing embedding **fails hard** instead of silently recomputing.

## Full results table (Table 1 / S1)

```bash
python scripts/analysis/build_master_nested_ci.py   # -> outputs/reports/master_nested_ci_20seeds_postfilter.csv
```

Two-tree aggregation: BCR/McPAS from the post-`is_legit`-filter tree; VDJdb/SAbDab from the grand-slam tree.

