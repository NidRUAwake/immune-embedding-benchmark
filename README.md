# Clone-aware immune-receptor embedding benchmark — code

Source code for the benchmark comparing protein language models against sequence alignment for
antigen-specificity retrieval of immune receptors, under clone-aware data splitting. This package
contains only the **canonical** scripts that produce the manuscript numbers (see `CODE_GUIDE.md`).

## Quick start
```bash
conda env create -f environment.yml && conda activate embedding_benchmark_v1
export OFFLINE_EMBED_STRICT=1
python reproduction/run_reproduction_canonical.py            # self-contained PASS check (no embeddings needed)
# strict 20-seed numeric check: python reproduction/run_reproduction_canonical.py --seeds 20
```
Run commands from the package root (some drivers add `scripts/` to the path relative to CWD).

## Environments
- **Main** (`environment.yml` / `environment.lock.yml`, conda env `embedding_benchmark_v1`): runs
  everything — data prep, all alignment/PLM methods, ESM2, tcrdist3, and even the SCEPTR *scoring*
  step (`scratch/run_sceptr_unified_eval.py` only loads SCEPTR `.npy` and uses tcrdist3, both here).
- **SCEPTR embedding env** (`environment_sceptr.txt`, isolated pip venv): needed ONLY to regenerate
  SCEPTR embeddings (`scratch/run_sceptr_embed.py`). SCEPTR pins torch 2.12 + numpy 2.x, which are
  incompatible with the main env, so it is kept separate. Skip it if you use the released `.npy`.

## Layout
- `scripts/benchmark/` — core library: `data.py` (slices + strict-20AA `is_legit` filter),
  `split.py` (clone-aware split), `embeddings.py`, `evaluate.py` (order-independent expected-R@1), `run.py`.
- `scripts/analysis/`, `scripts/task*`, `scratch/*.py` — the canonical drivers (one per manuscript artifact).
- `scratch/run_sceptr_embed.py` (+ `environment_sceptr.txt`) → `scratch/run_sceptr_unified_eval.py` — SCEPTR (Sec 3.5).
- `scripts/fig*.py` — figure generators.
- `reproduction/` — canonical reproduction harness (verified PASS).
- **`CODE_GUIDE.md` — START HERE**: canonical artifact→script→tree map + reviewer checklist.
- `REPRODUCIBILITY.md` — environment, per-number reproduction, Data Availability.
- `download_scripts/` — fetch raw databases (IEDB/VDJdb/SAbDab/McPAS).
- `results/Table_S1_Full_CI.csv` — released master results table.

## Data (not bundled)
Raw inputs and the precomputed embedding cache are obtained separately (size / redistribution):
raw data via `download_scripts/` (+ preprocessing); embeddings via the Zenodo data package
(see REPRODUCIBILITY.md). Always run with `OFFLINE_EMBED_STRICT=1` so a missing embedding fails hard.

## Full results table (Table 1 / S1)
Two-tree aggregation — BCR/McPAS from the post-`is_legit`-filter tree, VDJdb/SAbDab from the
grand-slam tree: `python scripts/analysis/build_master_nested_ci.py`
→ `outputs/reports/master_nested_ci_20seeds_postfilter.csv`.
