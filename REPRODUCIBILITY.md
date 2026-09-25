# Reproducibility guide

Published values are **20-seed nested-bootstrap aggregates** of order-independent
expected Recall@1. This guide explains how to rebuild them.

- **Paper:** "A germline shortcut in protein language model retrieval of adaptive immune receptors"
- **Code:** https://github.com/NidRUAwake/immune-embedding-benchmark
- **Start here for the artifact map:** [`CODE_GUIDE.md`](CODE_GUIDE.md)

## Read this first

Four rules decide whether your numbers match ours.

1. **Run from the repository root.** Several scripts resolve paths relative to the
   working directory. Running from a subdirectory can silently change results.
2. **Always set `OFFLINE_EMBED_STRICT=1`.** Without it, a missing embedding falls back
   to a zero vector and quietly lowers Recall@1. With it, the run fails immediately.
3. **Always pass `--export-predictions`.** The aggregator reads the per-query
   `*_predictions.csv` files. Without them it falls back to an order-dependent metric
   that does not match the published tables.
4. **All Recall@1 values are expected Recall@1**, which splits credit across tied
   top-1 hits. This is not the same as counting `rank == 0`; for Levenshtein at the
   CDR3 level the two differ by about 6 percentage points.

## Quick start

```bash
git clone https://github.com/NidRUAwake/immune-embedding-benchmark.git
cd immune-embedding-benchmark

conda env create -f environment.yml          # use this file, not environment.lock.yml
conda activate embedding_benchmark_v1

export OFFLINE_EMBED_STRICT=1
python reproduction/run_reproduction_canonical.py
```

The harness is self-contained: it bundles its own BCR input, needs no GPU and no
precomputed embeddings, and checks two things.

1. The clone-aware split leaks no clone between train and test.
2. The 20-seed mean expected Recall@1 for Levenshtein and BLOSUM62, at the CDR3 and
   full-variable-domain levels, matches the committed canonical values.

Use `--seeds N` to run fewer seeds for a faster smoke test. Only the default
(`--seeds 20`) reproduces the published numbers.

## Environment

Install with `environment.yml`. **Do not install from `environment.lock.yml`**: it is
a forensic record of the original machine's base environment, includes unrelated
packages, and its pins conflict, so `conda env create` fails with
`ResolutionImpossible`. Treat it as a reference only.

Core versions:

| Package | Version | Used for |
|---|---|---|
| Python | 3.10.13 | — |
| PyTorch | 2.0.1 | PLM inference |
| transformers | 4.31.0 | loads every PLM (ESM2, AntiBERTy, AbLang, TCR-BERT, AntiBERTa2, IgBert) |
| NumPy | 1.26.4 | — |
| pandas | 2.2.1 | — |
| SciPy | 1.11.1 | statistics |
| scikit-learn | 1.7.2 | linear probe, AUROC |
| statsmodels | 0.14.1 | TOST equivalence tests, Hedges' g |
| parasail | 1.3.4 | BLOSUM62 Smith-Waterman (`sw_striped_16`, gap 10/1) |
| rapidfuzz | 3.6.1 | canonical Levenshtein baseline and clone clustering |
| tcrdist3 | 0.2.2 | TCRdist and BCRdist baselines |
| rjieba | 0.2.1 | tokenizer dependency for AntiBERTa2 |

Two embedding steps need their own environment because their pins conflict with the
main one. **All scoring runs in the main environment**, so you can skip both if you use
the released embedding files.

| Environment | File | Needed only for |
|---|---|---|
| SCEPTR embedding | `environment_sceptr.txt` | `scratch/run_sceptr_embed.py` (torch 2.12, numpy 2.x) |
| TCR2vec embedding | `environment.revision.yml` | `scripts/analysis/revision_models/*_embed.py` (torch 1.13, tape-proteins) |

### External tools (not installed by pip)

| Tool | Version | Needed for |
|---|---|---|
| ANARCI | bioconda 2024.05.21 (or source 1.3) | IMGT numbering, **preprocessing only** |
| NCBI BLAST+ | 2.16.0 | BLAST baseline (Table S10) and the HIV-1 leakage audit |
| HMMER3 | ships with ANARCI | ANARCI backend |

Install ANARCI with `conda install -c bioconda anarci hmmer`. **Do not run
`pip install anarci==1.3`**: that version is not on PyPI, and the version that is
(`2026.2.13.2`) pulls in numpy 2.x and breaks the pinned stack.

### Pretrained weights

`scripts/benchmark/embeddings.py` loads all PLM weights through Hugging Face
transformers, not through the `ablang` or `antiberty` pip packages.

| Alias | Hugging Face weights | Dim |
|---|---|---|
| esm2-150m / 650m / 3b | `facebook/esm2_t30_150M_UR50D` / `t33_650M` / `t36_3B` | 640 / 1280 / 2560 |
| antiberty | `neulab/antiberty` | 512 |
| ablang | `msc-bioinformatics/AbLang` | 768 |
| tcr-bert | `wukevin/tcr-bert` | 768 |
| antiberta2 | `alchemab/antiberta2` | 1024 |
| igbert | `Exscientia/IgBert` | 1024 |

Weights download on first use into `$HF_HOME` (about 8 GB for the full ESM2 family).

Three models are not loaded through Hugging Face and need their own setup:

| Model | Where the weights come from | Scripts |
|---|---|---|
| ESM-C-300M | the `esm` package (`ESMC.from_pretrained("esmc_300m")`), in its own environment | `scratch/export_esmc_canonical_sequences.py` -> `scratch/external_task_esmc_canonical/extract_esmc.py` (GPU) -> `scratch/run_esmc_canonical_20seed.py` (scoring, main env) |
| SCEPTR | the `sceptr` pip package, in its own venv | `scratch/run_sceptr_embed.py` -> `scratch/run_sceptr_unified_eval.py` |
| TCR2vec / CDR3vec | checkpoints from the TCR2vec repository | `scripts/analysis/revision_models/` (see its README for the layout) |

All three follow the same pattern: generate embeddings in an isolated environment,
then score them in the main environment like every other model.

## Data

We do not redistribute the raw database dumps. Each database has its own license and
terms, so we point to the original source and the exact version instead.

| Database | License | Version used | Download |
|---|---|---|---|
| IEDB (BCR) | CC-BY-4.0 | downloaded 2026-05-09 | `download_scripts/iedb_download.sh` (manual web export) |
| VDJdb (TCR) | AGPL-3.0 | 2025-12-29 release | `download_scripts/vdjdb_download.sh` |
| SAbDab | CC-BY | downloaded 2026-05-05 | `download_scripts/sabdab_download.sh` (manual) |
| McPAS-TCR | per site terms | standardized 2026-05-19 | `download_scripts/mcpas_download.sh` (manual, Shiny app) |
| CoV-AbDab | free for academic use | 2024-02-08 release | `download_scripts/covabdab_download.sh` |
| IMGT IGHV/IGHJ germline | IMGT academic terms | reference sequences | IMGT/GENE-DB, for the germline-reversion test |

CoV-AbDab supplies the zero-shot BCR-native task (SARS-CoV-2 neutralization
discrimination, Supplementary Note S2.7 and Table S16). The analysis scripts read it
through `$COVABDAB_CSV`, which defaults to `data/covabdab.csv`.

**Two of the five sources are byte-reproducible from their official servers:**

- **VDJdb** publishes pinned release assets; `vdjdb_full.txt` has md5
  `4ab97ea73b42a04afeaf1957d3bf9894`.
- **CoV-AbDab** serves the 2024-02-08 release as a direct CSV; md5
  `4bcbcec3f35bc0cfb72535bbcf3dff08`. Both download scripts check the md5 for you.

IEDB, SAbDab and McPAS-TCR publish only their current release, and McPAS-TCR now
requires clicking through a web app, so an exact byte match with our snapshot is not
guaranteed. For those three, reproduce from the **license-permitted processed inputs**
we distribute and verify them with the published md5 checksums. Because every reported
number is a 20-seed aggregate, small input drift does not change any conclusion.

> **Preprocessing scripts are not included in this package.** It ships the canonical
> benchmark and analysis code only. Converting a raw download into a canonical input
> (`bcr_singlechain_vh.tsv`, `mcpas_standardized.tsv`,
> `sabdab_vj_annotated_paired.csv`) uses the preprocessing code in the full
> development repository. To reproduce the published numbers, start from the
> distributed processed inputs and embeddings.

## Reproducing the published numbers

### Step 1: per-seed runs

Twenty seeds: `[42 + 10*i for i in range(20)]`, i.e. 42, 52, ..., 232. Every run uses
`--test-size 0.2 --clone-threshold 0.95 --min-clones-per-label 2`.

Each database has its own label parameters. Using one set of parameters for all four
will not reproduce the tables.

All four commands share these flags:

```
--levels level1 level2 level3 level4 --models <model> --random-state <seed> \
--test-size 0.2 --clone-threshold 0.95 --export-predictions \
--offline-dir outputs/embeddings
```

Only the dataset selection and label parameters differ:

```bash
# BCR (IEDB): 127 test queries over 10 labels at seed 42
python scripts/benchmark/run.py --include-bcr --human-only \
  --bcr-file raw/iedb/bcr_singlechain_vh.tsv \
  --top-labels 10 --min-label-count 14 \
  <shared flags> \
  --output-dir outputs/phase3_filtered_tie_v2/bcr/<model>/seed_<seed>

# VDJdb (TCR): 199 test queries. "top 15 with at least 50 sequences" resolves to the
# 10 working epitopes used in the paper; do not change these two numbers.
python scripts/benchmark/run.py --include-tcr --human-only \
  --tcr-file raw/vdjdb/vdjdb_full.txt \
  --top-labels 15 --min-label-count 50 \
  <shared flags> \
  --output-dir outputs/phase3_grand_slam_tie_v2/tcr/<model>/seed_<seed>

# SAbDab: 59 test queries. Note the input is the annotated paired file, not the raw
# SAbDab summary.
python scripts/benchmark/run.py --include-sabdab --human-only \
  --sabdab-file outputs/intermediate/sabdab_vj_annotated_paired.csv \
  --top-labels 5 --min-label-count 25 \
  <shared flags> \
  --output-dir outputs/phase3_grand_slam_tie_v2/sabdab/<model>/seed_<seed>

# McPAS-TCR: 199 test queries over 10 labels. Note: top 10, not top 5.
# McPAS goes through the TCR path and does NOT take --human-only, because the
# download script already filtered to human.
python scripts/benchmark/run.py --include-tcr \
  --tcr-file outputs/intermediate/mcpas_standardized.tsv \
  --top-labels 10 --min-label-count 30 \
  <shared flags> \
  --output-dir outputs/phase3_filtered_tie_v2/mcpas/<model>/seed_<seed>
```

Each command above was verified to reproduce the committed seed-42 CDR3 value exactly
(see the anchor table below).

`<model>` is one of `levenshtein`, `blosum62`, `esm2-150m`, `esm2-650m`, `esm2-3b`,
`antiberty`, `ablang`.

**Which output tree.** BCR and McPAS go to `phase3_filtered_tie_v2/`, the tree built
after the strict 20-amino-acid `is_legit` filter. VDJdb and SAbDab go to
`phase3_grand_slam_tie_v2/`, where that filter changes nothing. Mixing the two is the
most common reason for mismatched numbers.

### Step 2: aggregate

```bash
python scripts/analysis/build_master_nested_ci.py
# -> outputs/reports/master_nested_ci_20seeds_postfilter.csv
```

This wrapper reads each database from its correct tree and is what produces Table 1 and
Supplementary Table S1. Calling `nested_bootstrap.py` directly also works, but its
default `--base-dir` is the pre-filter tree, so BCR and McPAS would come out wrong.

> **The aggregate carries about +/- 0.007 of Monte Carlo noise.** The two-level
> resampling in `nested_bootstrap_metrics()` is not seeded, so the third decimal place
> moves between runs (largest for SAbDab, whose per-seed test sets differ most in size).
> This is far below the confidence-interval width (0.04-0.09) and the ~9 point effects
> the paper reports. Do not use byte-identical CSVs as a correctness check; compare
> against the committed aggregate instead.

### Step 3: specific analyses

`CODE_GUIDE.md` maps every figure, table and supplementary note to the script that
produces it, including the clonotype ladder, the paired-chain analyses, the
germline-reversion test, the V-gene oracle, and the revision models
(`scripts/analysis/revision_models/`, see its own README).

## Check your run

Single-seed anchors, expected Recall@1 at the CDR3 level, seed 42:

| Dataset | Method | Test queries | Expected R@1 |
|---|---|---|---|
| BCR (IEDB) | Levenshtein | 127 | 0.3552 |
| BCR (IEDB) | BLOSUM62 | 127 | 0.4232 |
| VDJdb | BLOSUM62 | 199 | 0.3894 |
| McPAS-TCR | BLOSUM62 | 199 | 0.3777 |
| SAbDab | Levenshtein | 59 | 0.5565 |

Twenty-seed aggregates, BCR at the CDR3 level (Table 1 / Table S1):

| Method | Aggregate R@1 |
|---|---|
| Levenshtein | 0.446 [0.407, 0.487] |
| BLOSUM62 | 0.459 [0.417, 0.498] |

Single-seed and aggregate values are different quantities, and they are not expected to
be close. The aggregate is query-weighted across seeds, so for SAbDab in particular it
sits well below any individual seed.

### Alignment reproduces exactly; single-seed PLM values at the CDR3 do not

CDR3 sequences are short, so top-1 ties are common. Alignment scores are integers and
tie naturally; PLM embeddings tie too, because repeated or near-repeated CDR3s get
identical vectors and therefore identical cosine similarity. Expected Recall@1 scores a
tie group deterministically, but *which sequences form the group* is sensitive to tiny
numerical differences.

- **Affected:** the CDR3 level, and partly the clonotype and paratope levels. The full
  variable domain is essentially tie-free and reproduces per seed.
- **Size:** about 0.5 to 4 points per seed, with a random sign, so it cancels across 20
  seeds to under 1 point. `exp_recall@5` stays identical, confirming that only the
  top-1 tie group moves.
- **What this means:** verify **alignment** methods per seed, and verify **PLMs** by
  aggregating the archived predictions and comparing against Table S1. Do not diff a
  single-seed `pilot_results.csv` for a PLM.
- The split itself is fully deterministic. Train/test membership is identical for a
  given seed, and the harness verifies this with a checksum.

## Determinism

Set these before any run:

```bash
export OFFLINE_EMBED_STRICT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:16:8
```

| Component | How it is made deterministic |
|---|---|
| Clone-aware split | greedy clustering over sequence indices, not dictionary order |
| Ranking | `np.argsort(..., kind='stable')` |
| CDR3 extraction | IMGT numbering via ANARCI |
| Similarity | fixed metrics (rapidfuzz Levenshtein, parasail BLOSUM62, cosine) |
| Embedding | mean pooling over residue tokens |

Different hardware or operating systems can change the last digits by up to about
0.01%. This does not affect any reported conclusion.

## Runtime and hardware

One seed takes roughly 1-2 minutes for an alignment method, 3-5 minutes for ESM2-150M,
and up to 45 minutes for ESM2-3B. A full 20-seed sweep takes about 3-4 hours on a GPU.

Recommended: an NVIDIA A100 or similar (40 GB), 8 cores, 32 GB RAM, 50 GB of disk.
A 16 GB GPU is enough for everything except ESM2-3B. Alignment methods and all scoring
run on CPU.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Recall@1 is several points low across the board | An embedding was missing and silently became a zero vector. Set `OFFLINE_EMBED_STRICT=1` and rerun. |
| Aggregate does not match the tables | Predictions are missing (`--export-predictions`), or BCR/McPAS were read from the pre-filter tree. Use `build_master_nested_ci.py`. |
| BCR test set has 143 queries instead of 127 | `label_aliases.json` was not found, so label aliases were not merged. Run from the repository root. |
| `ResolutionImpossible` during install | You used `environment.lock.yml`. Install from `environment.yml`. |
| `tcrdist3 not installed` | `pip install tcrdist3` (it is in `environment.yml`). |
| CUDA out of memory | Lower `--batch-size`, or use a smaller model. |
| Model download fails | Check the network, set `HF_HOME` to a writable path, or predownload the weights. |

## Reporting problems

Open an issue at
https://github.com/NidRUAwake/immune-embedding-benchmark/issues with the full error
message, your OS / Python / CUDA versions, and the exact command you ran.

---

Following this guide reproduces the 20-seed nested-bootstrap Recall@1 values and their
95% confidence intervals, which are the numbers reported in the paper. Single-seed
CDR3-level values for PLMs are not expected to match byte-for-byte; see the tie
discussion above.
