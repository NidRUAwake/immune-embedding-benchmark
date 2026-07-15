# Reproduction package

`run_reproduction_canonical.py` reproduces the post-`is_legit`-filter, expected-R@1 numbers the
manuscript reports, end-to-end, from the canonical pipeline.

## `run_reproduction_canonical.py` — canonical, self-contained

```bash
conda env create -f ../environment.yml && conda activate embedding_benchmark_v1
export OFFLINE_EMBED_STRICT=1                 # set automatically by the script too
python run_reproduction_canonical.py          # ~a few minutes (20 seeds, alignment only)
# faster smoke: python run_reproduction_canonical.py --seeds 5
```

**What it does** — runs the BCR alignment methods (Levenshtein, BLOSUM62) at L1 and L4 end-to-end
through the SAME canonical pipeline as the manuscript: `build_pilot_slice` (strict-20AA `is_legit`
filter) → `clone_aware_split_fast` → order-independent expected-R@1 (`retrieval_metrics`).
Alignment needs no precomputed embeddings (parasail only), so this is fully self-contained.

**What it checks (PASS/FAIL):**
1. The clone-aware split is **deterministic** and has **zero within-label clone leakage** — the
   paper's central methodological claim.
2. 20-seed per-seed-mean expected-R@1 for Lev/BLOSUM at BCR L1 & L4 matches the committed
   canonical values (`expected/expected_bcr_canonical.csv`) within tolerance.

Expected result: **PASS** (verified — split deterministic, 0/127 within-label leak, all 4
alignment values reproduce to Δ=0.0000).

**Inputs** — uses `../raw/iedb/bcr_singlechain_vh.tsv` if present, else the bundled
`data/bcr_singlechain_vh.tsv` (the canonical post-ANARCI single-chain VH file).

## What this does NOT cover (and how to do it)

This harness validates the pipeline + the headline *alignment* numbers. The **PLM rows**
(ESM2/AntiBERTy/AbLang/ESM-C), the **nested-bootstrap CIs**, and the **full Table 1 / Table S1**
additionally require the precomputed embedding cache (`outputs/embeddings/`, large) and the other
dataset inputs. To reproduce those:

1. Obtain raw data + embeddings (see `../REPRODUCIBILITY.md` Data Availability / `download_scripts/`).
2. Run with `OFFLINE_EMBED_STRICT=1` so any cache miss fails hard.
3. Build the master table with the **two-tree** aggregator:
   `python ../scripts/analysis/build_master_nested_ci.py`
   → `outputs/reports/master_nested_ci_20seeds_postfilter.csv` (matches Table 1/S1; BCR/McPAS from
   the post-filter `phase3_filtered_tie_v2` tree, VDJdb/SAbDab from `phase3_grand_slam_tie_v2`).

The authoritative artifact→script→tree map and a reviewer checklist are in
**`../discussions/368_external_code_review_guide.md`**.
