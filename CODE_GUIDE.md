# 368 — External Code-Review Guide (final canonical state)

**Read this BEFORE `367_results_provenance_audit.md`.** Doc 367 is a chronological audit *log*
with mid-stream self-corrections (e.g. §14 raised an alarm that §15 retracted); reading it
top-to-bottom can mislead. This doc is the **final, non-chronological canonical state** for a
reviewer inspecting the code. Date: 2026-06-11. All retrieval R@1 = order-independent
**expected-R@1** (`scripts/benchmark/evaluate.py::retrieval_metrics`), 20 seeds, clone-aware split.

---

## 0. Canonical conventions (what "correct" means here)

| Aspect | Canonical | Stale / wrong (ignore) |
|---|---|---|
| BCR input | `raw/iedb/bcr_singlechain_vh.tsv` | `bcr_full_single_header*.tsv` |
| BCR/McPAS results tree | `outputs/phase3_filtered_tie_v2/` (post-`is_legit`-filter) | `outputs/phase3_grand_slam_tie_v2/{bcr,mcpas}` (05-28/30, pre-filter) |
| VDJdb/SAbDab tree | `outputs/phase3_grand_slam_tie_v2/{tcr,sabdab}` (filter is a no-op here) | — |
| Metric | expected-R@1 (`retrieval_metrics`, `exp_recall@1` / `pq_recall@1`) | single-pick `ranks==0` (Levenshtein L1 off by ~6pp) |
| Legitimacy filter | strict 20-AA `is_legit_sequence` in `build_pilot_slice` | — |
| Offline embeddings | run with `OFFLINE_EMBED_STRICT=1` (fail-hard on cache miss) | silent zero-vector |
| Aggregation | `nested_bootstrap.py --bootstrap-seed 42` (seeded, reproducible) | unseeded (~±0.007 MC noise) |

Three BCR trees and how they relate (367 §24): `phase3_grand_slam_tie_v2/bcr` = PRE-filter (stale);
`phase3_filtered_tie_v2/bcr` = POST-filter (canonical, what the manuscript + S9 use);
`phase3_grand_slam_postfilter/bcr` = an independent 2026-06-11 regen that reproduces
`phase3_filtered_tie_v2` **bit-identically** (cross-check).

---

## 1. Core library (review these first — everything depends on them)

- `scripts/benchmark/evaluate.py` — `retrieval_metrics` + `_expected_one`: the expected-R@1 under
  uniform tie resolution. **The single most important function to review.**
- `scripts/benchmark/split.py` — `clone_aware_split_fast`: clone-aware split (greedy CDR3-similarity
  clustering; index remap after label-collapse, see docstring contract).
- `scripts/benchmark/data.py` — `build_pilot_slice` (canonical slice; applies `is_legit_sequence`),
  `is_legit_sequence`, `clean_sequence` (note: `clean_sequence` is format-normalizing only — it does
  NOT filter; `is_legit_sequence` does the strict 20-AA filtering).
- `scripts/benchmark/embeddings.py` — `OfflineEmbedder` (honour `OFFLINE_EMBED_STRICT`), `get_embedder`.
- `scripts/benchmark/run.py` — orchestrates a (dataset, level, model, seed, split-strategy) run.

## 2. Canonical artifact → script map (what produced each manuscript number)

| Manuscript artifact | Canonical script | Source tree / file |
|---|---|---|
| Table 1, Table S1 (main grid L1/L4) | **`analysis/build_master_nested_ci.py`** (two-tree wrapper around `nested_bootstrap.py`) | `outputs/reports/master_nested_ci_20seeds_postfilter.csv` = BCR/McPAS from `phase3_filtered_tie_v2` (post-filter) + TCR/SAbDab from `phase3_grand_slam_tie_v2`. ⚠️ The older `grand_slam_tie_v2_nested_ci_20seeds.csv` is PRE-filter for BCR/McPAS (BCR L1 esm2 0.373 not 0.357) — superseded, do not use. |
| S1 L2/L2.5 (clonotype) | (vjfilter run) | `outputs/reports/l2_vjfilter_filtered_nested_ci.csv` |
| S3 / S1 L3.5 (framework control) | `run.py --levels level3.5` ; `analysis/run_20seeds_l35_padded.py` | filtered_tie_v2 / task282 (both expected-R@1) |
| S6 threshold sensitivity + figS4 | `analysis/run_threshold_sweep_tie_v2.py` | `outputs/reports/threshold_sweep_tie_v2.csv` |
| S7/S8 per-antigen | `analysis/per_antigen_from_canonical_tree.py` | filtered_tie_v2 |
| S9 SHM-stratified | `analysis/run_bcr_shm_stratified.py` | filtered_tie_v2 |
| S10 BLAST baseline | `task285_blast/run_blast_baseline.py` | `outputs/task285_blast_baseline_postfilter/` |
| SCEPTR + TCRdist3 head-to-head (Sec 3.5, Supp Methods §SCEPTR) | `scratch/run_sceptr_embed.py` (ISOLATED env: `pip install sceptr tidytcells` → offline `.npy`) → `scratch/run_sceptr_unified_eval.py` (5 methods — BLOSUM62/Lev/ESM2-150M/SCEPTR/TCRdist3 — scored on the common gene-mappable query subset, MAIN env) | SCEPTR/ESM2-150M offline `.npy` + `mcpas_standardized.tsv`/`vdjdb_full.txt` → `outputs/reports/sceptr_unified/*.csv`. Note: numbers here are on the common subset (e.g. VDJdb BLOSUM 0.311) and differ from the canonical full-set Table S1 (0.375); the eval needs `tcrdist3` (main env) but the embedding step needs the separate SCEPTR env. |
| S11 V-gene sharing rates | (sharing summary) | `outputs/reports/vgene_sharing_summary_20seed.csv` |
| S11 L4 R@1 + two-stage oracle + V-baseline | `analysis/s11_l4_vgene_postfilter.py` | filtered_tie_v2 |
| HIV LOO memorization (Note S2.3) | `task268a_hiv_loo.py` | build_pilot_slice + bNAb refs |
| Pooling controls (Note S2.2, figS2) | `analysis/cls_pooling_eval.py` | online ESM2 (verified == offline, cosine 1.0) |
| Germline-reversion test (Note S2.5, figS9) | `scratch/build_germline_framework_bcr.py` (builds germ-fw/germ-cdr cols) → `scratch/run_germfw_experiment.py` (BCR) / `scratch/run_germfw_sabdab.py` (SAbDab); `scratch/verify_germfw.py` (validity/leakage); `scratch/figS9_germfw_decomposition.py` (plot) | online ESM2 + `imgt/IGV.fasta` (IMGT IGHV germline) |
| OOD CDR3 + germline residual (Note S2.6, figS10) | `scratch/ood_cdr3_experiment.py` (ESM2 pseudo-perplexity vs per-query PLM−BLOSUM deficit) → `outputs/reports/ood_cdr3_perquery.csv` → `scripts/figS10_ood.py` (plot); `scratch/germline_residual_experiment.py` (somatic residual, HIV localization) | online ESM2 (EsmForMaskedLM for PPL) |
| OOD model-independent control (Note S2.6 "Model-independent check") | `scratch/ood_germline_distance.py` (germline V–J-join distance per CDR3 from IMGT IGHV/IGHJ; correlates with deficit/ppl) → `outputs/reports/ood_germline_distance_perCDR3.csv`. Result: does NOT reproduce gradient (ρ=+0.03, length-dominated) → ppl effect is compositional, not junction-size | imgt/IGV.fasta + IGJ.fasta (no model) |
| Fig 1 (clone-vs-random inflation) | `fig1_clone_vs_random_corrected.py` | phase3_grand_slam_postfilter/bcr_clone_vs_random |
| Fig 2 (design + BCR L1/L4 bars) | `figure1_2panel.py` | nested CI |
| Fig 3 (BCR ladder) | `fig3_bcr_ladder_corrected.py` | (hardcoded post-filter nested values) |
| figS1 (V-gene agreement) | `figS1_vgene_agreement_corrected.py` | |
| figS3 (CDR3-length-stratified) | `analysis/figS3_length_postfilter.py` (data) + `figS3_length_corrected.py` (plot) | filtered slices |
| figS5 (CDR3 AA composition vs UniProt) | `figS5_composition.py` | descriptive control, no embeddings (extracted from `scratch/recompute_supp_figs.py`) |
| figS6 (McPAS scaling) | `figS6_mcpas_scaling.py` | |
| figS7 (per-antigen gap) | `figS9_per_antigen_corrected.py` (writes figS7.pdf) | |
| figS8 (L3.5 trajectory) | `figS10_l35_corrected.py` (writes figS8.pdf) | |

> Figure-script naming drift: `figS9_…`→figS7.pdf and `figS10_…`→figS8.pdf (manuscript renumbered
> S1–S8; generator filenames kept old numbers). The output filename in each script is authoritative.

## 3. Superseded / non-canonical (do NOT use for paper numbers)

Marked in-file with a `DEPRECATED / NON-CANONICAL` header:
`antigen_stratification.py`, `run_20seeds_l35_ablation.py`, `run_20seeds_l35_levenshtein.py`,
`run_20seeds_l35_padded_levenshtein.py`, `run_l4_threshold_sweep.py`, `regen_table_s1.py`,
`fig2_clone_vs_random_corrected.py`, `run_20seeds_bcr_pooling.py` (retrieval bug → use
cls_pooling_eval.py), and the old `m3_macro_recall.py` / `m4_run_20_seeds.py` / `m5_run_highparam_20seeds.py`.

**General rule for the reviewer:** any script NOT in the §1–§2 tables above is exploratory /
historical (diagnostics `m1`–`m10`, `task2xx_*` one-offs, `legacy/`, `preprocessing/` audits). They
may read the old input or use `ranks==0`; they do **not** feed the manuscript. ~46 scripts still
reference `bcr_full_single_header*` (mostly in comments or in these exploratory files); the
manuscript-feeding defaults (`run.py`, `data.py`) and the §2 drivers are all canonical.

## 4. Reviewer checklist (concrete things to verify)

1. `retrieval_metrics::_expected_one` correctly computes expected R@1/R@5/MRR under uniform tie
   resolution (the c_top/t_top top-tie-group logic + the first-same-label tie-group expectation).
2. `clone_aware_split_fast` produces train/test with no shared clone (CDR3 similarity ≥ threshold),
   and the post-label-collapse index remap returns iloc-safe positions (docstring contract).
3. `build_pilot_slice` applies `is_legit_sequence` to the level-specific sequence before sampling.
4. Each §2 driver reads the canonical input/tree (filtered_tie_v2 for BCR/McPAS) and expected-R@1.
5. Reproduce end-to-end with `OFFLINE_EMBED_STRICT=1`. Easiest entry point: **`reproduction/run_reproduction_canonical.py`** —
   self-contained (alignment only, no embeddings), reproduces BCR Lev/BLOSUM L1/L4 + checks
   0 within-label clone leak; expected **PASS** (verified, Δ=0.0000). Or S6 via
   `run_threshold_sweep_tie_v2.py` → matches `threshold_sweep_tie_v2.csv` bit-exact.
6. Spot-check that figure-script hardcoded values match their source CSV/tree (367 §25 did this).

## 5. Known limitations honestly stated in the manuscript

L3.5 step does not by itself isolate the framework (→ pooling control); HIV per-antigen signal rests
largely on a single antigen; SHM strata pool non-independent instances (direction only); TCR
full-length is germline-reconstructed (directional); the L4 germline-shortcut is an association, not
a proven causal mechanism. These are stated in §3.3/§3.4/Note S2.4 and should remain.
