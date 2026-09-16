# Revision analyses: receptor-specific PLMs and a BCR-native functional task

Scripts added during peer-review revision to evaluate receptor-specific and
recent protein language models (PLMs) alongside the alignment/ESM2 baselines of
the main benchmark, and to add a functional BCR task. All use the **same
clone-aware split, 20 seeds, and expected-Recall@1 metric** as the main
pipeline (`scripts/benchmark/`), scored offline from precomputed embeddings.

Run every script from the **repository root** (they resolve internal modules via
`sys.path.insert(0, "scripts")` and read/write repo-relative paths).

## What each script covers

| Analysis | Embed | Score |
|---|---|---|
| TCR2vec (full-length V–CDR3–J) | `tcr2vec_embed.py` | `tcr2vec_score.py` |
| CDR3vec (CDR3-only) | `cdr3vec_embed.py` | `tcr2vec_score.py` |
| 128-d small variants (TCR2vec_small, CDR3vec_small) | `small_embed.py` | `tcr2vec_score.py` |
| TCRdb-pretrained TCR2vec | `tcrdb_embed.py` | `tcr2vec_score.py` |
| Paired α+β TCR2vec/CDR3vec | `paired_extract.py` → `paired_embed.py` | `paired_score.py` |
| TCR2vec sanity check | `tcr2vec_validate.py` | — |
| AntiBERTa2 / IgBert BCR ladder | `bcr_embed_focused.py` | `bcr_score_driver.py` |
| CoV-AbDab SARS-CoV-2 neutralization | `bcr_covabdab_build.py` → `bcr_covabdab_embed.py` | `bcr_covabdab_score.py` |

## Prerequisites (not bundled — external models/data)

**TCR2vec package** (the model code — provides `tcr2vec.model`, `tcr2vec.utils`,
`tcr2vec.cdr3_to_full_seq`, which the `*_embed.py` scripts import). Not on PyPI;
install into the `environment.revision.yml` env from the authors' repository:
`git clone https://github.com/jiangdada1221/TCR2vec && pip install -e ./TCR2vec`
(or add the clone to `PYTHONPATH`).

**Model weights** (`$TCR2VEC_WEIGHTS`, default `weights/tcr2vec_weights/`).
Obtain the TCR2vec/CDR3vec checkpoints from the TCR2vec release
(Jiang et al., 2023) and lay them out as:

```
$TCR2VEC_WEIGHTS/
  tcr2vec_full_dir/TCR2vec_120
  cdr3vec_dir/CDR3vec_120
  tcr2vec_small_dir/TCR2vec_small_128
  cdr3vec_small_dir/CDR3vec_small_128
  tcr2vec_tcrdb_dir/tcrdb_1.0
```

**Hugging Face models** (downloaded on first run into `$HF_HOME`, default
`hf_cache/`): `alchemab/antiberta2` (requires the `rjieba` package),
`Exscientia/IgBert`, `facebook/esm2_t30_150M_UR50D`. SCEPTR (Nagano et al., 2025)
is run in its own environment; see `scripts/analysis/` SCEPTR helpers.

**Data**: CoV-AbDab (`$COVABDAB_CSV`, default `data/covabdab.csv`) — the
2024-02-08 CoV-AbDab release (Raybould et al., 2021). Main-benchmark inputs are
read from the repository's canonical data paths, identical to `scripts/benchmark/`.

**Environments.** Two models need dependencies that conflict with the main
`environment.yml` and must run in **isolated** environments; the rest run in the
main env.

- **TCR2vec / CDR3vec** (`tcr2vec_embed.py`, `cdr3vec_embed.py`, `small_embed.py`,
  `tcrdb_embed.py`, `paired_embed.py`, `tcr2vec_validate.py`): isolated env with
  `tape-proteins==0.5`, `torch==1.13.1`, `numpy==1.23.5` (the TAPE IUPAC
  tokenizer; incompatible with the main `torch==2.0.1` / `numpy==1.26.4` pins).
  Scoring (`*_score.py`) reads the resulting offline embeddings and runs in the
  **main** env.
- **AntiBERTa2** (`bcr_embed_focused.py`): the `alchemab/antiberta2` RoFormer
  tokenizer imports `rjieba`. `rjieba` does **not** conflict with the main env,
  so it is added to `environment.yml`; AntiBERTa2 and IgBert
  (`Exscientia/IgBert`, a plain BERT needing no extra package) then run in the
  main env.
- **SCEPTR**: run in its own environment per the SCEPTR package.

All other dependencies match the project `environment.yml`.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `TCR2VEC_WEIGHTS` | `weights/tcr2vec_weights` | TCR2vec/CDR3vec checkpoint root |
| `REVISION_WORKDIR` | `revision_workdir` | scratch dir for intermediate `.npy`/metadata |
| `COVABDAB_CSV` | `data/covabdab.csv` | CoV-AbDab release CSV |
| `HF_HOME` | `hf_cache` | Hugging Face model/download cache |

Outputs (20-seed result CSVs) are written to `outputs/reports/`
(`covabdab_neutralization_20seed.csv`, `tcr2vec_{mcpas,tcr}_20seed.csv`,
`tcr2vec_paired_vdjdb_20seed.csv`), which back the revision's Supplementary
Table S15/S16 and Note S2.7.
