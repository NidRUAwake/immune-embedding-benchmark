# Revision analyses: receptor-specific PLMs + a BCR-native functional task

Scripts added during peer-review revision to benchmark receptor-specific and recent
protein language models (TCR2vec/CDR3vec, AntiBERTa2, IgBert) and a functional BCR
task (CoV-AbDab neutralization) against the main alignment/ESM2 baselines. Every
script reuses the main pipeline's **clone-aware split, 20 seeds, and expected
Recall@1** (`scripts/benchmark/`). Each analysis is an **embed** step (writes `.npy`)
followed by an offline **score** step.

> **Run from the repository root.** Scripts add `scripts/` to the path
> (`sys.path.insert(0, "scripts")`) and use repo-relative paths.

## Scripts

| Analysis | Embed | Score |
|---|---|---|
| TCR2vec (full-length V–CDR3–J) | `tcr2vec_embed.py` | `tcr2vec_score.py` |
| CDR3vec (CDR3-only) | `cdr3vec_embed.py` | `tcr2vec_score.py` |
| 128-d variants (TCR2vec_small, CDR3vec_small) | `small_embed.py` | `tcr2vec_score.py` |
| TCRdb-pretrained TCR2vec | `tcrdb_embed.py` | `tcr2vec_score.py` |
| Paired α+β TCR2vec/CDR3vec | `paired_extract.py` → `paired_embed.py` | `paired_score.py` |
| AntiBERTa2 / IgBert BCR ladder | `bcr_embed_focused.py` | `bcr_score_driver.py` |
| CoV-AbDab SARS-CoV-2 neutralization | `bcr_covabdab_build.py` → `bcr_covabdab_embed.py` | `bcr_covabdab_score.py` |
| TCR2vec load sanity check | `tcr2vec_validate.py` | — |

## Which environment runs what

Two model families need dependencies that conflict with the main `environment.yml`,
so their **embed** step runs in an isolated env. **Every `*_score.py` runs in the
main env** (it only reads the offline `.npy`).

| Step | Environment | Key dependencies |
|---|---|---|
| TCR2vec / CDR3vec embed | `environment.revision.yml` (isolated) | `tape-proteins==0.5`, `torch==1.13.1`, `numpy==1.23.5`, + the `tcr2vec` package |
| AntiBERTa2 / IgBert embed | main `environment.yml` | `rjieba` (AntiBERTa2's RoFormer tokenizer) |
| SCEPTR embed | its own env (see `scratch/run_sceptr_embed.py`) | — |
| All scoring | main `environment.yml` | reads the offline `.npy` |

## Setup (external models & data — not bundled)

**1 · TCR2vec package** — the model *code* (`tcr2vec.model` / `.utils` /
`.cdr3_to_full_seq`, imported by the `*_embed.py` scripts). Not on PyPI; install
into the isolated env from the authors' repo (or add the clone to `PYTHONPATH`):

```bash
git clone https://github.com/jiangdada1221/TCR2vec && pip install -e ./TCR2vec
```

**2 · Model weights** → `$TCR2VEC_WEIGHTS` (default `weights/tcr2vec_weights/`).
The checkpoints are not on PyPI or Hugging Face. Download them from the links in the
TCR2vec repository README (https://github.com/jiangdada1221/TCR2vec), which hosts
five pretrained models: the full-length `TCR2vec_120`, the CDR3-only `CDR3vec_120`,
their 128-dimensional `_small` counterparts, and a TCR2vec pretrained on TCRdb.
Unzip each one and lay them out as:

```
$TCR2VEC_WEIGHTS/
  tcr2vec_full_dir/TCR2vec_120
  cdr3vec_dir/CDR3vec_120
  tcr2vec_small_dir/TCR2vec_small_128
  cdr3vec_small_dir/CDR3vec_small_128
  tcr2vec_tcrdb_dir/tcrdb_1.0
```

**3 · Hugging Face models** — auto-downloaded to `$HF_HOME` on first run:
`alchemab/antiberta2`, `Exscientia/IgBert`, `facebook/esm2_t30_150M_UR50D`.

**4 · CoV-AbDab data** → `$COVABDAB_CSV` (default `data/covabdab.csv`): the
2024-02-08 CoV-AbDab release (Raybould et al., 2021). Main-benchmark inputs come
from the repo's canonical data paths, same as `scripts/benchmark/`.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `TCR2VEC_WEIGHTS` | `weights/tcr2vec_weights` | TCR2vec/CDR3vec checkpoint root |
| `REVISION_WORKDIR` | `revision_workdir` | scratch dir for intermediate `.npy` + metadata |
| `COVABDAB_CSV` | `data/covabdab.csv` | CoV-AbDab release CSV |
| `HF_HOME` | `hf_cache` | Hugging Face model / download cache |

## Outputs

20-seed result CSVs are written to `outputs/reports/`:

- `tcr2vec_mcpas_20seed.csv`, `tcr2vec_tcr_20seed.csv` — TCR2vec/CDR3vec ladders
- `tcr2vec_paired_vdjdb_20seed.csv` — paired α+β evaluation
- `covabdab_neutralization_20seed.csv` — CoV-AbDab functional task

These back Supplementary **Table S15 / S16** and **Note S2.7**.
