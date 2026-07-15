# REPRODUCIBILITY.md — 复现指南（投稿值 = 20-seed nested-bootstrap 聚合）

**最后更新**: 2026-06-04（下方多数命令为此日期；2026-06-11 post-filter 重构后的权威状态见下方红框）
**项目**: 免疫受体嵌入基准测试 (Immune Embedding Benchmark)
**论文**: "Clone-aware benchmarking points to a germline shortcut in protein language model retrieval of immune receptors"

> [!IMPORTANT]
> **⚠️ 2026-06-11 更新 — 先读这里。** 本指南正文（2026-06-04 版）早于 strict-20AA `is_legit` 过滤
> (2026-06-10) 与随后的 post-filter 重生成。当前权威状态见 `discussions/367_results_provenance_audit.md`（§0 canonical 约定、§24 三树关系、§28 metric 一致性）。要点：
> 1. **BCR/McPAS 主树 = `outputs/phase3_filtered_tie_v2/`（post-`is_legit`-filter）**，**不是** `phase3_grand_slam_tie_v2/`（05-30 pre-filter,stale,勿用）。两者 bit-identical 复现见 §24。
> 2. **手稿 canonical = `manuscript_BiB_v1.tex` + `supplementary_streamlined_v3.tex`**（streamlined_v4+v3 谱系的后续定稿；另有精简分支 `manuscript_OUPbioinformatics_lean_v1.tex`，投稿去向待定）。
> 3. **所有 R@1 = order-independent expected-R@1**（`retrieval_metrics`），**非** 单选 `ranks==0`（Levenshtein 在 L1 会差 ~6pp，见 §28）。
> 4. **复现时务必 `export OFFLINE_EMBED_STRICT=1`**：让 `OfflineEmbedder` 在任何序列未命中缓存时 fail-hard，杜绝零向量静默降级；canonical 运行均 100% 命中。
> 5. post-filter 后 BCR seed42 = n_train/n_test **534/120**（下文 556/120 为 pre-filter）。注：手稿/Table S2 报告的是**跨 20 seed 均值 ≈546/128**（带"≈"），此处 534/120 为 seed42 单值，二者差异是正常 seed 间波动，非矛盾。

---

## ⚠️ 当前权威状态（2026-06-04，先读这一节）

**稿件**：canonical = `outputs/manuscript_submission/manuscript_BiB_v1.tex` + `supplementary_streamlined_v3.tex`（streamlined_v4+v3 谱系的后续定稿；OUP 精简分支 `manuscript_OUPbioinformatics_lean_v1.tex` 待定投稿去向）。
**数值总表**：`submission/Supplementary_Tables/Table_S1_Full_CI.csv`（已排版为 Supplementary Table S1）。它是**多源 validated reports 手维护**的，不等于任何单个 CSV：L1/L3/L4 主网格列——**BCR/McPAS 来自 post-filter 树 `phase3_filtered_tie_v2/`**（VDJdb/SAbDab 不受 filter 影响，仍来自 `phase3_grand_slam_tie_v2/`），均经 `nested_bootstrap.py --bootstrap-seed 42`（expected-R@1 + tie，20 seeds，seeded 可复现）；**L2/L2.5 来自 `l2_vjfilter_filtered_nested_ci.csv`**（post-filter clonotype）；L3.5 post-filter（见 doc 368 §2）；paired/BCRdist/V-gene 各有报告（artifact→脚本映射见 **[368_external_code_review_guide.md](368_external_code_review_guide.md) §2**）。`regen_table_s1.py` 已废弃（会改坏 L2），勿用。
**逐数字溯源**：见 [discussions/manuscript_review_and_traceability.md](discussions/manuscript_review_and_traceability.md) Part 2（每个数值 → 来源文件 → 生成脚本）。

**复现源代码以 `scripts/benchmark/` 为准（开发树）。** `submission/scripts/` 镜像已于 2026-06-04 同步为当前 canonical（含配对 builders + expected-R@1，见 `submission/scripts/README_DRIVERS.md`）；同步前的旧镜像无法复现配对/clonotype 结果，故仍以开发树为最终权威。标签选择默认即"shared labels"（无 `--shared-labels` 这个 flag，只有 `--no-shared-labels` 关闭它）。

### 论文产物 → 驱动脚本 → 输出（关键映射）

> [!WARNING]
> **下表为 2026-06-04 版,已部分过时**:多行指向 post-filter 后**已标 DEPRECATED 的脚本**
> (`antigen_stratification.py`、`run_l4_threshold_sweep.py`、`run_20seeds_l35_ablation.py`、
> `run_20seeds_bcr_pooling.py` 等),且 BCR/McPAS 输出树应为 `phase3_filtered_tie_v2/`(非
> `phase3_grand_slam_tie_v2/`)。**权威的 artifact→脚本→树 映射以
> [`368_external_code_review_guide.md`](368_external_code_review_guide.md) §2 为准**,下表仅作历史参考。

| 产物 | 驱动 | 输出 |
|---|---|---|
| 主表 Table 1 / Supp S1（全库 × L1–L4） | `scripts/benchmark/run.py`（canonical 参数，见下）→ `scripts/analysis/nested_bootstrap.py` | `outputs/phase3_grand_slam_tie_v2/` → `Table_S1_Full_CI.csv` |
| Fig 2 / §3.1 随机-split 虚高 | `scratch/aggregate_split_inflation.py`（聚合 `outputs/phase3_grand_slam_tie_v2/bcr_clone_vs_random/`，20 seeds）。**勿用** `run_task_322C.py`（旧 5-seed 版，输出名 `split_inflation_expectedR1.csv`，doc 327 已 superseded） | `outputs/reports/split_inflation_tie_v2.csv` |
| §3.2 Hedges g | per-seed 计算 | `outputs/reports/hedges_g_effect_sizes_tie_v2.csv` |
| §3.2 linear probe | task269 脚本 | `outputs/task269_results/linear_probe_primary_summary.csv` |
| Fig 3 / §3.3 / S3 BCR ladder + L3.5 | `scripts/analysis/run_20seeds_l35_ablation.py` + `scratch/aggregate_subtrees.py` | `outputs/reports/subtrees_nested_ci.csv` |
| §3.3 / S11 V-gene oracle | `scripts/analysis/d1_vgene_stratified_retrieval.py` | `outputs/reports/vgene_sharing_summary_20seed.csv` |
| §3.3 pooling controls | `scripts/analysis/run_20seeds_bcr_pooling.py` | `outputs/reports/pooling_sensitivity_tie_v2.csv` |
| §3.4 / S7,S8 per-antigen | `scripts/analysis/antigen_stratification.py` · `scripts/figS9_per_antigen_corrected.py` | per-antigen 报告 |
| §3.4 HIV LOO 泄漏审计 | `scripts/task268a_hiv_loo.py` | `outputs/reports/hiv_leakage_hits_20seed.csv` |
| §3.4 / S9 SHM 分层 + within-antigen | `scripts/analysis/run_bcr_shm_stratified.py` + `scratch/within_antigen_shm.py` | `outputs/reports/shm_stratified_l4_tie_v2.csv`, `shm_perquery_bcr_l4.csv` |
| §3.5 / S6 阈值扫描 | `scripts/analysis/run_l4_threshold_sweep.py` + `scripts/analysis/aggregate_threshold_sweep.py`（canonical） | `outputs/reports/threshold_sweep_tie_v2.csv` |
| §3.2 / S10 BLAST L1 | `scripts/task285_blast/run_blast_baseline.py` | `outputs/reports/blast_l1_tie_v2.csv` |
| L2/L2.5 clonotype（单链） | `scratch/run_l2_vjfilter_20seed.py` | `outputs/reports/l2_vjfilter_20seed_nested_ci.csv` |
| Table 2 配对阶梯 + 配对 clonotype | `scratch/run_paired_clonotype_20seed.py` | `outputs/reports/tcr_paired_ladder_v2_nested_ci.csv`, `paired_clonotype_20seed_nested_ci.csv` |
| Table 2 / S5 TCRdist3（真实） | `scratch/run_tcrdist3_paired_20seed.py` | `outputs/reports/tcr_paired_ladder_tcrdist3_nested_ci.csv` |
| §3.5 native-TCR germline 签名 | `scratch/tcr_germline_signature.py` | `outputs/reports/tcr_germline_signature.csv` |
| **S1.3 BCRdist 负对照** | `scratch/run_bcrdist_canonical_20seed.py` | `outputs/reports/bcrdist_canonical_20seed.csv` |
| **S1 SAbDab 配对阶梯** | `scratch/run_sabdab_paired_ladder_canonical_20seed.py` | `outputs/reports/sabdab_paired_ladder_canonical_20seed.csv` |

> 作废脚本（污染，勿用）：`scratch/run_bcrdist_20seed.py`、`scratch/run_sabdab_paired_ladder_20seed.py`（绕过了 build_pilot_slice，见 [discussions/336](discussions/336_bcrdist_sabdab_paired_correction_delivery.md)）。
> 绘图脚本（`scripts/analysis/plot_paper_fig*.py`、`plot_fig*.py`）多数默认读旧树 `outputs/phase3_grand_slam`——**投稿图（`figures/fig1–3.pdf`）已 frozen**，无需重绘；如要重绘需把脚本指向 `tie_v2` 并核对锚点。

### 逐数值复现（四个数据集的精确命令）

通用：`conda activate embedding_benchmark_v1`（从 `environment.yml` 建；勿照抄本机绝对 python 路径——本机默认 `python` 是 base）。20 seeds = `[42 + 10*i for i in range(20)]`；clone 0.95；test_size 0.2；min_clones 2。先生成 embedding（`scripts/generate_embeddings.py` 或 `discussions/` 内 GPU 工单 → `outputs/embeddings/`），再跑下列命令；缺省 `--offline-dir outputs/embeddings` 复用离线 embedding。

**Step 0 — 数据获取（自行下载，见 §数据获取）**：`submission/download_scripts/{iedb,vdjdb,sabdab,mcpas}_download.sh` → 预处理（`scripts/preprocessing/` 等）产出 canonical 输入：`raw/iedb/bcr_singlechain_vh.tsv`、`raw/vdjdb/vdjdb_full.txt`、`outputs/intermediate/mcpas_standardized.tsv`、SAbDab + `outputs/intermediate/sabdab_vj_annotated_paired.csv`。

**Step 1 — 主 L1–L4 网格（Table 1 / Supp S1 核心）**：编排脚本 `scripts/analysis/m8_run_all_20seeds.py` **当前只含 BCR scenario**（TCR/SAbDab/McPAS 的 scenarios 已不在脚本里）——所以四库请用**下面的手动命令块**逐库跑（或自行把四库 scenarios 加回 m8）。下式以一个 seed/一个 model 为例，`<seed>∈{42,52,…,232}`，`<model>∈{levenshtein,blosum62,esm2-150m,esm2-650m,esm2-3b,antiberty,ablang}`：

```bash
# BCR (IEDB)  → seed42 (post-filter): n_train/n_test = 534/120  (pre-filter was 556/120)
python scripts/benchmark/run.py --include-bcr \
  --bcr-file raw/iedb/bcr_singlechain_vh.tsv --top-labels 10 --min-label-count 14 \
  --levels level1 level2 level3 level4 --models <model> --random-state <seed> \
  --human-only --test-size 0.2 --clone-threshold 0.95 --export-predictions \
  --offline-dir outputs/embeddings \
  --output-dir outputs/phase3_grand_slam_tie_v2/bcr/<model>/seed_<seed>

# VDJdb TCR  → seed42: 801/199, BLOSUM-L1=0.3894（已亲验）。用 10 个固定工作 epitope
# （top-15 ∩ 各粒度≥50 的交集，故须用 --forced-labels，不能用 --top-labels/--shared-labels）：
python scripts/benchmark/run.py --include-tcr \
  --tcr-file raw/vdjdb/vdjdb_full.txt --max-per-label-tcr 100 \
  --forced-labels SLLMWITQV,NLVPMVATV,KLGGALQAK,GILGFVFTL,VISNDVCAQV,GLCTLVAML,RAKFKQLL,ELAGIGILTV,AVFDRKSDAK,YLQPRTFLL \
  --levels level1 level2 level3 level4 --models <model> --random-state <seed> \
  --human-only --test-size 0.2 --clone-threshold 0.95 --export-predictions --offline-dir outputs/embeddings \
  --output-dir outputs/phase3_grand_slam_tie_v2/tcr/<model>/seed_<seed>

# SAbDab  → 251/59
python scripts/benchmark/run.py --include-sabdab \
  --top-labels 5 --min-label-count 25 --max-per-label-sabdab 100 \
  --levels level1 level2 level3 level4 --models <model> --random-state <seed> \
  --human-only --test-size 0.2 --clone-threshold 0.95 --export-predictions --offline-dir outputs/embeddings \
  --output-dir outputs/phase3_grand_slam_tie_v2/sabdab/<model>/seed_<seed>

# McPAS  → seed42: 801/199, BLOSUM-L1=0.387 (post-filter; was ~0.393 pre-filter)（恰与 VDJdb 同为 801/199，因同样 cap-100×相近标签数）。
# McPAS 以 standardized.tsv 当作 TCR 输入、无 --human-only。canonical 结果在 tie_v2/mcpas/（由 m8 编排），
# 命令同 VDJdb 但换输入与标签参数：
python scripts/benchmark/run.py --include-tcr \
  --tcr-file outputs/intermediate/mcpas_standardized.tsv --top-labels 5 --min-label-count 30 \
  --max-per-label-tcr 100 --levels level1 level2 level3 level4 \
  --models <model> --random-state <seed> --test-size 0.2 --clone-threshold 0.95 --export-predictions \
  --offline-dir outputs/embeddings \
  --output-dir outputs/phase3_grand_slam_tie_v2/mcpas/<model>/seed_<seed>
```

> **`--export-predictions` 是必选**：`nested_bootstrap.py` 读的是各 seed 的 `*_predictions.csv`（含逐 query `exp_recall@1`），**不读** `pilot_results.csv` 的 list 列。缺 predictions 时聚合会退化为 order-dependent 的 `rank==0`，与 Table S1 的 expected-R@1 口径不一致。四库命令都已带上。
（注：独立变体 `scripts/analysis/run_mcpas_benchmark.py` 默认输出到 `outputs/mcpas_benchmark/`，非 tie_v2 树。）

> **⚠️ L2 / L2.5（clonotype）勿对错锚点**：L2/L2.5 已重定义为 *clonotype 检索*——复用 **level1 的 CDR3 embedding** + 在检索时按 germline 基因过滤（`vj_filter`），所以离线跑 L2 用 `*_level1_*.npy` 是**正确的**；仓库里的 `*_level2_*.npy` / `*_level2.5_*.npy` 是**重定义前的旧输入（V/J 基因名串），已废弃**。**canonical L2/L2.5 数值来自 `scratch/run_l2_vjfilter_20seed.py` → `outputs/reports/l2_vjfilter_20seed_nested_ci.csv`**（已并入 Table S1：BCR L2 ESM2-150M = 0.400），**不是** `tie_v2/{ds}/.../pilot_results.csv` 里的 L2 行——后者是重定义前的旧值（如 BCR L2 = 0.275），勿用作对照锚点。用 `run.py --levels level2 level2.5` 现跑会得到**新**值（与 Table S1 一致）。

**Step 2 — 聚合为投稿值**：
```bash
python scripts/analysis/nested_bootstrap.py \
  --base-dir outputs/phase3_grand_slam_tie_v2 \
  --output-nested outputs/reports/grand_slam_tie_v2_nested_ci_20seeds.csv
# 该 CSV = submission/Supplementary_Tables/Table_S1_Full_CI.csv 的数值来源
```
> ✅ 自 doc 343 起，`nested_bootstrap.py` 的默认 `--base-dir` 已是 `outputs/phase3_grand_slam_tie_v2`，且默认 `--datasets bcr,tcr,sabdab,mcpas`（裸跑即只聚合四库，并打印 `SKIPPED non-canonical subdirs [...]` 列出 `bcr_backup_*`/`bcr_clone_vs_random`/`bcr_l35`/`*_l3excl` 等被跳过的子树）。要把这些非-canonical 子树也算进去才需 `--all-datasets`。核对：BCR Levenshtein L1 nested = **0.443 [0.398, 0.493]**，与 Table S1 一致（实跑 0.441，落在 CI 内）。
> ⚠️ 注意 `--datasets` 白名单只限**数据集**、不限 level：输出 CSV 仍含 `level2`/`level2.5` 行，那是 tie_v2 主树的**旧 L2（≈0.402）**；**投稿 L2/L2.5 以 `l2_vjfilter_20seed_nested_ci.csv` 为准（0.400/0.479）**，勿把该 CSV 的 level2 行当发表值（详见下「逐数值复现」L2 说明）。
> ⚠️ **聚合值含 ±0.007 蒙特卡洛噪声（不逐字节可复现）**：`nested_bootstrap_metrics()` 的两层重采样（`np.random.choice`）未播种 RNG，`nested_mean` 是 1000 次 bootstrap 分布的均值、本身是随机估计量，故重生成的 CSV 跨运行第三位小数可能漂 **±0.007**（实测两次全新跑最大偏差 0.0072，**全部集中在 sabdab**——因各 seed 测试集大小悬殊、query-weighting 方差最大；bcr 子集仅 0.0028）。**结论不受影响**（±0.007 ≪ CI 宽度 ~0.04–0.09 ≪ 核心 ~9pp 效应量），但严格的逐字节聚合复现请直接用已提交的 `grand_slam_tie_v2_nested_ci_20seeds.csv`（投稿快照），勿以"重生成 CSV 与之逐位相同"作为校验。复现执行报告见 [discussions/357_reproduction_run_report_20260608.md](discussions/357_reproduction_run_report_20260608.md)。

**Step 3 — 专项分析（机制/稳健性/配对/负对照）**：见上「论文产物 → 驱动脚本」映射表逐项跑。

**复现校验锚点（两种口径，勿混）**：
- *单 seed 切片自检*（确认切片逐位一致）：seed-42 同片普通 Levenshtein-L1 = BCR **0.4387** / SAbDab **0.5669**；BLOSUM62-L1 = BCR 0.454 / TCR 0.389 / McPAS 0.398（皆 seed-42 per-seed）。
- *投稿值（Table 1 / Supp S1）*：20-seed **nested-bootstrap** Levenshtein-L1 = BCR **0.443** / SAbDab **0.362**（query-weighted；SAbDab 单 seed 远高于此，因各 seed 测试集大小悬殊——见 doc 332 N1）。
- SAbDab 的 query-weighting 来自给 `nested_bootstrap_metrics` 喂 **per-query** `pq_recall@1` 数组（由 `retrieval_metrics` 返回）；`nested_bootstrap.py` 本身无独立 query-weight 开关。
- 任何新方法/数据集运行**必须走 `build_pilot_slice`** 并对上单 seed 锚点，否则会被去重/cap/alias 差异虚高（教训见 doc 336）。

### 外部复现前必须完成的打包项（已知缺口）
1. **公开仓库 + Zenodo DOI**：代码库已建于 `github.com/NidRUAwake/immune-embedding-benchmark`。投稿前仍需为数据 / 嵌入打 Zenodo DOI，并把数据可用性声明中的 DOI 占位符替换为真实 DOI。
2. **源代码统一（部分已做）**：canonical = `scripts/benchmark/`。`submission/scripts/` 镜像已于 **2026-06-04 同步**（含 8 个 paired builders + expected-R@1 evaluate；canonical 驱动复制到 `submission/scripts/analysis/drivers/`，见 `submission/scripts/README_DRIVERS.md`）。其余散在 `scratch/` 的驱动建议继续迁入 `scripts/analysis/drivers/` 并在映射表登记。
3. **预计算 embedding 不在 git**（体积）：需用 GPU 重新生成（`scripts/generate_embeddings.py` 及 `discussions/` 内的外部 GPU 工单）。
4. **环境安装**：用 `environment.yml`（已验证可解析）。`environment.lock.yml` 是原机 base 的取证记录，**不可** literal `conda env create`（tensorflow/numpy 冲突 → ResolutionImpossible）；版本表以 `environment.yml` 为准（如 pandas 2.2.1），lock 仅作脚注。
5. **稿件 TeX 不在代码 git 树**：canonical 稿件 `outputs/manuscript_submission/manuscript_BiB_v1.tex` + `supplementary_streamlined_v3.tex` 位于被 `.gitignore` 忽略的 `outputs/`——属"本机投稿包"，不是"公开复现代码包"。建库时需单独打包稿件/图，或纳入 submission 目录。

### 数据获取与再分发（提交策略）

**提交时不直接分发原始数据库 dump**——这既是各库许可/使用条款的要求，也是 Bioinformatics 的常规做法（数据可用性应指向**原始来源 + 版本**，而非再托管第三方库）。复现者按 `submission/download_scripts/` 从官方源自行下载对应版本，再跑预处理：

| 库 | 官方源 / 下载脚本 | 本文使用版本（Methods 已注明） |
|---|---|---|
| IEDB BCR | `submission/download_scripts/iedb_download.sh`（iedb.org） | 下载于 2026-05-09 |
| VDJdb TCR | `submission/download_scripts/vdjdb_download.sh`（GitHub **release 资产**，已修正 URL + pin tag） | 2025-12-29 release（下载 2026-04-27） |
| SAbDab | `submission/download_scripts/sabdab_download.sh`（OPIG，手动） | 下载于 2026-05-05 |
| McPAS-TCR | `submission/download_scripts/mcpas_download.sh`（**站点改 Shiny，需手动下载**） | 标准化于 2026-05-19 |
| IMGT IGHV/IGHJ germline（germline-reversion test, Note S2.5 / figS9） | IMGT/GENE-DB（imgt.org，人源 IGH）→ `imgt/IGV.fasta`, `imgt/IGJ.fasta`；`scratch/build_germline_framework_bcr.py` 读取 | 参考序列（按 IMGT 学术使用条款获取，不随包再分发） |

**随代码/Zenodo 一起发布（许可证允许的派生物）**：全部代码、`submission/download_scripts/`、预处理脚本、小派生表（`submission/Supplementary_Tables/Table_S1_Full_CI.csv` 等）；体积允许时可放预计算 embedding 与处理后中间文件（`outputs/intermediate/*`）。原始 `raw/` 与大体积 `outputs/` 不入 git（`.gitignore` 已忽略）。逐数据库的再分发条款建库前需逐一确认（IEDB/VDJdb 通常 CC-BY 类、可附 derived；McPAS/SAbDab 以指向源 + 版本为稳妥）。

#### 下载源实测与逐位复现状态（2026-06-08，见 [discussions/357](discussions/357_reproduction_run_report_20260608.md) P4）

> ⚠️ **许可前提**：不能一刀切把四库原始 dump 打进 Zenodo——各库条款限制再分发（见上）。因此**逐位复现靠"校验和 + 许可允许的派生物"，而非再托管原始 dump**：我们公布每份输入文件的 md5（见 [`submission/INPUT_CHECKSUMS.md5`](submission/INPUT_CHECKSUMS.md5)，可 `md5sum -c` 校验），复现者从官方源取到对应版本后自行核对字节；只随包分发许可允许的派生物（处理后输入表 / 数值 embedding，属转换性派生物）。

| 库 | 许可 | 官方源现状（实测） | 从下载逐位复现？ | 再分发决定 |
|---|---|---|---|---|
| IEDB BCR | CC-BY-4.0 | 网页手动导出，**只发 current**（无版本 pin） | ❌ 官方源已漂移，无法 byte-match 2026-05-09 快照 | 指向源+版本；CC-BY 允许，可随包附**派生输入表** + md5 |
| VDJdb TCR | AGPL-3.0 | GitHub **release 资产**；论文版 `2025-12-29` **仍在线**（旧脚本 `master/vdjdb.txt` URL 已 404，已修） | ✅ **已实测 md5 逐位一致**（`vdjdb_full.txt` = `4ab97ea73b42a04afeaf1957d3bf9894`） | 修好脚本即可纯从官方源复现；AGPL 允许，必要时也可附 |
| SAbDab | CC-BY | OPIG 网页手动，**只发 current** | ❌ 官方源已漂移 | 指向源+版本；CC-BY 允许附**派生输入表** + md5 |
| McPAS-TCR | 见站点条款（受限） | **旧静态 CSV URL 已死**；站点改为交互式 R Shiny 应用，下载需点 `downloadDB` 按钮（curl 不可自动化），**只发 current** | ❌ 官方源已无法脚本化获取 | 条款受限：**只指向源 + 版本 + 公布派生输入 md5**；原始 dump 不再分发，逐位复现以随包**派生输入** `mcpas_standardized.tsv` 为准 |

**结论**：仅 VDJdb 可纯从官方源逐位复现（已 md5 验证）；IEDB/SAbDab/McPAS 因"只发 current / 源站改版 / 条款受限"无法从官方源 byte-match——这三库的逐位复现以**随包分发的许可允许派生物（处理后输入 + embedding）+ md5 校验**为准，符合各库再分发条款。这与论文结论建立在 **20-seed nested 聚合**上一致：聚合值对输入的极小漂移稳健。

#### 可直接粘入手稿的 Data Availability 段（草拟）

> **Data availability.** All code, download scripts, preprocessing scripts, and small derived tables are available at https://github.com/NidRUAwake/immune-embedding-benchmark (code) and a Zenodo data package (DOI to be assigned upon archiving). The four source databases are used under their respective licenses (IEDB, CC-BY-4.0; VDJdb, AGPL-3.0; SAbDab, CC-BY; McPAS-TCR, per the McPAS website terms) and are **not redistributed as raw dumps**; instead we point to each original source and the exact version used (IEDB downloaded 2026-05-09; VDJdb 2025-12-29 release; SAbDab downloaded 2026-05-05; McPAS-TCR standardized 2026-05-19). VDJdb's 2025-12-29 release is byte-reproducible from its GitHub release asset (`vdjdb_full.txt`, md5 `4ab97ea7…`). Because IEDB, SAbDab, and McPAS-TCR serve only their current release (and McPAS-TCR is now an interactive web app), byte-exact retrieval of the original raw dumps from those sources is not guaranteed; to enable exact reproduction we provide the **license-permitted processed inputs and precomputed embeddings** together with md5 checksums for every input file. Reported results are 20-seed nested-bootstrap aggregates and are robust to negligible input drift.

---

## 快速开始

### 1. 从锁定环境复现

```bash
# 克隆仓库
git clone https://github.com/NidRUAwake/immune-embedding-benchmark.git
cd immune-embedding-benchmark

# 创建环境：用 environment.yml（已验证可解析）。
# environment.lock.yml 只是原机 base env 的取证记录，literal `conda env create` 会因
# tensorflow/numpy 等冲突 ResolutionImpossible —— 不要用它装环境。
conda env create -f environment.yml
conda activate embedding_benchmark_v1

# 验证环境
python -c "import torch, transformers; print(f'PyTorch {torch.__version__}'); print(f'Transformers {transformers.__version__}')"

# 强烈建议：让离线嵌入未命中时直接报错（避免零向量静默降级拉低 R@1）
export OFFLINE_EMBED_STRICT=1
```

### 2. 下载数据

```bash
# 注意：raw/ 与 outputs/ 不在 git（见 .gitignore）——git 只含代码 + 少量表/复现夹具。
# 原始数据需从各数据库官方下载（见文末 Data Availability / submission/download_scripts/），
# 或从随附 Zenodo 数据包获取。就位后应有（canonical BCR 输入是 bcr_singlechain_vh.tsv）：
ls raw/iedb/   # 期望含 bcr_singlechain_vh.tsv
ls raw/vdjdb/ ; ls raw/sabdab/ ; ls raw/mcpas/
```

### 3. 运行单个 seed 基准

```bash
# 在确定性模式下运行
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:16:8

# 冒烟测试（验证管线能跑）。注意：这不是 canonical 复现——
# run.py 无 --include-mcpas（McPAS 走 scripts/analysis/ 的专用驱动，见映射表）；
# 且 top-labels/min-count/cap 因库而异（BCR 10/14/150、SAbDab 5/25/100），
# 一条命令混跑多库会用同一组参数。canonical 复现请按各库参数单独跑（见下「canonical 运行参数」）。
python scripts/benchmark/run.py \
  --include-bcr --human-only \
  --bcr-file raw/iedb/bcr_singlechain_vh.tsv \
  --top-labels 10 --min-label-count 14 \
  --models blosum62 esm2-150m \
  --random-state 42 \
  --output-dir outputs/reproduce_seed42
```

### 4. 验证结果

```bash
# 对齐方法：可逐 seed 精确对锚点（如 BCR Levenshtein seed42 L1 = 0.4387）
grep -A1 level1 outputs/reproduce_seed42/pilot_results.csv | grep levenshtein  # → 0.4387
```

- **对齐（Lev/BLOSUM）**：单 seed 可对锚点（0.4387 / 0.3894 / 0.5669 / 0.398，见上）。
- **PLM（ESM2 等）**：**不要**逐字节 diff 单 seed `pilot_results.csv`——L1 因重复-CDR3 并列，现跑与归档 per-seed 会稳定差 ~3pp（`exp_recall@5` 一致）。PLM 的正确核验是**对归档 predictions 做 20-seed nested 聚合**并对 Table S1（如 BCR L1 ESM2 nested = 0.373），而非比单 seed pilot。
- **全长 L4 / 聚合值**：可复现（见「逐数值复现」Step 2）。

---

## 环境一致性

### 环境规范 (Python 包)

| 组件 | 版本 | 用途 |
|:-----|:-----|:-----|
| Python | 3.10.13 | 主要语言 |
| CUDA | 11.7 | GPU 加速 |
| PyTorch | 2.0.1 | 神经网络 |
| Transformers | 4.31.0 | 预训练模型 (ESM2 / AntiBERTy / AbLang / TCR-BERT 均经此加载) |
| NumPy | 1.26.4 | 数值计算 |
| Pandas | 2.2.1 | 数据处理 |
| SciPy | 1.11.1 | 统计 (TOST t 检验) |
| Scikit-learn | 1.2.2 | 机器学习工具 / 线性探针 |
| statsmodels | 0.14.1 | TOST 等价检验、Hedges' g |
| Parasail | 1.3.4 | **BLOSUM62 Smith-Waterman 比对** (`sw_striped_16`, gap 10/1) |
| Rapidfuzz | 3.6.1 | clone-aware 聚类相似度 **和规范 Levenshtein 基线** (`rapidfuzz.distance.Levenshtein` = 标准单位编辑距离,ins=del=sub=1) |
| Levenshtein / python-levenshtein | 0.23.0 | 仅辅助/legacy 脚本直接 import;规范 Levenshtein 数值来自 rapidfuzz(两者结果完全一致) |
| **tcrdist3** | **0.2.2** | **TCRdist / BCRdist 距离基线 (此前缺失)** |
| Biopython | 1.81 | 数据预处理 + 辅助分析(SHM / HIV-BLAST / contact residues);核心检索流程 `run.py` 不 import,复现已提交输入的主结果时不需要 |
| umap-learn | 0.5.3 | UMAP 诊断图 |
| tqdm | 4.65.0 | 进度条 |
| PyYAML | 6.0.1 | 配置读取 |
| joblib | 1.3.1 | 并行 / 缓存 |

### 外部工具 (非 pip 安装,必须单独安装)

| 工具 | 版本 | 用途 | 备注 |
|:-----|:-----|:-----|:-----|
| **ANARCI** | 源码 1.3 / bioconda 2024.05.21 | IMGT CDR 编号提取 | **仅数据预处理用** — 基准测试直接读取已提交的 `*_anarci.tsv`,只有从 raw 重新生成输入时才需要 ANARCI。**切勿 `pip install anarci==1.3`**:该版本不在 PyPI(PyPI 仅有 `2026.2.13.2`,且会拉入 numpy≥2 破坏整个固定栈)。请用 `conda install -c bioconda anarci hmmer`(bioconda 提供 2021.02.04 / 2024.05.21,IMGT 编号与论文所用源码 1.3 完全一致),或从源码安装 github.com/oxpig/ANARCI(其 setup.py 报告版本 1.3) |
| NCBI BLAST+ (`blastp` 2.16.0+, `makeblastdb` 2.15.0+) | 2.16.0 | 第二条比对基线 (Supplementary Table S9);HIV-1 泄漏审计 | ✅ **已实测确认 (2026-05-28)**:本机 `blastp` 唯一版本为 2.16.0+(build Jun 2024,早于结果生成的 2025-05),用它重跑 SABDAB L1 seeds 42/52/62,R@1/R@5/MRR **与已提交数值逐位一致 (误差<1e-12)**、BLAST DB 字节数也完全一致 → 论文原稿误写的 "v2.13" 已更正为 2.16.0,无版本漂移,Table S9 可字节复现 |
| HMMER3 | 随 ANARCI | ANARCI 的 HMM 比对后端 | conda 安装 `anarci` 时一并提供 |
| IMGT germline databases | ANARCI 内置 | CDR 编号参考 | ANARCI 包内置;若手动安装需运行其 `RScript`/germline 构建步骤 |

### 预训练模型权重 (运行时由 transformers 自动下载)

规范流水线 `scripts/benchmark/embeddings.py` 通过 HuggingFace **transformers** 加载下列权重(并非 `ablang`/`antiberty` pip 包):

| 别名 | HuggingFace 权重 | 维度 |
|:-----|:-----|:-----|
| esm2-150m / 650m / 3b | facebook/esm2_t30_150M_UR50D / t33_650M / t36_3B | 640 / 1280 / 2560 |
| antiberty | neulab/antiberty | 512 |
| ablang | msc-bioinformatics/AbLang | 768 |
| tcr-bert | wukevin/tcr-bert | 768 |

> ⚠️ 已生成的 embedding metadata 记录的 `resolved_model` 为 `alchemab/antiberty` 与 `RethinkX/AbLang`(与上表当前别名是同一权重的不同 HF 镜像)。如需从零重生成 embedding 而非使用 `outputs/embeddings/*.npy`,请确认镜像权重一致。
>
> ESM-C(§S2.3 / Fig S2B 敏感性分析)使用独立的 ESM-C 包,非主流水线必需。

### 完全冻结的依赖

`environment.lock.yml` 是原机 base env 的**取证记录**（含许多本项目不用的包，如 tensorflow，且 pin 互相冲突）。它**不能**用来装环境：literal `conda env create -f environment.lock.yml` 会 ResolutionImpossible。安装一律用 `environment.yml`：

```bash
conda env create -f environment.yml   # 不要用 environment.lock.yml
```

> 注意:锁定文件中曾包含 `ablang==0.3.1` 与 `ablang2`,但**规范的 AbLang/AntiBERTy embedding 来自 transformers + HF 权重**,这些 pip 包仅被 legacy 脚本 `scripts/analysis/benchmark_competitors.py` 使用,复现论文结果不需要它们。

### 操作系统支持

- ✅ Linux (主要测试平台)
- ✅ macOS (Intel/Apple Silicon)
- ⚠️ Windows (未测试，但理论上支持)

**注意**: 不同操作系统之间的浮点精度可能有 <0.01% 的差异。

---

## 确定性计算

### 关键设置

运行任何基准之前，设置这些环境变量:

```bash
# 禁用多线程以确保确定性
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Python hash 种子
export PYTHONHASHSEED=0

# CUDA 确定性
export CUBLAS_WORKSPACE_CONFIG=:16:8

# 把上述 export 放进运行前的 shell 即可（仓库未附 run_reproducible.sh 脚本）。
```

### 随机种子

所有基准都使用明确的随机种子:

```python
parser.add_argument("--random-state", type=int, default=42)
# 值: 42, 52, 62, ..., 232 (20 不同的种子)
```

20-seed 集合已发表在论文第 2.5 节。

### 确定性算法

| 算法 | 确定性措施 |
|:-----|:----------|
| Clone-aware split | 贪心聚类使用序列索引 (不依赖字典顺序) |
| Ranking | `np.argsort(..., kind='stable')` 用于一致的 tie-breaking |
| CDR3 extraction | 使用 IMGT 标准 + ANARCI 编号 |
| Similarity calculation | 相同的距离指标 (rapidfuzz Levenshtein / parasail BLOSUM62 / cosine) |
| Embedding generation | 相同的池化方法 (mean pooling) |

### CDR3 (L1) 的 tie-breaking 敏感性 (复现时的预期小差异)

CDR3-only (L1) 输入很短,会出现**大量并列(ties)**:对齐分数(Levenshtein / BLOSUM62)是整数、天然并列;**PLM(ESM2)也并列**——库内若有**重复 / 近重复 CDR3**，它们的 embedding 相同、cosine 完全相等，top-1 形成真并列。expected-R@1 对并列组按 \(c_{\text{top}}/t_{\text{top}}\) 给分(确定性),但并列组的构成对极小的相似度扰动敏感。

- **影响范围:** 仅 L1(以及部分 L2/L3);全长 L4 分数近乎唯一,可逐 seed 精确复现。**对齐与 PLM 在 L1 都受影响**(早期文档只提对齐,不完整)。
- **量级:** 每个 seed 约 ±0.5–4 pp(实测 BCR L1 ESM2-150M seed42 现跑 0.239 vs 归档 0.272,−3.3 pp;同 query 的 `exp_recall@5` 完全一致 → 排序主体未变,差异只在 top-1 并列组)。**符号随机**,在 20 seeds 上**相互抵消**:逐 seed 均值差 <1 pp,论文报告的 20-seed nested 均值远在其 95% CI 内。
- **投稿口径(重要):** 论文 Table S1 的值是用**已归档的 20 份 per-seed predictions** 做 nested 聚合(341 已逐位验证),**不是"现场重跑 20 次"**。从归档树复现 → 逐位一致;从零重跑 → PLM-L1 逐 seed 可能漂 ~3pp(根因还可能含归档时的 evaluate/embedding 版本),但聚合值与**所有结论**(L1 上 alignment 领先 ~9pp ≫ 此漂动)不受影响。
- **拆分本身是确定的:** 训练/测试集划分逐 seed 完全一致(`reproduction/` 用 test-set 标签 md5 验证)。

因此:**聚合数值(均值 + CI)可从归档 predictions 复现**;L1 的**逐 seed 数值不作逐字节复现承诺**(短序列 + 重复 CDR3 并列的固有性质,不是错误)。复现请用归档 `tie_v2` 树 + `nested_bootstrap.py`,而非逐 seed 比对 `pilot_results.csv`。

### 复现对照工具

`reproduction/` 目录提供一个快速对照脚本(仅比对方法,无需 GPU)。**注意：这是非-canonical 快检**——它用 bundled `bcr_full_single_header_anarci.tsv`（非 canonical 的 `bcr_singlechain_vh.tsv`）、只跑 3 个 seed（42/92/142）、对照旧的 `phase3_grand_slam` 树而非 `tie_v2`，因此它只验证**管线能跑且划分确定**，不复现投稿数值（投稿数值见上「canonical 运行参数」+ 映射表）：

```bash
python reproduction/run_reproduction.py    # exit 0 = 通过
```

它重跑 BCR(seeds 42/92/142)并对照已提交的标准结果,核验:拆分 md5 精确一致、L4 近乎精确、L1 在 tie-breaking 容差内(逐 seed 均值 <1 pp 偏差)。详见 `reproduction/README.md`。

---

## 数据一致性

### 原始数据来源与版本

| 数据集 | 来源 | 下载日期 | 版本 | 序列数 |
|:-----|:-----|:--------|:-----|:------|
| IEDB BCR | IEDB | 2026-05-09 | current | ~870 |
| VDJdb TCR | VDJdb | 2026-04-27 | 2025-12-29 | ~5000+ |
| SAbDab | SAbDab | 2026-05-05 | current | ~1500+ |
| McPAS | McPAS-TCR | 2026-05-19 | current | ~1000+ |

### 数据预处理一致性

1. **标签标准化**: `label_aliases.json` (固定映射表)
2. **序列过滤**: 长度 > 8 AA，无非标准字符
3. **CDR3 锚定移除**: 去除 C/F 锚定 (VDJdb/McPAS)
4. **去重**: 相同 CDR3 仅保留一条

所有预处理逻辑在 `scripts/benchmark/data.py` 中确定。

### 输出数据位置

```
outputs/phase3_grand_slam_tie_v2/
├── bcr/
│   ├── levenshtein/seed_{42..232}/
│   ├── blosum62/seed_{42..232}/
│   ├── esm2-150m/seed_{42..232}/
│   └── [其他方法]/seed_{42..232}/
├── tcr/
│   └── [方法]/seed_{42..232}/  # 20 seeds (canonical)
├── sabdab/
│   └── [方法]/seed_{42..232}/  # 20 seeds (canonical)
└── mcpas/
    └── [方法]/seed_{42..232}/  # 20 seeds (canonical)
# 注：四库均 20 seeds（42,52,…,232）。旧版/配对子树或残留 *_backup_* 目录可能 seed 数不同——
#     聚合时只取 dataset∈{bcr,tcr,sabdab,mcpas} 主子树。
```

每个目录包含:
- `pilot_results.csv` - 该 seed 的主要指标
- `*_predictions.csv` - 各 level 的详细预测

---

## 代码修复与改进

### Task 306: 数据补全与修复

#### Task 306A: TCR nested CI fix

**问题**: TCR 的 level 检测只查看第一个 seed，缺失其他 seeds 有的 levels

**修复**: 从所有 seeds 的并集中检测 levels

```python
# scripts/analysis/nested_bootstrap.py lines 108-130
levels = set()
for seed in seeds:  # ← 现在遍历所有 seeds
    seed_dir = model_dir / f"seed_{seed}"
    for f in seed_dir.glob("*_predictions.csv"):
        levels.add(level_part)
```

**影响**: TCR CI 报告现在包含所有应有的 levels

---

#### Task 306B: McPAS ESM2 fix

> **历史记录，已超越**：下述"McPAS 移除 `--offline-dir`、5 seeds"是早期状态。现状（2026-06）：McPAS **使用** `--offline-dir outputs/embeddings`（m8 里移除 offline 的逻辑已注释掉），且与其它三库一样跑 **20 seeds**；canonical McPAS 命令见上「逐数值复现」Step 1。

**问题**: 
1. McPAS 缺少 alpha 链导致 level1_paired/level4_paired 无效
2. --offline-dir 使用 VDJdb 缓存但 McPAS 序列不同

**修复**:
1. 检测 McPAS，条件性排除 paired builders
2. McPAS 时移除 --offline-dir

```python
# scripts/benchmark/data.py lines 204-225
is_mcpas = "mcpas" in tcr_file.lower()
if not is_mcpas:
    tcr_builders["level1_paired"] = ...
    tcr_builders["level4_paired"] = ...

# scripts/analysis/m8_run_all_20seeds.py lines 62-69
if s["name"] == "mcpas":
    cmd.remove("--human-only")
    cmd.pop(cmd.index("--offline-dir") + 1)  # 移除路径
    cmd.pop(cmd.index("--offline-dir"))      # 移除标志
```

**影响**（历史；现状见本节顶部"已超越"说明）: 当时 McPAS esm2-150m 完成 5 seeds；**现状为四库统一 20 seeds**。

---

#### Task 306C: BLOSUM62 20-seed 补全

**问题**: VDJdb TCR paired level 在某些 seeds 缺失数据

**修复**: 从现有 pilot_results.csv 提取缺失的 seed 数据

**影响**: Supplementary Table S4 现包含完整 20-seed 数据

---

### Bug 修复: Argsort 稳定性

**问题**: `np.argsort()` 在 tie 时的顺序可能随 NumPy 版本改变

**修复**: 使用稳定排序

```python
# scripts/benchmark/evaluate.py —— 见 expected-R@1 排序处（搜 `np.argsort(... kind='stable')`，勿依赖行号）
ranked = d_labels[np.argsort(sim_scores, kind='stable')[::-1]]
```

**影响**: 一致的排名即使在浮点值相同时

---

## 性能基准

### 运行时间估计

| 数据集 | Seed | 方法 | 运行时间 |
|:-----|:-----|:-----|:--------|
| BCR | 1 | blosum62 | ~2 分钟 |
| BCR | 1 | esm2-150m | ~5 分钟 |
| TCR | 1 | blosum62 | ~1 分钟 |
| VDJdb TCR | 1 | esm2-150m | ~3 分钟 |
| McPAS | 1 | esm2-650m | ~15 分钟 |
| McPAS | 1 | esm2-3b | ~45 分钟 |

**20-seed 完整运行**: ~3-4 小时 (使用 GPU)

### 硬件要求

**推荐**:
- GPU: NVIDIA A100 或等效 (40+ GB VRAM)
- CPU: 8+ 核心
- 内存: 32+ GB RAM
- 存储: 50+ GB SSD

**最小**:
- GPU: NVIDIA V100 或等效 (16+ GB VRAM)
- CPU: 4 核心
- 内存: 16 GB RAM
- 存储: 30+ GB SSD

---

## 验证与测试

### 单元测试

仓库当前未附正式单元测试套件（无 `scripts/tests/`）。可用的功能性验证是
`reproduction/run_reproduction.py`（见下「小规模验证运行」）——但注意它是**非-canonical 快检**。

### 小规模验证运行

```bash
# 快速验证: 单个 seed，3 个顶级标签，单个方法（仅冒烟，非 canonical）
python scripts/benchmark/run.py \
  --include-bcr --human-only \
  --bcr-file raw/iedb/bcr_singlechain_vh.tsv \
  --models blosum62 \
  --random-state 42 \
  --top-labels 3 \
  --min-label-count 5 \
  --output-dir outputs/quick_test

# 应在 <5 分钟内完成
```

### 数值验证

**勿用**上面的 `quick_test`（top-3/min-5）去对 canonical（top-10/min-14）——参数不同，必然不一致。正确做法是用**同参 canonical 命令**重跑一个**对齐**方法（确定性，可逐位对）：

```bash
# 用 canonical 参数重跑 BCR BLOSUM62 seed42（对齐方法，确定性）
python scripts/benchmark/run.py --include-bcr --human-only \
  --bcr-file raw/iedb/bcr_singlechain_vh.tsv --top-labels 10 --min-label-count 14 \
  --levels level1 --models blosum62 --random-state 42 \
  --test-size 0.2 --clone-threshold 0.95 --export-predictions \
  --output-dir outputs/verify_bcr_blosum_seed42

python << 'EOF'
import pandas as pd
new = pd.read_csv("outputs/verify_bcr_blosum_seed42/pilot_results.csv")
ref = pd.read_csv("outputs/phase3_grand_slam_tie_v2/bcr/blosum62/seed_42/pilot_results.csv")
r = lambda d: float(d[d.level=='level1']['recall@1'].iloc[0])
print(f"BLOSUM62 L1 seed42: rerun={r(new):.4f}  committed={r(ref):.4f}")
assert abs(r(new) - r(ref)) < 0.001, "alignment 应逐位一致"
print("✓ 对齐数值验证通过（≈0.454）")
EOF
```

> PLM（ESM2）的**单 seed** L1 不要这样断言——它因重复-CDR3 并列会差 ~3pp（见「CDR3 (L1) tie 敏感性」）；PLM 的正确验证是对**归档 predictions** 做 nested 聚合并对 Table S1（见「逐数值复现」Step 2）。

---

## 已知限制与注意事项

### 浮点差异

由于硬件和软件的不同，可能出现最多 **0.01% 的浮点差异**。这在:
- GPU vs CPU 计算
- 不同操作系统
- 不同 NumPy 版本

这些差异不影响结论，因为我们报告的是聚合的 20-seed 平均值。

### 模型加载

某些模型首次加载时会自动下载:

```
facebook/esm2_t30_150M_UR50D      (~600 MB)
facebook/esm2_t33_650M_UR50D      (~1.3 GB)
facebook/esm2_t36_3B_UR50D        (~5.9 GB)
```

默认缓存位置: `~/.cache/huggingface/`

要离线运行，预先下载这些模型:

```bash
python << 'EOF'
from transformers import AutoModel
models = [
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D"
]
for model_id in models:
    AutoModel.from_pretrained(model_id)
    print(f"Downloaded {model_id}")
EOF
```

### 数据可用性

- ✅ 代码: https://github.com/NidRUAwake/immune-embedding-benchmark
- ✅ 原始数据: 通过 IEDB, VDJdb, McPAS, SAbDab API
- ⚠️ 预计算嵌入: 太大，未包含 (可使用提供的脚本重新生成)
- ⚠️ 完整 20-seed 结果: 在 `outputs/phase3_grand_slam_tie_v2/` 中

---

## Docker 容器 (可选)

### 构建容器

```bash
# 创建 Dockerfile (见下方)
docker build -t immune-benchmark:reproducible .
```

### 运行基准

```bash
docker run --gpus all -it \
  -v $(pwd)/outputs:/workspace/outputs \
  immune-benchmark:reproducible \
  python scripts/benchmark/run.py \
    --include-bcr \
    --models blosum62 esm2-150m \
    --random-state 42 \
    --output-dir /workspace/outputs/docker_test
```

### Dockerfile 示例

```dockerfile
FROM nvidia/cuda:11.7.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y wget ca-certificates

# 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/miniconda/bin:$PATH"

WORKDIR /workspace

COPY environment.yml .
RUN conda env create -f environment.yml
# 注：还需在容器内安装 ANARCI 与 NCBI BLAST+ 2.16.0（非 pip，见「外部工具」节），
#     否则从 raw 重做预处理会缺工具。environment.lock.yml 不可用于安装（ResolutionImpossible）。

SHELL ["conda", "run", "-n", "embedding_benchmark_v1", "/bin/bash", "-c"]

COPY . .

# 设置确定性变量
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONHASHSEED=0
ENV CUBLAS_WORKSPACE_CONFIG=:16:8

ENTRYPOINT ["conda", "run", "-n", "embedding_benchmark_v1", "python"]
```

---

## 故障排除

### 错误: "tcrdist 未安装"

**解决方案**: 在执行前安装:
```bash
pip install tcrdist3
```

（注意: tcrdist3 已在 `environment.yml` 中；若缺失见「外部工具」节单独安装。勿用 environment.lock.yml 装环境。）

### 错误: "CUDA 内存不足"

**解决方案**:
1. 减少批大小 (--batch-size 参数)
2. 使用 CPU 模式 (未建议，会很慢)
3. 使用更小的模型 (esm2-150m 而不是 esm2-3b)

### 错误: "模型下载失败"

**解决方案**:
1. 检查网络连接
2. 手动设置 HF 缓存: `export HF_HOME=/custom/path`
3. 预下载模型 (见上方"数据可用性"部分)

### 浮点差异太大 (>0.01%)

**排查**:
1. 检查 Python 版本: `python --version` (应为 3.10.x)
2. 检查 PyTorch: `python -c "import torch; print(torch.__version__)"`
3. 检查 NumPy: `python -c "import numpy as np; print(np.__version__)"`
4. 检查环境变量是否设置正确

---

## 联系与支持

对于复现问题，请:

1. 检查此指南的"故障排除"部分
2. 查看 `CODE_REPRODUCIBILITY_AUDIT.md` 的技术细节
3. 查看 GitHub Issues: https://github.com/NidRUAwake/immune-embedding-benchmark/issues
4. 提交带有以下信息的问题:
   - 完整的错误消息
   - 环境信息 (OS, Python, CUDA)
   - 复现步骤

---

## 参考文献

**论文**: "Clone-aware benchmarking points to a germline shortcut in protein language model retrieval of immune receptors"  
**主仓库**: `github.com/NidRUAwake/immune-embedding-benchmark`（代码，已建）；Zenodo DOI 待分配（数据 / 嵌入包）  
**环境版本**: embedding_benchmark_v1（用 `environment.yml`）  
**文档修订**: 2026-06-04

---

**按本指南可复现 20-seed nested-bootstrap 聚合 R@1 与 95% CI（即投稿表中的值）。单 seed 的 L1 因整数对齐分数的 tie-breaking 不保证逐位一致——见「确定性计算」节；主结论建立在 nested bootstrap + expected-R@1 上，而非逐 seed L1 逐位复现。**

