
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import os

def load_seed_data(base_path, dataset, model, level, seeds):
    seed_data = []
    for seed in seeds:
        seed_dir = Path(base_path) / dataset / model / f"seed_{seed}"
        if not seed_dir.exists():
            continue
            
        # Exact filename matching (case-insensitive for dataset prefix)
        target_name = f"_{level}_{model}_predictions.csv".lower()
        prefix_to_check = dataset.lower() + "_"
            
        matching_files = [f for f in seed_dir.glob("*.csv") if f.name.lower().endswith(target_name) and f.name.lower().startswith(prefix_to_check)]
        
        # Filter to ensure exact level match (avoid level4 matching level4_paired)
        exact_matches = []
        for f in matching_files:
            filename = f.name.lower()
            suffix = f"_{model}_predictions.csv".lower()
            extracted_level = filename[len(prefix_to_check) : -len(suffix)]
            if extracted_level == level.lower():
                exact_matches.append(f)
        
        if exact_matches:
            file_path = exact_matches[0]
            df = pd.read_csv(file_path)
            # Prefer expected (fractional) per-query credit when present; the legacy 'rank'
            # column gives 0/1 hits which are order-dependent for tied scores. See doc 311.
            if {'exp_recall@1', 'exp_recall@5', 'exp_mrr'}.issubset(df.columns):
                seed_data.append({
                    'hits1': df['exp_recall@1'].values.astype(float),
                    'hits5': df['exp_recall@5'].values.astype(float),
                    'rr':    df['exp_mrr'].values.astype(float),
                })
            elif 'rank' in df.columns:
                seed_data.append({
                    'hits1': (df['rank'] == 0).values.astype(float),
                    'hits5': (df['rank'] < 5).values.astype(float),
                    'rr':    (1.0 / (df['rank'] + 1)).values.astype(float),
                })
        else:
            print(f"Warning: File with pattern {target_name} not found in {seed_dir}")
            
    return seed_data

def nested_bootstrap_metrics(seed_data, n_outer=1000):
    if not seed_data:
        return None
        
    n_seeds = len(seed_data)
    results = {
        'recall@1': [],
        'recall@5': [],
        'mrr': []
    }
    
    # Pre-extract arrays for speed
    hits1_list = [d['hits1'] for d in seed_data]
    hits5_list = [d['hits5'] for d in seed_data]
    rr_list = [d['rr'] for d in seed_data]
    
    for _ in range(n_outer):
        resampled_seed_indices = np.random.choice(n_seeds, size=n_seeds, replace=True)
        
        b_hits1, b_hits5, b_rr = [], [], []
        for s_idx in resampled_seed_indices:
            h1 = hits1_list[s_idx]
            h5 = hits5_list[s_idx]
            rr = rr_list[s_idx]
            
            n = len(h1)
            resample_idx = np.random.choice(n, size=n, replace=True)
            
            b_hits1.extend(h1[resample_idx])
            b_hits5.extend(h5[resample_idx])
            b_rr.extend(rr[resample_idx])
            
        results['recall@1'].append(np.mean(b_hits1))
        results['recall@5'].append(np.mean(b_hits5))
        results['mrr'].append(np.mean(b_rr))
        
    final_stats = {}
    for metric, values in results.items():
        final_stats[metric] = {
            'mean': np.mean(values),
            'lower': np.percentile(values, 2.5),
            'upper': np.percentile(values, 97.5)
        }
    return final_stats

def main():
    parser = argparse.ArgumentParser()
    # Canonical tree is the tie_v2 (expected-R@1) grid; the old "phase3_grand_slam" is superseded.
    # ⚠️ The DEFAULT base-dir below is PRE-`is_legit`-filter for BCR/McPAS. The canonical master
    # table (Table 1 / S1) is a TWO-TREE aggregation: BCR+McPAS from `outputs/phase3_filtered_tie_v2`
    # (post-filter), VDJdb+SAbDab from `outputs/phase3_grand_slam_tie_v2` (filter is a no-op there).
    # Do NOT aggregate BCR/McPAS from the default here for paper numbers — use
    # `scripts/analysis/build_master_nested_ci.py`, which does the correct two-tree assembly
    # -> outputs/reports/master_nested_ci_20seeds_postfilter.csv. See discussions/368 §2.
    parser.add_argument("--base-dir", default="outputs/phase3_grand_slam_tie_v2")
    parser.add_argument("--output-nested", default="outputs/reports/grand_slam_tie_v2_nested_ci_20seeds.csv")
    parser.add_argument("--output-comparison", default="outputs/reports/ci_comparison.csv")
    # By default aggregate only the four canonical datasets; backup / ablation subtrees
    # (bcr_backup_*, bcr_clone_vs_random, bcr_l35, *_broken_backup, ...) are NOT submission
    # numbers and must not be pooled into the master table. Pass --all-datasets to scan everything.
    parser.add_argument("--datasets", default="bcr,tcr,sabdab,mcpas",
                        help="comma-separated dataset subdirs to aggregate (canonical four by default)")
    parser.add_argument("--all-datasets", action="store_true",
                        help="scan every subdir under --base-dir (includes backup/ablation trees)")
    parser.add_argument("--bootstrap-seed", type=int, default=42,
                        help="seed for the two-level resampling RNG; makes the aggregate CSV "
                             "byte-reproducible across runs. The committed submission snapshot "
                             "was generated UNSEEDED, so it will differ by the documented "
                             "~±0.007 Monte-Carlo noise (well within CI).")
    args = parser.parse_args()

    # Seed once here so the whole aggregation (every dataset/model/level) is deterministic.
    np.random.seed(args.bootstrap_seed)

    base_path = Path(args.base_dir)
    present = [d.name for d in base_path.iterdir() if d.is_dir()]
    if args.all_datasets:
        datasets = present
    else:
        wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
        datasets = [d for d in wanted if d in present]
        skipped = [d for d in present if d not in datasets]
        if skipped:
            print(f"[nested_bootstrap] aggregating canonical datasets {datasets}; "
                  f"SKIPPED non-canonical subdirs {skipped} (use --all-datasets to include).")
    
    all_results = []
    
    for ds in datasets:
        print(f"Processing dataset: {ds}")
        models = [m.name for m in (base_path / ds).iterdir() if m.is_dir()]
        for model in models:
            # Detect seeds and levels
            model_dir = base_path / ds / model
            seeds = [int(s.name.split("_")[1]) for s in model_dir.glob("seed_*")]
            if not seeds: continue
            
            # Collect levels from ALL seeds (some seeds may have incomplete data)
            levels = set()
            ds_lower = ds.lower()
            model_lower = model.lower()
            suffix = f"_{model_lower}_predictions.csv"
            prefix_to_check = f"{ds_lower}_"
            
            for seed in seeds:
                seed_dir = model_dir / f"seed_{seed}"
                for f in seed_dir.glob("*_predictions.csv"):
                    # Extract level: Dataset_{level}_{model}_predictions.csv
                    # We want everything between Dataset_ and _{model}_predictions.csv
                    filename = f.name.lower()
                    if filename.startswith(prefix_to_check) and filename.endswith(suffix):
                        level_part = filename[len(prefix_to_check) : -len(suffix)]
                        levels.add(level_part)
            
            for level in sorted(list(levels)):
                print(f"  Model: {model}, Level: {level}, Seeds: {len(seeds)}")
                seed_data = load_seed_data(args.base_dir, ds, model, level, seeds)
                if not seed_data: continue
                
                nested_stats = nested_bootstrap_metrics(seed_data)
                if nested_stats:
                    for metric, stats in nested_stats.items():
                        all_results.append({
                            'dataset': ds,
                            'model': model,
                            'level': level,
                            'metric': metric,
                            'nested_mean': stats['mean'],
                            'nested_lower': stats['lower'],
                            'nested_upper': stats['upper'],
                            'nested_width': stats['upper'] - stats['lower']
                        })

    df_nested = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.output_nested), exist_ok=True)
    df_nested.to_csv(args.output_nested, index=False)
    print(f"Saved nested CIs to {args.output_nested}")
    
    # Comparison logic
    comparison = []
    for _, row in df_nested.iterrows():
        # Load pooled CI from existing bootstrap files
        pooled_file = base_path / row['dataset'] / row['model'] / f"bootstrap_{row['level']}.csv"
        if pooled_file.exists():
            df_pooled = pd.read_csv(pooled_file)
            pooled_row = df_pooled[df_pooled['metric'] == row['metric']]
            if not pooled_row.empty:
                p_lower = pooled_row['lower'].values[0]
                p_upper = pooled_row['upper'].values[0]
                p_width = p_upper - p_lower
                
                comparison.append({
                    'dataset': row['dataset'],
                    'model': row['model'],
                    'level': row['level'],
                    'metric': row['metric'],
                    'pooled_width': p_width,
                    'nested_width': row['nested_width'],
                    'width_ratio': row['nested_width'] / p_width if p_width > 0 else 1.0
                })
                
    df_comp = pd.DataFrame(comparison)
    df_comp.to_csv(args.output_comparison, index=False)
    print(f"Saved comparison to {args.output_comparison}")
    
    # Summary of comparison
    if not df_comp.empty:
        avg_ratio = df_comp['width_ratio'].mean()
        print(f"\nAverage CI width expansion: {avg_ratio:.2f}x")
        print(df_comp.groupby('dataset')['width_ratio'].mean())

if __name__ == "__main__":
    main()
