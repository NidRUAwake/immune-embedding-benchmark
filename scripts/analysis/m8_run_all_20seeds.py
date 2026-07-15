#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

def main():
    python_path = sys.executable
    out_dir = Path("outputs/phase3_grand_slam_tie_v2")
    
    # 20 seeds
    seeds = [42 + i * 10 for i in range(20)]
    
    scenarios = [
        {
            "name": "bcr",
            "flags": ["--include-bcr", "--bcr-file", "raw/iedb/bcr_singlechain_vh.tsv",
                      "--top-labels", "10", "--min-label-count", "14",
                      "--levels", "level1", "level2", "level3", "level4"],
            "models": ["levenshtein", "blosum62", "esm2-150m", "esm2-650m", "esm2-3b", "antiberty", "ablang"]
        }
    ]
    
    for s in scenarios:
        print(f"\n{'='*40}\nScenario: {s['name'].upper()}\n{'='*40}")
        for model in s["models"]:
            print(f"\n--- Model: {model} ---")
            for seed in seeds:
                seed_dir = out_dir / s["name"] / model / f"seed_{seed}"
                
                # Check if it already exists and has csv outputs
                if seed_dir.exists() and len(list(seed_dir.glob("*.csv"))) > 0:
                    print(f"Skipping model {model} seed {seed} (already exists)")
                    continue
                    
                print(f"  seed {seed}: running...")
                cmd = [
                    python_path, "scripts/benchmark/run.py",
                    *s["flags"],
                    "--models", model,
                    "--random-state", str(seed),
                    "--human-only",
                    "--test-size", "0.2",
                    "--clone-threshold", "0.95",
                    "--export-predictions",
                    "--offline-dir", "outputs/embeddings",
                    "--output-dir", str(seed_dir)
                ]
                
                # For McPAS: don't use cached embeddings (original run_mcpas_benchmark.py didn't either)
                # This ensures embeddings are computed specifically for McPAS sequences
                if s["name"] == "mcpas":
                    cmd.remove("--human-only")
                    # Keep --offline-dir to use pre-computed offline embeddings and run in seconds!
                    # offline_idx = cmd.index("--offline-dir")
                    # cmd.pop(offline_idx + 1)  # remove the path
                    # cmd.pop(offline_idx)      # remove the flag
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  ERROR at {s['name']} - {model} - seed {seed}:\n{result.stderr[-1000:]}")
                else:
                    print(f"  seed {seed}: done")

    print("\nAll tasks completed. Ready for nested_bootstrap.py!")

if __name__ == "__main__":
    main()
