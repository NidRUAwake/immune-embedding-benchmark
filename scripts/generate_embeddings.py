#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import torch

if __package__ in (None, ""):
    # Allow direct execution: `python scripts/generate_embeddings.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.embeddings import get_embedder, MODEL_ALIASES
from benchmark.data import build_dataset_specs as original_build_dataset_specs, load_table, filter_human_only

# Monkey-patch get_embedder to load ablang and antiberty offline using local package weights when --local-files-only is set
import benchmark.embeddings as be

original_get_embedder = be.get_embedder

class LocalAbLangEmbedder(be.Embedder):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        import ablang
        import torch
        self.device = torch.device(device)
        self.heavy_ablang = ablang.pretrained(chain="heavy", device=device)
        self.heavy_ablang.freeze()
        
    def encode(self, sequences):
        import numpy as np
        sequences = list(sequences)
        # Strip our artificial structural delimiters so AbLang receives the concatenated
        # amino-acid content: L3 "CDR1|CDR2|CDR3" -> CDR1CDR2CDR3; paired "VH:VL" -> VHVL.
        # (L2 clonotype strings still contain V/J gene-name digits/hyphens, which AbLang's
        # amino-acid tokenizer correctly rejects -> those fall back to zero, as intended.)
        cleaned_seqs = [
            str(s).upper().strip().replace("*", "X").replace("|", "").replace(":", "").replace("_", "")
            for s in sequences
        ]
        try:
            return self.heavy_ablang(cleaned_seqs, mode="seqcoding", align=False)
        except Exception as e:
            chunks = []
            for s in cleaned_seqs:
                try:
                    chunks.append(self.heavy_ablang([s], mode="seqcoding", align=False))
                except Exception:
                    chunks.append(np.zeros((1, 768), dtype=np.float32))
            return np.vstack(chunks)

class LocalAntiBERTyEmbedder(be.Embedder):
    def __init__(self, args=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        import os
        import antiberty
        import torch
        from transformers import AutoModel, BertTokenizer
        
        project_path = os.path.dirname(os.path.realpath(antiberty.__file__))
        trained_models_dir = os.path.join(project_path, 'trained_models')
        checkpoint_path = os.path.join(trained_models_dir, 'AntiBERTy_md_smooth')
        vocab_file = os.path.join(trained_models_dir, 'vocab.txt')
        
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False)
        self.model = AutoModel.from_pretrained(checkpoint_path, local_files_only=True).to(self.device).eval()
        self.batch_size = getattr(args, 'batch_size', 16) if args else 16
        self.pooling = getattr(args, 'pooling', 'mean') if args else 'mean'
        
    def encode(self, sequences):
        import numpy as np
        import torch
        sequences = list(sequences)
        chunks = []
        for start in range(0, len(sequences), self.batch_size):
            batch_seqs = sequences[start:start + self.batch_size]
            processed_seqs = [" ".join(list(str(s).upper().replace(" ", ""))) for s in batch_seqs]
            
            inputs = self.tokenizer(processed_seqs, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state
                
            if self.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                
            chunks.append(pooled.detach().cpu().numpy())
        return np.vstack(chunks)

def custom_get_embedder(model_name: str, args=None):
    model_lower = model_name.lower()
    local_files_only = getattr(args, 'local_files_only', False) if args else False
    
    if model_lower == "ablang" and local_files_only:
        print("  [PATCH] Using local AbLang package weights offline.", flush=True)
        return LocalAbLangEmbedder(device="cuda" if torch.cuda.is_available() else "cpu")
    elif model_lower == "antiberty" and local_files_only:
        print("  [PATCH] Redirecting AntiBERTy to local package weights offline.", flush=True)
        return LocalAntiBERTyEmbedder(args, device="cuda" if torch.cuda.is_available() else "cpu")
        
    return original_get_embedder(model_name, args)

be.get_embedder = custom_get_embedder
# IMPORTANT: main() calls the module-global `get_embedder` (bound at import on line 13 to
# the ORIGINAL function). Rebind it here so the offline AbLang/AntiBERTy routing actually
# takes effect; patching only `be.get_embedder` is not enough.
get_embedder = custom_get_embedder

def build_dataset_specs(args):
    specs = original_build_dataset_specs(args)
    for spec in specs:
        if spec.name == "BCR":
            from benchmark.data import combine_first_available, clean_sequence
            spec.sequence_builders["level4_paired"] = lambda df: (
                combine_first_available(df, ["Chain 1_Protein Sequence", "Chain 1_Full Sequence"]).fillna("") + ":" +
                combine_first_available(df, ["Chain 2_Protein Sequence", "Chain 2_Full Sequence"]).fillna("")
            ).map(clean_sequence)
    return specs

def parse_args():
    parser = argparse.ArgumentParser(description="Offline feature extraction pipeline.")
    parser.add_argument("--bcr-file", default="raw/iedb/bcr_full_single_header.tsv")
    parser.add_argument("--tcr-file", default="raw/vdjdb/vdjdb_full.txt")
    parser.add_argument("--sabdab-file", default="raw/sabdab/sabdab_paired_clean_human_full.csv")
    parser.add_argument("--bcr-label-col", default="Epitope_Source Molecule")
    parser.add_argument("--tcr-label-col", default="antigen.epitope")
    parser.add_argument("--sabdab-label-col", default="label")
    parser.add_argument("--tcr-level4-col", default=None)
    parser.add_argument("--output-dir", default="outputs/embeddings")
    parser.add_argument("--models", nargs="+", default=["esm2-35m", "antiberty"])
    parser.add_argument("--include-bcr", action="store_true")
    parser.add_argument("--include-tcr", action="store_true")
    parser.add_argument("--include-sabdab", action="store_true")
    parser.add_argument("--human-only", action="store_true", help="Filter to human receptor sequences only.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pooling", choices=["mean", "cls", "layer-weighted"], default="mean")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--levels", nargs="+", default=None, help="Specific levels to generate (e.g., level1 level4)")
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    specs = build_dataset_specs(args)
    for spec in specs:
        print(f"Loading {spec.name} data from {spec.path}...", flush=True)
        try:
            df = load_table(spec)
        except Exception as e:
            print(f"Skipping {spec.name}: {e}", flush=True)
            continue
        if args.human_only:
            before = len(df)
            df = filter_human_only(df, spec)
            print(f"{spec.name} human-only filter: {before} -> {len(df)} rows", flush=True)
            
        for level, builder in spec.sequence_builders.items():
            if args.levels and level not in args.levels:
                continue
            seqs = builder(df).dropna()
            unique_seqs = list(seqs.unique())
            
            if not unique_seqs:
                continue
                
            print(f"Generating features for {spec.name} {level} ({len(unique_seqs)} unique sequences)", flush=True)
            
            for model_name in args.models:
                print(f"  -> Model: {model_name} (pooling={args.pooling})", flush=True)
                resolved_model_name = MODEL_ALIASES.get(model_name.lower(), model_name)
                
                safe_model_name = model_name.replace("/", "_")
                prefix = f"{spec.name.lower()}_{level}_{safe_model_name}_{args.pooling}"
                
                npy_path = out_dir / f"{prefix}.npy"
                json_path = out_dir / f"{prefix}_metadata.json"
                
                if npy_path.exists() and json_path.exists():
                    print(f"     Already exists, skipping: {npy_path}")
                    continue
                
                embedder = get_embedder(model_name, args)
                embeddings = embedder.encode(unique_seqs)
                np.save(npy_path, embeddings)
                
                metadata = {
                    "dataset": spec.name,
                    "level": level,
                    "model": model_name,
                    "resolved_model": resolved_model_name,
                    "pooling": args.pooling,
                    "n_sequences": len(unique_seqs),
                    "embedding_dim": embeddings.shape[1],
                    "sequence_order": unique_seqs
                }
                
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                print(f"     Saved {embeddings.shape} array to {npy_path}")

if __name__ == "__main__":
    main()
