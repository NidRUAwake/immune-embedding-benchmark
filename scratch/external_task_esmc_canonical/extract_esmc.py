"""
ESM-C 300M embedding extraction script.
Usage examples:
  python extract_esmc.py --input bcr_level1_sequences.csv \
      --output bcr_level1_esmc_mean.npy --pooling mean --gpu 0
  python extract_esmc.py --input bcr_level1_sequences.csv \
      --output bcr_level1_esmc_cls.npy --pooling cls --gpu 0
  python extract_esmc.py --input bcr_level4_sequences.csv \
      --output bcr_level4_esmc_mean.npy --pooling mean --gpu 0
  python extract_esmc.py --input bcr_level4_sequences.csv \
      --output bcr_level4_esmc_cls.npy --pooling cls --gpu 0
  python extract_esmc.py --input bcr_level4_sequences.csv \
      --cdr-mask bcr_level4_cdr_masks.csv \
      --output bcr_level4_esmc_cdr_masked.npy --pooling cdr_masked --gpu 0
"""

import argparse, gc, json, time
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from esm.models.esmc import ESMC


def load_model(gpu: int):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = ESMC.from_pretrained("esmc_300m").to(device).eval()
    hidden_dim = model.embed.weight.shape[1]   # 960 for esmc_300m
    print(f"ESM-C 300M loaded | device={device} | hidden_dim={hidden_dim}")
    return model, device, hidden_dim


def embed_batch(model, sequences, device, pooling, cdr_masks=None):
    """
    sequences : list[str]  (no empty strings; replace with 'A' before calling)
    cdr_masks : list[np.ndarray of float32, shape (len(seq),)]  — required for cdr_masked
    returns   : np.ndarray shape [B, hidden_dim]
    """
    tokenizer = model.tokenizer
    tokens = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = tokens["input_ids"].to(device)        # [B, L_pad]
    attn_mask = tokens["attention_mask"].to(device)   # [B, L_pad]

    with torch.no_grad():
        out = model(sequence_tokens=input_ids)
    hidden = out.embeddings  # [B, L_pad, D]  — includes CLS and EOS positions

    results = []
    for i in range(len(sequences)):
        seq_len = int(attn_mask[i].sum())       # actual tokens incl. CLS + EOS
        emb_i = hidden[i, :seq_len, :]          # [seq_len, D]

        # ESM-C outputs BFloat16; cast to float32 before numpy conversion
        emb_i = emb_i.float()

        if pooling == "cls":
            # CLS token is position 0
            vec = emb_i[0].cpu().numpy()

        elif pooling == "mean":
            # Average over all non-padding tokens (includes CLS and EOS)
            vec = emb_i.mean(dim=0).cpu().numpy()

        elif pooling == "cdr_masked":
            # Residue tokens occupy positions 1 … seq_len-2 (excl. CLS=0 and EOS=-1)
            residue_embs = emb_i[1:-1]          # [L_residues, D]
            mask = cdr_masks[i]                 # float32 array, len = len(sequence)

            mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(-1)
            r_len = residue_embs.shape[0]
            # align mask length to actual residue count (truncation guard)
            if mask_t.shape[0] > r_len:
                mask_t = mask_t[:r_len]
            elif mask_t.shape[0] < r_len:
                pad = torch.zeros(r_len - mask_t.shape[0], 1, device=device)
                mask_t = torch.cat([mask_t, pad], dim=0)

            if mask_t.sum() == 0:
                vec = residue_embs.mean(dim=0).cpu().numpy()  # fallback: full mean
            else:
                vec = ((residue_embs * mask_t).sum(dim=0) / mask_t.sum()).cpu().numpy()

        results.append(vec)
    return np.array(results, dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    required=True, help="Input CSV (seq_idx,sequence)")
    p.add_argument("--output",   required=True, help="Output .npy path")
    p.add_argument("--pooling",  choices=["mean", "cls", "cdr_masked"], required=True)
    p.add_argument("--cdr-mask", default=None,  help="CDR mask CSV (required for cdr_masked)")
    p.add_argument("--gpu",      type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    if args.pooling == "cdr_masked" and args.cdr_mask is None:
        raise ValueError("--cdr-mask is required when --pooling cdr_masked")

    model, device, hidden_dim = load_model(args.gpu)

    df = pd.read_csv(args.input)
    sequences = df["sequence"].tolist()
    print(f"Total sequences: {len(sequences)}")

    all_cdr_masks = None
    if args.pooling == "cdr_masked":
        mask_df = pd.read_csv(args.cdr_mask)
        all_cdr_masks = [
            np.array(list(map(float, s.split())), dtype=np.float32)
            for s in mask_df["cdr_mask"].tolist()
        ]
        assert len(all_cdr_masks) == len(sequences), "Mask count must equal sequence count"
        print(f"Loaded {len(all_cdr_masks)} CDR masks")

    all_embeddings, errors = [], []
    t0 = time.time()

    for i in tqdm(range(0, len(sequences), args.batch_size)):
        batch_seqs = sequences[i:i + args.batch_size]
        safe_batch = [s if isinstance(s, str) and len(s) > 0 else "A" for s in batch_seqs]
        batch_masks = all_cdr_masks[i:i + args.batch_size] if all_cdr_masks is not None else None

        try:
            emb = embed_batch(model, safe_batch, device, args.pooling, batch_masks)
            # zero-out entries that were originally invalid
            for j, s in enumerate(batch_seqs):
                if not isinstance(s, str) or len(s) == 0:
                    emb[j] = np.zeros(hidden_dim, dtype=np.float32)
                    errors.append({"index": i + j, "error": "empty/invalid sequence"})
            all_embeddings.append(emb)
        except Exception as e:
            print(f"\nERROR at batch start={i}: {e}")
            all_embeddings.append(np.zeros((len(batch_seqs), hidden_dim), dtype=np.float32))
            errors.append({"index": i, "error": str(e)})

        if i > 0 and i % (args.batch_size * 20) == 0:
            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - t0
    embeddings = np.vstack(all_embeddings)
    print(f"\nShape: {embeddings.shape} | Time: {elapsed:.1f}s")

    np.save(args.output, embeddings)

    meta = {
        "model": "esmc_300m",
        "pooling": args.pooling,
        "hidden_dim": hidden_dim,
        "input_file": args.input,
        "n_sequences": len(sequences),
        "embedding_dim": int(embeddings.shape[1]),
        "batch_size": args.batch_size,
        "n_errors": len(errors),
        "errors": errors[:20],
        "total_time_seconds": round(elapsed, 1),
        "device": str(device),
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(args.gpu)
        meta["gpu_memory_peak_gb"] = round(
            torch.cuda.max_memory_allocated(args.gpu) / 1024**3, 2)

    meta_path = args.output.replace(".npy", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {args.output} + {meta_path}")


if __name__ == "__main__":
    main()
