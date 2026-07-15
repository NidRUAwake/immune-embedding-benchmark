import os
import json
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional
from pathlib import Path

MODEL_ALIASES = {
    "esm2-35m": "facebook/esm2_t12_35M_UR50D",
    "esm2-150m": "facebook/esm2_t30_150M_UR50D",
    "esm2-650m": "facebook/esm2_t33_650M_UR50D",
    "esm2-3b": "facebook/esm2_t36_3B_UR50D",
    "antiberty": "neulab/antiberty",
    "ablang": "msc-bioinformatics/AbLang",
    "tcr-bert": "wukevin/tcr-bert"
}

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"

ATCHLEY_FACTORS = {
    'A': [-0.591, -1.302, -0.733,  1.570, -0.146],
    'C': [-1.343,  0.465, -0.862, -1.020, -0.255],
    'D': [ 1.050,  0.302, -3.656, -0.259, -3.242],
    'E': [ 1.357, -1.453,  1.477,  0.113, -0.837],
    'F': [-1.006, -0.590,  1.891, -0.397,  0.412],
    'G': [-0.384,  1.652,  1.330,  1.045,  2.064],
    'H': [ 0.336, -0.417, -1.673, -1.474, -0.078],
    'I': [-1.239, -0.547,  2.131,  0.393,  0.816],
    'K': [ 1.831, -0.561,  0.533, -0.277,  1.648],
    'L': [-1.019, -0.987, -1.505,  1.266, -0.912],
    'M': [-0.663, -1.524,  2.219, -1.005,  1.212],
    'N': [ 0.945,  0.828,  1.299, -0.169,  0.933],
    'P': [ 0.189,  2.081, -1.628,  0.421, -1.392],
    'Q': [ 0.931, -0.179, -3.005, -0.503, -1.853],
    'R': [ 1.538, -0.055,  1.502,  0.440,  2.897],
    'S': [-0.228,  1.399, -4.760,  0.670, -2.647],
    'T': [-0.032,  0.326,  2.213,  0.908,  1.313],
    'V': [-1.337, -0.279, -0.544,  1.242, -1.262],
    'W': [-0.595,  0.009,  0.672, -2.128, -0.184],
    'Y': [ 0.260,  0.830,  3.097, -0.838,  1.512],
}

class Embedder(ABC):
    @abstractmethod
    def encode(self, sequences: Iterable[str]) -> np.ndarray:
        pass

class AlignmentEmbedder(Embedder):
    """Passes sequences as strings (np.ndarray of object) for downstream distance matrices."""
    def encode(self, sequences: Iterable[str]) -> np.ndarray:
        return np.array(list(sequences), dtype=object)

class CompositionEmbedder(Embedder):
    """Encodes sequences based on amino acid composition or physical properties."""
    def __init__(self, method: str):
        self.method = method
        
    def encode(self, sequences: Iterable[str]) -> np.ndarray:
        sequences = list(sequences)
        if self.method == "onehot":
            return self._encode_onehot(sequences)
        elif self.method == "atchley":
            return self._encode_atchley(sequences)
        else:
            raise ValueError(f"Unknown composition method: {self.method}")

    def _encode_onehot(self, sequences: List[str]) -> np.ndarray:
        aa_to_idx = {aa: i for i, aa in enumerate(AA_VOCAB)}
        embeddings = np.zeros((len(sequences), len(AA_VOCAB)), dtype=np.float32)
        for row_idx, seq in enumerate(sequences):
            for aa in seq:
                if (idx := aa_to_idx.get(aa)) is not None: 
                    embeddings[row_idx, idx] += 1.0
            if (norm := embeddings[row_idx].sum()) > 0: 
                embeddings[row_idx] /= norm
        return embeddings

    def _encode_atchley(self, sequences: List[str]) -> np.ndarray:
        embeddings = np.zeros((len(sequences), 5), dtype=np.float32)
        for row_idx, seq in enumerate(sequences):
            factors = [ATCHLEY_FACTORS.get(aa, [0]*5) for aa in seq if aa in ATCHLEY_FACTORS]
            if factors: 
                embeddings[row_idx] = np.mean(factors, axis=0)
        return embeddings

class TransformerEmbedder(Embedder):
    """Extracts features using a pre-trained protein language model."""
    def __init__(self, model_name: str, args=None):
        from transformers import AutoModel, AutoTokenizer
        self.batch_size = getattr(args, 'batch_size', 16) if args else 16
        self.pooling = getattr(args, 'pooling', 'mean') if args else 'mean'
        local_files_only = getattr(args, 'local_files_only', False) if args else False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Resolve alias
        hf_path = MODEL_ALIASES.get(model_name.lower(), model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path, local_files_only=local_files_only, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(hf_path, local_files_only=local_files_only, trust_remote_code=True).to(self.device).eval()

    def encode(self, sequences: Iterable[str]) -> np.ndarray:
        sequences = list(sequences)
        # TCR-BERT (wukevin/tcr-bert) tokenizer splits on whitespace, expecting
        # amino acids to be space-separated. Solid strings tokenize to [CLS,UNK,SEP].
        # See discussions/319, 320 for the diagnosis.
        if "tcr-bert" in getattr(self.tokenizer, "name_or_path", "").lower():
            sequences = [
                " ".join(list(str(s).upper().replace(" ", ""))) if s else s
                for s in sequences
            ]
        chunks = []
        for start in range(0, len(sequences), self.batch_size):
            batch_seqs = sequences[start:start + self.batch_size]
            max_len = 64 if "tcr-bert" in getattr(self.tokenizer, "name_or_path", "").lower() else 1024
            inputs = self.tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
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

class OfflineEmbedder(Embedder):
    """Loads pre-computed embeddings from disk with robust sequence matching."""
    def __init__(self, npy_path: str):
        self.npy_path = Path(npy_path)
        if not self.npy_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {npy_path}")
            
        self.data = np.load(npy_path)
        
        # Load metadata
        meta_path = self.npy_path.parent / (self.npy_path.stem + "_metadata.json")
        self.seq_to_idx = None
        self.cleaned_seq_to_idx = None
        self.stripped_seq_to_idx = None
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                seq_list = meta.get('sequence_order') or meta.get('sequences')
                if seq_list:
                    self.seq_to_idx = {s: i for i, s in enumerate(seq_list)}
                    
                    # Add cleaned mapping for robust lookup
                    try:
                        from benchmark.data import clean_sequence
                    except ImportError:
                        def clean_sequence(s): return str(s).upper().strip().replace("-", "").replace(".", "")
                        
                    self.cleaned_seq_to_idx = {}
                    for i, s in enumerate(seq_list):
                        cleaned = clean_sequence(s)
                        if cleaned and cleaned not in self.cleaned_seq_to_idx:
                            self.cleaned_seq_to_idx[cleaned] = i
                            
                    # Add stripped mapping for robust anchor residues handling (C at start, F/W/Y at end)
                    self.stripped_seq_to_idx = {}
                    for i, s in enumerate(seq_list):
                        if isinstance(s, str) and len(s) >= 3:
                            s_upper = s.upper()
                            stripped = s_upper
                            if stripped.startswith('C'):
                                stripped = stripped[1:]
                            if stripped and stripped[-1] in 'FWY':
                                stripped = stripped[:-1]
                            if stripped and stripped not in self.stripped_seq_to_idx:
                                self.stripped_seq_to_idx[stripped] = i

    def encode(self, sequences: Iterable[str]) -> np.ndarray:
        sequences = list(sequences)
        if self.seq_to_idx is None:
             if len(sequences) != len(self.data):
                raise ValueError(f"Length mismatch and no metadata: {len(sequences)} vs {len(self.data)}")
             return self.data

        dim = self.data.shape[1]
        out = np.zeros((len(sequences), dim), dtype=self.data.dtype)
        
        found_count = 0
        for i, s in enumerate(sequences):
            # 1. Direct match
            if s in self.seq_to_idx:
                out[i] = self.data[self.seq_to_idx[s]]
                found_count += 1
            # 2. Cleaned match
            elif self.cleaned_seq_to_idx is not None:
                try:
                    from benchmark.data import clean_sequence
                except ImportError:
                    def clean_sequence(s): return str(s).upper().strip().replace("-", "").replace(".", "")
                
                cs = clean_sequence(s)
                if cs in self.cleaned_seq_to_idx:
                    out[i] = self.data[self.cleaned_seq_to_idx[cs]]
                    found_count += 1
                # 3. Stripped match (input stripped vs metadata stripped)
                elif self.stripped_seq_to_idx is not None and s in self.stripped_seq_to_idx:
                    out[i] = self.data[self.stripped_seq_to_idx[s]]
                    found_count += 1
                # 4. Strip the input sequence and match against stripped metadata
                elif self.stripped_seq_to_idx is not None:
                    s_upper = s.upper()
                    stripped_s = s_upper
                    if stripped_s.startswith('C'):
                        stripped_s = stripped_s[1:]
                    if stripped_s and stripped_s[-1] in 'FWY':
                        stripped_s = stripped_s[:-1]
                    if stripped_s in self.stripped_seq_to_idx:
                        out[i] = self.data[self.stripped_seq_to_idx[stripped_s]]
                        found_count += 1
                    elif ":" in s:
                        # Paired chain fallback (using stripped_seq_to_idx first)
                        parts = s.split(":")
                        if len(parts) == 2:
                            h_idx = self.stripped_seq_to_idx.get(parts[0]) or self.seq_to_idx.get(parts[0])
                            l_idx = self.stripped_seq_to_idx.get(parts[1]) or self.seq_to_idx.get(parts[1])
                            if h_idx is not None and l_idx is not None:
                                out[i] = (self.data[h_idx] + self.data[l_idx]) / 2.0
                                found_count += 1
                elif ":" in s:
                    # Paired chain fallback
                    parts = s.split(":")
                    if len(parts) == 2:
                        h_idx = self.seq_to_idx.get(parts[0])
                        l_idx = self.seq_to_idx.get(parts[1])
                        if h_idx is not None and l_idx is not None:
                            out[i] = (self.data[h_idx] + self.data[l_idx]) / 2.0
                            found_count += 1

        if len(sequences) > 0:
            pct = (found_count / len(sequences)) * 100
            print(f"    [LOG] OfflineEmbedder: Found {found_count}/{len(sequences)} sequences ({pct:.1f}%).")
            if found_count < len(sequences):
                # Misses are left as ZERO vectors, whose cosine similarity is 0 to everything and
                # can create spurious ties / inflated ranks. Make this loud so a reproduction run
                # never silently scores on zero-vector embeddings. Set OFFLINE_EMBED_STRICT=1 to fail hard.
                import warnings, os as _os
                msg = (f"OfflineEmbedder({self.npy_path.name}): {len(sequences)-found_count} of "
                       f"{len(sequences)} sequences NOT found -> left as zero vectors. "
                       f"Reproduction should require 100% hit; check the level->.npy mapping.")
                if _os.environ.get("OFFLINE_EMBED_STRICT") == "1":
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning)

        return out

def get_embedder(model_name: str, args=None) -> Embedder:
    """Factory method to construct the appropriate Embedder."""
    # 1. Check if model_name is an actual path to a .npy file
    if isinstance(model_name, str) and model_name.endswith(".npy") and os.path.exists(model_name):
        return OfflineEmbedder(model_name)
    
    # 2. Check if we should use offline embeddings from a directory (provided in args)
    offline_dir = getattr(args, 'offline_dir', None)
    if offline_dir:
        # This logic is usually handled by the caller (run.py) by passing the .npy path directly,
        # but we keep this here as a fallback or if run.py passes the directory.
        pass

    # 3. Standard model dispatch
    model_lower = model_name.lower()
    if model_lower in ("onehot", "atchley"):
        return CompositionEmbedder(model_lower)
    elif model_lower in ("levenshtein", "blosum62", "tcrdist"):
        return AlignmentEmbedder()
    elif any(x in model_lower for x in ["esm2", "tcr-bert", "antiberty", "ablang"]):
        return TransformerEmbedder(model_name, args)
    else:
        # Try to treat as a generic transformer model name
        try:
            return TransformerEmbedder(model_name, args)
        except Exception as e:
            raise ValueError(f"Unknown model or failed to load: {model_name}. Error: {e}")
