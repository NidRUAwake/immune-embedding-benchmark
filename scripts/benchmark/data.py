import json
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, List

def strip_tcr_cdr3_anchors(seq: str) -> str:
    """
    Strip IMGT anchor residues from TCR CDR3 sequences stored in VDJdb/McPAS format.
    VDJdb and McPAS store CDR3 including the conserved 104C (N-terminal) and 118F/W/Y
    (C-terminal) anchor residues. Per IMGT convention and our benchmark definition,
    CDR3 is defined as residues 105-117, excluding these anchors.

    Case normalization (seq.upper()) is applied first to handle McPAS data quality
    issues: 24 records contain lowercase letters (21 with lowercase 'f' at C-terminal
    anchor position, 2 with internal lowercase 'y', 1 non-sequence entry filtered
    downstream by clean_sequence). VDJdb CDR3 sequences are confirmed fully uppercase
    (0/226,917 records contain lowercase), so upper() is a no-op for VDJdb inputs.
    This fix was officially adopted as mainline on 2026-05-18 (D144/D146).
    """
    if not isinstance(seq, str) or len(seq) < 3:
        return seq
    seq = seq.upper()
    if seq.startswith('C'):
        seq = seq[1:]
    if seq and seq[-1] in 'FWY':
        seq = seq[:-1]
    return seq


# ============================================================================
# Constants & Vocab
# ============================================================================

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA_VOCAB + "BJXZ" + "|_:*-." + "0123456789")

# IMGT Lookup Tables (Lazy Loading)
_trv_cdr = {}
_trv_full = {}
_trj_full = {}

def load_imgt_lookups():
    global _trv_cdr, _trv_full, _trj_full
    lookup_dir = Path("outputs/imgt_lookup")
    if (lookup_dir / "trv_cdr_lookup.json").exists():
        with open(lookup_dir / "trv_cdr_lookup.json", "r") as f:
            _trv_cdr = json.load(f)
    if (lookup_dir / "trv_full_lookup.json").exists():
        with open(lookup_dir / "trv_full_lookup.json", "r") as f:
            _trv_full = json.load(f)
    if (lookup_dir / "trj_full_lookup.json").exists():
        with open(lookup_dir / "trj_full_lookup.json", "r") as f:
            _trj_full = json.load(f)

def load_label_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {"aliases": {}, "non_protein_labels": []}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Global configuration (can be re-initialized by run.py)
LABEL_CONFIG = load_label_config("label_aliases.json")

def get_alias_map(config: dict) -> dict:
    alias_map = {}
    for canonical, aliases in config.get("aliases", {}).items():
        for alias in aliases:
            alias_map[alias.lower()] = canonical
    return alias_map

LABEL_MERGE_MAP = get_alias_map(LABEL_CONFIG)
NON_PROTEIN_LABELS = set(config.lower() for config in LABEL_CONFIG.get("non_protein_labels", []))

def init_label_config(config_path: str):
    """Re-initialize global label config from a specific path."""
    global LABEL_CONFIG, LABEL_MERGE_MAP, NON_PROTEIN_LABELS
    LABEL_CONFIG = load_label_config(config_path)
    LABEL_MERGE_MAP = get_alias_map(LABEL_CONFIG)
    NON_PROTEIN_LABELS = set(config.lower() for config in LABEL_CONFIG.get("non_protein_labels", []))

import re
def normalize_gene_name(name):
    """Normalize TCR gene names by removing leading zeros from family/gene only. 
    TRBV06-05*01 -> TRBV6-5*01.
    """
    if not name or pd.isna(name): return name
    name = str(name).strip().upper()
    
    # Handle 'S' nomenclature: TRBV23S1 -> TRBV23-1
    name = name.replace("S", "-")
    
    # Split by allele if present
    if "*" in name:
        base, allele = name.split("*", 1)
        # Normalize only the base
        parts = re.split(r'([0-9]+)', base)
        norm_base = "".join(str(int(p)) if p.isdigit() else p for p in parts)
        # Re-attach allele (keep leading zero in allele)
        return f"{norm_base}*{allele.zfill(2)}"
    else:
        # No allele, normalize whole thing
        parts = re.split(r'([0-9]+)', name)
        return "".join(str(int(p)) if p.isdigit() else p for p in parts)

# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class DatasetSpec:
    name: str
    path: Path
    sep: str
    label_col: str
    sequence_builders: dict[str, Callable[[pd.DataFrame], pd.Series]]

@dataclass
class PilotSlice:
    dataset: str
    level: str
    label_col: str
    sequence_col: str
    dataframe: pd.DataFrame

# ============================================================================
# Label & Sequence Cleaning
# ============================================================================

def normalize_labels(series: pd.Series, case_normalize: bool = True) -> pd.Series:
    """Normalize, clean, and merge labels (without filtering non-protein antigens)."""
    def _norm(value: object) -> Optional[str]:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return None
            
        if case_normalize:
            text = text.lower()
            
        # Synonym Merging
        text = LABEL_MERGE_MAP.get(text, text)
        return text
        
    return series.map(_norm)

def assign_antigen_type(series: pd.Series) -> pd.Series:
    """Assign 'protein' or 'non-protein' based on the label."""
    return series.map(lambda x: "non-protein" if x and x.lower() in NON_PROTEIN_LABELS else "protein")

def clean_sequence(seq: object) -> Optional[str]:
    """Remove invalid amino acids and format sequence."""
    if pd.isna(seq):
        return None
    text = str(seq).strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return None
    cleaned = "".join(ch for ch in text if ch in AA_SET)
    return cleaned or None

_STD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def is_legit_sequence(s) -> bool:
    """A legitimate model input: every delimiter-separated segment (split on '|' for
    multi-CDR levels and ':' for paired chains) is non-empty and contains ONLY the 20
    standard amino acids. Drops records carrying stop codons ('*'), ambiguity codes
    (X/B/Z/J/U/O), residual gaps, or an empty segment (a missing CDR, or a single-chain
    record in a paired representation). Strict 20-AA rule; see the data-legitimacy audit."""
    if not isinstance(s, str) or s == "":
        return False
    for seg in s.replace(":", "|").split("|"):
        if seg == "" or (set(seg) - _STD_AA):
            return False
    return True

def combine_first_available(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Fallback mechanism to build sequences from available columns."""
    values = pd.Series([None] * len(df), index=df.index, dtype="object")
    for col in columns:
        if col not in df.columns:
            continue
        candidate = df[col].map(clean_sequence)
        values = values.where(values.notna(), candidate)
    return values

def combine_first_available_raw(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Fallback mechanism to pull raw strings (no clean_sequence)."""
    values = pd.Series([None] * len(df), index=df.index, dtype="object")
    for col in columns:
        if col not in df.columns:
            continue
        candidate = df[col]
        values = values.where(values.notna(), candidate)
    return values

def normalize_gene_value(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NA", "U", "UNKNOWN"}:
        return None
    return text


# ============================================================================
# Specs & Spec Loading
# ============================================================================

def build_dataset_specs(args) -> List[DatasetSpec]:
    """Build BCR/TCR/SAbDab dataset specs from CLI args."""
    specs = []
    
    # BCR
    if getattr(args, "include_bcr", False):
        bcr_builders = {
            "level1": lambda df: combine_first_available(df, ["Chain 1_CDR3 ANARCI", "Chain 1_CDR3 Curated", "Chain 1_CDR3 Calculated"]).map(clean_sequence),
            "level2": lambda df: combine_first_available(df, ["Chain 1_CDR3 ANARCI", "Chain 1_CDR3 Curated", "Chain 1_CDR3 Calculated"]).map(clean_sequence),
            "level2.5": lambda df: combine_first_available(df, ["Chain 1_CDR3 ANARCI", "Chain 1_CDR3 Curated", "Chain 1_CDR3 Calculated"]).map(clean_sequence),
            "level3": lambda df: (combine_first_available(df, ["Chain 1_CDR1_anarci", "Chain 1_CDR1 Curated", "Chain 1_CDR1 Calculated"]).fillna("") + "|" +
                                 combine_first_available(df, ["Chain 1_CDR2_anarci", "Chain 1_CDR2 Curated", "Chain 1_CDR2 Calculated"]).fillna("") + "|" +
                                 combine_first_available(df, ["Chain 1_CDR3_anarci", "Chain 1_CDR3 ANARCI", "Chain 1_CDR3 Curated", "Chain 1_CDR3 Calculated"]).fillna("")).map(clean_sequence),
            "level4": lambda df: combine_first_available(df, ["Chain 1_Protein Sequence", "Chain 1_Full Sequence"]).map(clean_sequence),
        }
        # Optional BCR levels from the chain-composition cleanup (doc 324). Gated so they
        # only activate on the file that actually carries the data, and so the main m8 BCR
        # run (which iterates sequence_builders) does NOT accidentally run them on the
        # single-chain file. They are evaluated on dedicated files / explicit --levels:
        #   level3.5      : length-matched framework + native CDR3 (CDR1/CDR2 -> glycine).
        #                   Gated on the precomputed `Chain 1_L35_anarci` column (single-chain VH file only).
        #   level4_paired : heavy VH : light VL concatenation. Gated on the dedicated
        #                   bcr_paired_hl file (the 197 H+L records).
        _bcr_path = str(getattr(args, "bcr_file", "raw/iedb/bcr_singlechain_vh.tsv"))
        try:
            _bcr_cols = set(pd.read_csv(_bcr_path, sep="\t", nrows=0).columns)
        except Exception:
            _bcr_cols = set()
        if "Chain 1_L35_anarci" in _bcr_cols and "paired" not in _bcr_path.lower():
            bcr_builders["level3.5"] = lambda df: combine_first_available(df, ["Chain 1_L35_anarci"]).map(clean_sequence)
        if "paired" in _bcr_path.lower():
            bcr_builders["level4_paired"] = lambda df: (
                combine_first_available(df, ["Chain 1_Protein Sequence", "Chain 1_Full Sequence"]).fillna("") + ":" +
                combine_first_available(df, ["Chain 2_Protein Sequence", "Chain 2_Full Sequence"]).fillna("")
            ).map(clean_sequence)
        specs.append(DatasetSpec(
            name="BCR",
            path=Path(getattr(args, "bcr_file", "raw/iedb/bcr_singlechain_vh.tsv")),
            sep="\t",
            label_col=getattr(args, "bcr_label_col", "Epitope_Source Molecule"),
            sequence_builders=bcr_builders,
        ))

    # TCR
    if getattr(args, "include_tcr", False):
        load_imgt_lookups()
        tcr_file = getattr(args, "tcr_file", "raw/vdjdb/vdjdb_full.txt")
        is_mcpas = "mcpas" in tcr_file.lower()
        
        tcr_builders = {
            "level1": lambda df: combine_first_available(df, ["cdr3.beta", "cdr3.alpha"]).map(strip_tcr_cdr3_anchors).map(clean_sequence),
            "level2": lambda df: combine_first_available(df, ["cdr3.beta", "cdr3.alpha"]).map(strip_tcr_cdr3_anchors).map(clean_sequence),
            "level2.5": lambda df: combine_first_available(df, ["cdr3.beta", "cdr3.alpha"]).map(strip_tcr_cdr3_anchors).map(clean_sequence),
            "level3": _tcr_l3,
            "level4": _tcr_l4,
        }
        
        if not is_mcpas:
            _paired_cdr3 = lambda df: (combine_first_available(df, ["cdr3.beta"]).map(strip_tcr_cdr3_anchors).fillna("") + ":" + combine_first_available(df, ["cdr3.alpha"]).map(strip_tcr_cdr3_anchors).fillna("")).map(clean_sequence)
            tcr_builders["level1_paired"] = _paired_cdr3
            # L2/L2.5 paired are clonotype-retrieval variants (two-stage: germline-gene
            # filter on BOTH chains, then rank by the paired CDR3). They embed the paired
            # CDR3 (the L1_paired input); the V/J gene filter is applied at retrieval time
            # (run.py vj_filter), exactly as for the single-chain L2/L2.5.
            tcr_builders["level2_paired"] = _paired_cdr3
            tcr_builders["level2.5_paired"] = _paired_cdr3
            tcr_builders["level3_paired"] = lambda df: (_tcr_l3(df) + ":" + _tcr_l3_alpha(df)).map(clean_sequence)
            tcr_builders["level4_paired"] = lambda df: (_tcr_l4(df) + ":" + _tcr_l4_alpha(df)).map(clean_sequence)
        
        specs.append(DatasetSpec(
            name="mcpas" if is_mcpas else "TCR",
            path=Path(tcr_file),
            sep="\t",
            label_col=getattr(args, "tcr_label_col", "antigen.epitope"),
            sequence_builders=tcr_builders,
        ))

    # SAbDab
    if getattr(args, "include_sabdab", False):
        sabdab_builders = {
            "level1": lambda df: combine_first_available(df, ["cdr3_heavy"]).map(clean_sequence),
            "level2": lambda df: combine_first_available(df, ["cdr3_heavy"]).map(clean_sequence),
            "level2.5": lambda df: combine_first_available(df, ["cdr3_heavy"]).map(clean_sequence),
            "level3": lambda df: (combine_first_available(df, ["cdr1_heavy"]).fillna("") + "|" +
                                 combine_first_available(df, ["cdr2_heavy"]).fillna("") + "|" +
                                 combine_first_available(df, ["cdr3_heavy"]).fillna("")).map(clean_sequence),
            "level4": lambda df: combine_first_available(df, ["sequence_heavy"]).map(clean_sequence),
            "level4_paired": lambda df: (combine_first_available(df, ["sequence_heavy"]) + ":" +
                                        combine_first_available(df, ["sequence_light"])).map(clean_sequence),
            # Paired CDR3 (heavy:light) — used as embedding input for L1/L2/L2.5 paired.
            # Requires cdr3_light from sabdab_vj_annotated_paired.csv.
            "level1_paired": lambda df: (combine_first_available(df, ["cdr3_heavy"]).fillna("") + ":" +
                                         combine_first_available(df, ["cdr3_light"]).fillna("")).map(clean_sequence),
            "level2_paired": lambda df: (combine_first_available(df, ["cdr3_heavy"]).fillna("") + ":" +
                                         combine_first_available(df, ["cdr3_light"]).fillna("")).map(clean_sequence),
            "level2.5_paired": lambda df: (combine_first_available(df, ["cdr3_heavy"]).fillna("") + ":" +
                                            combine_first_available(df, ["cdr3_light"]).fillna("")).map(clean_sequence),
            # Paired all-CDR (heavy:light CDR1|CDR2|CDR3 concatenation).
            "level3_paired": lambda df: (
                combine_first_available(df, ["cdr1_heavy"]).fillna("") + "|" +
                combine_first_available(df, ["cdr2_heavy"]).fillna("") + "|" +
                combine_first_available(df, ["cdr3_heavy"]).fillna("") + ":" +
                combine_first_available(df, ["cdr1_light"]).fillna("") + "|" +
                combine_first_available(df, ["cdr2_light"]).fillna("") + "|" +
                combine_first_available(df, ["cdr3_light"]).fillna("")
            ).map(clean_sequence),
        }
        # Use the paired-annotated file (includes light-chain CDRs and V/J genes) when
        # any paired level is requested; fall back to the base file otherwise.
        sabdab_path = getattr(args, "sabdab_file", "raw/sabdab/sabdab_paired_clean_human_full.csv")
        annotated_path = "outputs/intermediate/sabdab_vj_annotated_paired.csv"
        if Path(annotated_path).exists():
            sabdab_path = annotated_path
        specs.append(DatasetSpec(
            name="SAbDab",
            path=Path(sabdab_path),
            sep=",",
            label_col=getattr(args, "sabdab_label_col", "label"),
            sequence_builders=sabdab_builders,
        ))

    return specs

# ============================================================================
# Loading & Slicing
# ============================================================================

def load_table(spec: DatasetSpec) -> pd.DataFrame:
    """Load dataframe from disk."""
    path = spec.path.expanduser().resolve()
    if not path.exists(): 
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, sep=spec.sep, low_memory=False)


def _is_human_series(series: pd.Series) -> pd.Series:
    """Return boolean mask for human annotations in a text-like column."""
    s = series.fillna("").astype(str).str.strip().str.lower()
    return (
        s.str.contains("ncbitaxon_9606", regex=False)
        | s.str.contains("homo sapiens", regex=False)
        | s.str.contains("homosapiens", regex=False)
        | (s == "human")
    )


def filter_human_only(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """
    Keep only human receptor rows.
    - BCR: rely on chain organism annotation columns.
    - TCR: rely on `species` when available.
    """
    if df.empty:
        return df

    if spec.name == "BCR":
        candidate_cols = [
            "Chain 1_Organism IRI",
            "Chain 2_Organism IRI",
            "Chain 1_Organism Name",
            "Chain 2_Organism Name",
        ]
    elif spec.name == "SAbDab":
        candidate_cols = ["hmm_species", "heavy_species", "light_species"]
    else:
        candidate_cols = [
            "species",
            "Species",
            "host.species",
        ]

    masks = []
    for col in candidate_cols:
        if col in df.columns:
            masks.append(_is_human_series(df[col]))

    if not masks:
        # No species metadata -> keep original to avoid silent data loss.
        return df

    combined = masks[0].copy()
    for m in masks[1:]:
        combined = combined | m
    return df[combined].copy()

def compute_shared_labels(
    df: pd.DataFrame, 
    spec: DatasetSpec, 
    top_labels: int, 
    min_label_count: int, 
    label_contains: Optional[str] = None
) -> List[str]:
    """Ensures consistent label subset across granularities."""
    labels = normalize_labels(df[spec.label_col])
    per_level_eligible: List[set[str]] = []
    
    for level, builder in spec.sequence_builders.items():
        if spec.name == "BCR" and level not in ["level1", "level4"]:
            continue
        sequences = builder(df)
        work = pd.DataFrame({"label": labels, "sequence": sequences}).dropna()
        if label_contains:
            work = work[work["label"].str.contains(label_contains, case=False, na=False)]
        counts = work["label"].value_counts()
        per_level_eligible.append(set(counts[counts >= min_label_count].index))
        
    if not per_level_eligible: 
        return []
    shared = set.intersection(*per_level_eligible)
    if len(shared) < 2: 
        return []
    
    first_level = list(spec.sequence_builders.keys())[0]
    work = pd.DataFrame({"label": labels, "sequence": spec.sequence_builders[first_level](df)}).dropna()
    counts = work[work["label"].isin(shared)]["label"].value_counts()
    return counts.head(top_labels).index.tolist()

def build_pilot_slice(
    df: pd.DataFrame, 
    spec: DatasetSpec, 
    level: str, 
    max_per_label: int, 
    random_state: int, 
    forced_labels: Optional[List[str]],
    label_contains: Optional[str] = None,
) -> Optional[PilotSlice]:
    """Build a dataset slice for a specific level of granularity."""
    labels = normalize_labels(df[spec.label_col])
    sequences = spec.sequence_builders[level](df)
    split_sequences = spec.sequence_builders["level1"](df) if "level1" in spec.sequence_builders else sequences

    v_gene = pd.Series([None] * len(df), index=df.index, dtype="object")
    j_gene = pd.Series([None] * len(df), index=df.index, dtype="object")
    if spec.name == "BCR":
        v_gene = combine_first_available_raw(
            df,
            ["Chain 1_Vgene_anarci", "Chain 1_Curated V Gene", "Chain 1_Calculated V Gene"],
        )
        j_gene = combine_first_available_raw(
            df,
            ["Chain 1_Jgene_anarci", "Chain 1_Curated J Gene", "Chain 1_Calculated J Gene"],
        )
    elif spec.name in {"TCR", "mcpas"} and level in {"level2_paired", "level2.5_paired"}:
        # Paired clonotype: the gene key spans BOTH chains. The vj_filter then requires the
        # query and candidate to share the same beta+alpha V genes (L2.5_paired) or the same
        # beta+alpha V and J genes (L2_paired). A row missing any chain gene is left null so
        # evaluate.py scores it as a miss (no valid clonotype).
        vb = df.get("v.beta",  pd.Series([None] * len(df))).map(normalize_gene_name)
        va = df.get("v.alpha", pd.Series([None] * len(df))).map(normalize_gene_name)
        jb = df.get("j.beta",  pd.Series([None] * len(df))).map(normalize_gene_name)
        ja = df.get("j.alpha", pd.Series([None] * len(df))).map(normalize_gene_name)
        v_gene = pd.Series(
            [f"{b}|{a}" if isinstance(b, str) and isinstance(a, str) else None for b, a in zip(vb, va)],
            index=df.index, dtype="object")
        j_gene = pd.Series(
            [f"{b}|{a}" if isinstance(b, str) and isinstance(a, str) else None for b, a in zip(jb, ja)],
            index=df.index, dtype="object")
    elif spec.name == "SAbDab" and level in {"level2_paired", "level2.5_paired"}:
        # Paired clonotype for SAbDab: composite key = heavy_v|light_v (and heavy_j|light_j
        # for L2_paired). A row missing any chain gene is scored as a miss.
        vh = df.get("heavy_v_gene", pd.Series([None] * len(df))).map(normalize_gene_name)
        vl = df.get("light_v_gene", pd.Series([None] * len(df))).map(normalize_gene_name)
        jh = df.get("heavy_j_gene", pd.Series([None] * len(df))).map(normalize_gene_name)
        jl = df.get("light_j_gene", pd.Series([None] * len(df))).map(normalize_gene_name)
        v_gene = pd.Series(
            [f"{h}|{l}" if isinstance(h, str) and isinstance(l, str) else None for h, l in zip(vh, vl)],
            index=df.index, dtype="object")
        j_gene = pd.Series(
            [f"{h}|{l}" if isinstance(h, str) and isinstance(l, str) else None for h, l in zip(jh, jl)],
            index=df.index, dtype="object")
    elif spec.name in {"TCR", "mcpas"}:
        v_gene = df.get("v.beta", pd.Series([None] * len(df))).copy()
        j_gene = df.get("j.beta", pd.Series([None] * len(df))).copy()
        v_gene = v_gene.map(normalize_gene_name)
        j_gene = j_gene.map(normalize_gene_name)
    elif spec.name == "SAbDab":
        v_gene = df.get("heavy_v_gene", pd.Series([None] * len(df))).copy()
        j_gene = df.get("heavy_j_gene", pd.Series([None] * len(df))).copy()

    v_gene = v_gene.map(normalize_gene_value)
    j_gene = j_gene.map(normalize_gene_value)
    # Clonotype matching (L2/L2.5) is at the V/J GENE level, not the allele level:
    # collapse the allele suffix so e.g. IGHV3-23*04 and IGHV3-23*01 count as the same V gene.
    # Paired clonotype keys are "beta|alpha"; collapse the allele of each chain separately.
    _to_gene = lambda g: ("|".join(p.split("*")[0] for p in g.split("|")) if "|" in g else g.split("*")[0]) if isinstance(g, str) else g
    v_gene = v_gene.map(_to_gene)
    j_gene = j_gene.map(_to_gene)
    work = pd.DataFrame({
        "label": labels, 
        "sequence": sequences,
        "split_sequence": split_sequences,
        "antigen_type": assign_antigen_type(labels),
        "v_gene": v_gene,
        "j_gene": j_gene,
    }).dropna(subset=["label", "sequence", "split_sequence"])

    # Strict 20-AA legitimacy filter: drop records whose level-specific sequence is not a
    # clean amino-acid string (stop codons, ambiguity codes, gaps, or empty segments). This
    # is applied per level, so a record clean at one level but illegitimate at another (e.g.
    # X only in the full-chain framework) is dropped only where its input is malformed.
    work = work[work["sequence"].map(is_legit_sequence)]

    if label_contains:
        work = work[work["label"].str.contains(label_contains, case=False, na=False)]
    
    if forced_labels is not None:
        if len(forced_labels) == 0:
            return None
        work = work[work["label"].isin(forced_labels)]
    
    if work.empty or work["label"].nunique() < 2:
        return None

    sampled_parts = []
    for label in work["label"].unique():
        label_df = work[work["label"] == label].copy()
        if len(label_df) > max_per_label:
            label_df = label_df.sample(max_per_label, random_state=random_state)
        sampled_parts.append(label_df)

    pilot_df = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sequence_col = f"{spec.name.lower()}_{level}_sequence"
    pilot_df = pilot_df.rename(columns={"sequence": sequence_col})
    
    return PilotSlice(spec.name, level, spec.label_col, sequence_col, pilot_df)

# ============================================================================
# TCR Helper Builders
# ============================================================================

def _tcr_l3(df: pd.DataFrame) -> pd.Series:
    """Reconstruct CDR1|CDR2|CDR3 for TCR using IMGT germline CDR1/CDR2."""
    vgenes = df.get("v.beta", pd.Series(["U"]*len(df))).fillna("U")
    cdr3s = combine_first_available(df, ["cdr3.beta", "cdr3.alpha"]).fillna("")
    parts = []
    for vg, c3 in zip(vgenes, cdr3s):
        norm_vg = normalize_gene_name(vg)
        
        # Multi-stage fallback
        candidates = [norm_vg, norm_vg.split("*")[0] + "*01"]
        base = norm_vg.split("*")[0]
        if "-" not in base:
            candidates.append(base + "-1*01")
        else:
            candidates.append(base.split("-")[0] + "*01")
            
        key = next((c for c in candidates if c in _trv_cdr), None)
        info = _trv_cdr.get(key, {}) if key else {}
        cdr1 = info.get("cdr1", "")
        cdr2 = info.get("cdr2", "")
        
        # Ensure CDR3 doesn't have duplicate anchors if possible
        # Su Ming says VDJdb is clean, but let's be robust
        clean_c3 = c3
        if clean_c3.startswith("C") and len(clean_c3) > 5: clean_c3 = clean_c3[1:]
        if clean_c3.endswith(("F", "W", "Y")) and len(clean_c3) > 5: clean_c3 = clean_c3[:-1]

        # Records with no resolvable germline V-gene have no CDR1/CDR2 -> they are not a
        # valid "three-CDR" (L3) input. Return None so build_pilot_slice drops them from L3
        # only (L1/L2/L4 builders are unaffected). This avoids a format-clustering confound
        # where CDR3-only records segregate from full-3-CDR records in the L3 retrieval pool.
        # ~10% of VDJdb / ~5% of McPAS records are CDR3-only deposits (no V-gene); see doc 324.
        if not cdr1 and not cdr2:
            parts.append(None)
        else:
            parts.append(f"{cdr1}|{cdr2}|{clean_c3}")
    return pd.Series(parts, index=df.index).map(lambda s: clean_sequence(s) if isinstance(s, str) else None)

def _tcr_l3_alpha(df: pd.DataFrame) -> pd.Series:
    """Reconstruct CDR1|CDR2|CDR3 for the TCR alpha chain (mirror of _tcr_l3)."""
    vgenes = df.get("v.alpha", pd.Series(["U"]*len(df))).fillna("U")
    cdr3s = df.get("cdr3.alpha", pd.Series([""]*len(df))).fillna("")
    parts = []
    for vg, c3 in zip(vgenes, cdr3s):
        norm_vg = normalize_gene_name(vg)
        candidates = [norm_vg, norm_vg.split("*")[0] + "*01"]
        base = norm_vg.split("*")[0]
        if "-" not in base:
            candidates.append(base + "-1*01")
        else:
            candidates.append(base.split("-")[0] + "*01")
        key = next((c for c in candidates if c in _trv_cdr), None)
        info = _trv_cdr.get(key, {}) if key else {}
        cdr1 = info.get("cdr1", "")
        cdr2 = info.get("cdr2", "")
        clean_c3 = c3
        if clean_c3.startswith("C") and len(clean_c3) > 5: clean_c3 = clean_c3[1:]
        if clean_c3.endswith(("F", "W", "Y")) and len(clean_c3) > 5: clean_c3 = clean_c3[:-1]
        parts.append(f"{cdr1}|{cdr2}|{clean_c3}")
    return pd.Series(parts, index=df.index).map(clean_sequence)


def _tcr_l4(df: pd.DataFrame) -> pd.Series:
    """Reconstruct approximate full-length TCR beta chain."""
    vgenes = df.get("v.beta", pd.Series(["U"]*len(df))).fillna("U")
    jgenes = df.get("j.beta", pd.Series(["U"]*len(df))).fillna("U")
    cdr3s = combine_first_available(df, ["cdr3.beta", "cdr3.alpha"]).fillna("")
    parts = []
    for vg, jg, c3 in zip(vgenes, jgenes, cdr3s):
        norm_vg = normalize_gene_name(vg)
        norm_jg = normalize_gene_name(jg)
        
        # V-fallback
        v_cands = [norm_vg, norm_vg.split("*")[0] + "*01"]
        v_base = norm_vg.split("*")[0]
        if "-" not in v_base: v_cands.append(v_base + "-1*01")
        else: v_cands.append(v_base.split("-")[0] + "*01")
        v_key = next((c for c in v_cands if c in _trv_full), None)
        
        # J-fallback
        j_cands = [norm_jg, norm_jg.split("*")[0] + "*01"]
        j_base = norm_jg.split("*")[0]
        if "-" not in j_base: j_cands.append(j_base + "-1*01")
        else: j_cands.append(j_base.split("-")[0] + "*01")
        j_key = next((c for c in j_cands if c in _trj_full), None)
        
        v_seq = _trv_full.get(v_key, "") if v_key else ""
        j_seq = _trj_full.get(j_key, "") if j_key else ""
        
        if v_seq and j_seq and c3:
            # Strip anchors from CDR3 as they are provided by V/J templates
            clean_c3 = c3
            if clean_c3.startswith("C"): clean_c3 = clean_c3[1:]
            if clean_c3.endswith(("F", "W", "Y")): clean_c3 = clean_c3[:-1]
            
            # V[1..104] + CDR3[105..117] + J[118..end]
            full = v_seq + clean_c3 + j_seq
            parts.append(full)
        else:
            parts.append(None)
            
    return pd.Series(parts, index=df.index).map(clean_sequence)

def _tcr_l4_alpha(df: pd.DataFrame) -> pd.Series:
    """Reconstruct approximate full-length TCR alpha chain."""
    vgenes = df.get("v.alpha", pd.Series(["U"]*len(df))).fillna("U")
    jgenes = df.get("j.alpha", pd.Series(["U"]*len(df))).fillna("U")
    cdr3s = df.get("cdr3.alpha", pd.Series([""]*len(df))).fillna("")
    parts = []
    for vg, jg, c3 in zip(vgenes, jgenes, cdr3s):
        if not c3:
            parts.append("")
            continue
            
        norm_vg = normalize_gene_name(vg)
        norm_jg = normalize_gene_name(jg)
        
        # V-fallback
        v_cands = [norm_vg, norm_vg.split("*")[0] + "*01"]
        v_base = norm_vg.split("*")[0]
        if "-" not in v_base: v_cands.append(v_base + "-1*01")
        else: v_cands.append(v_base.split("-")[0] + "*01")
        v_key = next((c for c in v_cands if c in _trv_full), None)
        
        # J-fallback
        j_cands = [norm_jg, norm_jg.split("*")[0] + "*01"]
        j_base = norm_jg.split("*")[0]
        if "-" not in j_base: j_cands.append(j_base + "-1*01")
        else: j_cands.append(j_base.split("-")[0] + "*01")
        j_key = next((c for c in j_cands if c in _trj_full), None)
        
        v_seq = _trv_full.get(v_key, "") if v_key else ""
        j_seq = _trj_full.get(j_key, "") if j_key else ""
        
        if v_seq and j_seq and c3:
            clean_c3 = c3
            if clean_c3.startswith("C"): clean_c3 = clean_c3[1:]
            if clean_c3.endswith(("F", "W", "Y")): clean_c3 = clean_c3[:-1]
            full = v_seq + clean_c3 + j_seq
            parts.append(full)
        else:
            parts.append(None)
            
    return pd.Series(parts, index=df.index).map(clean_sequence)

def load_mcpas_tcr(path: str) -> pd.DataFrame:
    """
    Load McPAS-TCR human data and standardize to VDJdb-compatible format.
    Strips IMGT anchor residues from CDR3 (McPAS stores C...F/W/Y format).
    Returns DataFrame with columns: cdr3.beta, antigen.epitope
    """
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df = df[df['cdr3'].notna() & df['Epitope.peptide'].notna()].copy()
    df['cdr3.beta'] = df['cdr3'].map(strip_tcr_cdr3_anchors).map(clean_sequence)
    df['antigen.epitope'] = df['Epitope.peptide'].str.lower().str.strip()
    return df[['cdr3.beta', 'antigen.epitope']].drop_duplicates('cdr3.beta')

