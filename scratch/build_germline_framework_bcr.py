#!/usr/bin/env python3
"""Build a germline-framework BCR L4 variant to decompose the PLM's L3->L4 framework gain
into a germline-framework component vs a framework-SHM component (concern #2 / germline shortcut).

germ-fw sequence = germline FW1 + native CDR1 + germline FW2 + native CDR2 + germline FW3
                   + native CDR3 + native FW4
i.e. the framework (FR1-FR3) is reverted to the assigned germline V allele, while the CDRs
(and the short J-encoded FW4) are kept native. If a PLM scores germ-fw == native L4, the
framework's contribution does not depend on its somatic mutations -> it is a germline shortcut;
if germ-fw < native L4, framework SHM carried real (antigen-shaped) signal.

Germline FR residues come from the IMGT-gapped IGHV reference (imgt/IGV.fasta): in IMGT V-region
gapped format the i-th character is IMGT position i, so FR1=1-26, CDR1=27-38, FR2=39-55,
CDR2=56-65, FR3=66-104. Adds column 'Chain 1_germfw' to raw/iedb/bcr_singlechain_vh.tsv in place
(matching how 'Chain 1_L35_anarci' was added). Idempotent.
"""
import re
from pathlib import Path
import pandas as pd

TSV = "raw/iedb/bcr_singlechain_vh.tsv"
IGV = "imgt/IGV.fasta"
AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_ighv_framework():
    """{allele: (FW1, FW2, FW3, CDR1, CDR2)} for human IGHV, from the IMGT-gapped reference
    (IMGT positions: FR1 1-26, CDR1 27-38, FR2 39-55, CDR2 56-65, FR3 66-104)."""
    out = {}
    allele = None; species = None; region = None; buf = []
    def flush():
        if allele and species == "Homo sapiens" and region == "V-REGION" and buf:
            g = "".join(buf)
            fw1 = g[0:26].replace(".", "")
            cdr1 = g[26:38].replace(".", "")
            fw2 = g[38:55].replace(".", "")
            cdr2 = g[55:65].replace(".", "")
            fw3 = g[65:104].replace(".", "")
            # keep only clean AA (germline refs are clean; guard anyway)
            if all(set(x) <= AA for x in (fw1, fw2, fw3, cdr1, cdr2)) and fw1 and fw3 and cdr1 and cdr2:
                out[allele] = (fw1, fw2, fw3, cdr1, cdr2)
    for line in open(IGV):
        line = line.rstrip("\n")
        if line.startswith(">"):
            flush()
            parts = line[1:].split("|")
            allele = parts[1] if len(parts) > 1 else None
            species = parts[2].strip() if len(parts) > 2 else None
            region = parts[4].strip() if len(parts) > 4 else None
            buf = []
        else:
            buf.append(line.strip())
    flush()
    return out


def pick_allele(call, lut):
    if not isinstance(call, str) or not call:
        return None
    c = call.split(",")[0].strip()
    if c in lut:
        return c
    gene = c.split("*")[0]
    if f"{gene}*01" in lut:
        return f"{gene}*01"
    alleles = sorted(a for a in lut if a.split("*")[0] == gene)
    return alleles[0] if alleles else None


def main():
    lut = load_ighv_framework()
    print(f"Loaded {len(lut)} human IGHV alleles with framework segments")
    df = pd.read_csv(TSV, sep="\t", dtype=str)
    vh = df["Chain 1_Protein Sequence"].fillna("").astype(str)
    vcall = df["Chain 1_Vgene_anarci"] if "Chain 1_Vgene_anarci" in df else pd.Series([None]*len(df))

    def cdr(row, n):
        for col in (f"Chain 1_CDR{n}_anarci", f"Chain 1_CDR{n} ANARCI",
                    f"Chain 1_CDR{n} Curated", f"Chain 1_CDR{n} Calculated"):
            if col in df.columns:
                v = row.get(col)
                if isinstance(v, str) and v.strip():
                    return re.sub(r"[^A-Z]", "", v.strip().upper())
        return ""

    germfw = []; germcdr = []; n_fw = 0; n_cdr = 0; fw_changes = []
    for i, row in df.iterrows():
        nat = re.sub(r"[^A-Z]", "", vh[i].upper())
        c1, c2, c3 = cdr(row, 1), cdr(row, 2), cdr(row, 3)
        al = pick_allele(vcall[i], lut)
        if not (nat and c1 and c2 and c3 and al):
            germfw.append(""); germcdr.append(""); continue
        fw1, fw2, fw3, gc1, gc2 = lut[al]
        # positional split of native VH into FW1 | c1 | FW2 | c2 | FW3 | c3 | FW4
        i1 = nat.find(c1)
        rest = nat[i1 + len(c1):] if i1 >= 0 else ""
        i2 = rest.find(c2)
        rest2 = rest[i2 + len(c2):] if i2 >= 0 else ""
        i3 = rest2.find(c3)
        ok = (i1 >= 0 and i2 >= 0 and i3 >= 0)
        nfw1, nfw2, nfw3 = (nat[:i1], rest[:i2], rest2[:i3]) if ok else ("", "", "")
        fw4 = rest2[i3 + len(c3):] if ok else (nat[nat.rfind(c3) + len(c3):] if nat.rfind(c3) >= 0 else "")
        # germ-fw  : germline FR1-3 + native CDR1/2/3 + native FW4
        s_fw = fw1 + c1 + fw2 + c2 + fw3 + c3 + fw4
        # germ-cdr : native FR + germline CDR1/2 + native CDR3 + native FW4
        s_cdr = (nfw1 + gc1 + nfw2 + gc2 + nfw3 + c3 + fw4) if ok else ""
        germfw.append(s_fw if (set(s_fw) <= AA and len(s_fw) >= 90) else "")
        germcdr.append(s_cdr if (ok and set(s_cdr) <= AA and len(s_cdr) >= 90) else "")
        if germfw[-1]:
            n_fw += 1
            j = nat.rfind(c3); nat_fr = nat[:j] if j >= 0 else nat; germ_fr = fw1 + c1 + fw2 + c2 + fw3
            fw_changes.append(sum(a != b for a, b in zip(nat_fr, germ_fr)) + abs(len(nat_fr) - len(germ_fr)))
        if germcdr[-1]:
            n_cdr += 1
    df["Chain 1_germfw"] = germfw
    df["Chain 1_germcdr"] = germcdr
    df.to_csv(TSV, sep="\t", index=False)
    import numpy as np
    fc = np.array(fw_changes)
    print(f"Built germ-fw for {n_fw}/{len(df)} and germ-cdr for {n_cdr}/{len(df)} records.")
    print(f"FR residues reverted to germline: mean {fc.mean():.1f}, median {np.median(fc):.0f}, "
          f"max {fc.max()}, %==0: {100*(fc==0).mean():.0f}%")
    ex = df[df['Chain 1_germcdr'] != ""].iloc[0]
    print("example native  :", re.sub(r'[^A-Z]','',str(ex['Chain 1_Protein Sequence']).upper())[:80])
    print("example germ-fw :", ex['Chain 1_germfw'][:80])
    print("example germ-cdr:", ex['Chain 1_germcdr'][:80])


if __name__ == "__main__":
    main()
