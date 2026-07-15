#!/bin/bash
set -euo pipefail

# Download McPAS-TCR data
# Source: https://friedmanlab.weizmann.ac.il/McPAS-TCR/
# License: Check website for usage terms
#
# ⚠️ MANUAL DOWNLOAD REQUIRED (verified 2026-06-08).
# The old static CSV URL (http://friedmanlab.weizmann.ac.il/McPAS-TCR/data/McPAS-TCR.csv)
# is DEAD (302 -> 404). The site is now an interactive R Shiny app and the
# database is served through a "Download Database" button (Shiny downloadHandler
# id="downloadDB") that needs a live WebSocket session — it is NOT fetchable via
# a plain curl/wget. Because McPAS also serves only "current" (no version pin),
# byte-exact reproduction of the paper's 2026-05-19 snapshot must use the snapshot
# bundled with the Zenodo data package, not a fresh download.
#
# Manual steps:
#   1. Visit https://friedmanlab.weizmann.ac.il/McPAS-TCR/
#   2. Click the "Download Database" button (downloads McPAS-TCR.csv).
#   3. Place it at raw/McPAS/McPAS-TCR.csv
#   4. Run this script again to do the Human filter + standardization below.

mkdir -p raw/McPAS

if [ ! -f raw/McPAS/McPAS-TCR.csv ]; then
  echo "ERROR: raw/McPAS/McPAS-TCR.csv not found."
  echo "Please download it manually via the Shiny app (see header), then re-run."
  exit 1
fi

# Filter for Human as expected by the pipeline.
python -c "
import pandas as pd
df = pd.read_csv('raw/McPAS/McPAS-TCR.csv', encoding='ISO-8859-1', low_memory=False)
df[df['Species'] == 'Human'].to_csv('raw/McPAS/McPAS-TCR_human.tsv', sep='\t', index=False)
"
echo "Wrote raw/McPAS/McPAS-TCR_human.tsv"
echo "NOTE: the canonical pipeline input is outputs/intermediate/mcpas_standardized.tsv,"
echo "      produced by the McPAS standardization step (see scripts/preprocessing/)."
