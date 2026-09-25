#!/bin/bash
set -euo pipefail

# Download CoV-AbDab (Coronavirus Antibody Database)
# Source:  https://opig.stats.ox.ac.uk/webapps/covabdab/
# Paper:   Raybould et al. (2021), Bioinformatics
# License: free for academic use, per the CoV-AbDab website terms
#
# Used for the zero-shot BCR-native task (SARS-CoV-2 neutralization discrimination,
# Supplementary Note S2.7 / Table S16).
#
# The paper uses the PINNED 2024-02-08 release, which OPIG serves as a direct CSV
# at the URL below. Verified 2026-09-25: this file is byte-identical (md5
# 4bcbcec3f35bc0cfb72535bbcf3dff08) to the copy used for the published numbers,
# so CoV-AbDab is byte-reproducible from its official source.
#
# NOTE: OPIG replaces the file when a new release is published. If the md5 below
# no longer matches, you have a newer release; the published numbers correspond to
# the 2024-02-08 file only.

COVABDAB_FILE="CoV-AbDab_080224.csv"          # 080224 = 2024-02-08
EXPECTED_MD5="4bcbcec3f35bc0cfb72535bbcf3dff08"
DEST_DIR="data"
DEST="${DEST_DIR}/covabdab.csv"

mkdir -p "${DEST_DIR}"

echo "Downloading CoV-AbDab ${COVABDAB_FILE} ..."
curl -L --fail -o "${DEST}" \
  "https://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/${COVABDAB_FILE}"

ACTUAL_MD5=$(md5sum "${DEST}" | cut -d' ' -f1)
echo "Downloaded ${DEST}"
echo "  expected md5: ${EXPECTED_MD5}"
echo "  actual   md5: ${ACTUAL_MD5}"

if [ "${ACTUAL_MD5}" = "${EXPECTED_MD5}" ]; then
  echo "OK: byte-identical to the release used in the paper."
else
  echo "WARNING: md5 differs. OPIG has probably published a newer release."
  echo "         The published numbers correspond to the 2024-02-08 file."
fi

echo
echo "The analysis scripts read this file via \$COVABDAB_CSV (default: ${DEST})."
echo "Next: python scripts/analysis/revision_models/bcr_covabdab_build.py <work.csv>"
