#!/bin/bash
set -euo pipefail

# Download VDJdb TCR data
# Source: https://github.com/antigenomics/vdjdb-db  (GitHub releases)
# License: AGPL-3.0
#
# The paper uses the PINNED 2025-12-29 release. VDJdb publishes the compiled
# database as a release ASSET (zip), NOT as a file at the repo root — the old
# `master/vdjdb.txt` URL now 404s, and `master` would also drift to the latest
# release (e.g. 2026-06-03 "Summer release"). Always pin the release tag.
#
# Verified 2026-06-08: the `vdjdb_full.txt` inside vdjdb-2025-12-29.zip is
# byte-identical (md5 4ab97ea73b42a04afeaf1957d3bf9894) to the committed
# raw/vdjdb/vdjdb_full.txt. So VDJdb is byte-reproducible from this URL.

VDJDB_RELEASE="2025-12-29"
mkdir -p raw/vdjdb

echo "Downloading VDJdb release ${VDJDB_RELEASE} ..."
curl -L -o ${TMPDIR:-/tmp}/vdjdb-${VDJDB_RELEASE}.zip \
  "https://github.com/antigenomics/vdjdb-db/releases/download/${VDJDB_RELEASE}/vdjdb-${VDJDB_RELEASE}.zip"

echo "Extracting vdjdb_full.txt (the FULL export with paired alpha/beta columns"
echo "the pipeline needs — NOT the slim vdjdb.txt) ..."
unzip -o -j ${TMPDIR:-/tmp}/vdjdb-${VDJDB_RELEASE}.zip vdjdb_full.txt -d raw/vdjdb/

echo "Done. raw/vdjdb/vdjdb_full.txt"
echo "Verify: md5sum raw/vdjdb/vdjdb_full.txt  # expect 4ab97ea73b42a04afeaf1957d3bf9894"
