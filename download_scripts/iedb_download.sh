#!/bin/bash

# Download IEDB B-cell assay data
# Source: https://www.iedb.org/
# License: CC-BY-4.0

mkdir -p raw/iedb
echo "Downloading IEDB data requires manual steps:"
echo "1. Visit https://www.iedb.org/database_export_v3.php"
echo "2. Locate the 'B Cell Receptor' dataset under the 'Receptor' section."
echo "3. Download the CSV format export."
echo "4. Place the downloaded file into the 'raw/iedb/' directory and rename it to 'bcell_full_export.csv'."
