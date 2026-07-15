#!/bin/bash

# Download SAbDab Structural Antibody Database data
# Source: https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/
# License: CC-BY

mkdir -p raw/sabdab
echo "Downloading SAbDab data requires manual steps:"
echo "1. Visit https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/"
echo "2. Download the latest summary file (sabdab_summary_all.tsv)."
echo "3. Place the downloaded file into the 'raw/sabdab/' directory."
