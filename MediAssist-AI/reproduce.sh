#!/usr/bin/env bash
# usage: bash reproduce.sh /path/to/original/csv/folder
set -euo pipefail
SRC="${1:?give the folder that contains records_final.csv, voc_final.csv, ... }"
cd "$(dirname "$0")"
python3 -I code/build_package.py --source-dir "$SRC"
python3 -I code/write_reports.py
python3 -I integration/run_synthetic_tests.py
