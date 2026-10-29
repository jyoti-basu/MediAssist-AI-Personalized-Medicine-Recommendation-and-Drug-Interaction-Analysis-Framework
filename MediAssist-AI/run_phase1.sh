#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 code/run_phase1.py
python3 integration/run_end_to_end_demo.py
python3 -m unittest discover -s tests -v
python3 integration/run_synthetic_tests.py
jupyter nbconvert --to notebook --execute --inplace MediAssist_AI_Phase1.ipynb --ExecutePreprocessor.timeout=300
