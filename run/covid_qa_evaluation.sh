#!/usr/bin/env bash
# Evaluation only: full supplied document unless --contexts is provided.
set -euo pipefail
cd "$(dirname "$0")/.."
python -u test/run_covid_qa_evaluation.py "$@"
