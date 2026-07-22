#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
INSTALL_DEPS="${INSTALL_DEPS:-True}"
UPGRADE_PIP="${UPGRADE_PIP:-True}"
DATASET_PREFIX_VALUE="${DATASET_PREFIX_VALUE:-}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "python executable not found: $PYTHON_BIN"
    exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

if [ "$UPGRADE_PIP" = "True" ]; then
    python -m pip install --upgrade pip setuptools wheel
fi

if [ "$INSTALL_DEPS" = "True" ]; then
    pip install -r "$REPO_ROOT/src/requirements.txt"
fi

cat <<EOF
bootstrap completed

repo root:
  $REPO_ROOT

venv:
  $VENV_DIR

activate:
  source "$VENV_DIR/bin/activate"
EOF

if [ -n "$DATASET_PREFIX_VALUE" ]; then
    cat <<EOF

recommended dataset prefix:
  export DATASET_PREFIX="$DATASET_PREFIX_VALUE"
EOF
fi

cat <<EOF

example commands:
  DATASET=longbench-hotpotqa PREPROCESS_SUBDIR=sent SPLITTER=sentence CACHEABLE_CHUNK_SIZE=None RETRIEVABLE_CHUNK_SIZE=1024 ./run/preprocess.sh
  DATASET=longbench-hotpotqa CHUNK_SIZE=1024 ./run/preprocess_vanilla.sh
  DATASET=longbench-hotpotqa DATA_SUBDIR=sent TOP_K=5 GLOBAL_TOP_R=0.10 EVAL_USE_PAST_CACHE=True ./run/eval.sh
  python run/grid_search/eval.py run/grid_search/grid.yaml
  python run/grid_search/eval.py run/grid_search/grid_vanilla.yaml
EOF
