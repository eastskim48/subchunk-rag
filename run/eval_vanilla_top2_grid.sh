#!/usr/bin/env bash
set -euo pipefail
: "${VANILLA_TOP2_CODE:?}"
: "${VANILLA_TOP2_CONFIG:?}"
: "${OUTPUT_FILE:?}"
python -u "$VANILLA_TOP2_CODE/test/run_vanilla_top2_grid.py" --config "$VANILLA_TOP2_CONFIG" --output "$OUTPUT_FILE" 2>&1 | tee "${OUTPUT_FILE}.live.log"
