#!/usr/bin/env bash
set -euo pipefail
: "${GRANULARITY_CODE:?}"
: "${GRANULARITY_CONFIG:?}"
: "${OUTPUT_FILE:?}"
runner="$GRANULARITY_CODE/test/run_granularity_grid.py"
if [ "${SCORER}" = denseon ]; then
    python -u "$runner" --config "$GRANULARITY_CONFIG" --action prepare_dense
fi
# Tee preserves live progress while the existing grid runner captures stdout.
python -u "$runner" --config "$GRANULARITY_CONFIG" --action run --output "$OUTPUT_FILE" 2>&1 | tee "${OUTPUT_FILE}.live.log"
