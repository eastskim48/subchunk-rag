#!/usr/bin/env bash
echo "This legacy experiment has been retired" >&2
exit 1
set -euo pipefail
exec python -u test/eval_jina_chunk_regions.py
