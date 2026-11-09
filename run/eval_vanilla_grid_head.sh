#!/bin/bash
# Only verify DB completion, then enter the unchanged HEAD vanilla evaluation script.
set -euo pipefail
: "${DB_DIR:?DB_DIR must be explicit in the grid case}"
: "${VANILLA_HEAD_CODE:?VANILLA_HEAD_CODE must point to the frozen HEAD snapshot}"
/home/dongseob/miniconda3/bin/python - "$DB_DIR" <<'PY'
import json
import os
from pathlib import Path
import sqlite3
import sys
p=Path(sys.argv[1])
assert (p/'subchunks.json').is_file()
manifest=json.loads((p/'build_manifest.json').read_text())
con=sqlite3.connect((p/'chroma.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
count=con.execute('SELECT count(*) FROM embeddings').fetchone()[0]
if os.environ.get('VANILLA_REUSE_SENTENCE_DB') == '1':
    assert manifest['splitter']=='sentence'
    assert manifest['retrievable_chunk_size']==512
    assert manifest['embedding_backend']=='bge_small_v1_5'
    assert count>0
    for key in ('subchunk_start','subchunk_end'):
        assert con.execute('SELECT count(*) FROM embedding_metadata WHERE key=?',(key,)).fetchone()[0]==count
else:
    ready=json.loads((p.parent/'vanilla_ready_20261028.json').read_text())
    assert Path(ready['db']).resolve()==p.resolve()
    assert manifest['splitter']=='fixed_size'
    assert count==ready['entries']>0
con.close()
PY
cd "$VANILLA_HEAD_CODE"
exec bash run/eval.sh
