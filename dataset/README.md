# Dataset preparation

Only the four baseline datasets below and their shared preparation helpers are
kept in this directory.

## Four baseline datasets

These entrypoints reproduce the existing custom construction rules. Each output
root contains `documents/`, `questions/query.jsonl`, `answers/answer.jsonl`, and
source/provenance metadata. Physical datasets, source caches, retrieval DBs, and
encoder artifacts stay under `DATASET_PREFIX`; logs stay under `outputs/temp/`.
Dataset directories are not overwritten. Failed staging/source files are kept
for inspection; there is no automatic dataset repair or question exclusion.

Set a parent directory explicitly, for example:

```bash
export DATASET_PREFIX=.
```

### NQ

```bash
bash dataset/get_nq.sh
```

This downloads the pinned DAPR NaturalQuestions test and official NQ-open
validation inputs using the existing source-size/checksum checks. The custom
verbatim-question intersection retains 2,390 questions and all answer aliases,
with all 108,626 DAPR parent documents. This is not the full official NQ benchmark.

### BioASQ

Download the original ZIP manually:

1. Register or sign in at the [BioASQ participant area](https://participants-area.bioasq.org/datasets/).
2. Download the **Task 14b training dataset** ZIP from the [training dataset page](https://participants-area.bioasq.org/Tasks/14b/trainingDataset/).
3. Pass the downloaded ZIP path with the required `--archive` option; do not unpack it first:

```bash
bash dataset/get_bioasq.sh --archive /path/to/BioASQ-training14b.zip
```

The script does not log in or download the BioASQ ZIP. It reads this local archive
and automatically performs the remaining question preparation and linked PubMed
document download/construction.

The ZIP member `BioASQ-training14b/training14b.json` must match the preserved
source checksum. The existing summary-only exclusion retains 4,366 of 5,729
training questions, including their complete original annotations and nested
answer items. The corpus still uses document references from all 5,729 questions.
It is a custom training-source evaluation, not an official held-out test.

The script fetches the same 58,048 distinct linked PubMed IDs in batches of 200,
retains raw XML, and writes title plus abstract sections joined by newline. The
existing corpus requires 58,012 returned records. Missing and unexpected IDs are
recorded without substituting records or changing the question population.
If the returned record count differs, construction stops.

To reproduce the original collected document text, supply the preserved records:

```bash
bash dataset/get_bioasq.sh \
    --archive /path/to/BioASQ-training14b.zip \
    --pubmed-records /path/to/pubmed_records.jsonl
```

Without this input, PubMed supplies its current title/abstract records; their
identity to the historical text is not guaranteed. Source paths and checksums
are recorded. `--source-dir` selects the XML/source cache directory.

### NewsQA

Download both original archives and construct the dataset:

```bash
bash dataset/get_newsqa.sh
```

The annotation ZIP is fetched from Microsoft's original download address. The
article archive is fetched from a pinned historical revision of the CNN/DailyMail
dataset repository. Both downloads must match their fixed source checksums:

- `newsqa-data-v1.zip`: `5b542995d1d46efd9d90b16a5aee3b04f3fe17c9ee2a2d5094e9f978175357bd`
- `cnn_stories.tgz`: `e8fbc0027e54e0a916abd9c969eb35f708ed1467d7ef4e3b17a56739d65cb200`

To reuse local archives or download from explicit addresses instead:

```bash
bash dataset/get_newsqa.sh \
    --annotations-zip /path/to/newsqa-data-v1.zip \
    --stories-archive /path/to/cnn_stories.tgz
```

`--annotations-url` and `--stories-url` override download addresses; automatic
downloads still require the original checksums. Sources are cached alongside
the dataset under `newsqa-sources` unless `--source-dir` is supplied.

The small story-correction and test-story-list files are downloaded from the
fixed Maluuba/newsqa commit `d5bb9e9640e2ed7a31e209393376549d737d276b`.
Alternatively pass `--official-repo-dir` pointing to a checkout at that commit.
Public NewsQA downloads do not require account credentials.

All 12,744 annotated article bodies are reconstructed with the existing official
text corrections. The existing single-consensus implementation retains 4,293
question/answer pairs from the 5,126 raw-answerable test rows, preserving their
original row IDs and order. Both active and `_full` files contain all 4,293
records; this entrypoint does not recreate the historical 400-question sample.
This is the project's custom consensus population, not the full official test.

### COVID-QA

```bash
bash dataset/get_covid_qa.sh
```

The original deepset COVID-QA JSON is downloaded and checked against the
preserved source checksum. All 2,019 questions and 147 document contexts are
retained as the existing custom full-set test, not an official held-out split.
The native `test.json` is byte-identical to the source; `test.jsonl` preserves
original question fields. Common query/answer files preserve original IDs,
order, question text, and every answer text. Documents are the original contexts,
with the same 147-document shared search corpus.

To use a preserved raw download:

```bash
bash dataset/get_covid_qa.sh --source-file /path/to/COVID-QA.json
```

### Common preprocessing

Complete the [environment setup](../README.md#setup), including GPU FAISS and
Llama model access, before running these builds.

```bash
bash run/preprocess.sh \
    --dataset nq --dataset-root "$DATASET_PREFIX/dapr-nq-open"
bash run/preprocess.sh \
    --dataset bioasq --dataset-root "$DATASET_PREFIX/bioasq14b-linked-documents"
bash run/preprocess.sh \
    --dataset newsqa --dataset-root "$DATASET_PREFIX/newsqa"
bash run/preprocess.sh \
    --dataset covid-qa --dataset-root "$DATASET_PREFIX/covid-qa-full-test"
```

Each command performs:

1. The existing sentence preprocessor creates the source 512-Llama-token search
   chunks using BGE-small, initially with the ordinary 180-Llama-token sentence cap.
2. The existing offline metadata rebuild copies the DB and applies the separate
   LateOn 100-token sentence cap and 180-token subchunk construction. Source indexed
   texts, retrieval vectors, and the graph are preserved in this copy.
3. The existing materializer encodes each sentence with LateOn and stores 128-
   dimensional vectors using 2-bit residual compression and seed 123. It uses the
   existing path that first stores 16-bit floating-point (FP16) vectors and its validation.

The initial upsert/embedding batch defaults are 5461/5461 for NQ and NewsQA,
256/128 for BioASQ, and 256/256 for COVID-QA, following the stored construction
settings. The new indexes are rebuilds; they are not the original physical
indexes. Each build records its settings. `--db-batch-size`,
`--embedding-batch-size` and `--encoding-batch-size` make batch sizes explicit;
changed insertion batches may produce a different approximate search graph.
Selection rules and evaluation metrics are unchanged.
Retrieval-vector construction and the LateOn materializer both use CUDA.

Use `--dry-run` to save/print the command plan without loading models or touching
dataset artifacts. `--source-db` explicitly skips source-index construction and
reuses an existing idle retrieval DB. `--stage retrieval`, `--stage subchunks`, or
`--stage embeddings` runs only that stage. Existing output paths are refused;
there is no implicit resume, overwrite, GPU fallback, or evaluation run.

The final relative paths are:

```text
sent-bge-small-v1.5-512-splitlong100-lateon/db
sent-bge-small-v1.5-512-splitlong100-lateon/colbert_window
```

Use `COLBERT_WINDOW_DIR` explicitly when reusing an older artifact with another
name, such as `colbert_lateon_sentence_2bit`.
