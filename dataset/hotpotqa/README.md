# HotpotQA dataset preparation

This directory builds the project's **custom** HotpotQA distractor-development
full-Wikipedia retrieval-augmented generation dataset. It is not the official
HotpotQA ten-context evaluation protocol.

Run the complete download and dataset-construction path with:

```bash
./dataset/get_hotpotqa.sh
```

The default output is `/mnt/nvme1/datasets/hotpotqa`. Override the common
dataset parent or this dataset's exact output directory when needed:

```bash
DATASET_PREFIX=/path/to/datasets ./dataset/get_hotpotqa.sh
HOTPOTQA_OUTPUT_DIR=/exact/output/path ./dataset/get_hotpotqa.sh
```

Additional arguments are passed to `hotpotqa/prepare.py`, for example
`--scan-workers 16`. Set `PYTHON_BIN` to select a non-default Python
interpreter.

The command performs these stages in order:

1. Download and verify the official HotpotQA distractor development JSON.
2. Download and verify the official processed October 2017 Wikipedia archive.
3. Extract the exact custom 66,705-document corpus.
4. Validate the official supporting-fact annotations against that corpus.
5. Write `questions/query.jsonl`, `answers/answer.jsonl`, and
   `dataset_info/evidence_labels.jsonl`.

Downloads are retained by default in a sibling directory named
`hotpotqa-sources`. A verified file is reused, and an interrupted `.part`
download is resumed. Completed construction stages are verified and skipped on
rerun. An incomplete stage is not overwritten.

To reuse manually downloaded official sources:

```bash
HOTPOTQA_OUTPUT_DIR=/path/to/datasets/hotpotqa ./dataset/get_hotpotqa.sh \
  --dev-path /path/to/hotpot_dev_distractor_v1.json \
  --wiki-archive /path/to/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2
```

The resulting dataset root contains the `documents/`, `questions/`, and
`answers/` inputs consumed by the existing preprocessing and evaluation code.
Database and ColBERT artifact construction remain separate GPU stages.

For the repository's current `/mnt/nvme1/datasets/hotpotqa` layout, the
existing materialization pipeline can be run after this preparation command:

```bash
bash run/run_hotpotqa_bge_materialization_pipeline.sh
```

That pipeline builds the configured Vanilla databases, the sentence-split
database, and its ColBERT artifact. It is not invoked automatically by the
dataset preparation command. The historical pipeline explicitly sets the code
option `COLBERT_WINDOW_CENTER_UNIT=subchunk`; it does not use the generic
`subchunk_only` default. Choose the artifact build command according to the
experiment condition instead of treating those two encodings as equivalent.
