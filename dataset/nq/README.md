# DAPR-NQ/NQ-open dataset preparation

This directory builds the project's custom retrieval-augmented generation
(RAG) dataset from the exact-question intersection of the DAPR NaturalQuestions
test split and the official NQ-open development split. It is not an official
Natural Questions benchmark configuration.

Run the complete download and CPU-side construction path from the repository
root:

```bash
DATASET_PREFIX=. ./dataset/get_nq.sh
```

This writes the dataset to `./dapr-nq-open` and retains verified source files
under `./dapr-nq-open-sources`. `DATASET_PREFIX` selects the parent directory
of both directories. Override the exact output or source directory when needed:

```bash
NQ_OUTPUT_DIR=/exact/output/path ./dataset/get_nq.sh
./dataset/get_nq.sh --source-dir /exact/source/path
```

The command downloads these immutable inputs:

- DAPR `UKPLab/dapr`, `NaturalQuestions` configuration, test split, revision
  `67ae3daa13596700976d20605630f5f9db3bd732`;
- the official Google Research `NQ-open.dev.jsonl` file at commit
  `a7d6452c0905c7772e9fbbb9a20b5fcab07c668f`.

Every source file is checked against its fixed byte size and SHA-256 before
construction. Interrupted `.part` downloads are resumed when the source server
supports byte ranges.

The custom construction applies these exact rules:

1. Retain all 108,626 DAPR parent documents and all 2,682,017 passages.
2. Join DAPR-NQ test and NQ-open development records by verbatim question text.
3. Preserve DAPR test order for the 2,390 matching questions.
4. Copy every official NQ-open answer alias without normalization or
   deduplication.
5. Preserve all 2,971 positive DAPR qrels as passage evidence metadata.

The resulting dataset root contains:

```text
documents/
questions/query.jsonl
answers/answer.jsonl
dataset_info/evidence_labels.jsonl
dataset_info/manifest.json
```

Database and ColBERT artifact construction remain separate GPU stages.
