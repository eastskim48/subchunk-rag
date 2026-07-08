# Codebase Guide

## Key Paths

### `run/`
- Store experiment scripts here.
- Use this directory for reproducible execution entry points.

### `test/`
- Store test code here.
- Use this directory for:
  - small feasibility checks
  - debugging with minimal examples
  - unit tests for methods

## Model

### Prompting
- Write prompts in a format that the current Llama-3.1-Instruct model can easily follow.
- Preserve the current cache reuse pipeline.
- Do not break the current prompt/cache structure:

`[Chunk Cache][Chunk Cache][System Prompt][Query]`

- Prefer prompt changes that are compatible with KV reuse.
- Avoid prompt formats that require rebuilding the whole prefix.

### Model choice
- Keep using the current model unless explicitly requested otherwise.
- Do not switch to another model without user approval.

## Dataset

- Default dataset path:

`/mnt/nvme1/datasets/longbench-hotpotqa`

Given `base_path = /mnt/nvme1/datasets/longbench-hotpotqa`:

- Questions: `base_path + /questions/query.jsonl`
- Answers: `base_path + /answers/answer.jsonl`
- Documents: `base_path + /documents/`
- Vector database: `base_path + /db/`
- KV caches: `base_path + /$chunk_size/`

### Official Dataset Fidelity
- Do not write custom local dataset builders that drop, collapse, rename, normalize, deduplicate, or reinterpret official dataset fields unless the user explicitly asks for that exact transformation.
- When using LongBench, preserve official fields such as `answers` exactly. Do not replace an answer-alias list with `answers[0]`.
- If local files must be generated from an official dataset, copy the required official fields verbatim and document any unavoidable structural change before running experiments.
- Do not silently create convenience formats that differ from the official benchmark format.

## Working Style
- Start with a very small sample for feasibility and debugging.
- Prefer minimal, local changes.
- Reuse existing interfaces when possible.
- Ask the user before modifying external code or reference library code.
- Prefer changes in `src/`, `run/`, and `test/` first.
- Do not create large temporary files or persistent cache artifacts under the project root by default.
- Prefer `/mnt/nvme1/dongseob/tmp/` for large intermediate outputs, reusable test caches, and other non-source artifacts unless reproducibility requires a documented project-local path.
- Do not use `/tmp` for project dependencies, repositories, caches, smoke artifacts, or experiment outputs. The official ColBERT source is managed as the `third_party/ColBERT` git submodule by default. ColBERT embedding/window artifacts remain dataset-local under paths such as `$DATASET_PATH/sent/colbert_window`.

## Explicit-Scope Discipline
- Do not implement behavior the user did not explicitly request.
- Do not add fallback logic, heuristic recovery, fuzzy matching, extra metrics, altered scoring, prompt changes, dataset handling changes, or evaluation-flow changes unless the user explicitly asks for them.
- If a change seems useful but was not requested, explain the option and ask before implementing it.
- Evaluation code is especially strict: preserve the requested metric definition exactly, even if another variant looks more robust or convenient.
- If the user asks to add a field, metric, option, or behavior, only add it. Do not remove, rename, replace, or reinterpret existing fields, metrics, options, or behavior unless the user explicitly asks for that removal or rename.

## Refactoring Discipline
- When refactoring, do not guess and make changes beyond what was explicitly requested.
- Do only what was asked.
- If a requested change requires an additional structural decision or scope expansion, ask before doing it.
