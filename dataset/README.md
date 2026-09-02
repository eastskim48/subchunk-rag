# Dataset preparation

Dataset-specific source download and CPU-side construction entry points live
under this directory. Each dataset directory must leave a dataset root with
`documents/`, `questions/query.jsonl`, and `answers/answer.jsonl` so database
and candidate-artifact materialization can remain separate GPU stages.

Available entry points:

- `get_hotpotqa.sh`: custom HotpotQA distractor-development full-Wikipedia
  corpus and RAG inputs. Future dataset entry points follow the same
  `get_<dataset>.sh` naming convention.
- `get_hotpot_full.sh`: the same HotpotQA queries and answers with every page
  in the official processed October 2017 Wikipedia archive as the shared
  retrieval corpus.
- `get_nq.sh`: custom DAPR-NQ test / official NQ-open development
  exact-question intersection with all DAPR NaturalQuestions parent documents.
