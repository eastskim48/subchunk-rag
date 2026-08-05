# Evaluation Protocol

This document is the current canonical protocol for running and comparing
evaluation results.

## Evaluation-Governance Invariant

Evaluation datasets and metrics implement only the exact rules explicitly
specified by the user. If the user explicitly requests an official protocol,
it is reproduced without local additions. Dataset inspection must never be
used to invent or silently introduce filtering, exclusion, annotation repair,
fallback matching, evidence scoping, denominator changes, aggregation rules,
or any other subjective policy. An observed anomaly must be preserved and
reported only as a concrete fact. Unless the user first asks for treatment
options or explicitly instructs a treatment, the agent must not invent,
consider, propose, recommend, compare, or seek approval for a treatment.
Approval of an agent-originated policy is not a substitute because the agent
must not originate the policy. Describing a choice as clean, gold, correct,
robust, official, or custom does not authorize an unspecified treatment.

## Entry Points

- Default single-run entry point: `run/eval.sh`.
- Grid-search entry point: `python run/grid_search/eval.py <grid.yaml>`.
- GPU-locked grid wrapper: `run/run_grid.sh <grid.yaml> [more-grid.yaml ...]`.
- Python evaluation entry point used by `run/eval.sh`:
  `src/entrypoint/eval.py`.
- Answer scoring implementation: `src/acc_metric.py`.
- Retrieval/evidence-only entry point:
  `run/eval_retrieval_only.sh`.
- Python implementation used by the retrieval/evidence-only shell:
  `src/entrypoint/eval_retrieval_only.py`.
- Gold-evidence oracle reader entry point:
  `run/eval_oracle.sh`.

`run/eval.sh` resolves datasets in this order:

1. `$DATASET_PREFIX/$DATASET` when `DATASET_PREFIX` is set.
2. `/mnt/nvme1/datasets/$DATASET` when that directory exists.
3. `$DATASET` as a direct path/name fallback.

## Default Runtime Settings

The current `run/eval.sh` defaults are:

- `DATASET=longbench-hotpotqa`
- `MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct`
- `DATA_SUBDIR=sent`
- `CACHE_SUBDIR=cache`
- `DB_DIR=$DATASET_PATH/$DATA_SUBDIR/db`
- `CACHE_DIR=$DATASET_PATH/$DATA_SUBDIR/$CACHE_SUBDIR`
- `QUERY_FILE=$DATASET_PATH/questions/query.jsonl`
- `ANSWER_FILE=$DATASET_PATH/answers/answer.jsonl`
- `CHROMA_EMBED_BACKEND=bge_small_v1_5`
- `DENSE_EMBED_DIR=$DATASET_PATH/$DATA_SUBDIR/dense_embed`
- `COLBERT_WINDOW_DIR=$DATASET_PATH/$DATA_SUBDIR/colbert_window`
- `EVAL_USE_PAST_CACHE=False`
- `MODEL_LOAD_IN_4BIT=False`
- `MEASURE_PROMPT_STATS=True`
- `RETRIEVAL_INCLUDE_DOCUMENTS=True`
- `USE_CLEANER=False`
- `EVAL_BSZ=4`
- `TOP_K=20`
- `TOTAL_NUM` unset: evaluate every record in the query file
- `MAX_NEW_TOKENS=20`
- `COMPRESS_METHOD=dense`
- `RETAIN_TOKEN_RATIO=0.1`

Changing any of these can change the evaluated workload, model behavior, or
timing condition. Report changed values with every result table.

Generation runs report two post-generation length diagnostics. `avg output
lens` is the average Python Unicode-character length of the final
post-processed prediction string. `avg output token lens (approx)` retokenizes
that same final prediction with the active model tokenizer, without adding
special tokens, and averages the resulting lengths. Grid summaries expose
these values as `avg_output_lens` and `avg_output_token_lens_approx`, including
in `summary2`. The token value is approximate rather than the exact number of
generation steps because decoding removes special tokens and response
post-processing can remove generated text before retokenization. Neither
diagnostic changes answer scoring, generation, or the evaluated population.

Generation also preserves each request's exact `generated_token_count` in the
prediction JSONL. This count includes the first generated end-of-sequence token
when present and excludes padding after it. `summary2` reports its
request-level mean as `avg_generated_token_count`. It additionally reports
`avg_decode_step_sec`, computed as the sum of measured decode time divided by
the number of autoregressive decode iterations after the prefill-produced first
token. This is a batch execution cost per decode step, not a per-request token
latency; requests that finish early leave the active batch while the remaining
requests continue.

For historical generation runs created before these fields existed,
`run/backfill_output_char_length_to_summary2.py` can recover only
`avg_output_lens` from the preserved `avg output lens` log line. It validates
each `summary2` row against the corresponding detailed summary row and does not
infer exact generated-token counts or decode-step times. Retrieval-only runs
without generation logs are left unchanged.

For generation through `run/eval.sh`, setting `TOTAL_NUM` to a positive integer
limits evaluation to that many records from the beginning of the query file.
Omitting it, or setting it to an empty value, evaluates the complete query
file. Results with an explicit limit are not directly comparable to full-file
results unless the evaluated populations are otherwise made identical.
`TOTAL_NUM` is not accepted by the retrieval-only evidence entry point.

`QUERY_FILE` and `ANSWER_FILE` can override the default query and answer paths
without changing dataset-root resolution. When unset, the previous
`$DATASET_PATH/questions/query.jsonl` and
`$DATASET_PATH/answers/answer.jsonl` paths remain in effect.

`COMPRESS_METHOD=dense` is the materialized dense-embedding selector. It uses
the same query embedding, stored sentence embeddings, and deduplication as the
former `compare_all_materialized` implementation, but now selects under the
shared token-budget policy. Earlier candidate-count-ratio dense results are not
directly comparable. The online `compare` and `compare_all` baselines and all
three old factory names are intentionally unsupported.

## Dataset And Scoring

Predictions are scored by `src/acc_metric.py` using dataset-aware evaluators.
The supported dataset names and scoring families are:

- `longbench-hotpotqa`: HotpotQA-style official EM/F1 with yes/no/noanswer
  special handling.
- `longbench-2wiki`: Hotpot-style EM/F1.
- `longbench-musique`: Hotpot-style EM/F1.
- `longbench-triviaqa`: TriviaQA-style alias-aware EM/F1.
- `triviaqa-unfiltered-wikipedia`: the same official TriviaQA alias-aware
  EM/F1 over the local Wikipedia-only unfiltered-dev condition.
- `longbench-qasper`: LongBench QA-style alias-aware EM/F1.
- `longbench-narrativeqa`: LongBench QA-style alias-aware EM/F1.
- `conditionalqa`: official ConditionalQA answer-only permutation EM/F1.
- `dapr-nq-open` (alias `nq-open`): official NQ-open normalized exact match
  over answer aliases, with auxiliary token F1.
- `newsqa`: NewsQA's official SQuAD v1.1 normalized EM and token-overlap F1.

For LongBench-family evaluators, ground-truth `answers` lists are treated as
valid aliases. If `answers` is absent, scoring falls back to `answer`.

ConditionalQA instead uses the answer-only portion of the official evaluator
from upstream commit `77bd295952daf415548b3244db10880d3d55cfe0`. Its
`answers` entries are distinct answers that must all be produced, not aliases.
Scoring pads missing predicted answers, selects the best permutation between
predictions and references, and applies the official exponential penalty for
extra predictions. The local RAG conversion intentionally omits condition
annotations, so conditional EM/F1 is not reported.

The current generation interface returns one free-form answer string per
query. It is represented as one official predicted answer with an empty
condition list; an empty generation is represented as no predicted answer. It
is not split heuristically on commas, newlines, or conjunctions. Consequently,
a single correct prediction for a question with `N` distinct gold answers
receives at most `1/N` answer EM/F1. Structured multi-answer generation would
change the prompt/output protocol and must be reported as a separate evaluation
condition.

The local `/mnt/nvme1/datasets/hotpotqa` directory is a custom corpus and label
construction, not an official HotpotQA evaluation layout. The directory name
selects the existing `dataset=hotpotqa` answer evaluator; it does not make the
custom corpus construction an official HotpotQA retrieval setting.
It starts from every title in every `context` entry of the official HotpotQA
distractor development set, including distractors, and joins those titles to
the official processed October 1, 2017 Wikipedia corpus. The first join key is
Unicode NFKC normalization plus case-insensitive exact equality. A fallback
key removes every non-alphanumeric character after the same normalization and
again requires exact equality. No fuzzy matching is used. If one key identifies
multiple Wikipedia page IDs, the title is recorded as ambiguous and no page is
selected.

The current build contains 7,405 queries, 73,700 context-title occurrences,
and 66,581 unique requested titles. Of these, 66,447 titles identify one page
and 134 case-insensitive exact titles each identify two pages; no title is
missing and no fallback-key match was needed. Both pages from every ambiguous
exact-title join are included in the corpus while the join remains labeled
ambiguous. Ten page IDs occur under case variants of more than one ambiguous
title, so the ambiguous joins add 258 unique pages. The resulting 66,705 pages
are written as one plain-text file per page. The source JSON records are also
preserved in `dataset_info/documents_raw.jsonl`. Plain-text construction
concatenates sentences within an official paragraph without a separator,
removes only HTML anchor tags, unescapes HTML entities, drops empty paragraphs,
and joins paragraphs with two newlines.

The recorded 512-chunk estimate uses the current fixed-size preprocessing
semantics with the Llama-3.1-8B-Instruct tokenizer: no tokenizer special
tokens, no overlap, and 511 document tokens per chunk because the prompt-visible
two-newline suffix consumes one token of the 512-token chunk budget. This gives
88,125,750 document tokens and 208,225 chunks over all 66,705 pages.

`dataset_info/title_to_documents.json` is the direct title inverted index for
this custom corpus. It maps each of the 66,581 verbatim HotpotQA context titles
to one or two records containing the official page ID, Wikipedia title, and
relative document filename. The 134 ambiguous exact titles map to both included
documents. `dataset_info/document_file_to_title.json` provides the reverse
mapping for all 66,705 document files.

Supporting-fact presence was validated for all 18,005 official
`(title, sentence_index)` labels in the 7,405-query development set. The
validator looks up the verbatim context title, reads the context sentence at
the labeled index, applies the same HTML-anchor removal and entity unescaping
used to construct the stored document plaintext, and performs a
case-sensitive contiguous substring search. Sentence-boundary whitespace is
stripped for the primary result; a secondary result normalizes whitespace
only. No fuzzy matching, case folding, semantic matching, or label repair is
used.

The primary trimmed-exact check finds 17,519 facts. Whitespace normalization
finds another 484, giving 18,003 of 18,005 facts in the joined documents. The
remaining two records are source-annotation issues rather than missing title
joins: one `Benedict of Nursia` context sentence contains malformed nested HTML
and a textual typo relative to the official page, and one
`Jimmy Butler (basketball)` label uses sentence index 902 although that context
contains five sentences. Detailed per-query results are stored in
`dataset_info/supporting_fact_validation.jsonl`; the aggregate result and
protocol are in `dataset_info/supporting_fact_validation_summary.json`.

The custom runnable input conversion preserves all 7,405 official development
examples in unchanged source order:

- `questions/query.jsonl` assigns contiguous local IDs `0..7404`, stores the
  official question verbatim as `query`, and preserves the official `_id` as
  `source_id`.
- `answers/answer.jsonl` stores the single official answer verbatim as both
  `answer` and the sole element of `answers`.
- `dataset_info/evidence_labels.jsonl` stores one aligned record per query.
  Each valid supporting fact includes the exact stored-document substring,
  its half-open `[start, end)` character span, page ID, document filename,
  title, official sentence index, source sentence text, text-match mode, and
  source-structure alignment mode.
- `dataset_info/rag_input_validation_manifest.json` records the construction
  policy, aggregate counts, output SHA-256 hashes, and both source annotation
  errors.

Evidence spans are selected structurally rather than by taking an arbitrary
occurrence of repeated text. The converter reconstructs the stored document
from its official raw Wikipedia paragraphs, first aligns the complete
HotpotQA context sentence list to one source paragraph, and then applies the
official supporting-fact sentence index. When a sibling context sentence is
malformed, it permits a fallback only if one same-length source paragraph
uniquely has the labeled sentence at the same index and has the highest number
of other sentences matching at their positions. This fallback is used for two
valid `Benedict of Nursia` sentence-1 facts associated with the already
recorded malformed sibling annotation.

The output contains 18,003 valid evidence facts: 17,519 trimmed-exact text
matches and 484 whitespace-normalized matches whose spans are mapped back to
the original stored characters. The two invalid official facts are excluded
from `evidence_texts` but preserved in each affected record's
`invalid_supporting_facts` and in the manifest. Every query retains at least
one valid supporting fact. This conversion changes no official question or
answer text, but retrieval against the 66,705-page joined corpus remains a
custom evaluation protocol and is not directly comparable to evaluation on
the original ten context passages per query.

`run/grid_search/grid_hotpotqa.yaml` is the custom generation grid for this
7,405-query distractor-development/full-Wikipedia-join condition. It copies
the seven retrieval/compression cases and all model, prompt, generation, and
batch settings from `grid_newsqa_bge_rag.yaml`: two ColBERT sliding-region
cases (top-k 40/ratio 0.15 and top-k 20/ratio 0.25) and five Vanilla cases
(128/top-k 10 and 20, 256/top-k 10 and 20, and 512/top-k 20). It uses
Llama-3.1-8B-Instruct in FP16, `raw_chunk_first`, no cleaner, no past-key/value
cache, batch size 1 per case, and at most 20 generated tokens. With no
`TOTAL_NUM` override, all 7,405 query and answer rows are evaluated. Answer
scoring uses the existing `dataset=hotpotqa` HotpotQA normalized exact match
and token-overlap F1 evaluator. This custom open-corpus retrieval condition is
not directly comparable to HotpotQA evaluation restricted to each example's
ten provided context passages.

`run/grid_search/grid_hotpotqa_evidence.yaml` is the corresponding custom
retrieval/evidence-only grid. It copies all 23 Vanilla and ColBERT
sliding-region cases from `grid_newsqa_bge_evidence.yaml`, uses retrieval-only
batch size 32, and evaluates all 7,405 queries because retrieval-only
evaluation has no population truncation setting. Stable local IDs join
`questions/query.jsonl` to `dataset_info/evidence_labels.jsonl`; query strings
must also match exactly or evaluation stops. The scorer compares every stored
evidence passage with the complete reconstructed retrieval or compressed
context using the shared `text_evidence_exact` formulas described below.
This is a custom 18,003-passage denominator: the two invalid official
supporting-fact annotations recorded in the validation manifest are absent
from `evidence_texts` and therefore receive neither a hit nor a zero. The grid
must not be reported as evidence recall over all 18,005 official HotpotQA
supporting-fact annotations.

`run/grid_search/grid_hotpotqa_sampled.yaml` is a separate custom generation
grid over the 200 custom HotpotQA rows whose verbatim query strings exactly
match LongBench-HotpotQA, ordered by the LongBench-HotpotQA query sequence.
It reads `questions/query_sampled.jsonl` and
`answers/output_sampled.jsonl`, preserves the original custom HotpotQA stable
IDs and answer records, and explicitly sets `TOTAL_NUM=200`. It copies the
current active model, prompt, retrieval, compression, and batch settings from
`grid_hotpotqa.yaml`; the currently commented Vanilla-512 case remains
commented. Retrieval still uses the custom 66,705-page joined corpus, so this
is not an official LongBench-HotpotQA condition. Its aggregate answer scores
are directly comparable across cases within this sampled grid, but not to the
7,405-query full-population grid because the evaluated query population
differs.

`run/grid_search/grid_hotpotqa_sampled_direct_subchunk.yaml` is a custom
generation grid for direct sentence-subchunk retrieval on the same 200-query
population. It has one case:
`sent-direct-bge-small-v1.5-splitlong180`, top-k 80, no compression, and
evaluation batch size 1. The query, answer, model, prompt, generation, cleaner,
past-cache, and scoring settings are identical to the active Vanilla-512/top-k
80 case in `grid_hotpotqa_sampled.yaml`; the retrieval DB is the only
changed component. Each retrieved row is one sentence subchunk from the custom
one-off direct-subchunk DB. This does not change the HotpotQA answer metric or
evaluated population. Answer scores are directly comparable with the sampled
Vanilla cases, while top-k denotes different text units and therefore does not
imply an equal retrieved-token budget.

`run/grid_search/grid_hotpotqa_sampled_direct_subchunk_evidence.yaml`
is the retrieval/evidence-only companion to the direct-subchunk generation
grid. It uses the same custom 200-query file, direct sentence-subchunk DB,
top-k 80, batch size 1, and no compression. Stable IDs select the corresponding
records from the full evidence-label file, and exact query strings must match.
The evaluated sampled population contains 463 stored evidence passages. It
uses the existing `text_evidence_exact` scorer without changing the
metric or denominator rules.

`run/grid_search/grid_hotpotqa_sampled_evidence.yaml` is the corresponding
custom retrieval/evidence-only grid for the same 200-query population. It uses
the same eight active Vanilla retrieval cases as
`grid_hotpotqa_sampled.yaml`: 128- and 256-token chunks at top-k 10 and 20,
and 512-token chunks at top-k 10, 20, 40, and 80. It reads
`questions/query_sampled.jsonl` and joins the preserved stable IDs to the full
`dataset_info/evidence_labels.jsonl`; query strings must also match exactly.
It uses retrieval-only batch size 32 and the existing `text_evidence_exact`
scorer. The evidence results are directly comparable across cases within this
sampled evidence grid, but not to the 7,405-query full-population evidence grid
because the evaluated query population differs.

The local NewsQA QA set uses only official test questions for which
`NewsQaDataset.get_consensus_answer` returns a character span after the
official loader's out-of-range endpoint clipping. This produces 4,293
question/answer pairs from the preceding 5,126-row raw-non-`None` test
conversion. Each `answers` list contains exactly the single consensus span.
The active files are a seed-42 sample of 400 pairs from those 4,293 records;
source IDs are preserved and are therefore not contiguous. NewsQA answer
scoring ports the SQuAD v1.1 evaluator specified by the NewsQA paper:
lowercase, remove ASCII punctuation and the English articles `a`/`an`/`the`,
normalize whitespace, then compute exact match and token-overlap F1. Local
scores are represented in `[0, 1]`; the upstream script multiplies the same
averages by 100 for display.

The earlier raw-answer files are preserved under
`query_raw_non_none_{full,seed42_400}.jsonl` and
`answer_raw_annotations_{full,seed42_400}.jsonl`. Results using those raw
annotator answers are not directly comparable to the consensus-set QA
results because both the evaluated question population and reference-answer
policy differ. Existing non-NewsQA dataset evaluators are unchanged.

The `grid_narrativeQA.yaml` condition is a custom open-corpus NarrativeQA
evaluation. Its active `query.jsonl` and `answer.jsonl` contain all 200
LongBench NarrativeQA rows in LongBench order and preserve each LongBench
answer list exactly. Exact question plus exact answer-reference matching joins
the rows to 182 unique official NarrativeQA validation QAP rows associated
with 20 source documents; 18 LongBench question/QAP rows are repeated. The
active files use contiguous IDs 0--199 and preserve the LongBench `_id`, source
QAP ID, and source document ID as metadata. The join is recorded in
`splits/test/dataset_info/longbench_join.jsonl`; the directory name `test` is
retained only for runner-path compatibility even though the source QAP rows
are from the official NarrativeQA validation split.

The prior 1,000-row seed-42 sample from the 10,557 official test questions is
no longer active. The complete official test files remain unchanged as
`query_full.jsonl`, `answer_full.jsonl`, and
`query_documents_full.jsonl`, and its prior sample manifest is preserved as
`active_sample_manifest_seed42_1000.json`. The active LongBench join manifest
is `active_sample_manifest.json`.

Each active question retrieves globally from the shared index of all 1,572
original NarrativeQA stories. It does not use
`splits/test/dataset_info/query_documents.jsonl` to restrict retrieval to the
question's associated story. This is a custom LongBench-query-aligned
original-corpus condition, not unchanged LongBench NarrativeQA and not the
official NarrativeQA full-story setting, which supplies the associated story.
The local `narrativeqa` evaluator reports LongBench-style normalized
alias-aware exact match and token-overlap F1; it is not the official
NarrativeQA BLEU/ROUGE/METEOR evaluation. Results are directly comparable only
across methods that use this same 200-row active input, original 1,572-story
corpus, prompt, generation limit, and scorer. They are not directly comparable
to official known-story NarrativeQA results or the previous 1,000-row custom
test-query condition.

The local `triviaqa-unfiltered-wikipedia` data preserves all 11,313 official
TriviaQA unfiltered dev question/answer pairs in `query_full.jsonl` and
`answer_full.jsonl`. Its active `query.jsonl` and `answer.jsonl` are the same
1,000 row indices sampled uniformly without replacement with Python
`random.Random(42)`, then restored to source order without renumbering IDs.
Every method compared on this condition must use this identical active sample.
The retrieval corpus is the union of the 16,078 packaged Wikipedia
`EntityPages` documents; Web `SearchResults` documents are excluded. Therefore,
results on the active sample are directly comparable across methods using this
same corpus, but not directly comparable to results over all 11,313 questions
or to the complete Wikipedia-plus-Web unfiltered condition. No evidence labels
are defined for this local dataset. Dataset name
`triviaqa-unfiltered-wikipedia` selects the existing official TriviaQA
alias-aware EM/F1 evaluator; this registration changes only name resolution,
not normalization or scoring.

The local `dapr-nq-open` evaluation set is the 2,390-query exact-question
intersection of DAPR-NQ test and official NQ-open dev; it is not the full
NQ-open dev set. Its primary answer metric ports the official FiD/DPR evaluator
at commit `fe769f30e3714e22476910ee39ea0054dd7921de`: lowercase the prediction
and each answer alias, remove ASCII punctuation and the English articles
`a`/`an`/`the`, normalize whitespace, then report exact match if any alias
matches. The upstream evaluator reports exact match only. The local `f1` field
is an auxiliary maximum token-overlap F1 over aliases using the same
normalization and must not be described as an official NQ-open metric. This
registration adds a previously unsupported dataset and therefore does not
change or invalidate results from existing dataset evaluators.

The DAPR-NQ gold-evidence oracle reader condition is a separate diagnostic
protocol. `run/eval_oracle.sh` bypasses approximate retrieval and
compression, looks up each query's labeled DAPR passages by exact query text,
removes exact duplicate passage texts while preserving label order, and places
only those passages in the existing QA prompt. It does not add titles,
neighboring sentences, or parent-document text. The prompt, generation, and
NQ-open answer evaluator are otherwise unchanged. These EM/F1 values estimate
reader performance given labeled evidence; their retrieval/compression timing
is not comparable to end-to-end RAG timing. `TOP_K=1` in this runner is only a
required engine placeholder and does not limit the number of labeled evidence
passages returned.

The oracle context source is implemented as
`GoldEvidenceVectorDB` in `src/gold_evidence_vectordb.py`. This name records
that it substitutes for the engine's `VectorDB` interface; it does not perform
vector search. Renaming the former `GoldEvidenceDB` implementation changes no
label loading, context construction, prompt, metric, or output, so results
before and after the rename remain directly comparable when all experiment
inputs match.

`grid_dapr_nq_gold_evidence_oracle.yaml` does not override `PROMPT_FORMAT` and
therefore uses the `raw_chunk_first` default from `run/eval_oracle.sh`.
`grid_dapr_nq_bge_rag_sllm.yaml` likewise uses the `raw_chunk_first` default
from `run/eval.sh`. Historical results whose run names contain `-chat-` used
`chat_system_user`; they are not directly comparable with these raw-prompt
conditions because prompt serialization can change model behavior.

`USE_CLEANER=False` is the current default in `run/eval.sh`. Results with
different cleaner settings are not directly comparable unless the prediction
post-processing equivalence is explicitly checked.

## Post-hoc Generation Metrics

`run/add_generation_metrics_to_summary2.py` computes auxiliary generation
metrics from completed prediction JSONL files without rerunning retrieval or
generation. It appends `rougeL_f1`, `bertscore_precision`,
`bertscore_recall`, `bertscore_f1`, and `bertscore_hash` to the existing
`summary2` CSV.

The current NewsQA `newsqa-bge-rag-consensus-bsz-0726-chat` and DAPR-NQ
`dapr-nq-bge-rag-tuned-bsz-0725` and `dapr-nq-bge-rag-bsz16-0724`
additions use:

- `rouge-score==0.1.2`, ROUGE-L F1, and Porter stemming;
- `bert-score==0.3.13`;
- `roberta-large`, layer 17, no inverse-document-frequency weighting, and no
  baseline rescaling;
- BERTScore hash
  `roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.49.0)`.

For multiple answer aliases, ROUGE-L selects the alias with maximum ROUGE-L
F1 independently. BERTScore selects the alias with maximum BERTScore F1 and
uses that alias's precision, recall, and F1. The reported values are macro
averages over questions. Empty predictions receive the BERTScore package's
raw zero score.

These columns are auxiliary local metrics, not the official primary NewsQA or
NQ-open metrics. Existing EM/F1 values and prediction files are unchanged.
Post-hoc values are directly comparable only when the reference population,
alias policy, package/model hash, stemming, IDF, and baseline-rescaling
settings match. Each run directory contains
`posthoc-generation-metrics.json`, which records the metric configuration and
SHA-256 hashes of the answer and prediction files.

## Retrieval/Evidence-Only Metrics

`run/eval_retrieval_only.sh` evaluates retrieval and optional
compression against evidence labels without running large language model
(LLM) generation. It always uses the project's custom `text_evidence_exact`
metric.

The evidence file is passed as `EVIDENCE_FILE` and defaults to
`$DATASET_PATH/evidence_labels.json`. For the current original-context
benchmarks, the 200 queries are the same LongBench HotpotQA/2Wiki queries, exact
matched to original validation records and paired with original supporting
document/fact labels.

The metric compares each gold evidence passage directly with the final
retrieved or compressed context text. It does not use cacheable
identifiers, retrieved chunk identifiers, source token spans, source character
spans, reconstructed source text, or source-document metadata. It reports:

- `char_exact_recall`: fraction of gold passages whose complete
  whitespace-normalized character sequence occurs contiguously in the context.
  This is the primary passage-level metric.
- `token_exact_recall`: fraction of gold passages whose complete
  `MODEL_NAME` token sequence occurs contiguously in the context.
- `all_char_exact` and `all_token_exact`: fraction of queries
  for which every gold evidence passage is exactly contained.
- `char_contiguous_partial_recall` and
  `token_contiguous_partial_recall`: auxiliary recall based only on the single
  longest exact contiguous substring shared by the gold passage and context.
  Disconnected matches are never combined.
- conditional exact retention: among passages exactly contained after
  retrieval, the fraction still exactly contained after compression.
- conditional partial retention: compressed longest-contiguous overlap divided
  by retrieved longest-contiguous overlap, capped at the retrieved overlap.

Character matching collapses each whitespace run to one ASCII space before
matching, so newline/space formatting differences do not cause a false
negative, but inserting or deleting a separator between words still changes
the sequence. Token matching applies the same whitespace normalization and
prepends one space before independently tokenizing the complete gold passage
and context. The common leading space makes the gold passage's first token use
the same word-boundary convention as an occurrence inside a larger context.
The complete gold token sequence must then occur as consecutive token IDs.

As of 2026-07-31, the character/token partial-recall-at-threshold fields and
the worst-passage partial-recall field are no longer produced. The remaining
exact, mean contiguous-partial, any/all, and conditional-retention formulas are
unchanged, so old and new outputs are directly comparable on those retained
fields only.

This metric requires only
`evidence_texts` and `evidence_passage_ids`; source-document files and
source-position metadata are not evaluation inputs.

Retrieval-only evidence evaluation always reads every record in `QUERY_FILE`.
It has no `TOTAL_NUM` environment variable or `--total_num` command-line
argument. The grid runner's automatic batch-size search uses the internal
`EVAL_PROBE_QUERY_LIMIT`/`--probe_query_limit` path only for its temporary probe
processes; the selected batch size's actual evaluation receives neither probe
setting and processes the complete query file.

Every new retrieval-only evidence run writes a run-local `eval_details` JSONL
file. Each query record includes `retrieved_context_text` and
`compressed_context_text`, containing the complete context strings used by the
evidence scorer. These strings are the prompt-visible retrieved and selected
text, excluding the system prompt and question. They allow labels and
text-evidence scoring rules to be replayed without rerunning retrieval or
compression. The grid result records the file path in `DETAILS_FILE`.
Historical detail files created before this schema change contain scores but
not these context strings and are not replayable in the same way.

Results produced by either the former source-span reconstruction metric or the
discarded non-contiguous LCS implementation are not directly comparable with
`text_evidence_exact`. The current metric measures exact full-passage
containment as its primary outcome and uses only contiguous overlap for its
partial-match diagnostics.

The former `legacy_text` path, including Rouge-L, supporting-document, and
supporting-subchunk overlap metrics, has been removed. Existing
`text_evidence_exact` results remain directly comparable before and after this
removal when all experiment inputs match because its label loading, context
construction, scoring, and output fields did not change. Former `legacy_text`
results are not directly comparable with `text_evidence_exact` results.

All answer evaluators require the prediction count to equal the ground-truth
record count. A mismatch raises `ValueError`; neither side is truncated and no
partial score is emitted. Results from completed runs whose counts already
matched are unchanged and directly comparable. Previously scored partial runs
that depended on automatic truncation do not satisfy the current evaluation
invariant.

New prediction logs preserve the stable ID from each query record. Before
answer scoring, every prediction ID is compared with the ground-truth ID at the
same row. A missing or unequal ID raises `ValueError`; equal counts with a
different row order are not scored. Historical prediction logs without IDs do
not satisfy this invariant and cannot be rescored through the current answer
evaluation entry point without a separately verified ID-bearing conversion.

### Custom NewsQA Consensus-Answer Evidence Protocol

`grid_newsqa_bge_evidence.yaml` uses a custom NewsQA retrieval diagnostic. It
is not an official NewsQA evidence-evaluation protocol. The active run
`newsqa-bge-consensus-evidence-0731` evaluates the 4,293 official test
questions retained by the repository's existing consensus-answer
materialization:

- `QUERY_FILE` is `questions/query.jsonl`.
- `EVIDENCE_FILE` is `dataset_info/evidence.jsonl`.
- Each question has exactly one evidence string: the single non-empty answer
  returned by NewsQA's official `get_consensus_answer()` logic and already
  stored in the aligned `answers/answer.jsonl`.
- Questions classified by the official consensus procedure as no-answer, bad
  question, or unresolved disagreement are absent from this 4,293-row
  population.
- Labels are joined to queries by the stable integer `id`, not query text.
  This preserves 14 additional rows across 9 repeated query strings without
  conflating their labels.
- Evidence containment is always scored against the complete reconstructed
  retrieved or compressed context string. No source-document filter is
  available in the evaluation path.
- Exact and partial character/token matching formulas are unchanged from
  `text_evidence_exact`.

The label artifact is produced reproducibly by
`test/prepare_newsqa_consensus_evidence.py`; its manifest records input/output
SHA-256 hashes, the record count, duplicate-query counts, and the construction
policy.

The earlier `newsqa-bge-raw-evidence-0726` run evaluates 5,126 raw annotator
spans with a different question population and label multiplicity. Its outputs
remain preserved, but they are not directly comparable with the 4,293-row
single-consensus-answer results. DAPR-NQ evidence results are also not directly
comparable as the same metric target because DAPR-NQ labels passages, whereas
this custom NewsQA protocol labels an answer span.

ColBERT compression queries optionally use `COLBERT_QUERY_MAXLEN` and
`COLBERT_QUERY_TRUNCATION_SIDE`. `right` preserves the query head and `left`
preserves the query tail when truncation is necessary. Unless explicitly
overridden, the artifact's `query_maxlen` and right truncation remain the
default.

`COLBERT_QUERY_MINLEN` enables adaptive query length. The query encoder still
tensorizes up to `COLBERT_QUERY_MAXLEN`, but only
`max(MINLEN, min(MAXLEN, content_WordPieces + 3))` query-vector rows
participate in MaxSim. Thus batch padding or `[MASK]` expansion beyond the
per-query effective length is excluded. The intended ColBERTv2-compatible
adaptive setting uses `COLBERT_QUERY_MINLEN=32`; it preserves the checkpoint's
q32 expansion floor while treating `COLBERT_QUERY_MAXLEN` as a content-driven
cap. If `COLBERT_QUERY_MINLEN` is unset, the official fixed-length behavior is
preserved and all `MAXLEN` query-vector rows participate.

Changing max length, min length, or truncation side is a query-encoding
ablation. It does not change dense retrieval but can change compression
selection, so the output records all three values and results are not directly
comparable as identical configurations.

## Prompt And Cache Modes

`EVAL_USE_PAST_CACHE=False` is cache-off mode. It constructs the full visible
prompt and runs regular HuggingFace generation.

`EVAL_USE_PAST_CACHE=True` is cache-on mode. It loads reusable KV cache
artifacts from `CACHE_DIR` and decodes with prefilled cache state.

Cache-on and cache-off results are directly comparable only when all visible
prompt content, retrieval settings, compression settings, model, generation
settings, and scoring settings match. Cache-on also requires cache artifacts to
match the evaluated dataset, `DATA_SUBDIR`, cacheable units, model/tokenizer,
and cache construction settings.

`provence` and `exit` currently emit compressed prompt text directly and are
cache-off only. They must not be run with `EVAL_USE_PAST_CACHE=True`.

`TOP_K=0` is a context-free cache-off baseline and must not be used with
`EVAL_USE_PAST_CACHE=True`.

## Retrieval And Artifacts

`DB_DIR` determines the vector retrieval database. `DATA_SUBDIR` is only a
convention for selecting dataset-local subdirectories; the actual DB path is
`DB_DIR`.

DB preprocessing buffers retrievable chunks and upserts them in batches.
`DB_BATCH_SIZE` defaults to 256 and is recorded in `build_manifest.json`.
Changing it leaves the generated chunk texts, metadata, IDs, and insertion
order unchanged, but Chroma's approximate HNSW graph is not guaranteed to be
identical across different insertion batch boundaries. Vanilla and compression
methods in one comparison must use the same physical DB. Results from DBs built
with different batch sizes are not guaranteed to have identical retrieval
rankings, especially for near-tied candidates.

For `CHROMA_EMBED_BACKEND=default`, DB preprocessing explicitly records
`CHROMA_EMBED_DEVICE` and `CHROMA_EMBED_BATCH_SIZE`. CPU and CUDA use the same
MiniLM ONNX graph, tokenizer, 256-WordPiece truncation, mean pooling, and
normalization, but floating-point embeddings are not bit-identical. DBs built
on different providers are therefore not retrieval-identical conditions.
Runtime `ChromaDB(db_dir)` pins query encoding to CPU; build-time CUDA settings
do not make online query encoding use the GPU. The runtime constructor has no
device option; CUDA is available only through the preprocessing-only
`ChromaDB.for_build(...)` path. Runtime GPU query encoding is outside the
evaluation protocol and must not be enabled.

When `DEDUPLICATE_DOCUMENTS_BY_HASH=True` is used during preprocessing, exact
duplicate source documents are removed from the vector DB by normalized full
text hash while the local `documents/` directory remains unchanged. This changes
the retrieval corpus and can change retrieved chunk IDs, so results from a
deduplicated DB are not directly comparable to results from the original DB
unless the corpus condition is reported. ColBERT-window artifacts must be
rebuilt against the deduplicated DB before running ColBERT-based compression.

`CHROMA_EMBED_BACKEND` controls how Chroma collections are opened:

- When the variable is unset, `bge_small_v1_5` is selected.
- The explicitly named `default` backend is the main paper MiniLM setting. It
  opens the collection with Chroma's ONNX `all-MiniLM-L6-v2` embedding
  implementation. Because this path does not pass an explicit HNSW
  configuration, Chroma 1.5.7 uses `ef_construction=100`, `ef_search=100`, and
  `M=16` (`max_neighbors=16`). `chroma_default` is an alias for this same path.
- `bge_m3`, `bge_small_v1_5`, and `e5_small_v2` use the project's
  `DenseTextEmbedder`/SentenceTransformers implementation and an explicit HNSW
  configuration: `ef_construction=200`, `ef_search=200`, and `M=32`
  (`max_neighbors=32`).

The small-model backends follow their published model-specific encoding rules:

- `bge_small_v1_5` uses CLS pooling, no passage prefix, and
  `Represent this sentence for searching relevant passages: ` for queries.
- `e5_small_v2` uses masked mean pooling and the `passage: ` / `query: `
  prefixes.

Both use normalized 384-dimensional embeddings and a maximum sequence length
of 512. Results from the older July 7 BGE-small analysis script are not directly
comparable: that script passed BGE-small through the former generic mean-pooling
path. The ConditionalQA comparison and the July 24 HotpotQA DBs use the correct
model-specific pooling.

Every backend above performs dense vector retrieval. In this documentation,
"Chroma default MiniLM backend" means `default`; "SentenceTransformers
retrieval backends" means `bge_m3`, `bge_small_v1_5`, and `e5_small_v2`.
Neither term means the `dense` context-compression method. Changing the
retrieval backend changes the retrieval condition and must be reported as a
separate retriever setting.

Changing the implicit default does not alter grids that explicitly set
`CHROMA_EMBED_BACKEND`. A DB built with MiniLM still requires
`CHROMA_EMBED_BACKEND=default`; opening it under the new implicit
`bge_small_v1_5` selection is not a compatible retrieval condition. Results
from MiniLM and BGE-small DBs are not directly comparable as the same
retriever.

Dense selectors use `DENSE_EMBED_DIR`. ColBERT-window selectors use
`COLBERT_WINDOW_DIR` and `COLBERT_MODEL_NAME`. Runtime ColBERT query encoding is
CPU-only, and initialization asserts that the checkpoint parameters are
actually on CPU. The artifact query length and right truncation are the
defaults; the query-shape ablation overrides documented above may change them.
Results produced before the 2026-07-23 runtime-device enforcement must not be
reported as CPU-query-encoding results merely because the wrapper requested
CPU: when CUDA was visible, upstream ColBERT could still place the checkpoint
and query tensors on CUDA. Those historical results are not directly comparable
for compression latency, and their selections can also differ because the CUDA
path used mixed precision while the enforced CPU path uses full precision.

ColBERT-window artifact construction is CUDA-only and uses the same
`COLBERT_MODEL_NAME` checkpoint variable as runtime. The optional tensorization
cross-check is disabled by fixed build policy rather than exposed as an
experiment environment variable; this does not change stored tensor values or
runtime scoring.

`COMPRESS_METHOD=colbert_subchunk` selects the global ColBERT subchunk
selector. This factory key replaces the former `colbert_window` key; the old
alias is intentionally unsupported. It walks globally ranked subchunks under
the budget configured by exactly one of `RETAIN_TOKEN_RATIO` or
`FINAL_TOKEN_BUDGET`.

Earlier `colbert_subchunk` candidate-count-ratio results and strict-fit token
budget results are not directly comparable to the current add-then-stop token
budget policy.

For `colbert_sliding_region`, v3 data artifacts store per-retrieval-chunk region
specs in `colbert_window/data/region_payloads.json`. They must be rebuilt after any
change to region-spec construction. In particular, results produced before the
skip-over packing fix in commit `6713d96` are not directly comparable to
results produced after rebuilding those sidecar specs. The split JSON sidecars are
mandatory. A missing chunk, mismatched cacheable-id list, or region budget
different from the artifact window budget is an error; runtime region
recomputation is intentionally unsupported.

Splitting the corpus-wide ID, window-membership, and region mappings into
`cacheable_rows.json`, `window_ids.json`, and `region_payloads.json` changes only
artifact storage and initialization. Production runtime eagerly loads only the
cacheable-row and region mappings into dicts. It does not load build-only window
membership. This does not change candidate text, ColBERT
vectors, region construction, scoring, selection, prompt construction, or
metrics, so pre-migration and v3 results are directly comparable when all experiment
inputs and stored values are otherwise identical. The v3 runtime intentionally
has no legacy reader; legacy artifacts must be rebuilt.

The coarse retrieval chunk size must be strictly larger than the stored
ColBERT region/window budget. The former additional 100-token heuristic buffer
was removed on 2026-07-23 because it had no artifact or scoring invariant
behind it and prevented the valid coarse-256/window-180 condition. This change
only enables configurations in that former buffer range; it does not change
region construction or selection for existing coarse-512 results.

### ConditionalQA Dense-Embedder Retrieval Comparison

`test/analyze_conditionalqa_embedder_retrieval.py` compares strict fixed chunks
from the existing ConditionalQA vanilla databases. It validates source-token
span continuity and source-text equality before building a separate Chroma
index for each embedder and chunk size. Passage embeddings are built on CUDA;
runtime query embeddings are deliberately computed on CPU.

The evaluated configurations are:

- `sentence-transformers/all-MiniLM-L6-v2`, no text prefixes, maximum sequence
  length 256.
- `intfloat/e5-small-v2`, `query: ` and `passage: ` prefixes, maximum sequence
  length 512.
- `BAAI/bge-small-en-v1.5`, its retrieval query instruction and no passage
  prefix, maximum sequence length 512.

All models use their SentenceTransformers pooling configuration, normalized
embeddings, the same Chroma HNSW configuration, and the same insertion order.
The primary metric is non-whitespace character coverage of DAPR source-span
evidence; Llama-3.1-8B token coverage is reported as a secondary metric.
Changing the embedder creates a new retrieval baseline, so these results are
not directly comparable to indexes built with a different pooling rule,
prefix, maximum length, or insertion order.

`RETRIEVAL_INCLUDE_DOCUMENTS=False` is allowed only for methods whose prompt
text can be reconstructed from cacheable metadata. Do not use it for methods
that require original retrieved document text unless that path has been
explicitly checked.

## Compression And Budget Controls

`COMPRESS_METHOD` selects the retrieval/compression variant. Empty
`COMPRESS_METHOD` means no downstream compressor; retrieved units are used
directly.

The unused `COMPRESS_METHOD=front` baseline has been removed. Existing results
from that method remain historical baseline results, but the configuration is
no longer accepted. The OpenAI-backed `COMPRESS_METHOD=summ` baseline remains
available and its behavior is unchanged.

`FINAL_TOKEN_BUDGET` controls the absolute prompt-token target for `dense`,
`colbert_subchunk`, `colbert_sliding_region`, and `rerank_and_region`.

`RETAIN_TOKEN_RATIO` computes the final token budget per query as a ratio of
retrieved context tokens. Exactly one of `RETAIN_TOKEN_RATIO` and
`FINAL_TOKEN_BUDGET` must be set for those four methods. Setting both or neither
is a configuration error.

For all fixed-budget and retained-ratio paths, final budget accounting uses
the prompt-visible passage text format, `text.strip() + "\n\n"`, tokenized with
the evaluated `MODEL_NAME` tokenizer. It must not use ColBERT artifact vector
row counts or a separate ColBERT/window-artifact
tokenizer, because those can differ from the LLM prompt tokenizer and can be
truncated by the ColBERT document max-length. Results produced before this
budget-accounting fix are not directly comparable to results produced after the
fix.

New DB/materialization outputs store this per-cacheable prompt-visible length as
`prompt_token_count` plus its `prompt_tokenizer_name` inside each
`cacheables_json` payload. Query-time selection uses that stored integer
only when `prompt_tokenizer_name` exactly matches the runtime `MODEL_NAME`; it
falls back to runtime `MODEL_NAME` tokenization when the count is missing or the
tokenizer does not match.

The final token budget is a soft target. Each selector adds the next complete
ranked candidate and then stops when accumulated prompt-visible tokens reach or
exceed the budget. Dense and `colbert_subchunk` can exceed the target by the
last subchunk; region methods can exceed it by the novel subchunks in the last
region. No method splits or skips a higher-scoring candidate merely because it
crosses the target.

Fixed-token-budget runs and retained-ratio runs are not directly comparable
unless the resulting input lengths are reported and intentionally matched.

## Generation And Timing

The generation workload is controlled by:

- `MODEL_NAME`
- `MODEL_LOAD_IN_4BIT`
- `EVAL_BSZ`
- `MAX_NEW_TOKENS`
- `TOTAL_NUM`
- `EVAL_USE_PAST_CACHE`
- cache and prompt construction settings

`MODEL_LOAD_IN_4BIT=False` is the current latency-oriented default. 4-bit and
fp16 runs are different inference backends and should be reported separately.

`EVAL_BSZ` affects both throughput and model behavior. Quality metrics from
different batch sizes should not be treated as identical unless equivalence has
been checked.

### Longest-Prompt Stress Selection For Throughput Batch Size

`test/probe_max_bsz_and_eval.py` implements a custom batch-size selection
protocol for cache-off throughput runs. It replaces the older capped-prefix
binary probe when an exact matching batch-size-1 result already exists.

- The input grid must define `max_prompt_bsz_probe.source_run_dir`, pointing to
  a completed batch-size-1 grid run with prediction JSONL files.
- For every eval case, the utility matches exactly one successful source result
  using the case fields other than `EVAL_BSZ` and probe-only controls.
- It reconstructs every source prompt from the saved ordered `ctxs`, question,
  current system prompt, and `raw_chunk_first`. The recomputed mean token length
  must match the source result's recorded `avg_nocache_model_input_len` after
  rounding to four decimal places, or the utility stops.
- The longest reconstructed prompt determines that case's stress length. The
  initial candidate is the configured `target_padded_tokens` divided by
  `max_prompt_tokens + max_new_tokens`, rounded down to `step`, unless the case
  explicitly sets `PROBE_INITIAL_BSZ`.
- The utility repeats the same longest prompt for every batch row. Hugging Face
  generation does not deduplicate identical rows, so this materializes the
  intended batch-by-maximum-length LLM tensors and KV cache.
- The probe uses the same model name and precision as the target grid. It
  reproduces the project's split generation path and forces exactly the
  configured number of new tokens with `min_new_tokens`, preventing early EOS
  from understating peak memory.
- On CUDA OOM, the utility clears probe allocations, subtracts `step` from the
  candidate, and retries. It does not probe upward after a success.
- After every case succeeds, it writes `max_bsz_probe.jsonl` and
  `selected_grid.yaml` under the target run directory. Unless `--probe-only` is
  supplied, it launches the standard grid runner with the selected fixed batch
  sizes. `--dry-run` performs source matching, prompt reconstruction, and
  initial-candidate calculation without loading the GPU model.
- The command accepts one or more grid YAML paths. Multiple grids execute
  sequentially under the same outer GPU lock; each grid runs in an isolated
  child process. As in `run/run_grid.sh` grid mode, later grids still run if an
  earlier grid fails, and the final status is nonzero if any grid failed. This
  orchestration does not combine cases or results across run directories.
- Rerunning the same command and `run_name` resumes completed work. Existing
  `max_bsz_probe.jsonl` records must form the ordered case prefix `0..N-1`.
  Before reuse, every record is checked against the current case mapping,
  source output path, reconstructed maximum and mean prompt lengths, generation
  length, padded-token target, initial batch size, selected batch size, and a
  successful probe attempt. Newly written records also persist and validate the
  probe step and minimum/maximum batch-size bounds. A mismatch or malformed
  record stops the run rather than silently reusing it.
- After restoring that completed prefix, only the remaining probe cases run.
  If every probe is already complete, the utility does not initialize CUDA or
  reload the LLM. An existing `selected_grid.yaml` must exactly equal the grid
  reconstructed from the restored records. The standard grid runner then skips
  exact matching successful evaluation cases recorded in `results.jsonl` and
  reruns missing or failed cases. A probe or evaluation case interrupted before
  its success record was appended restarts that case from its beginning; work
  inside an incomplete case is not checkpointed.

Because the next larger batch is not tested, the selected value is a
conservative stress-tested batch size, not a proven mathematical maximum.
Throughput results selected by this protocol are directly comparable only when
the source-prompt construction, `target_padded_tokens`, decrement step, model,
precision, generation length, GPU state, and all ordinary eval settings are
reported and matched.

The timing fields printed by `src/engine.py` are batch averages unless otherwise
stated:

- retrieval time
- compression time
- prompt build time
- prompt stats time
- generate extra time
- prefill time
- decode time
- total time per batch

CUDA synchronization is applied around model prefill/decode timing windows in
`src/model.py`. Latency comparisons should be made on an otherwise idle GPU, or
under an explicitly documented GPU lock/exclusive-process setup.

ColBERT-based compression always clears process-local retrieved-vector and
sliding-region-spec caches before every measured compression batch. This
preserves intra-batch deduplication while preventing reuse across measured
batches. It also always runs one dummy query encoding after compressor
initialization, moving the first-forward cost into setup rather than the first
measured compression batch. These are fixed evaluation policies rather than
runtime options.

`COMPRESS_METHOD=colbert_chunk_rerank` is a fixed-retrieval-chunk reranking
baseline. It uses the same vector DB retrieval result as vanilla RAG, scores the
retrieved fixed chunks with materialized ColBERT chunk representations, and
keeps the top `COLBERT_CHUNK_RERANK_KEEP` chunks.

For this baseline:

- offline ColBERT chunk materialization time is not counted in eval latency.
- online query encoding, artifact lookup, ColBERT MaxSim scoring, reranking,
  prompt construction, and LLM inference are counted.
- use `colbert_fixed_chunk_docmax512` artifacts, not the older
  `colbert_fixed_chunk` artifacts built with ColBERT default `doc_maxlen=180`.
- kept chunks are prompted in ColBERT reranked order.

`COLBERT_RERANK_KEEP` controls the coarse-chunk keep count for both reranking
methods below. Both score a retrieved coarse chunk by loading all of its
subchunk representations from the matching DB's `colbert_window` artifact,
concatenating their token-vector pools, and applying one query-to-pool ColBERT
MaxSim score. Neither method uses the separate
`colbert_fixed_chunk_docmax512` artifact.

- `COMPRESS_METHOD=colbert_rerank` keeps the top-K coarse chunks and prompts
  their complete contents in reranked order.
- `COMPRESS_METHOD=rerank_and_region` keeps the top-K coarse chunks as an
  eligibility set, then constructs, scores, and selects sliding regions only
  inside those chunks. For retained-ratio runs, the token budget is computed
  from the kept coarse chunks rather than the original retrieval top-k set.

The removed `colbert_window_chunk_rerank` name has the same scoring/output
semantics as `colbert_rerank`, so those results are directly comparable after
renaming the configuration field. The former fixed-artifact pre-filter changed
its coarse scoring artifact and chunk alignment, so its results are not
directly comparable to the new `rerank_and_region`. The post-filter method was
removed and has no replacement.

`ttft_per_batch_avg_sec` in grid summaries is derived as:

```text
time_per_batch_avg_sec - decode_per_batch_avg_sec - generate_extra_per_batch_avg_sec
```

It is a reconstructed batch-average latency component, not an independently
instrumented server-side TTFT.

## Output Files And Grid Summaries

### Custom HotpotQA Sampled-200 Labeled-Evidence Oracle

`run/grid_search/grid_hotpotqa_sampled_gold_evidence_oracle.yaml` defines a
custom oracle evaluation protocol. It is not the official full HotpotQA
benchmark context setting.

- The query set and order come from
  `questions/query_sampled.jsonl` (200 examples).
- Each query is matched exactly by its `query` string to
  `dataset_info/evidence_labels.jsonl`.
- The context contains every corresponding `evidence_texts` value in its stored
  order. The protocol performs no retrieval, compression, sorting, title
  insertion, document expansion, evidence repair, or fallback.
- The sampled subset has been verified to contain no repeated evidence text
  within an example, so the existing exact-text deduplication in
  `GoldEvidenceVectorDB` does not alter these 200 contexts.
- `PROMPT_FORMAT=raw_chunk_first` serializes the context as evidence passages,
  followed by the system prompt, `Question: <query>`, and `Answer:`. It does not
  apply a chat template.
- Generation uses `meta-llama/Llama-3.1-8B-Instruct` in FP16
  (16-bit floating point), batch size 1, at most 20 new tokens, cache-off
  inference, and `USE_CLEANER=False`.
- Answers are scored against `answers/output_sampled.jsonl` through the same
  HotpotQA answer evaluator used by the sampled retrieval runs.

Answer exact match and F1 (token-overlap F1 score) use the same sampled queries,
answers, model, generation settings, prompt format, and scoring path as the
corresponding sampled retrieval runs. The results are not directly comparable
as an identical retrieval configuration because this protocol supplies labeled
evidence instead of retrieved context; report it as a custom oracle baseline.

`run/eval.sh` writes prediction JSONL to `OUTPUT_FILE` when set, otherwise to
`outputs/eval-...jsonl` with suffixes derived from method, `TOP_K`, cache mode,
LLM precision, model tag, and only the selection controls effective for that
method. Retained-ratio runs use an `rtr` tag and absolute-budget runs use an
`ftb` tag. Rerank-only methods have no token-budget tag.

Prediction and answer JSONL parsing treats only LF (`U+000A`) as the record
delimiter. Other Unicode line-separator characters, including `U+0085` NEXT
LINE, remain part of JSON string values such as retrieved source text. This
delimiter rule changes no prediction, answer normalization, metric, or scoring
formula; it prevents embedded source characters from being misclassified as
record boundaries.

`run/grid_search/eval.py` writes:

- `manifest.json`
- `events.log`
- per-run logs under `logs/`
- `results.jsonl`
- `failures.jsonl`
- `summary-<dataset>.csv`
- `summary-<dataset>-summary2.csv`

`summary2` keeps a compact table with method, budget controls, input length,
EM/F1, prefill/decode latency, elapsed time, total batch time, compression time,
derived TTFT, and retrieval time.

## Comparability Rules

Treat two results as directly comparable only when all of the following match:

- dataset and dataset files
- answer scoring path and `USE_CLEANER`
- model, tokenizer, precision/backend, and generation settings
- `EVAL_BSZ`, generation-only `TOTAL_NUM`, and `MAX_NEW_TOKENS`
- cache-on/cache-off mode
- prompt template and visible prompt serialization
- retrieval DB, embedding backend, `TOP_K`, and retrieval artifacts
- compression method and all method-specific environment variables
- fixed ColBERT inter-batch cache clearing and query-encoder warmup policy
- final budget controls such as `FINAL_TOKEN_BUDGET`,
  `COLBERT_CHUNK_RERANK_KEEP`, and `RETAIN_TOKEN_RATIO`
- GPU/timing environment when reporting latency or throughput

If any of these differ, report the changed condition explicitly and treat the
result as a separate ablation or baseline.

### Experimental Sliding-Region Group Ordering

`COLBERT_REGION_GROUP_ORDER` is a temporary experiment control for
`colbert_sliding_region` and `rerank_and_region`. Its supported values are:

- `retrieval`: preserve the original coarse-retrieval chunk order.
- `max`: order selected coarse chunks by the maximum score among selected
  regions that contributed at least one previously unselected subchunk.
- `sum`: order selected coarse chunks by the sum of those contributing region
  scores.

All three modes preserve region selection, overlap deduplication, token budget,
selected subchunks, and source order within each coarse chunk. They change only
the order of coarse-chunk groups in the LLM prompt. Results from different
values are therefore prompt-order ablations and are not directly comparable as
identical evaluation configurations. Non-retrieval modes add an `rgo<mode>` tag
to the default output filename. This control is experimental and should be
removed after the ordering ablation is concluded.

## When This Document Must Change

Update this document whenever evaluation behavior changes in a way that affects:

- dataset handling
- retrieval flow
- prompt format
- cache validity
- generation settings
- timing definitions
- metrics or scoring
- output schema used by grid summaries
- result comparability

Historical one-off notes and old bug-fix logs should go to
`docs/eval_protocol_history.md` or `docs/handoff.md`, not into this canonical
protocol.
