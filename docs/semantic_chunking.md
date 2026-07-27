# Sentence Chunking And Length Analysis

This note describes the current sentence-level chunking path and summarizes the
HotpotQA / 2Wiki sentence length analyses that motivated later subchunk and
region-selection experiments.

## Current Sentence Chunking Algorithm

The active sentence splitting logic is implemented in
`src/materialize/splitter/base.py`.

The sentence path is:

1. Load the source document text.
2. Tokenize the full document with the LLM tokenizer.
3. Split the text into sentence strings with `blingfire.text_to_sentences`.
4. Apply a conservative post-pass that merges obvious false sentence boundaries.
5. Locate each sentence back in the original source text.
6. Use tokenizer offset mappings to map each sentence span to token start/end
   positions.
7. Create one `CacheableChunk` per sentence.
8. If `retrievable_chunk_size` is set, create fixed-size retrievable windows and
   attach every overlapping sentence cacheable to each window.

The conservative boundary merge handles cases where `blingfire` splits after
short abbreviations or title/legal fragments. Examples include endings such as:

- `v.`
- `vs.`
- `No.`
- `Mr.`, `Mrs.`, `Ms.`, `Dr.`, `Prof.`, `Sr.`, `Jr.`
- `U.S.`, `U.K.`, `D.C.`
- very short initial fragments such as `A.` or `J. K.`

This merge is intentionally narrow. It is not a semantic merge policy; it only
fixes likely sentence-boundary artifacts.

## Sentence Cacheables And Retrievable Chunks

For `splitter=sentence`, each sentence becomes a cacheable/evidence unit:

- id format: `filename::sent_{idx}`
- text: stripped source sentence text
- `chunk_start` / `chunk_end`: token span in the original document
- `sentence_ids`: one id for the sentence itself
- `sentence_texts`: one sentence text

When a retrievable chunk size is set, retrieval units are token windows over the
original document. A sentence cacheable is attached to a retrievable window when
its token span overlaps the window:

```text
sentence_start < window_end and sentence_end > window_start
```

This means retrieval can remain coarse while candidate evidence units remain
sentence-level.

### Boundary Sentence Duplication

Retrievable chunks are fixed token windows, but sentence cacheables are not cut
to fit those windows. If a sentence crosses a retrievable-window boundary, the
same sentence cacheable is attached to both neighboring retrievable chunks.

Example with `retrievable_chunk_size=512`:

```text
retrievable window A: [0, 512)
retrievable window B: [512, 1024)
sentence S:          [500, 530)
```

Because `S` overlaps both windows, it is included in both:

```text
S.start < A.end and S.end > A.start
S.start < B.end and S.end > B.start
```

This is intentional. It avoids splitting a sentence into two partial evidence
units only because it happened to straddle a retrieval boundary. The trade-off
is that the same sentence id can appear in more than one retrieved chunk's
candidate list.

Downstream compressors and prompt builders should therefore deduplicate by
cacheable id/text when they need a final unique evidence set. The retrievable
chunk is a search container; the sentence cacheable is the evidence unit.

### ChromaDB Layout

`ChromaDB.store()` stores each retrievable chunk as one Chroma row:

- Chroma `id`: retrievable chunk id, e.g. `doc.txt::ret_0`.
- Chroma `document`: retrievable window text decoded from the source token
  window.
- Chroma `metadata`:
  - `parent_doc_id`
  - `source_token_start`
  - `source_token_end`
  - `chunk_size`
  - `token_count`
  - `cache_unit`
  - `cacheables_json`

`cacheables_json` is a serialized list of every `CacheableChunk` attached to
that retrievable window. For sentence splitting, those cacheables are sentence
objects with ids such as `doc.txt::sent_17`.

When a boundary sentence overlaps two retrievable windows, Chroma stores the
same sentence cacheable payload in both rows' `cacheables_json`. There is still
only one logical evidence id, but it is reachable through either retrieval
window. At query time, `ChromaDB.find_top_k_docs()` deserializes
`cacheables_json` and returns `RetrievableChunk` objects whose `cacheables`
fields contain those candidate sentence units.

## Semantic Splitter Path

`splitter=semantic` reuses the same parsed sentence units, then groups their
indices with a `UnitGrouper`.

The current semantic grouping framework is:

1. Parse the document into `ParsedUnit` sentence units.
2. Pass parsed units to a grouper.
3. The grouper returns groups of unit indices.
4. Each group is joined into one semantic subchunk.
5. The semantic subchunk span runs from the first grouped sentence token to the
   last grouped sentence token.

Current grouper implementations include:

- `identity`: keeps each parsed unit separate; this is used by
  `SentenceWiseSplitter`.
- `pronoun_dp_128`: dynamic programming merge that rewards keeping
  pronoun-starting sentences with nearby context, under a token budget.
- `coref_pronoun_dp_128`: similar goal, but uses `fastcoref` to check whether a
  leading pronoun has an antecedent inside the candidate span.

`SentenceWiseSplitter` and `SemanticSplitter` are compatibility wrappers over
the same `ParsedUnitSplitter` orchestration. Their existing long-unit handling
and cacheable ID formats remain different and unchanged.

Important status note: `UnitGrouper` and the semantic grouper variants are
currently outdated relative to the main project direction. They remain in the
codebase as experimental/prototype paths, but current documentation and paper
framing should focus on sentence evidence units, retrieval-bounded candidate
selection, and contextualized sentence/subchunk representations. Do not treat
the grouper variants as the active method unless they are explicitly refreshed
and re-evaluated under the current evaluation protocol.

## Full-Dataset Sentence Length Distribution

Earlier full-dataset analysis measured sentence lengths after `blingfire`
splitting.

Measurement setup:

- dataset split: full document collections
- sentence split: `blingfire.text_to_sentences`
- tokenizer: `meta-llama/Llama-3.1-8B-Instruct`
- token count: `tokenizer.encode(sentence, add_special_tokens=False)`

| Dataset | docs | sentences | mean | median | p90 | p95 | p99 | max | <=32 | <=64 | <=128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TriviaQA | 2,742 | 70,463 | 30.90 | 25 | 51 | 66 | 135.0 | 2,158 | 67.92% | 94.72% | 98.91% |
| HotpotQA | 1,722 | 73,539 | 34.31 | 28 | 59 | 75 | 129.0 | 1,887 | 60.62% | 92.31% | 98.99% |
| 2Wiki | 1,986 | 39,806 | 34.98 | 28 | 60 | 78 | 158.95 | 737 | 60.56% | 91.64% | 98.40% |

Takeaways:

- HotpotQA and 2Wiki have very similar sentence-length profiles.
- Most sentences are short:
  - about 61% of HotpotQA / 2Wiki sentences are `<=32` tokens.
  - over 91% are `<=64` tokens.
  - over 98% are `<=128` tokens.
- Extreme max values are outliers and should not drive the main chunking policy.

## HotpotQA Vs 2Wiki `sent-ret512` Sentence Length Check

A later analysis compared sentence lengths in the `sent-ret512`
ColBERT-window artifact view for HotpotQA and 2Wiki.

Purpose:

- Check whether HotpotQA's weaker sliding-region behavior was caused by a bias
  toward unusually long or unusually short sentences.
- Verify that similar statistics were not caused by accidentally reading the
  same artifact for both datasets.

Sanity check:

- Sampled source texts were different.
- Unique normalized sentence hash overlap was only `34` sentences.
- Overlap rates:
  - vs 2Wiki unique sentences: `0.114%`
  - vs HotpotQA unique sentences: `0.047%`
  - Jaccard: `0.033%`

Artifact-wide source/Llama-token sentence lengths:

| Dataset | mean | p50 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 35.46 | 28 | 61 | 79 | 163 | 782 |
| HotpotQA | 34.76 | 28 | 59 | 75 | 134 | 1,891 |

After duplicate sentence removal:

| Dataset | unique count | mean | p50 | p90 | p95 | p99 | >=128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 29,696 | 35.40 | 28 | 61 | 78 | 152 | 1.52% |
| HotpotQA | 71,952 | 34.91 | 28 | 59 | 75 | 135 | 1.11% |

Stored ColBERT center-vector lengths after ColBERT `doc_mask`:

| Dataset | mean | p50 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 27.93 | 23 | 50 | 64 | 118 | 176 |
| HotpotQA | 28.03 | 24 | 49 | 62 | 104 | 176 |

Window sentence counts:

| Dataset | mean | p50 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 6.71 | 7 | 9 | 10 | 16 | 37 |
| HotpotQA | 6.77 | 7 | 9 | 10 | 15 | 30 |

Candidate prompt units from `sent-ret512 TOP_K=20`:

| Dataset | mean | p50 | p90 | p95 | p99 | >=128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 36.22 | 28 | 62 | 81 | 183 | 1.83% |
| HotpotQA | 37.47 | 29 | 62 | 81 | 170 | 1.57% |

## Selected Sliding-Region Prompt Unit Lengths

Selected prompt units from `colbert_sliding_region` are longer than original
sentence cacheables because each selected prompt unit is a region, not a single
sentence.

Budget `1300`:

| Dataset | selected unit mean | p50 | p90 | selected units/query | selected token sum/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 107.02 | 99 | 191 | 11.89 | 1,272.45 |
| HotpotQA | 99.34 | 79 | 186 | 12.52 | 1,243.21 |

Budget `2400`:

| Dataset | selected unit mean | p50 | p90 | selected units/query | selected token sum/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 107.73 | 95 | 191 | 21.84 | 2,352.34 |
| HotpotQA | 98.89 | 77 | 185 | 23.23 | 2,297.32 |

Interpretation:

- HotpotQA does not show an artifact-wide tendency toward unusually long or
  unusually short sentences compared with 2Wiki.
- Retrieved candidate distributions are also similar.
- HotpotQA selected sliding-region prompt units are slightly shorter than 2Wiki,
  not longer.
- Therefore HotpotQA's weaker sliding-region advantage is unlikely to be caused
  primarily by raw sentence length distribution.

The more likely causes are task/evidence structure, retrieval coverage, document
scale, and region scoring or prompt assembly behavior.

## Related Document-Length Finding

Sentence lengths are similar, but document lengths are not.

Under the same `sent-ret512` artifact view:

| Dataset | docs | mean | p50 | p75 | p90 | p95 | p99 | max | <=512 | <=1024 | >=2048 | >=4096 | >=8192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 1,986 | 709.7 | 411 | 791 | 1,495 | 2,013 | 6,027 | 12,717 | 59.0% | 82.2% | 4.9% | 2.0% | 0.6% |
| HotpotQA | 1,722 | 1,480.1 | 732 | 1,746 | 3,624 | 5,464 | 10,511 | 14,151 | 38.7% | 60.0% | 21.0% | 8.3% | 2.2% |

This suggests that HotpotQA's weaker behavior is more plausibly related to
longer documents, more retrieval-window fragmentation, evidence localization,
or title/context propagation than to sentence length itself.

## Practical Implications

- Sentence-level units are already short enough for fine-grained evidence
  selection.
- A semantic merge policy should be conservative; aggressive merging can quickly
  erase the token savings from sentence-level evidence selection.
- `128` tokens is a reasonable first cap for sentence-group experiments because
  it covers more than 98% of individual sentences in HotpotQA and 2Wiki.
- For ColBERT-window methods, the selected prompt unit may be a region even when
  the center evidence unit is a sentence. Length analysis must distinguish:
  - source sentence length.
  - stored center-vector length after ColBERT masking.
  - candidate prompt unit length.
  - actual selected prompt region length.
- Do not explain HotpotQA vs 2Wiki differences with sentence length alone unless
  a later analysis contradicts the current evidence.
