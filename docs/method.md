# Method: Two-Phase Contextualized Subchunk Selection

The main paper method is `colbert_sliding_region` over Chroma `default`
retrieval DBs, e.g. `sent-default-512`.

The method separates coarse retrieval from fine-grained prompt construction:

- Retrieval unit: fixed-size Chroma retrievable chunks.
- Evidence unit: sentence/subchunk cacheables attached to retrieved chunks.
- Scoring unit: ColBERT token vectors for contextualized center sentences.
- Prompt unit: selected sliding regions assembled from retrieved sentence
  cacheables.

The system has two phases:

1. Offline phase: build the retrieval DB and ColBERT window artifact.
2. Online phase: retrieve chunks, locate ColBERT vectors, score sliding regions,
   select under budget, and run LLM inference.

## Offline Phase

### 1. Sentence Cacheables

Documents are first split into sentence-level cacheables.

The sentence splitter:

1. reads the source document text.
2. splits with `blingfire.text_to_sentences`.
3. applies conservative boundary repair for abbreviation-like splits.
4. maps each sentence back to source character offsets.
5. maps character spans to LLM-token start/end offsets using tokenizer offset
   mappings.

Each sentence becomes a `CacheableChunk`:

- `id`: `doc_id::sent_{idx}`
- `text`: sentence text
- `parent_doc_id`: source document id
- `chunk_start`, `chunk_end`: token span in the source document
- `sentence_ids`: the sentence id
- `sentence_texts`: the sentence text

The sentence cacheable is the basic evidence unit.

### 2. Chroma Retrieval DB

The vector DB is built from fixed-size retrievable token windows, not from
individual ColBERT vectors.

For a retrievable chunk size such as `512`, each source document is divided into
token windows:

```text
doc.txt::ret_0 = [0, 512)
doc.txt::ret_1 = [512, 1024)
doc.txt::ret_2 = [1024, 1536)
...
```

Every sentence cacheable whose token span overlaps a retrievable window is
attached to that window:

```text
sentence_start < window_end and sentence_end > window_start
```

This means a boundary sentence that crosses a retrievable-window boundary is
attached to both neighboring windows. The sentence is not split into partial
evidence units.

### 3. Chroma Row Layout

Each retrievable window is stored as one Chroma row:

- Chroma `id`: retrievable chunk id, e.g. `doc.txt::ret_1`.
- Chroma `document`: retrievable window text.
- Chroma `metadata`:
  - `parent_doc_id`
  - `window_token_start`
  - `window_token_end`
  - `chunk_size`
  - `token_count`
  - `cache_unit`
  - `cacheables_json`

`cacheables_json` is a serialized list of the sentence `CacheableChunk` payloads
attached to that retrievable chunk.

This Chroma metadata is what connects DB retrieval results to subchunks. A
retrieved Chroma row directly exposes the candidate sentence ids and texts
available for downstream ColBERT scoring.

### 4. ColBERT Artifact Scope

ColBERT embeddings are materialized by parent document, not by retrievable
chunk.

The artifact layout is:

```text
colbert_window/
  index.json
  docs/
    doc_0.txt.pt
    doc_1.txt.pt
    ...
```

`index.json` maps each parent document id to its `.pt` payload file.

Each parent document payload stores:

- `doc_id`
- `cacheable_ids`
- `cacheable_texts`
- `window_texts`
- `window_selected_indices`
- `window_addition_order`
- `window_truncated_center`
- `center_token_vectors`
- `embedding_dim`

The key alignment is:

```text
cacheable_ids[i] <-> center_token_vectors[i]
```

Therefore, once runtime knows a retrieved sentence cacheable id, it can locate
the corresponding ColBERT vector by:

1. finding the parent document id.
2. loading that parent document `.pt` payload.
3. building `cacheable_id -> center_token_vectors` from the payload.
4. selecting vectors for the cacheables attached to the retrieved Chroma row.

### 5. Offline ColBERT Window Construction

For the main sentence-centered artifact, ColBERT does not encode each sentence
alone. For every center sentence, it builds a context window around that center.

The default main artifact uses a ColBERT document length budget of `180` tokens.
This is measured with the official ColBERT document tokenizer, not the LLM
prompt tokenizer.

For each center sentence:

1. start with the center sentence.
2. account for ColBERT document token overhead.
3. alternately try to add the left and right neighboring sentences.
4. keep a neighbor only if the resulting token count stays within the `180`
   token budget.
5. continue until no side can add more sentences.

If the center sentence alone is already too long, it is marked as truncated and
encoded as the center-only text. The input text is center-only in this case, but
the official ColBERT tensorization can still truncate tokens to its document
maximum length.

The encoder input is the full window text. However, after encoding, the artifact
stores only the ColBERT token vectors whose offsets overlap the center sentence.

This is the core contextualization step:

- representation context: center sentence plus neighboring sentences.
- stored vector target: center sentence only.
- prompt/evidence unit: still the center sentence or a selected region built
  from center sentence units.

### 6. Why Document-Level ColBERT Files Matter

Because ColBERT contextualization is parent-document scoped, each parent
document `.pt` file contains all center sentence vectors for that document.

This avoids creating separate embedding files per retrievable chunk and makes
the same sentence vector reusable when:

- a sentence appears in multiple retrievable chunks because it crosses a
  boundary.
- multiple retrieved chunks come from the same parent document.
- different queries retrieve different windows from the same parent document.

The trade-off is that runtime loads a whole parent document payload when any
retrieved chunk from that parent document needs ColBERT vectors.

## Online Phase

### 1. Retrieval

At query time, Chroma retrieves top-k retrievable chunks.

For the main paper setting, the DB backend is Chroma `default`, i.e. the
lightweight default/MiniLM-style retrieval configuration.

Each retrieved row is deserialized into a `RetrievableChunk`:

- `id`: retrievable chunk id.
- `text`: retrieved window text.
- `metadata`: parent document/window metadata.
- `cacheables`: sentence cacheables from `cacheables_json`.

No sentence splitting is performed from raw retrieved text at this stage. The
candidate subchunks are already carried by the Chroma row metadata.

### 2. Locating ColBERT Vectors

For every retrieved chunk, the compressor calls `artifact.vectors_for_doc(doc)`.

The artifact loader resolves the parent document id using:

1. `doc.metadata["parent_doc_id"]`, if present.
2. otherwise `cacheable.parent_doc_id`.
3. otherwise the prefix before `::ret_` in the retrieved chunk id.

Then it loads the parent document payload:

```python
self.doc_cache[doc_id] = torch.load(path, map_location="cpu")
```

The payload is cached in memory by parent document id. Therefore, if the same
query retrieves several chunks from the same parent document, or later queries
touch the same parent document, that `.pt` file is loaded only once per eval
process.

Important behavior:

- the whole corpus artifact is not loaded at startup.
- the whole parent document `.pt` file is loaded when needed.
- there is no eviction policy in the current `doc_cache`.
- memory use can grow with the number of unique parent documents touched during
  an eval run.

### 3. Retrieval-Bounded Candidate Pool

`colbert_sliding_region` only considers sentence cacheables attached to the
retrieved chunks.

For each retrieved chunk:

1. collect its sentence cacheables.
2. load the parent-document ColBERT vectors.
3. select vectors matching the retrieved cacheable ids.

The method does not score every sentence in the corpus. Coarse Chroma retrieval
first bounds the candidate set, and ColBERT scoring only reranks evidence inside
the retrieved chunks.

### 4. Online Sliding Regions

For each retrieved chunk, the method constructs sliding regions over the
retrieved sentence cacheables.

This uses the same centered-window helper/rule as offline materialization, but
it is applied online to the sentence cacheables attached to the retrieved chunk.
It does not reuse the offline artifact's `window_selected_indices` as the final
prompt regions.

The online region construction is:

1. choose each sentence as a center.
2. build a local region by adding neighboring retrieved sentences within the
   `COLBERT_SLIDING_WINDOW_TOKEN_BUDGET` or artifact window budget.
3. skip duplicate regions with the same selected sentence-index tuple.

Each region is represented as a synthetic `CacheableChunk`:

- id: `{retrievable_chunk_id}::sliding_region_{center_idx}`
- text: concatenated region text.
- `sentence_ids`: source sentence ids included in the region.
- `sentence_texts`: source sentence texts included in the region.

The region text is what can later enter the final prompt. The score, however,
is computed from the ColBERT vectors of the source sentences included in that
region. These source-sentence vectors are the offline materialized center
vectors; the online phase does not run ColBERT document encoding for every
region.

### 5. Query Encoding

The query is encoded online with the same ColBERT checkpoint used to build the
artifact.

The runtime checks that `COLBERT_MODEL_NAME` matches the artifact
`checkpoint_name`/`model_name`.

Only query encoding happens online. Document/window contextualization was
already done offline.

### 6. Region Scoring

The intended region score is equivalent to ColBERT MaxSim over all sentence
token vectors included in the region:

```text
score(region, query)
  = sum over query tokens q [
      max over document tokens d in region sim(q, d)
    ]
```

Naively, every region would concatenate its sentence token vectors and run
MaxSim independently. That repeats the same sentence-level similarity work many
times because neighboring sliding regions overlap heavily.

The current implementation uses memoization:

1. For each query and each candidate sentence, compute:

   ```text
   per_sentence_scores[s][q]
     = max over tokens d in sentence s sim(q, d)
   ```

2. Cache this vector by `(retrieved_chunk_index, cacheable_id)`.
3. For a region containing sentences `s1, s2, ...`, compute:

   ```text
   region_scores_per_query_token[q]
     = max(per_sentence_scores[s1][q],
           per_sentence_scores[s2][q],
           ...)
   ```

4. Sum over query tokens.

This is mathematically equivalent to MaxSim over the concatenated region token
vectors, because `max` over a union of sentence token sets is the same as
`max` over the per-sentence maxima.

The optimization reduces repeated computation for overlapping regions while
preserving the intended ColBERT score.

### 7. Ranking And Budgeted Selection

After scoring, regions are sorted by descending score.

The main grid uses retained-token-ratio budgets through `RETAIN_TOKEN_RATIO`.
When this is set, the final token budget is:

```text
ceil(RETAIN_TOKEN_RATIO * retrieved_context_token_count)
```

`retrieved_context_token_count` is computed over deduplicated retrieved
sentence cacheables. In the current implementation this budget accounting uses
`self.encoder.token_count(...)`, i.e. the ColBERT document tokenizer with its
document-token overhead. It is therefore a ColBERT-side token budget proxy, not
an exact LLM prompt-token budget.

If `RETAIN_TOKEN_RATIO` is not set, the method can use
`COLBERT_FINAL_TOKEN_BUDGET`. If neither is set, selection falls back to
`GLOBAL_TOP_R` over ranked regions.

The runtime priority of the budget controls is:

| Setting state | Final selection controller | Ignored for final selection |
| --- | --- | --- |
| `RETAIN_TOKEN_RATIO` is set | Per-query retained-ratio budget | `COLBERT_FINAL_TOKEN_BUDGET`, `GLOBAL_TOP_R` |
| `RETAIN_TOKEN_RATIO` unset, `COLBERT_FINAL_TOKEN_BUDGET` set | Fixed absolute token budget | `GLOBAL_TOP_R` |
| Neither `RETAIN_TOKEN_RATIO` nor `COLBERT_FINAL_TOKEN_BUDGET` set | Ratio keep count from `GLOBAL_TOP_R` | none |

`GLOBAL_TOP_R` is therefore not a token-budget controller when either
`RETAIN_TOKEN_RATIO` or `COLBERT_FINAL_TOKEN_BUDGET` is active. It only controls
the final number of selected ranked regions in the no-final-budget path.

`COLBERT_SLIDING_WINDOW_TOKEN_BUDGET` is a different control. It limits the size
of each candidate sliding region during online region construction; it does not
set the final selected prompt budget. If unset, the online region builder uses
the artifact `window_token_budget`, normally `180`.

### 8. Duplicate Sentence Removal Under Budget

Budgeted selection removes duplicate source sentences while walking ranked
regions.

The algorithm keeps:

- `selected_sentence_ids`
- `used_tokens`
- `selected_cacheables`

For each ranked region:

1. iterate through the region's source sentence indices.
2. skip a sentence if its id is already in `selected_sentence_ids`.
3. estimate its token length with the ColBERT tokenizer.
4. add it if it fits the remaining budget.
5. create a deduplicated region cacheable from the newly added sentences.

This is important because:

- boundary sentences may appear in multiple retrieved chunks.
- nearby sliding regions overlap heavily.
- without deduplication, the prompt could repeat the same evidence and waste
  token budget.

The final selected prompt units are region cacheables whose text is the
concatenation of newly selected, non-duplicate source sentences.

### 9. Prompt Assembly And Inference

After compression, the original retrieved chunk list is replaced with cloned
`RetrievableChunk` objects whose `cacheables` contain only the selected region
cacheables.

The downstream prompt builder then serializes those selected cacheables into the
LLM context. In the current main paper setting this is cache-off LLM inference:

- retrieval: Chroma default DB.
- compression: `colbert_sliding_region`.
- prompt: selected region text.
- generation: Llama-3.1-8B-Instruct fp16 by default.

The expected benefit is lower LLM input length while preserving evidence
coverage and reducing distractor exposure.

## Summary

The method works because the two phases share stable ids:

```text
Chroma retrieved row
  -> cacheables_json
  -> cacheable ids
  -> parent_doc_id
  -> parent document ColBERT .pt file
  -> cacheable_id -> center_token_vectors
  -> sliding-region score
  -> budgeted selected region prompt
```

The offline phase pays the document contextualization cost once. The online
phase keeps retrieval coarse, restricts ColBERT scoring to retrieved candidates,
memoizes overlapping region scores, removes duplicate sentences under budget,
and sends only selected evidence regions to the LLM.
