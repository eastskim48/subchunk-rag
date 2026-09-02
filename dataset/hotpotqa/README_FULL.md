# HotpotQA complete-Wikipedia corpus

`dataset/get_hotpot_full.sh` constructs a separate **custom** HotpotQA
distractor-development retrieval dataset whose shared corpus contains every
page in the official processed October 1, 2017 Wikipedia archive.

Run:

```bash
./dataset/get_hotpot_full.sh
```

Defaults:

- output: `/mnt/nvme1/datasets/hotpotqa-full`
- shared official downloads: `/mnt/nvme1/datasets/hotpotqa-sources`
- expected Wikipedia pages: 5,486,212
- HotpotQA queries: the 7,405 official distractor-development queries in their
  original order

Override the paths with `DATASET_PREFIX`, `HOTPOT_FULL_OUTPUT_DIR`, or
`HOTPOTQA_SOURCE_DIR`. Additional arguments such as `--scan-workers 16` are
forwarded to `hotpotqa/prepare_full.py`.

The corpus generator writes one `documents/doc_<page_id>.txt` file for every
Wikipedia page because the current DB preprocessor consumes a flat document
directory. It records completion per inner Wikipedia shard under
`dataset_info/full_wikipedia_shards/`. If construction is interrupted, rerun
the same command: the persistent `.hotpotqa-full.building` directory is checked
and completed shards are skipped.

This script constructs CPU-side dataset inputs only. It does not build a vector
DB or ColBERT artifact. Those remain separate GPU stages. Materializing and
indexing 5,486,212 individual document files is substantially larger than the
66,705-document `get_hotpotqa.sh` corpus and requires correspondingly more
storage, inodes, preprocessing time, and index capacity.

The query, answer, and labeled-evidence construction rules are unchanged from
the 66,705-document custom corpus. Only the shared retrieval corpus changes.
Results from the two corpora must therefore retain distinct run names and must
not be presented as the same retrieval condition.
