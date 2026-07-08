import time
import os
from itertools import islice
from tqdm import tqdm
from typing import List
import torch
from deepspeed.ops.op_builder import GDSBuilder, AsyncIOBuilder
from transformers import DynamicCache
from chunk import RetrievableChunk

from vectordb import VectorDB
from utils import parse_json_query, file_read, restore_tensor_shape
from model import LLMModel
from compressor.factory import compress_docs, initialize_compressor
import json
from cache_utils import (
    load_caches_from_rchunks,
    load_caches_from_doc_batch,
    concat_caches,
    pad_past_key_values,
    concat_caches_single,
)


class QueryProcessor:
    def __init__(
        self,
        model: LLMModel,
        vectordb: VectorDB,
        query_file: str,
        cache_dir: str,
        top_k: int = 4,
        use_past_cache: bool = True,
        output_file: str = "out.jsonl",
        compress_method=None,
        disable_rope: bool = False,
    ):
        self.query_file = query_file
        self.cache_dir = cache_dir
        self.top_k = top_k
        self.use_past_cache = use_past_cache
        self.vectordb = vectordb
        self.model = model
        self.output_file = output_file
        self.compress = compress_method
        self.disable_rope = disable_rope
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if self.top_k == 0 and self.use_past_cache:
            raise ValueError(
                "top_k=0 is a context-free cache-off baseline; set use_past_cache=False."
            )
        self.measure_prompt_stats = os.environ.get(
            "MEASURE_PROMPT_STATS", "True"
        ).strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }
        if self.compress is not None:
            print(f"using compress method: {self.compress}")
        if self.use_past_cache and self.compress in {"provence", "exit"}:
            raise ValueError(
                f"compress_method='{self.compress}' emits compressed text directly and is cache-off only; "
                "set use_past_cache=False."
            )

        # self.gds_handle = GDSBuilder().load().gds_handle() # TODO
        # self.aio_handle = AsyncIOBuilder().load().aio_handle() # TODO

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def _run_batch(self, batch_queries, batch_bsz: int, max_new_tokens: int):
        batch_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        batch_top_k_docs = self.vectordb.find_top_k_docs(
            top_k=self.top_k, queries=batch_queries
        )
        retrieval_time = time.perf_counter() - retrieval_start
        retrieval_timings = getattr(self.vectordb, "last_find_timings", {}) or {}

        compress_start = time.perf_counter()
        batch_top_k_docs = compress_docs(
            batch_queries, batch_top_k_docs, option=self.compress
        )
        compress_time = time.perf_counter() - compress_start

        cache_time = 0.0
        prompt_build_time = 0.0
        prompt_stats_time = 0.0
        generate_call_time = 0.0
        cache_lengths = []
        cache_prompt_lengths = []
        cached_rates = []
        batch_padded_cache_lengths = []
        batch_padded_prefill_total_lengths = []
        nocache_prompt_stats = None

        if self.use_past_cache:
            cache_load_start = time.perf_counter()
            initial_position_offset = 1 if self.model.use_front_bos_cache else 0
            if batch_bsz == 1:
                caches = []
                cached_lengths = []
                for docs in batch_top_k_docs:
                    request_caches, cached_length = load_caches_from_rchunks(
                        self.cache_dir,
                        docs,
                        apply_rotary_shift=not self.disable_rope,
                        initial_position_offset=initial_position_offset,
                    )
                    if self.model.use_front_bos_cache:
                        request_caches = [
                            self.model._get_front_bos_cache()
                        ] + request_caches
                        cached_length += 1
                    caches.append(request_caches)
                    cached_lengths.append(cached_length)
            else:
                caches, cached_lengths = load_caches_from_doc_batch(
                    self.cache_dir,
                    batch_top_k_docs,
                    apply_rotary_shift=not self.disable_rope,
                    initial_position_offset=initial_position_offset,
                )
                if self.model.use_front_bos_cache:
                    bos_cache = self.model._get_front_bos_cache()
                    caches = [[bos_cache] + request_caches for request_caches in caches]
                    cached_lengths = [
                        cached_length + 1 for cached_length in cached_lengths
                    ]

            if batch_bsz == 1:
                past_kv_caches = DynamicCache.from_legacy_cache(
                    concat_caches_single(caches[0])
                )
            else:
                past_kv_caches_no_pad = concat_caches(caches)
                past_kv_caches = pad_past_key_values(
                    past_kv_caches_no_pad[0],
                    past_kv_caches_no_pad[1],
                )
            cache_lengths = list(cached_lengths)
            if len(cached_lengths) > 0:
                batch_padded_cache_lengths.append(max(cached_lengths))
            cache_time = time.perf_counter() - cache_load_start

            prompt_build_start = time.perf_counter()
            batch_inputs = [
                self.seperate_query_and_doc(docs, q)
                for docs, q in zip(batch_top_k_docs, batch_queries)
            ]
            query_prompts = [query_prompt for _, query_prompt in batch_inputs]
            tokenizer_kwargs = dict(
                return_tensors="pt",
                padding=True,
                truncation=True,
                padding_side="right",
            )
            if self.model.use_front_bos_cache:
                tokenizer_kwargs["add_special_tokens"] = False
            tokenized_queries = self.model.tokenizer(
                query_prompts,
                **tokenizer_kwargs,
            )
            query_lengths = tokenized_queries["attention_mask"].sum(dim=1).tolist()
            padded_query_length = int(tokenized_queries["input_ids"].shape[1])
            cache_prompt_lengths = [
                cached_length + query_length
                for cached_length, query_length in zip(cached_lengths, query_lengths)
            ]
            if len(cached_lengths) > 0:
                batch_padded_prefill_total_lengths.append(
                    max(cached_lengths) + padded_query_length
                )
            cached_rates = [
                cached_length / (cached_length + query_length)
                for cached_length, query_length in zip(cached_lengths, query_lengths)
                if (cached_length + query_length) > 0
            ]
            prompt_build_time = time.perf_counter() - prompt_build_start
            generate_call_start = time.perf_counter()
            time_log, generated_texts = self.model.generate_response(
                batch_inputs,
                past_kv_caches=past_kv_caches,
                past_lengths=cached_lengths,
                max_new_tokens=max_new_tokens,
            )
            generate_call_time = time.perf_counter() - generate_call_start
        else:
            prompt_build_start = time.perf_counter()
            batch_inputs = [
                self.concatenate_query_and_doc(
                    batch_top_k_docs[idx], batch_queries[idx]
                )
                for idx in range(len(batch_queries))
            ]
            prompt_build_time = time.perf_counter() - prompt_build_start
            if self.measure_prompt_stats:
                prompt_stats_start = time.perf_counter()
                nocache_prompt_stats = self._measure_nocache_prompt_stats(
                    batch_top_k_docs,
                    batch_queries,
                    batch_inputs,
                )
                prompt_stats_time = time.perf_counter() - prompt_stats_start
            generate_call_start = time.perf_counter()
            time_log, generated_texts = self.model.generate_response(
                batch_inputs,
                max_new_tokens=max_new_tokens,
            )
            generate_call_time = time.perf_counter() - generate_call_start

        batch_latency = time.perf_counter() - batch_start
        generate_extra_time = max(
            generate_call_time - time_log.prefill - time_log.decode, 0.0
        )
        return {
            "batch_top_k_docs": batch_top_k_docs,
            "generated_texts": generated_texts,
            "time_log": time_log,
            "batch_latency": batch_latency,
            "retrieval_time": retrieval_time,
            "retrieval_query_time": float(retrieval_timings.get("query_time", 0.0)),
            "retrieval_postprocess_time": float(
                retrieval_timings.get("postprocess_time", 0.0)
            ),
            "retrieval_cacheable_deserialize_time": float(
                retrieval_timings.get("cacheable_deserialize_time", 0.0)
            ),
            "compress_time": compress_time,
            "cache_time": cache_time,
            "prompt_build_time": prompt_build_time,
            "prompt_stats_time": prompt_stats_time,
            "generate_extra_time": generate_extra_time,
            "cache_lengths": cache_lengths,
            "cache_prompt_lengths": cache_prompt_lengths,
            "cached_rates": cached_rates,
            "batch_padded_cache_lengths": batch_padded_cache_lengths,
            "batch_padded_prefill_total_lengths": batch_padded_prefill_total_lengths,
            "model_input_lengths": list(time_log.model_input_lengths or []),
            "nocache_prompt_stats": nocache_prompt_stats,
        }

    def process_query(
        self, bsz: int = 1, max_new_tokens: int = 100, total_num: int = 100
    ):
        process_start = time.perf_counter()
        elapsed = 0.0
        retrieval_elapsed = 0.0
        retrieval_query_elapsed = 0.0
        retrieval_postprocess_elapsed = 0.0
        retrieval_cacheable_deserialize_elapsed = 0.0
        cache_elapsed = 0.0
        compress_elapsed = 0.0
        prompt_build_elapsed = 0.0
        prompt_stats_elapsed = 0.0
        generate_extra_elapsed = 0.0
        batch_count = 0.0
        processed_queries = 0
        prefill_time = 0.0
        decode_time = 0.0
        cache_lens = []
        cache_prompt_lens = []
        cached_rates = []
        batch_padded_cache_lens = []
        batch_padded_prefill_total_lens = []
        output_lens = []
        nocache_runtime_prompt_lens = []
        nocache_query_lens = []
        nocache_retained_chunk_counts = []
        batch_latencies = []
        warmup_time = 0.0
        warmup_queries = 0

        setup_start = time.perf_counter()
        if self.compress is not None:
            initialize_compressor(option=self.compress)
        setup_time = time.perf_counter() - setup_start

        with open(self.query_file, encoding="utf-8") as f:
            warmup_line = next(f, None)
            if warmup_line is not None:
                warmup_query = parse_json_query(warmup_line)
                warmup_result = self._run_batch(
                    batch_queries=[warmup_query],
                    batch_bsz=1,
                    max_new_tokens=max_new_tokens,
                )
                warmup_time = warmup_result["batch_latency"]
                warmup_queries = 1

        with open(self.query_file, encoding="utf-8") as f:
            batch_queries = []
            for line in tqdm(islice(f, total_num), total=total_num):
                parsed_query = parse_json_query(line)
                batch_queries.append(parsed_query)

                if len(batch_queries) == bsz:
                    batch_count += 1
                    current_batch_size = len(batch_queries)
                    processed_queries += current_batch_size
                    batch_result = self._run_batch(
                        batch_queries=batch_queries,
                        batch_bsz=bsz,
                        max_new_tokens=max_new_tokens,
                    )
                    retrieval_elapsed += batch_result["retrieval_time"]
                    retrieval_query_elapsed += batch_result["retrieval_query_time"]
                    retrieval_postprocess_elapsed += batch_result[
                        "retrieval_postprocess_time"
                    ]
                    retrieval_cacheable_deserialize_elapsed += batch_result[
                        "retrieval_cacheable_deserialize_time"
                    ]
                    compress_elapsed += batch_result["compress_time"]
                    cache_elapsed += batch_result["cache_time"]
                    prompt_build_elapsed += batch_result["prompt_build_time"]
                    prompt_stats_elapsed += batch_result["prompt_stats_time"]
                    generate_extra_elapsed += batch_result["generate_extra_time"]
                    cache_lens.extend(batch_result["cache_lengths"])
                    cache_prompt_lens.extend(batch_result["cache_prompt_lengths"])
                    cached_rates.extend(batch_result["cached_rates"])
                    batch_padded_cache_lens.extend(
                        batch_result["batch_padded_cache_lengths"]
                    )
                    batch_padded_prefill_total_lens.extend(
                        batch_result["batch_padded_prefill_total_lengths"]
                    )
                    if not self.use_past_cache:
                        nocache_runtime_prompt_lens.extend(
                            batch_result["model_input_lengths"]
                        )
                    prompt_stats = batch_result["nocache_prompt_stats"]
                    if prompt_stats is not None:
                        nocache_query_lens.extend(prompt_stats["query_prompt_lens"])
                        nocache_retained_chunk_counts.extend(
                            prompt_stats["retained_chunk_counts"]
                        )

                    batch_latency = batch_result["batch_latency"]
                    time_log = batch_result["time_log"]
                    generated_texts = batch_result["generated_texts"]
                    batch_top_k_docs = batch_result["batch_top_k_docs"]
                    batch_latencies.append(batch_latency)
                    elapsed += batch_latency
                    output_lens.extend(len(g) for g in generated_texts)

                    with open(self.output_file, "a", encoding="utf-8") as lf:
                        for query, docs, text in zip(
                            batch_queries, batch_top_k_docs, generated_texts
                        ):
                            log_entry = {
                                "question": query,
                                "ctxs": [
                                    {"title": "", "text": doc.text} for doc in docs
                                ],
                                "prediction": text,
                            }
                            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                    batch_queries = []
                    prefill_time += time_log.prefill
                    decode_time += time_log.decode

            if batch_queries:
                batch_count += 1
                current_batch_size = len(batch_queries)
                processed_queries += current_batch_size
                batch_result = self._run_batch(
                    batch_queries=batch_queries,
                    batch_bsz=current_batch_size,
                    max_new_tokens=max_new_tokens,
                )
                retrieval_elapsed += batch_result["retrieval_time"]
                retrieval_query_elapsed += batch_result["retrieval_query_time"]
                retrieval_postprocess_elapsed += batch_result[
                    "retrieval_postprocess_time"
                ]
                retrieval_cacheable_deserialize_elapsed += batch_result[
                    "retrieval_cacheable_deserialize_time"
                ]
                compress_elapsed += batch_result["compress_time"]
                cache_elapsed += batch_result["cache_time"]
                prompt_build_elapsed += batch_result["prompt_build_time"]
                prompt_stats_elapsed += batch_result["prompt_stats_time"]
                generate_extra_elapsed += batch_result["generate_extra_time"]
                cache_lens.extend(batch_result["cache_lengths"])
                cache_prompt_lens.extend(batch_result["cache_prompt_lengths"])
                cached_rates.extend(batch_result["cached_rates"])
                batch_padded_cache_lens.extend(
                    batch_result["batch_padded_cache_lengths"]
                )
                batch_padded_prefill_total_lens.extend(
                    batch_result["batch_padded_prefill_total_lengths"]
                )
                if not self.use_past_cache:
                    nocache_runtime_prompt_lens.extend(
                        batch_result["model_input_lengths"]
                    )
                prompt_stats = batch_result["nocache_prompt_stats"]
                if prompt_stats is not None:
                    nocache_query_lens.extend(prompt_stats["query_prompt_lens"])
                    nocache_retained_chunk_counts.extend(
                        prompt_stats["retained_chunk_counts"]
                    )

                batch_latency = batch_result["batch_latency"]
                time_log = batch_result["time_log"]
                generated_texts = batch_result["generated_texts"]
                batch_top_k_docs = batch_result["batch_top_k_docs"]
                batch_latencies.append(batch_latency)
                elapsed += batch_latency
                output_lens.extend(len(g) for g in generated_texts)

                with open(self.output_file, "a", encoding="utf-8") as lf:
                    for query, docs, text in zip(
                        batch_queries, batch_top_k_docs, generated_texts
                    ):
                        log_entry = {
                            "question": query,
                            "ctxs": [{"title": "", "text": doc.text} for doc in docs],
                            "prediction": text,
                        }
                        lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                prefill_time += time_log.prefill
                decode_time += time_log.decode

        avg_cache_time_per_query = (
            cache_elapsed / batch_count if cache_elapsed > 0 else 0
        )
        avg_retrieval_time_per_batch = (
            retrieval_elapsed / batch_count if batch_count > 0 else 0
        )
        avg_retrieval_query_time_per_batch = (
            retrieval_query_elapsed / batch_count if batch_count > 0 else 0
        )
        avg_retrieval_postprocess_time_per_batch = (
            retrieval_postprocess_elapsed / batch_count if batch_count > 0 else 0
        )
        avg_retrieval_cacheable_deserialize_time_per_batch = (
            retrieval_cacheable_deserialize_elapsed / batch_count
            if batch_count > 0
            else 0
        )
        avg_compress_time_per_batch = (
            compress_elapsed / batch_count if batch_count > 0 else 0
        )
        avg_prompt_build_time_per_batch = (
            prompt_build_elapsed / batch_count if batch_count > 0 else 0
        )
        avg_prompt_stats_time_per_batch = (
            prompt_stats_elapsed / batch_count if batch_count > 0 else 0
        )
        avg_generate_extra_time_per_batch = (
            generate_extra_elapsed / batch_count if batch_count > 0 else 0
        )
        print(
            f"retrieval time per batch | total: {retrieval_elapsed:.4f}, avg: {avg_retrieval_time_per_batch:.4f}",
            flush=True,
        )
        print(
            f"retrieval query time per batch | total: {retrieval_query_elapsed:.4f}, "
            f"avg: {avg_retrieval_query_time_per_batch:.4f}",
            flush=True,
        )
        print(
            f"retrieval postprocess time per batch | total: {retrieval_postprocess_elapsed:.4f}, "
            f"avg: {avg_retrieval_postprocess_time_per_batch:.4f}",
            flush=True,
        )
        print(
            f"retrieval cacheable deserialize time per batch | total: {retrieval_cacheable_deserialize_elapsed:.4f}, "
            f"avg: {avg_retrieval_cacheable_deserialize_time_per_batch:.4f}",
            flush=True,
        )
        print(
            f"compress time per batch | total: {compress_elapsed:.4f}, avg: {avg_compress_time_per_batch:.4f}",
            flush=True,
        )
        print(
            f"prompt build time per batch | total: {prompt_build_elapsed:.4f}, avg: {avg_prompt_build_time_per_batch:.4f}",
            flush=True,
        )
        print(
            f"prompt stats time per batch | total: {prompt_stats_elapsed:.4f}, avg: {avg_prompt_stats_time_per_batch:.4f}",
            flush=True,
        )
        print(
            f"generate extra time per batch | total: {generate_extra_elapsed:.4f}, avg: {avg_generate_extra_time_per_batch:.4f}",
            flush=True,
        )
        print(
            f"cache load time per batch | total: {cache_elapsed:.4f}, avg: {avg_cache_time_per_query:.4f}",
            flush=True,
        )
        if len(cache_lens) > 0:
            print(f"avg valid cache len: {sum(cache_lens)/len(cache_lens):.4f}")
        if len(batch_padded_cache_lens) > 0:
            print(
                f"avg padded prefill cache len: "
                f"{sum(batch_padded_cache_lens) / len(batch_padded_cache_lens):.4f}"
            )
        if len(batch_padded_prefill_total_lens) > 0:
            print(
                f"avg padded prefill input len: "
                f"{sum(batch_padded_prefill_total_lens) / len(batch_padded_prefill_total_lens):.4f}"
            )
        if len(cache_prompt_lens) > 0:
            print(
                f"avg valid prefill input len: {sum(cache_prompt_lens) / len(cache_prompt_lens):.4f}"
            )
        if len(cached_rates) > 0:
            print(f"avg cached rate: {sum(cached_rates) / len(cached_rates):.4f}")
        if len(nocache_runtime_prompt_lens) > 0:
            print(
                f"avg no-cache model input len: "
                f"{sum(nocache_runtime_prompt_lens) / len(nocache_runtime_prompt_lens):.4f}"
            )
        if len(nocache_query_lens) > 0:
            print(
                f"avg no-cache query-only len: {sum(nocache_query_lens) / len(nocache_query_lens):.4f}"
            )
        if len(nocache_retained_chunk_counts) > 0:
            print(
                f"avg no-cache retained chunks: "
                f"{sum(nocache_retained_chunk_counts) / len(nocache_retained_chunk_counts):.4f}"
            )
        elif not self.use_past_cache and not self.measure_prompt_stats:
            print("no-cache prompt stats: disabled")
        print(
            f"prefill per batch | total: {prefill_time:.4f}, avg: {prefill_time / batch_count:.4f}",
            flush=True,
        )
        print(
            f"decode per batch | total: {decode_time:.4f}, avg: {decode_time / batch_count:.4f}",
            flush=True,
        )
        other_time = elapsed - prefill_time - decode_time
        print(
            f"other time per batch | total: {other_time:.4f}, avg: {other_time / batch_count:.4f}",
            flush=True,
        )
        print(f"avg output lens: {sum(output_lens)/len(output_lens)}")
        print(
            f"time per batch| total: {elapsed:.4f}, avg: {elapsed / batch_count:.4f}",
            flush=True,
        )
        if warmup_queries > 0:
            print(
                f"warmup | queries: {warmup_queries}, batch_size: 1, time: {warmup_time:.4f}",
                flush=True,
            )
        process_total = time.perf_counter() - process_start
        measured_run_time = elapsed
        return {
            "processed_queries": processed_queries,
            "processed_batches": int(batch_count),
            "setup_time": setup_time,
            "run_time": measured_run_time,
            "process_total_time": process_total,
            "warmup_time": warmup_time,
            "warmup_queries": warmup_queries,
        }

    def _extract_passages(self, docs: List[RetrievableChunk]):
        passages = []
        for doc in docs:
            passages.extend(
                cacheable.text for cacheable in getattr(doc, "cacheables", [])
            )
        return passages

    def concatenate_query_and_doc(self, docs: List[RetrievableChunk], query: str):
        passages = self._extract_passages(docs)
        return self.model.prompt_processor.build_cache_aligned_qa_prompt(
            query=query, passages=passages
        )

    def seperate_query_and_doc(self, docs: List[RetrievableChunk], query: str):
        passages = self._extract_passages(docs)
        return passages, self.build_query_prompt(query)

    def build_query_prompt(self, query: str):
        return self.model.prompt_processor.build_query_prompt(query)

    def _measure_nocache_prompt_stats(
        self,
        batch_docs: List[List[RetrievableChunk]],
        batch_queries: List[str],
        prompts: List[str],
    ):
        tokenizer = self.model.tokenizer
        original_truncation_side = tokenizer.truncation_side
        try:
            tokenizer.truncation_side = "left"
            runtime_prompt_lens = [
                len(
                    tokenizer(prompt, add_special_tokens=True, truncation=True)[
                        "input_ids"
                    ]
                )
                for prompt in prompts
            ]
        finally:
            tokenizer.truncation_side = original_truncation_side

        query_prompt_lens = [
            len(
                tokenizer(
                    self.build_query_prompt(query),
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
            for query in batch_queries
        ]

        retained_chunk_counts = []
        for docs, query_prompt_len, runtime_prompt_len in zip(
            batch_docs, query_prompt_lens, runtime_prompt_lens
        ):
            formatted_passages = []
            for doc in docs:
                for cacheable in getattr(doc, "cacheables", []):
                    chunk_text = cacheable.text
                    if chunk_text and chunk_text.strip():
                        formatted_passages.append(
                            self.model.prompt_processor.format_passage_chunk(chunk_text)
                        )

            available_passage_tokens = max(runtime_prompt_len - query_prompt_len, 0)
            chunk_token_lens = [
                len(
                    tokenizer(chunk_text, add_special_tokens=False, truncation=False)[
                        "input_ids"
                    ]
                )
                for chunk_text in formatted_passages
            ]
            retained_count = 0
            retained_tokens = 0
            for chunk_len in reversed(chunk_token_lens):
                if retained_tokens + chunk_len > available_passage_tokens:
                    break
                retained_tokens += chunk_len
                retained_count += 1
            retained_chunk_counts.append(retained_count)

        return {
            "runtime_prompt_lens": runtime_prompt_lens,
            "query_prompt_lens": query_prompt_lens,
            "retained_chunk_counts": retained_chunk_counts,
        }
