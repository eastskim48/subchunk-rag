"""End-to-end retrieval, compression, and full-prompt inference engine."""

import time
import os
from itertools import islice
from tqdm import tqdm
from typing import List, Optional
from chunk import RetrievableChunk

from vectordb import VectorDB
from utils import parse_json_query_record
from model import LLMModel
from compressor.factory import compress_docs, initialize_compressor
import json


class QueryProcessor:
    """Execute batched evaluation queries and record timing/output metadata."""

    def __init__(
        self,
        model: LLMModel,
        vectordb: VectorDB,
        query_file: str,
        top_k: int = 4,
        output_file: str = "out.jsonl",
        compress_method=None,
    ):
        self.query_file = query_file
        self.top_k = top_k
        self.vectordb = vectordb
        self.model = model
        self.output_file = output_file
        self.compress = compress_method
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
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
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def _run_batch(self, batch_queries, max_new_tokens: int):
        """Run retrieval through generation for one evaluation batch."""

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

        prompt_build_start = time.perf_counter()
        batch_inputs = [
            self.concatenate_query_and_doc(batch_top_k_docs[idx], batch_queries[idx])
            for idx in range(len(batch_queries))
        ]
        prompt_build_time = time.perf_counter() - prompt_build_start
        prompt_stats_time = 0.0
        nocache_prompt_stats = None
        if self.measure_prompt_stats:
            prompt_stats_start = time.perf_counter()
            nocache_prompt_stats = self._measure_nocache_prompt_stats(
                batch_top_k_docs, batch_queries, batch_inputs
            )
            prompt_stats_time = time.perf_counter() - prompt_stats_start
        generate_call_start = time.perf_counter()
        time_log, generated_texts = self.model.generate_response(
            batch_inputs, max_new_tokens=max_new_tokens
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
            "prompt_build_time": prompt_build_time,
            "prompt_stats_time": prompt_stats_time,
            "generate_extra_time": generate_extra_time,
            "model_input_lengths": list(time_log.model_input_lengths or []),
            "generated_token_counts": list(time_log.generated_token_counts or []),
            "decode_steps": int(time_log.decode_steps),
            "nocache_prompt_stats": nocache_prompt_stats,
        }

    def process_query(
        self,
        bsz: int = 1,
        max_new_tokens: int = 100,
        total_num: Optional[int] = None,
    ):
        """Evaluate input queries in batches and write reproducible records."""

        process_start = time.perf_counter()
        elapsed = 0.0
        retrieval_elapsed = 0.0
        retrieval_query_elapsed = 0.0
        retrieval_postprocess_elapsed = 0.0
        retrieval_cacheable_deserialize_elapsed = 0.0
        compress_elapsed = 0.0
        prompt_build_elapsed = 0.0
        prompt_stats_elapsed = 0.0
        generate_extra_elapsed = 0.0
        batch_count = 0.0
        processed_queries = 0
        prefill_time = 0.0
        decode_time = 0.0
        output_lens = []
        output_token_lens_approx = []
        generated_token_counts = []
        batch_decode_steps = []
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
                warmup_query = parse_json_query_record(warmup_line)["query"]
                warmup_result = self._run_batch(
                    batch_queries=[warmup_query],
                    max_new_tokens=max_new_tokens,
                )
                warmup_time = warmup_result["batch_latency"]
                warmup_queries = 1

        with open(self.query_file, encoding="utf-8") as f:
            batch_queries = []
            batch_query_ids = []
            for line in tqdm(islice(f, total_num), total=total_num):
                query_record = parse_json_query_record(line)
                batch_query_ids.append(query_record["id"])
                batch_queries.append(query_record["query"])

                if len(batch_queries) == bsz:
                    batch_count += 1
                    current_batch_size = len(batch_queries)
                    processed_queries += current_batch_size
                    batch_result = self._run_batch(
                        batch_queries=batch_queries,
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
                    prompt_build_elapsed += batch_result["prompt_build_time"]
                    prompt_stats_elapsed += batch_result["prompt_stats_time"]
                    generate_extra_elapsed += batch_result["generate_extra_time"]
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
                    current_generated_token_counts = batch_result[
                        "generated_token_counts"
                    ]
                    batch_top_k_docs = batch_result["batch_top_k_docs"]
                    batch_latencies.append(batch_latency)
                    batch_decode_steps.append(batch_result["decode_steps"])
                    elapsed += batch_latency
                    generated_token_counts.extend(current_generated_token_counts)
                    output_lens.extend(len(g) for g in generated_texts)
                    output_token_lens_approx.extend(
                        len(
                            self.model.tokenizer.encode(
                                g,
                                add_special_tokens=False,
                            )
                        )
                        for g in generated_texts
                    )

                    with open(self.output_file, "a", encoding="utf-8") as lf:
                        for query_id, query, docs, text, generated_token_count in zip(
                            batch_query_ids,
                            batch_queries,
                            batch_top_k_docs,
                            generated_texts,
                            current_generated_token_counts,
                        ):
                            log_entry = {
                                "id": query_id,
                                "question": query,
                                "ctxs": self._ctxs_for_output(docs),
                                "prediction": text,
                                "generated_token_count": generated_token_count,
                            }
                            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                    batch_queries = []
                    batch_query_ids = []
                    prefill_time += time_log.prefill
                    decode_time += time_log.decode

            if batch_queries:
                batch_count += 1
                current_batch_size = len(batch_queries)
                processed_queries += current_batch_size
                batch_result = self._run_batch(
                    batch_queries=batch_queries,
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
                prompt_build_elapsed += batch_result["prompt_build_time"]
                prompt_stats_elapsed += batch_result["prompt_stats_time"]
                generate_extra_elapsed += batch_result["generate_extra_time"]
                nocache_runtime_prompt_lens.extend(batch_result["model_input_lengths"])
                prompt_stats = batch_result["nocache_prompt_stats"]
                if prompt_stats is not None:
                    nocache_query_lens.extend(prompt_stats["query_prompt_lens"])
                    nocache_retained_chunk_counts.extend(
                        prompt_stats["retained_chunk_counts"]
                    )

                batch_latency = batch_result["batch_latency"]
                time_log = batch_result["time_log"]
                generated_texts = batch_result["generated_texts"]
                current_generated_token_counts = batch_result["generated_token_counts"]
                batch_top_k_docs = batch_result["batch_top_k_docs"]
                batch_latencies.append(batch_latency)
                batch_decode_steps.append(batch_result["decode_steps"])
                elapsed += batch_latency
                generated_token_counts.extend(current_generated_token_counts)
                output_lens.extend(len(g) for g in generated_texts)
                output_token_lens_approx.extend(
                    len(
                        self.model.tokenizer.encode(
                            g,
                            add_special_tokens=False,
                        )
                    )
                    for g in generated_texts
                )

                with open(self.output_file, "a", encoding="utf-8") as lf:
                    for query_id, query, docs, text, generated_token_count in zip(
                        batch_query_ids,
                        batch_queries,
                        batch_top_k_docs,
                        generated_texts,
                        current_generated_token_counts,
                    ):
                        log_entry = {
                            "id": query_id,
                            "question": query,
                            "ctxs": self._ctxs_for_output(docs),
                            "prediction": text,
                            "generated_token_count": generated_token_count,
                        }
                        lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                prefill_time += time_log.prefill
                decode_time += time_log.decode

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
        elif not self.measure_prompt_stats:
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
            "avg output token lens (approx): "
            f"{sum(output_token_lens_approx) / len(output_token_lens_approx)}"
        )
        print(
            "avg generated token count: "
            f"{sum(generated_token_counts) / len(generated_token_counts)}"
        )
        total_decode_steps = sum(batch_decode_steps)
        avg_decode_step_sec = (
            decode_time / total_decode_steps if total_decode_steps > 0 else 0.0
        )
        print(f"avg decode step sec: {avg_decode_step_sec}")
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

    def _ctxs_for_output(self, docs: List[RetrievableChunk]):
        return [
            {"title": "", "text": passage} for passage in self._extract_passages(docs)
        ]

    def concatenate_query_and_doc(self, docs: List[RetrievableChunk], query: str):
        passages = self._extract_passages(docs)
        return self.model.prompt_processor.build_qa_prompt(
            query=query, passages=passages
        )

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
