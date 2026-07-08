import json
import os
import torch
from tqdm import tqdm
from typing import List
import time
from deepspeed.ops.op_builder import AsyncIOBuilder
from deepspeed.ops.op_builder import GDSBuilder

from vectordb import VectorDB
from model import LLMModel
from chunk import Chunk, CacheableChunk, RetrievableChunk
from materialize.splitter import (
    FixedSizeSplitter,
    PNMappedSentenceWiseSplitter,
    SentenceWiseSplitter,
    ResolvedSentenceWiseSplitter,
    SemanticSplitter,
    build_merger,
)
from materialize.subchunk_embeds import CompareEmbeddingWriter
from embedding_utils import BGE_M3_MODEL

# from utils import file_write


class DocumentPreprocessor:
    def __init__(
        self,
        vectordb: VectorDB,
        model: LLMModel,
        docs_dir: str,
        cache_dir: str,
        cacheable_chunk_size: int | None = 1024,
        retrievable_chunk_size: int | None = None,
        batch_size: int = 1,
        dummy_bos_count: int = 0,
        splitter: str = "fixed_size",
        merger: str | None = None,
        materialize_cache: bool = True,
        materialize_db: bool = True,
        materialize_compare_embeds: bool = True,
        compare_embed_dir: str | None = None,
        compare_embed_model: str = BGE_M3_MODEL,
        compare_embed_overwrite: bool = False,
        sentence_cache_token_format: str = "legacy",
        resume_from_cache: bool = False,
        materialize_doc_ids_file: str | None = None,
        sentence_resolver: str = "openai",
        openai_model: str = "gpt-4o-mini",
        fastcoref_model_name: str = "biu-nlp/f-coref",
        pn_mapping_dir: str | None = None,
    ):
        self.docs_dir = docs_dir
        self.cache_dir = cache_dir
        self.cacheable_chunk_size = cacheable_chunk_size
        self.retrievable_chunk_size = retrievable_chunk_size
        self.batch_size = batch_size
        self.dummy_bos_count = dummy_bos_count
        self.splitter_name = splitter
        self.merger_name = merger
        self.materialize_cache = materialize_cache
        self.materialize_db = materialize_db
        self.materialize_compare_embeds = materialize_compare_embeds
        self.compare_embed_dir = compare_embed_dir
        self.compare_embed_model = compare_embed_model
        self.compare_embed_overwrite = compare_embed_overwrite
        self.sentence_cache_token_format = sentence_cache_token_format
        self.resume_from_cache = resume_from_cache
        self.materialize_doc_ids_file = materialize_doc_ids_file
        self.sentence_resolver = sentence_resolver
        self.openai_model = openai_model
        self.fastcoref_model_name = fastcoref_model_name
        self.pn_mapping_dir = pn_mapping_dir
        if self.materialize_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        if self.materialize_compare_embeds and self.compare_embed_dir:
            os.makedirs(self.compare_embed_dir, exist_ok=True)
        self.vectordb = vectordb
        self.model = model
        self.total_doc_tokens = 0
        self.processed_doc_count = 0
        self.docs_over_chunk_size = 0
        self.total_chunk_count = 0
        self.max_chunk_tokens = 0
        self.skipped_existing_chunk_count = 0
        self.rebuilt_invalid_chunk_count = 0
        self.materialize_doc_ids = None
        if self.materialize_doc_ids_file:
            with open(self.materialize_doc_ids_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            doc_ids = payload.get("doc_ids")
            if not isinstance(doc_ids, list):
                raise ValueError(
                    "materialize_doc_ids_file must contain a JSON object with a 'doc_ids' list"
                )
            self.materialize_doc_ids = set(doc_ids)
        if self.dummy_bos_count > 0 and self.model.tokenizer.bos_token_id is None:
            raise ValueError(
                "dummy_bos_count requires tokenizer.bos_token_id to be set"
            )
        if self.splitter_name not in {
            "fixed_size",
            "fixed_subchunk",
            "sentence",
            "resolved_sentence",
            "pn_sentence",
            "semantic",
        }:
            raise ValueError(f"unsupported splitter: {self.splitter_name}")
        if self.merger_name is not None and self.splitter_name != "semantic":
            raise ValueError("merger can only be set when splitter=semantic")
        if self.splitter_name == "semantic" and self.merger_name is None:
            raise ValueError("splitter=semantic requires merger to be set")
        if self.sentence_cache_token_format not in {
            "legacy",
            "space_prefix_newline_suffix",
            "merged_space_prefix_newline_suffix",
            "merged_space_prefix_space_newline_suffix",
        }:
            raise ValueError(
                f"unsupported sentence_cache_token_format: {self.sentence_cache_token_format}"
            )
        self.visible_prefix_token_ids = self.model.tokenizer.encode(
            self.model.PASSAGE_PREFIX,
            add_special_tokens=False,
        )
        self.visible_suffix_token_ids = self.model.tokenizer.encode(
            "\n\n",
            add_special_tokens=False,
        )
        self.space_token_ids = self.model.tokenizer.encode(
            " ", add_special_tokens=False
        )
        if len(self.space_token_ids) != 1:
            raise ValueError(
                f"expected a single-token space separator, got ids={self.space_token_ids}"
            )
        if len(self.visible_suffix_token_ids) != 1:
            raise ValueError(
                f"expected a single-token newline suffix, got ids={self.visible_suffix_token_ids}"
            )
        self.visible_token_overhead = len(self.visible_prefix_token_ids) + len(
            self.visible_suffix_token_ids
        )
        if self.cacheable_chunk_size is None and self.retrievable_chunk_size is None:
            if self.splitter_name not in {
                "sentence",
                "resolved_sentence",
                "pn_sentence",
                "semantic",
            }:
                raise NotImplementedError(
                    "splitter='fixed_size' requires cacheable_chunk_size and retrievable_chunk_size to both be set to the same integer"
                )
            self.content_chunk_size = None
        elif (
            self.cacheable_chunk_size is None
            and self.retrievable_chunk_size is not None
        ):
            if self.splitter_name not in {
                "sentence",
                "resolved_sentence",
                "pn_sentence",
                "semantic",
            }:
                raise NotImplementedError(
                    "cacheable_chunk_size=None with retrievable_chunk_size set is only supported for sentence/resolved_sentence/semantic splitters"
                )
            self.content_chunk_size = None
        elif (
            self.cacheable_chunk_size is not None
            and self.retrievable_chunk_size is not None
        ):
            if (
                self.splitter_name in {"fixed_size", "fixed_subchunk"}
                and self.cacheable_chunk_size <= self.visible_token_overhead
            ):
                raise ValueError(
                    f"cacheable_chunk_size ({self.cacheable_chunk_size}) must be larger than visible token overhead "
                    f"({self.visible_token_overhead})"
                )
            self.content_chunk_size = (
                self.cacheable_chunk_size - self.visible_token_overhead
                if self.splitter_name in {"fixed_size", "fixed_subchunk"}
                else None
            )
        else:
            raise NotImplementedError(
                "mixed None/non-None retrievable_chunk_size and cacheable_chunk_size combinations are not supported for this splitter"
            )

        if self.splitter_name == "sentence":
            self.splitter = SentenceWiseSplitter(
                docs_dir=self.docs_dir,
                model=self.model,
                cacheable_chunk_size=self.cacheable_chunk_size,
                retrievable_chunk_size=self.retrievable_chunk_size,
                content_chunk_size=self.content_chunk_size,
            )
        elif self.splitter_name == "resolved_sentence":
            self.splitter = ResolvedSentenceWiseSplitter(
                docs_dir=self.docs_dir,
                model=self.model,
                cacheable_chunk_size=self.cacheable_chunk_size,
                retrievable_chunk_size=self.retrievable_chunk_size,
                content_chunk_size=self.content_chunk_size,
                sentence_resolver=self.sentence_resolver,
                openai_model=self.openai_model,
                fastcoref_model_name=self.fastcoref_model_name,
            )
        elif self.splitter_name == "pn_sentence":
            if not self.pn_mapping_dir:
                raise ValueError(
                    "splitter=pn_sentence requires pn_mapping_dir to be set"
                )
            self.splitter = PNMappedSentenceWiseSplitter(
                docs_dir=self.docs_dir,
                model=self.model,
                cacheable_chunk_size=self.cacheable_chunk_size,
                retrievable_chunk_size=self.retrievable_chunk_size,
                content_chunk_size=self.content_chunk_size,
                pn_mapping_dir=self.pn_mapping_dir,
            )
        elif self.splitter_name == "semantic":
            self.splitter = SemanticSplitter(
                docs_dir=self.docs_dir,
                model=self.model,
                cacheable_chunk_size=self.cacheable_chunk_size,
                retrievable_chunk_size=self.retrievable_chunk_size,
                content_chunk_size=self.content_chunk_size,
                merger=build_merger(self.merger_name, tokenizer=self.model.tokenizer),
            )
        else:
            self.splitter = FixedSizeSplitter(
                docs_dir=self.docs_dir,
                model=self.model,
                cacheable_chunk_size=self.cacheable_chunk_size,
                retrievable_chunk_size=self.retrievable_chunk_size,
                content_chunk_size=self.content_chunk_size,
            )
        self.compare_embed_writer = None
        if self.materialize_compare_embeds:
            if not self.compare_embed_dir:
                raise ValueError(
                    "compare_embed_dir must be set when materialize_compare_embeds=True"
                )
            self.compare_embed_writer = CompareEmbeddingWriter(
                output_dir=self.compare_embed_dir,
                embedding_model=self.compare_embed_model,
                embedding_batch_size=self.batch_size,
                cache_unit=(
                    "sentence"
                    if self.splitter_name
                    in {"sentence", "resolved_sentence", "pn_sentence", "semantic"}
                    else "token"
                ),
                overwrite=self.compare_embed_overwrite,
            )

    def process_documents(self):
        start_time = time.time()
        files = os.listdir(self.docs_dir)
        print(f"Processing {len(files)} documents...")
        pending_chunks = []

        for filename in tqdm(files):
            cacheable_chunks, retrievable_chunks = self.split_document(filename)
            if not cacheable_chunks:
                continue
            if self.materialize_db:
                self.vectordb.store(retrievable_chunks)
            should_materialize_doc = (
                self.materialize_doc_ids is None or filename in self.materialize_doc_ids
            )
            if not should_materialize_doc:
                continue
            if self.compare_embed_writer is not None:
                self.compare_embed_writer.write_document(filename, cacheable_chunks)
            if self.materialize_cache:
                if self.resume_from_cache:
                    cacheable_chunks = self._filter_chunks_for_resume(cacheable_chunks)
                    if not cacheable_chunks:
                        continue
                pending_chunks.extend(cacheable_chunks)
                while len(pending_chunks) >= self.batch_size:
                    current_batch = pending_chunks[: self.batch_size]
                    self.save_kv_cache(current_batch)
                    pending_chunks = pending_chunks[self.batch_size :]

        if self.materialize_cache and pending_chunks:
            self.save_kv_cache(pending_chunks)
        if self.compare_embed_writer is not None:
            self.compare_embed_writer.finalize()

        end_time = time.time()
        elapsed_time = end_time - start_time

        if self.processed_doc_count > 0:
            avg_doc_tokens = self.total_doc_tokens / self.processed_doc_count
            print(f"Average document length: {avg_doc_tokens:.2f} tokens.")
            if self.cacheable_chunk_size is not None:
                print(
                    f"Documents longer than cacheable_chunk_size ({self.cacheable_chunk_size}): {self.docs_over_chunk_size}"
                )
        if self.total_chunk_count > 0:
            avg_chunks = self.total_chunk_count / self.processed_doc_count
            print(f"Average chunks per document: {avg_chunks:.2f}.")
            print(f"Max chunk length observed: {self.max_chunk_tokens} tokens.")
        if self.resume_from_cache:
            print(f"Skipped existing valid chunks: {self.skipped_existing_chunk_count}")
            print(
                f"Rebuilt invalid existing chunks: {self.rebuilt_invalid_chunk_count}"
            )

        print(f"Processing completed in {elapsed_time:.2f} seconds.")

    def _cache_path_for_chunk(self, chunk: Chunk) -> str:
        return os.path.join(self.cache_dir, f"{chunk.id}.pt")

    def _has_valid_existing_cache(self, chunk: Chunk) -> bool:
        cache_path = self._cache_path_for_chunk(chunk)
        if not os.path.exists(cache_path):
            return False
        try:
            torch.load(cache_path, weights_only=True, map_location="cpu")
            return True
        except Exception:
            try:
                os.remove(cache_path)
            except OSError:
                pass
            self.rebuilt_invalid_chunk_count += 1
            return False

    def _filter_chunks_for_resume(self, chunks: List[Chunk]) -> List[Chunk]:
        pending = []
        for chunk in chunks:
            if self._has_valid_existing_cache(chunk):
                self.skipped_existing_chunk_count += 1
                continue
            pending.append(chunk)
        return pending

    def split_document(self, filename: str):
        result = self.splitter.split_document(filename)
        self.total_doc_tokens += result.token_count
        self.processed_doc_count += 1
        if (
            self.cacheable_chunk_size is not None
            and result.token_count > self.cacheable_chunk_size
        ):
            self.docs_over_chunk_size += 1
        if result.chunks:  # cacheable chunk
            self.total_chunk_count += len(result.chunks)
            self.max_chunk_tokens = max(self.max_chunk_tokens, result.max_chunk_tokens)
        if self.splitter_name == "fixed_size":
            retrievable_chunks = [
                RetrievableChunk(
                    id=chunk.id,
                    text=chunk.text,
                    cacheables=[chunk],
                    chunk_size=chunk.chunk_size,
                    token_count=max(
                        0, int(chunk.chunk_end or 0) - int(chunk.chunk_start or 0)
                    ),
                    cache_unit="token",
                )
                for chunk in result.chunks
            ]
        else:
            retrievable_chunks = list(
                result.retrievable_chunks
                or (
                    []
                    if result.retrievable_chunk is None
                    else [result.retrievable_chunk]
                )
            )
        cache_unit = (
            "sentence"
            if self.splitter_name
            in {"sentence", "resolved_sentence", "pn_sentence", "semantic"}
            else "token"
        )
        for retrievable_chunk in retrievable_chunks:
            if retrievable_chunk.cache_unit is None:
                retrievable_chunk.cache_unit = cache_unit
        return result.chunks, retrievable_chunks

    def _build_cache_inputs(self, chunks: List[CacheableChunk], padding_side: str):
        visible_sequences = [
            self._build_visible_chunk_token_ids(chunk) for chunk in chunks
        ]
        max_length = max(len(sequence) for sequence in visible_sequences)
        input_ids = []
        attention_masks = []
        pad_token_id = int(self.model.tokenizer.pad_token_id)
        for sequence in visible_sequences:
            pad_len = max_length - len(sequence)
            if padding_side == "left":
                padded_sequence = [pad_token_id] * pad_len + sequence
                attention_mask = [0] * pad_len + [1] * len(sequence)
            else:
                padded_sequence = sequence + [pad_token_id] * pad_len
                attention_mask = [1] * len(sequence) + [0] * pad_len
            input_ids.append(padded_sequence)
            attention_masks.append(attention_mask)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        if self.dummy_bos_count == 0:
            return {
                "input_ids": input_ids.to("cuda"),
                "attention_mask": attention_mask.to("cuda"),
            }

        bos_prefix = torch.full(
            (input_ids.shape[0], self.dummy_bos_count),
            int(self.model.tokenizer.bos_token_id),
            dtype=input_ids.dtype,
        )
        bos_mask = torch.ones(
            (attention_mask.shape[0], self.dummy_bos_count),
            dtype=attention_mask.dtype,
        )
        input_ids = torch.cat([bos_prefix, input_ids], dim=1)
        attention_mask = torch.cat([bos_mask, attention_mask], dim=1)
        return {
            "input_ids": input_ids.to("cuda"),
            "attention_mask": attention_mask.to("cuda"),
        }

    def _build_visible_chunk_token_ids(self, chunk: Chunk) -> List[int]:
        chunk_token_ids = self.model.tokenizer.encode(
            chunk.text, add_special_tokens=False
        )
        if (
            self.splitter_name in {"sentence", "pn_sentence", "semantic"}
            and self.sentence_cache_token_format == "space_prefix_newline_suffix"
        ):
            return (
                self.space_token_ids + chunk_token_ids + self.visible_suffix_token_ids
            )
        if (
            self.splitter_name in {"sentence", "pn_sentence", "semantic"}
            and self.sentence_cache_token_format == "merged_space_prefix_newline_suffix"
        ):
            merged_prefix_chunk_token_ids = self.model.tokenizer.encode(
                f" {chunk.text}",
                add_special_tokens=False,
            )
            return merged_prefix_chunk_token_ids + self.visible_suffix_token_ids
        if (
            self.splitter_name in {"sentence", "pn_sentence", "semantic"}
            and self.sentence_cache_token_format
            == "merged_space_prefix_space_newline_suffix"
        ):
            merged_prefix_chunk_token_ids = self.model.tokenizer.encode(
                f" {chunk.text}",
                add_special_tokens=False,
            )
            merged_suffix_token_ids = self.model.tokenizer.encode(
                " \n\n",
                add_special_tokens=False,
            )
            if len(merged_suffix_token_ids) != 1:
                raise ValueError(
                    f"expected a single-token space-newline suffix, got ids={merged_suffix_token_ids}"
                )
            return merged_prefix_chunk_token_ids + merged_suffix_token_ids
        return (
            self.visible_prefix_token_ids
            + chunk_token_ids
            + self.visible_suffix_token_ids
        )

    def save_kv_cache(self, chunks: List[CacheableChunk]):
        inputs = self._build_cache_inputs(chunks, padding_side="right")
        with torch.no_grad():
            output = self.model.model(**inputs, use_cache=True)

        cache = output.past_key_values.to_legacy_cache()
        attention_mask = inputs["attention_mask"]
        sequence_lengths = attention_mask.sum(dim=1).tolist()

        for batch_idx, (chunk, sequence_length) in enumerate(
            zip(chunks, sequence_lengths)
        ):
            sample_cache = []
            seq_len = int(sequence_length)
            for layer in cache:
                key = layer[0][
                    batch_idx : batch_idx + 1, :, self.dummy_bos_count : seq_len, :
                ].contiguous()
                value = layer[1][
                    batch_idx : batch_idx + 1, :, self.dummy_bos_count : seq_len, :
                ].contiguous()
                sample_cache.append((key, value))
            torch.save(sample_cache, os.path.join(self.cache_dir, f"{chunk.id}.pt"))
