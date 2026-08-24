"""Model-based Provence and EXIT sentence-selection baselines."""

import os
import re
from contextlib import redirect_stderr, redirect_stdout
from typing import List

import numpy as np
import spacy
import torch
from dotenv import load_dotenv
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from chunk import CacheableChunk, RetrievableChunk
from compressor.base import Compressor

if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e4m3fn"):
    # PEFT 0.19 references torch.float8_e8m0fnu, but torch 2.6.0 does not expose it.
    # Alias it to an available float8 dtype so EXIT adapter loading can proceed.
    torch.float8_e8m0fnu = torch.float8_e4m3fn

try:
    from peft import PeftModel
except (
    ImportError
):  # pragma: no cover - exercised only when EXIT is requested without PEFT installed
    PeftModel = None


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


class OfficialSentenceSelector(Compressor):
    """Shared batch/output adapter for selectors operating on sentence units."""

    def _get_chunk_texts(self, doc: RetrievableChunk) -> List[str]:
        return [
            cacheable.text
            for cacheable in getattr(doc, "cacheables", [])
            if isinstance(cacheable.text, str) and cacheable.text.strip()
        ]

    @staticmethod
    def _get_document_text(doc: RetrievableChunk) -> str:
        text = getattr(doc, "text", None)
        return text if isinstance(text, str) and text.strip() else ""

    def _get_document_texts(self, doc: RetrievableChunk) -> List[str]:
        text = self._get_document_text(doc)
        return [text] if text else []

    def _build_text_document(
        self, doc: RetrievableChunk, text: str, source: str
    ) -> RetrievableChunk:
        cloned = doc.clone()
        if not isinstance(text, str) or not text.strip():
            cloned.cacheables = []
            return cloned
        cloned.cacheables = [
            CacheableChunk(
                id=f"{doc.id}::{source}",
                text=text,
            )
        ]
        return cloned

    def _select_sentence_indices(self, chunk_texts: List[str], query: str) -> List[int]:
        raise NotImplementedError

    def _select_sentence_indices_batch(
        self, chunk_texts_per_doc: List[List[str]], query: str
    ) -> List[List[int]]:
        return [
            self._select_sentence_indices(chunk_texts, query)
            for chunk_texts in chunk_texts_per_doc
        ]

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        summarized_batches = []
        for docs, query in zip(batch_top_k_docs, batch_queries):
            chunk_texts_per_doc = [self._get_chunk_texts(doc) for doc in docs]
            selected_indices_per_doc = self._select_sentence_indices_batch(
                chunk_texts_per_doc, query
            )
            summarized_docs = []
            for doc, chunk_texts, selected_indices in zip(
                docs, chunk_texts_per_doc, selected_indices_per_doc
            ):
                if not chunk_texts:
                    summarized_docs.append(doc.clone())
                    continue
                summarized_docs.append(
                    self._build_selected_document(doc, selected_indices)
                )
            summarized_batches.append(summarized_docs)
        return summarized_batches

    def compress(self, document_text: str, query: str) -> str:
        del query
        return document_text


class ProvenceCompressor(OfficialSentenceSelector):
    """Select context with the official Provence pruning model."""

    _RERANK_TOP_K = 5

    def __init__(self):
        super().__init__()
        load_dotenv()

        self.model_name = os.getenv(
            "PROVENCE_MODEL_NAME", "naver/provence-reranker-debertav3-v1"
        )
        self.threshold = float(os.getenv("PROVENCE_THRESHOLD", "0.5"))
        self.batch_size = int(os.getenv("PROVENCE_BATCH_SIZE", "256"))
        reorder_value = os.getenv("PROVENCE_REORDER", "False").strip().lower()
        if reorder_value not in {"true", "false"}:
            raise ValueError(
                "PROVENCE_REORDER must be exactly True or False "
                f"(case-insensitive), got {reorder_value!r}"
            )
        self.reorder = reorder_value == "true"
        self.device = os.getenv(
            "PROVENCE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(
            f"provence compression enabled. Initializing model: {self.model_name} on {self.device}"
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        ).to(self.device)

    def _extract_pruned_context(self, result) -> str:
        if isinstance(result, dict):
            for key in ("pruned_context", "pruned_passage", "compressed_context"):
                value = result.get(key)
                if isinstance(value, str):
                    return value
            for key in ("pruned_contexts", "pruned_passages", "compressed_contexts"):
                value = result.get(key)
                if isinstance(value, list) and value and isinstance(value[0], str):
                    return value[0]
        if isinstance(result, tuple):
            for item in result:
                if isinstance(item, str):
                    return item
                if isinstance(item, list) and item and isinstance(item[0], str):
                    return item[0]
        if isinstance(result, list) and result and isinstance(result[0], str):
            return result[0]
        if isinstance(result, str):
            return result
        raise ValueError("Unable to extract pruned context from Provence output.")

    def _extract_pruned_contexts(self, result, expected_len: int) -> List[str]:
        if expected_len == 0:
            return []
        if isinstance(result, dict):
            for key in (
                "pruned_context",
                "pruned_contexts",
                "pruned_passages",
                "compressed_contexts",
            ):
                value = result.get(key)
                if isinstance(value, list):
                    extracted = []
                    for item in value:
                        if isinstance(item, list):
                            extracted.append(
                                " ".join(part for part in item if isinstance(part, str))
                            )
                        else:
                            extracted.append(
                                self._extract_pruned_context(item)
                                if not isinstance(item, str)
                                else item
                            )
                    return extracted
            return [self._extract_pruned_context(result)]
        if isinstance(result, list):
            if len(result) == expected_len:
                extracted = []
                for item in result:
                    if isinstance(item, list):
                        extracted.append(
                            self._extract_pruned_context(item[0]) if item else ""
                        )
                    else:
                        extracted.append(
                            self._extract_pruned_context(item)
                            if not isinstance(item, str)
                            else item
                        )
                return extracted
            if len(result) == 1 and expected_len == 1:
                item = result[0]
                return [
                    (
                        self._extract_pruned_context(item)
                        if not isinstance(item, str)
                        else item
                    )
                ]
        return [self._extract_pruned_context(result)]

    def _greedy_match_indices(
        self, chunk_texts: List[str], pruned_context: str
    ) -> List[int]:
        normalized_pruned = _normalize_text(pruned_context)
        if not normalized_pruned:
            return []

        selected_indices = []
        cursor = 0
        for idx, sentence in enumerate(chunk_texts):
            normalized_sentence = _normalize_text(sentence)
            if not normalized_sentence:
                continue
            found_at = normalized_pruned.find(normalized_sentence, cursor)
            if found_at == -1:
                continue
            selected_indices.append(idx)
            cursor = found_at + len(normalized_sentence)
        return selected_indices

    def _select_sentence_indices(self, chunk_texts: List[str], query: str) -> List[int]:
        context = " ".join(chunk_texts)
        with (
            open(os.devnull, "w") as devnull,
            redirect_stdout(devnull),
            redirect_stderr(devnull),
        ):
            result = self.model.process(
                question=query,
                context=context,
                threshold=self.threshold,
                always_select_title=True,
                enable_warnings=False,
                reorder=False,
            )
        pruned_context = self._extract_pruned_context(result)
        return self._greedy_match_indices(chunk_texts, pruned_context)

    def _select_sentence_indices_batch(
        self, chunk_texts_per_doc: List[List[str]], query: str
    ) -> List[List[int]]:
        non_empty_docs = [
            (idx, chunk_texts)
            for idx, chunk_texts in enumerate(chunk_texts_per_doc)
            if chunk_texts
        ]
        selections = [[] for _ in chunk_texts_per_doc]
        if not non_empty_docs:
            return selections

        contexts = [[" ".join(chunk_texts)] for _, chunk_texts in non_empty_docs]
        questions = [query] * len(contexts)
        with (
            open(os.devnull, "w") as devnull,
            redirect_stdout(devnull),
            redirect_stderr(devnull),
        ):
            results = self.model.process(
                question=questions,
                context=contexts,
                threshold=self.threshold,
                batch_size=self.batch_size,
                always_select_title=True,
                enable_warnings=False,
                reorder=False,
            )
        pruned_contexts = self._extract_pruned_contexts(results, len(contexts))
        for (doc_idx, chunk_texts), pruned_context in zip(
            non_empty_docs, pruned_contexts
        ):
            selections[doc_idx] = self._greedy_match_indices(
                chunk_texts, pruned_context
            )
        return selections

    def _compress_texts_batch_with_doc_indices(
        self,
        batch_chunk_texts_per_doc: List[List[List[str]]],
        batch_queries: List[str],
    ) -> tuple[List[List[str]], List[List[int]]]:
        compressed_batches = [
            ["" for _ in chunk_texts_per_doc]
            for chunk_texts_per_doc in batch_chunk_texts_per_doc
        ]
        output_doc_indices = [
            list(range(len(chunk_texts_per_doc)))
            for chunk_texts_per_doc in batch_chunk_texts_per_doc
        ]
        active_query_indices = []
        active_doc_indices = []
        questions = []
        contexts = []
        for query_idx, (chunk_texts_per_doc, query) in enumerate(
            zip(batch_chunk_texts_per_doc, batch_queries)
        ):
            doc_indices = [
                doc_idx
                for doc_idx, chunk_texts in enumerate(chunk_texts_per_doc)
                if chunk_texts
            ]
            if not doc_indices:
                continue
            active_query_indices.append(query_idx)
            active_doc_indices.append(doc_indices)
            questions.append(query)
            contexts.append(
                [" ".join(chunk_texts_per_doc[doc_idx]) for doc_idx in doc_indices]
            )

        if not questions:
            return compressed_batches, output_doc_indices

        with (
            open(os.devnull, "w") as devnull,
            redirect_stdout(devnull),
            redirect_stderr(devnull),
        ):
            results = self.model.process(
                question=questions,
                context=contexts,
                threshold=self.threshold,
                batch_size=self.batch_size,
                always_select_title=True,
                enable_warnings=False,
                reorder=False,
            )
        if not isinstance(results, dict):
            raise ValueError("Provence batch output must be a dictionary.")
        pruned_batches = results.get("pruned_context")
        if not isinstance(pruned_batches, list) or len(pruned_batches) != len(
            questions
        ):
            actual = (
                len(pruned_batches)
                if isinstance(pruned_batches, list)
                else type(pruned_batches).__name__
            )
            raise ValueError(
                "Provence returned an unexpected number of query-level outputs: "
                f"expected {len(questions)}, got {actual}."
            )

        reranking_batches = results.get("reranking_score")
        if self.reorder and (
            not isinstance(reranking_batches, list)
            or len(reranking_batches) != len(questions)
        ):
            actual = (
                len(reranking_batches)
                if isinstance(reranking_batches, list)
                else type(reranking_batches).__name__
            )
            raise ValueError(
                "Provence returned an unexpected number of query-level "
                f"reranking scores: expected {len(questions)}, got {actual}."
            )

        for active_idx, (pruned_contexts, doc_indices) in enumerate(
            zip(pruned_batches, active_doc_indices)
        ):
            if not isinstance(pruned_contexts, list) or len(pruned_contexts) != len(
                doc_indices
            ):
                actual = (
                    len(pruned_contexts)
                    if isinstance(pruned_contexts, list)
                    else type(pruned_contexts).__name__
                )
                raise ValueError(
                    "Provence returned an unexpected number of document outputs "
                    f"for active query {active_idx}: expected {len(doc_indices)}, "
                    f"got {actual}."
                )
            if not all(isinstance(text, str) for text in pruned_contexts):
                raise ValueError("Provence document outputs must all be strings.")
            query_idx = active_query_indices[active_idx]
            if self.reorder:
                reranking_scores = reranking_batches[active_idx]
                if not isinstance(reranking_scores, (list, tuple, np.ndarray)) or len(
                    reranking_scores
                ) != len(doc_indices):
                    actual = (
                        len(reranking_scores)
                        if isinstance(reranking_scores, (list, tuple, np.ndarray))
                        else type(reranking_scores).__name__
                    )
                    raise ValueError(
                        "Provence returned an unexpected number of document "
                        f"reranking scores for active query {active_idx}: "
                        f"expected {len(doc_indices)}, got {actual}."
                    )
                ranked_positions = np.argsort(reranking_scores)[::-1][
                    : self._RERANK_TOP_K
                ]
                output_doc_indices[query_idx] = [
                    doc_indices[int(position)] for position in ranked_positions
                ]
                compressed_batches[query_idx] = [
                    pruned_contexts[int(position)] for position in ranked_positions
                ]
                continue
            for doc_idx, pruned_context in zip(doc_indices, pruned_contexts):
                compressed_batches[query_idx][doc_idx] = pruned_context
        return compressed_batches, output_doc_indices

    def _compress_texts_batch(
        self,
        batch_chunk_texts_per_doc: List[List[List[str]]],
        batch_queries: List[str],
    ) -> List[List[str]]:
        compressed_batches, _ = self._compress_texts_batch_with_doc_indices(
            batch_chunk_texts_per_doc,
            batch_queries,
        )
        return compressed_batches

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        if len(batch_top_k_docs) != len(batch_queries):
            raise ValueError(
                "Provence requires one retrieved-document batch per query: "
                f"got {len(batch_top_k_docs)} document batches and "
                f"{len(batch_queries)} queries."
            )
        batch_chunk_texts_per_doc = [
            [self._get_document_texts(doc) for doc in docs] for docs in batch_top_k_docs
        ]
        compressed_text_batches, output_doc_indices = (
            self._compress_texts_batch_with_doc_indices(
                batch_chunk_texts_per_doc,
                batch_queries,
            )
        )
        compressed_batches = []
        for docs, chunk_texts_per_doc, compressed_texts, doc_indices in zip(
            batch_top_k_docs,
            batch_chunk_texts_per_doc,
            compressed_text_batches,
            output_doc_indices,
        ):
            compressed_docs = []
            for doc_idx, compressed_text in zip(doc_indices, compressed_texts):
                doc = docs[doc_idx]
                chunk_texts = chunk_texts_per_doc[doc_idx]
                if not chunk_texts:
                    compressed_docs.append(
                        self._build_text_document(doc, "", "provence")
                    )
                    continue
                compressed_docs.append(
                    self._build_text_document(doc, compressed_text, "provence")
                )
            compressed_batches.append(compressed_docs)
        return compressed_batches


class EXITCompressor(OfficialSentenceSelector):
    """Select sentences using the EXIT yes/no relevance classifier."""

    def __init__(self):
        super().__init__()
        load_dotenv()

        if PeftModel is None:
            raise ImportError(
                "PEFT is required for EXIT. Install the 'peft' package before using compress_method='exit'."
            )

        self.adapter_name = os.getenv("EXIT_MODEL_NAME", "doubleyyh/exit-gemma-2b")
        self.base_model_name = os.getenv("EXIT_BASE_MODEL_NAME", "google/gemma-2b-it")
        self.device = os.getenv(
            "EXIT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.threshold = float(os.getenv("EXIT_THRESHOLD", "0.5"))
        self.max_input_tokens = int(os.getenv("EXIT_MAX_INPUT_TOKENS", "2048"))
        self.batch_size = int(os.getenv("EXIT_BATCH_SIZE", "8"))

        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("EXIT requires CUDA for its fixed 4-bit model load")

        print(
            "exit compression enabled. "
            f"Initializing 4-bit base model: {self.base_model_name}, "
            f"adapter: {self.adapter_name}"
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map={"": self.device},
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
        )
        self.model = PeftModel.from_pretrained(base_model, self.adapter_name)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        self.sentence_splitter = spacy.load(
            "en_core_web_sm",
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
        )
        self.sentence_splitter.enable_pipe("senter")

    @staticmethod
    def _build_prompt(query: str, context: str, sentence: str) -> str:
        return (
            "user\n"
            f"Query: {query}\n"
            f"Full context: {context}\n"
            f"Sentence: {sentence}\n"
            "Is this sentence useful in answering the query? "
            'Answer only "Yes" or "No".\n\n'
            "model\n"
        )

    def _score_prompts(self, prompts: List[str]) -> List[float]:
        if not prompts:
            return []
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded, logits_to_keep=1)
        logits = outputs.logits[:, -1, :][:, [self.yes_token_id, self.no_token_id]]
        scores = torch.softmax(logits, dim=-1)[:, 0]
        return [float(score) for score in scores.detach().cpu()]

    def _select_indices_from_scores(self, scores: List[float]) -> List[int]:
        selected = [idx for idx, score in enumerate(scores) if score >= self.threshold]
        if selected:
            return selected
        return [max(range(len(scores)), key=lambda idx: scores[idx])] if scores else []

    def _split_document_sentences(self, document_text: str) -> List[str]:
        return [
            sentence.text.strip()
            for sentence in self.sentence_splitter(document_text).sents
            if sentence.text.strip()
        ]

    def _select_sentence_indices(self, chunk_texts: List[str], query: str) -> List[int]:
        context = " ".join(chunk_texts)
        prompts = [
            self._build_prompt(query=query, context=context, sentence=sentence)
            for sentence in chunk_texts
        ]
        scores = []
        for start in range(0, len(prompts), self.batch_size):
            scores.extend(self._score_prompts(prompts[start : start + self.batch_size]))
        return self._select_indices_from_scores(scores)

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        compressed_batches = [
            [self._build_text_document(doc, "", "exit") for doc in docs]
            for docs in batch_top_k_docs
        ]
        prompt_records = []
        prompts = []

        for query_idx, (docs, query) in enumerate(zip(batch_top_k_docs, batch_queries)):
            for doc_idx, doc in enumerate(docs):
                context = self._get_document_text(doc)
                if not context:
                    continue
                chunk_texts = self._split_document_sentences(context)
                if not chunk_texts:
                    continue
                for sentence_idx, sentence in enumerate(chunk_texts):
                    prompt_records.append(
                        (query_idx, doc_idx, sentence_idx, chunk_texts)
                    )
                    prompts.append(
                        self._build_prompt(
                            query=query, context=context, sentence=sentence
                        )
                    )

        flat_scores = []
        for start in range(0, len(prompts), self.batch_size):
            flat_scores.extend(
                self._score_prompts(prompts[start : start + self.batch_size])
            )

        scores_by_doc = {}
        chunk_texts_by_doc = {}
        for (query_idx, doc_idx, sentence_idx, chunk_texts), score in zip(
            prompt_records, flat_scores
        ):
            key = (query_idx, doc_idx)
            chunk_texts_by_doc[key] = chunk_texts
            scores_by_doc.setdefault(key, [None] * len(chunk_texts))[
                sentence_idx
            ] = score

        for (query_idx, doc_idx), scores in scores_by_doc.items():
            chunk_texts = chunk_texts_by_doc[(query_idx, doc_idx)]
            dense_scores = [float(score) for score in scores if score is not None]
            selected_indices = self._select_indices_from_scores(dense_scores)
            selected_text = " ".join(chunk_texts[idx] for idx in selected_indices)
            original_doc = batch_top_k_docs[query_idx][doc_idx]
            compressed_batches[query_idx][doc_idx] = self._build_text_document(
                original_doc, selected_text, "exit"
            )

        return compressed_batches
