import os
import re
from contextlib import redirect_stderr, redirect_stdout
from typing import List

import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

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
    def _get_chunk_texts(self, doc: RetrievableChunk) -> List[str]:
        return [
            cacheable.text
            for cacheable in getattr(doc, "cacheables", [])
            if isinstance(cacheable.text, str) and cacheable.text.strip()
        ]

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
    def __init__(self):
        super().__init__()
        load_dotenv()

        self.model_name = os.getenv(
            "PROVENCE_MODEL_NAME", "naver/provence-reranker-debertav3-v1"
        )
        self.threshold = float(os.getenv("PROVENCE_THRESHOLD", "0.5"))
        self.batch_size = int(os.getenv("PROVENCE_BATCH_SIZE", "256"))
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
                enable_warnings=False,
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
                enable_warnings=False,
            )
        pruned_contexts = self._extract_pruned_contexts(results, len(contexts))
        for (doc_idx, chunk_texts), pruned_context in zip(
            non_empty_docs, pruned_contexts
        ):
            selections[doc_idx] = self._greedy_match_indices(
                chunk_texts, pruned_context
            )
        return selections

    def _compress_texts_batch(
        self, chunk_texts_per_doc: List[List[str]], query: str
    ) -> List[str]:
        non_empty_docs = [
            (idx, chunk_texts)
            for idx, chunk_texts in enumerate(chunk_texts_per_doc)
            if chunk_texts
        ]
        compressed_texts = ["" for _ in chunk_texts_per_doc]
        if not non_empty_docs:
            return compressed_texts

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
                enable_warnings=False,
            )
        pruned_contexts = self._extract_pruned_contexts(results, len(contexts))
        for (doc_idx, _), pruned_context in zip(non_empty_docs, pruned_contexts):
            compressed_texts[doc_idx] = pruned_context
        return compressed_texts

    def compress_batch_top_k_docs(
        self, batch_top_k_docs: List[List[RetrievableChunk]], batch_queries: List[str]
    ):
        compressed_batches = []
        for docs, query in zip(batch_top_k_docs, batch_queries):
            chunk_texts_per_doc = [self._get_chunk_texts(doc) for doc in docs]
            compressed_texts = self._compress_texts_batch(chunk_texts_per_doc, query)
            compressed_docs = []
            for doc, chunk_texts, compressed_text in zip(
                docs, chunk_texts_per_doc, compressed_texts
            ):
                if not chunk_texts:
                    compressed_docs.append(doc.clone())
                    continue
                compressed_docs.append(
                    self._build_text_document(doc, compressed_text, "provence")
                )
            compressed_batches.append(compressed_docs)
        return compressed_batches


class EXITCompressor(OfficialSentenceSelector):
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

        print(
            "exit compression enabled. "
            f"Initializing base model: {self.base_model_name}, adapter: {self.adapter_name}"
        )

        model_kwargs = {}
        if self.device == "cuda" and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs,
        )
        self.model = PeftModel.from_pretrained(base_model, self.adapter_name).to(
            self.device
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

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
            [doc.clone() for doc in docs] for docs in batch_top_k_docs
        ]
        prompt_records = []
        prompts = []

        for query_idx, (docs, query) in enumerate(zip(batch_top_k_docs, batch_queries)):
            for doc_idx, doc in enumerate(docs):
                chunk_texts = self._get_chunk_texts(doc)
                if not chunk_texts:
                    continue
                context = " ".join(chunk_texts)
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
