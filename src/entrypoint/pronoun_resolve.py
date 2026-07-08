import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fire
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from materialize.splitter.base import SentenceWiseSplitter
from materialize.splitter.resolution import (
    build_openai_client,
    resolve_leading_pronouns_with_fastcoref,
    resolve_pronouns_with_openai,
)


class TokenizerOnlyModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def write_manifest(
    output_mapping_dir: str,
    resolver: str,
    openai_model: str,
    fastcoref_model_name: str,
    tokenizer_model: str,
    retrievable_chunk_size: int,
):
    manifest = {
        "resolver": resolver,
        "openai_model": openai_model,
        "fastcoref_model_name": fastcoref_model_name,
        "tokenizer_model": tokenizer_model,
        "retrievable_chunk_size": retrievable_chunk_size,
    }
    manifest_path = Path(output_mapping_dir) / "_pn_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def build_retrievable_windows(
    filename: str, token_ids, sentence_views, retrievable_chunk_size: int
):
    windows = []
    for window_idx, window_start in enumerate(
        range(0, len(token_ids), retrievable_chunk_size)
    ):
        window_end = min(window_start + retrievable_chunk_size, len(token_ids))
        sentence_ids = [
            f"{filename}::sent_{sent_idx}"
            for sent_idx, sentence_view in enumerate(sentence_views)
            if sentence_view.token_start < window_end
            and sentence_view.token_end > window_start
        ]
        if not sentence_ids:
            continue
        windows.append(
            {
                "id": f"{filename}::ret_{window_idx}",
                "window_token_start": window_start,
                "window_token_end": window_end,
                "sentence_ids": sentence_ids,
            }
        )
    return windows


def main(
    input_docs_dir: str,
    output_mapping_dir: str,
    resolver: str = "openai",
    openai_model: str = "gpt-4o-mini",
    fastcoref_model_name: str = "biu-nlp/f-coref",
    tokenizer_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    retrievable_chunk_size: int = 1024,
    overwrite: bool = False,
    num_workers: int = 1,
):
    os.makedirs(output_mapping_dir, exist_ok=True)
    write_manifest(
        output_mapping_dir,
        resolver,
        openai_model,
        fastcoref_model_name,
        tokenizer_model,
        retrievable_chunk_size,
    )
    failure_log_path = Path(output_mapping_dir) / "_pn_failures.jsonl"

    openai_client = None
    coref_model = None
    thread_local = threading.local()
    failure_lock = threading.Lock()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, padding_side="right")
    splitter = SentenceWiseSplitter(
        docs_dir=input_docs_dir,
        model=TokenizerOnlyModel(tokenizer),
        cacheable_chunk_size=None,
        retrievable_chunk_size=retrievable_chunk_size,
        content_chunk_size=None,
    )
    if resolver == "openai":
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        if num_workers <= 1:
            openai_client = build_openai_client(project_root)
    elif resolver == "fastcoref":
        from fastcoref import FCoref
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        coref_model = FCoref(device=device, model_name_or_path=fastcoref_model_name)
    elif resolver != "none":
        raise ValueError(f"unsupported resolver: {resolver}")

    filenames = sorted(
        name
        for name in os.listdir(input_docs_dir)
        if os.path.isfile(os.path.join(input_docs_dir, name))
    )
    print(f"Resolving pronouns for {len(filenames)} documents...")

    def resolve_single_document(filename: str):
        input_path = os.path.join(input_docs_dir, filename)
        output_path = os.path.join(output_mapping_dir, f"{filename}.json")
        if not overwrite and os.path.exists(output_path):
            return

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        sentence_views = splitter._build_sentence_views(text, token_ids)
        sentence_texts = [
            sentence_view.text.strip() for sentence_view in sentence_views
        ]
        try:
            if resolver == "none":
                rewritten = list(sentence_texts)
            elif resolver == "fastcoref":
                rewritten, _ = resolve_leading_pronouns_with_fastcoref(
                    sentence_texts, coref_model
                )
            else:
                client = openai_client
                if client is None:
                    client = getattr(thread_local, "openai_client", None)
                    if client is None:
                        client = build_openai_client(project_root)
                        thread_local.openai_client = client
                rewritten, _ = resolve_pronouns_with_openai(
                    sentence_texts, client, openai_model
                )
        except Exception as exc:
            rewritten = list(sentence_texts)
            failure_payload = {
                "filename": filename,
                "resolver": resolver,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            with failure_lock:
                with failure_log_path.open("a", encoding="utf-8") as log_f:
                    log_f.write(json.dumps(failure_payload, ensure_ascii=False) + "\n")
        sentence_records = []
        for sent_idx, (sentence_view, original_text, resolved_text) in enumerate(
            zip(sentence_views, sentence_texts, rewritten)
        ):
            sentence_records.append(
                {
                    "sentence_id": f"{filename}::sent_{sent_idx}",
                    "original_text": original_text,
                    "resolved_text": resolved_text,
                    "char_start": sentence_view.char_start,
                    "char_end": sentence_view.char_end,
                    "token_start": sentence_view.token_start,
                    "token_end": sentence_view.token_end,
                }
            )
        payload = {
            "format": "pn_mapping_v1",
            "filename": filename,
            "resolver": resolver,
            "sentence_views": sentence_records,
            "retrievable_windows": build_retrievable_windows(
                filename=filename,
                token_ids=token_ids,
                sentence_views=sentence_views,
                retrievable_chunk_size=retrievable_chunk_size,
            ),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    if resolver == "openai" and num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(resolve_single_document, filename)
                for filename in filenames
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()
    else:
        for filename in tqdm(filenames):
            resolve_single_document(filename)


if __name__ == "__main__":
    fire.Fire(main)
