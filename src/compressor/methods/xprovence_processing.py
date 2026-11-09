"""Batched preprocessing for the native XProvence first-sentence-title route.

Input assembly and selection follow naver/xprovence-reranker-bgem3-v2,
revision f2798d357213c40164bdd79557d6d2e43ca7d8d2. Model forward is unchanged.
"""

import importlib
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def prepare_inputs(model, questions, contexts):
    """Preserve upstream sentence text, token IDs, block and sentence boundaries."""
    module = importlib.import_module(type(model).__module__)
    tokenizer = model.tokenizer
    start = time.perf_counter()
    flat = [
        (qi, ci, text)
        for qi, docs in enumerate(contexts)
        for ci, text in enumerate(docs)
    ]
    sentence_texts = []
    sentence_ranges = []
    for doc in module.nlp.pipe((r[2] for r in flat), batch_size=100, n_process=1):
        begin = len(sentence_texts)
        sentence_texts.extend(
            (" " if j else "") + sentence.text.strip()
            for j, sentence in enumerate(doc.sents)
        )
        sentence_ranges.append((begin, len(sentence_texts)))
    split_seconds = time.perf_counter() - start
    start = time.perf_counter()
    query_ids = tokenizer(
        [module.normalize(q) for q in questions],
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    sentence_ids = (
        tokenizer(
            sentence_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        if sentence_texts
        else []
    )
    tokenize_seconds = time.perf_counter() - start
    dataset = SimpleNamespace(
        ids=[],
        qis=[],
        cis=[],
        sis=[],
        sent_coords=[],
        cntx_coords=[],
        pad_idx=tokenizer.pad_token_id,
    )
    for (qi, ci, _), (begin, end) in zip(flat, sentence_ranges):
        query = query_ids[qi]
        # These are the upstream process() query-length conditions.
        if not query or len(query) >= model.max_len - 3:
            continue
        prefix = [tokenizer.cls_token_id] + query + [tokenizer.sep_token_id]
        left_0 = len(prefix)
        max_len = model.max_len - left_0 - 1
        pieces = [
            ids[offset : offset + max_len]
            for ids in sentence_ids[begin:end]
            for offset in range(0, len(ids), max_len)
        ]
        sequence = list(prefix)
        coordinates = []
        block = 0

        def append_block():
            dataset.ids.append(torch.tensor(sequence + [tokenizer.eos_token_id]))
            dataset.qis.append(qi)
            dataset.cis.append(ci)
            dataset.sis.append(block)
            dataset.sent_coords.append(coordinates)
            dataset.cntx_coords.append([coordinates[0][0], coordinates[-1][1]])

        for piece in pieces:
            if len(sequence) + len(piece) <= model.max_len - 1:
                coordinates.append([len(sequence), len(sequence) + len(piece)])
                sequence.extend(piece)
            else:
                if len(sequence) > left_0:
                    append_block()
                sequence = prefix + piece
                coordinates = [[left_0, left_0 + len(piece)]]
                block += 1
        if len(sequence) > left_0:
            append_block()
    return dataset, {
        "sentence_split_seconds": split_seconds,
        "tokenize_seconds": tokenize_seconds,
    }


def process_contexts(model, questions, contexts, threshold, batch_size):
    """Native adapter settings: first sentence title, title kept, no reordering."""
    module = importlib.import_module(type(model).__module__)
    start = time.perf_counter()
    dataset, timings = prepare_inputs(model, questions, contexts)
    timings["preprocess_seconds"] = time.perf_counter() - start
    selected = [[{0: text} for text in docs] for docs in contexts]
    ranking = [[None for _ in docs] for docs in contexts]
    post_seconds = 0.0
    with torch.no_grad():
        for begin in range(0, len(dataset.ids), batch_size):
            end = begin + batch_size
            ids_list = dataset.ids[begin:end]
            ids = pad_sequence(
                ids_list, batch_first=True, padding_value=dataset.pad_idx
            ).to(model.device)
            mask = ids != dataset.pad_idx
            outputs = model.forward(ids, mask)
            start = time.perf_counter()
            # Keep the original CPU softmax to preserve floating-point values.
            token_preds = (
                torch.softmax(outputs["compression_logits"].cpu(), dim=-1)[..., 1]
                > threshold
            ).numpy()
            ranking_scores = outputs["ranking_scores"].cpu().numpy()
            if ranking_scores.ndim == 0:
                ranking_scores = ranking_scores[None]
            for tokens, predictions, score, qi, ci, si, coords in zip(
                ids_list,
                token_preds,
                ranking_scores,
                dataset.qis[begin:end],
                dataset.cis[begin:end],
                dataset.sis[begin:end],
                dataset.sent_coords[begin:end],
            ):
                selected_mask = module.sentence_rounding(
                    predictions,
                    np.array(coords),
                    threshold=threshold,
                    always_select_title=si == 0,
                )
                selected[qi][ci][si] = tokens[selected_mask[: len(tokens)]]
                if si == 0:
                    ranking[qi][ci] = score
            post_seconds += time.perf_counter() - start
    start = time.perf_counter()
    positions = []
    token_lists = []
    for qi, docs in enumerate(selected):
        for ci, parts in enumerate(docs):
            if isinstance(parts[0], str):
                selected[qi][ci] = parts[0]
            else:
                positions.append((qi, ci))
                token_lists.append(
                    torch.cat([tokens for _, tokens in sorted(parts.items())]).tolist()
                )
    # Transformers batch_decode itself is a Python loop; use the Rust batch API.
    decoded = (
        model.tokenizer.backend_tokenizer.decode_batch(
            token_lists, skip_special_tokens=True
        )
        if token_lists
        else []
    )
    for (qi, ci), text in zip(positions, decoded):
        selected[qi][ci] = text
    compression = [
        [
            (len(original) - len(text)) / len(original) * 100
            for original, text in zip(originals, compressed)
        ]
        for originals, compressed in zip(contexts, selected)
    ]
    timings["postprocess_seconds"] = post_seconds + time.perf_counter() - start
    model._xprovence_processing_timings = timings
    return {
        "pruned_context": selected,
        "reranking_score": ranking,
        "compression_rate": compression,
    }
