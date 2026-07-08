from dataclasses import dataclass

import os
import time
from collections import defaultdict
from typing import Dict, List, Optional
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    DynamicCache,
    AutoModelForCausalLM,
    logging,
    BitsAndBytesConfig,
)
import torch
from deepspeed.ops.op_builder import GDSBuilder, AsyncIOBuilder
from typing import Tuple
from cache_utils import shift_rotary_cache
from prompt import PromptProcessor


@dataclass
class MatKVTimeLog:
    prefill: float
    decode: float
    model_input_lengths: Optional[List[int]] = None


class LLMModel:
    SYSTEM_PROMPT = (
        "You answer questions using only the provided passages. "
        "Return only the shortest exact answer phrase supported by the passages. "
        "Do not explain, do not repeat the question, and do not output options or full sentences."
    )
    PASSAGE_PREFIX = ""

    def __init__(
        self,
        model_name: str,
        disable_rope: bool = False,
        use_front_bos_cache: bool = False,
        load_in_4bit: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.disable_rope = disable_rope
        self.use_front_bos_cache = use_front_bos_cache
        if load_in_4bit is None:
            load_in_4bit = os.environ.get(
                "MODEL_LOAD_IN_4BIT", "False"
            ).strip().lower() in {"true", "1", "yes", "y", "on"}
        self.load_in_4bit = bool(load_in_4bit)
        self._front_bos_cache = None
        self._load_model()

    def _load_model(self):
        print(f"LOADING MODEL {self.model_name} ...", flush=True)
        init_time = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="right"
        )

        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto" if self.load_in_4bit else "cuda",
        }
        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        print(
            f"MODEL LOAD MODE: {'bnb_4bit_nf4' if self.load_in_4bit else 'fp16'}",
            flush=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prompt_processor = PromptProcessor(
            tokenizer=self.tokenizer,
            system_prompt=self.SYSTEM_PROMPT,
            passage_prefix=self.PASSAGE_PREFIX,
        )

        print(time.perf_counter() - init_time, flush=True)
        print(f"MODEL LOADED", flush=True)

    @staticmethod
    def post_process_response(text):
        # FIXME: temporal solution
        text = text.replace("Answer: ", "").strip()
        for marker in ("Question: ", "Passage: ", "\n", "\r"):
            if marker in text:
                text = text.split(marker)[0].strip()
        return text

    @staticmethod
    def _to_legacy_cache(past_key_values):
        if isinstance(past_key_values, DynamicCache):
            return past_key_values.to_legacy_cache()
        return past_key_values

    @staticmethod
    def _stack_request_caches(request_caches):
        if len(request_caches) == 0:
            return None

        num_layers = len(request_caches[0])
        stacked = []
        for layer_idx in range(num_layers):
            keys = torch.cat([cache[layer_idx][0] for cache in request_caches], dim=0)
            values = torch.cat([cache[layer_idx][1] for cache in request_caches], dim=0)
            stacked.append((keys, values))
        return tuple(stacked)

    def _get_front_bos_cache(self):
        if self._front_bos_cache is not None:
            return self._front_bos_cache

        bos_id = self.tokenizer.bos_token_id
        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        self._front_bos_cache = output.past_key_values.to_legacy_cache()
        return self._front_bos_cache

    def _build_front_bos_batch_cache(self, past_key_values, past_lengths: List[int]):
        legacy_cache = self._to_legacy_cache(past_key_values)
        max_past_length = int(legacy_cache[0][0].shape[2])
        bos_cache = self._get_front_bos_cache()
        request_caches = []

        for batch_idx, past_length in enumerate(past_lengths):
            left_pad = max_past_length - int(past_length)
            request_layers = []
            for key, value in legacy_cache:
                request_layers.append(
                    (
                        key[
                            batch_idx : batch_idx + 1, :, left_pad:max_past_length, :
                        ].contiguous(),
                        value[
                            batch_idx : batch_idx + 1, :, left_pad:max_past_length, :
                        ].contiguous(),
                    )
                )
            shifted_request = shift_rotary_cache(
                tuple(request_layers), position_offset=1
            )
            combined_layers = []
            for (bos_key, bos_value), (req_key, req_value) in zip(
                bos_cache, shifted_request
            ):
                combined_layers.append(
                    (
                        torch.cat([bos_key, req_key], dim=2),
                        torch.cat([bos_value, req_value], dim=2),
                    )
                )
            request_caches.append(tuple(combined_layers))

        padded_caches, _ = self._pad_request_caches(request_caches)
        adjusted_lengths = [int(length) + 1 for length in past_lengths]
        return DynamicCache.from_legacy_cache(padded_caches), adjusted_lengths

    @staticmethod
    def _select_cache_rows(past_key_values, row_indices):
        if isinstance(row_indices, torch.Tensor):
            if row_indices.dtype == torch.bool:
                if row_indices.numel() == 0 or not torch.any(row_indices):
                    return None
            elif row_indices.numel() == 0:
                return None
        elif len(row_indices) == 0:
            return None
        cache_is_dynamic = isinstance(past_key_values, DynamicCache)
        legacy_cache = LLMModel._to_legacy_cache(past_key_values)
        selected = []
        for key, value in legacy_cache:
            selected.append((key[row_indices], value[row_indices]))
        selected_cache = tuple(selected)
        if cache_is_dynamic:
            return DynamicCache.from_legacy_cache(selected_cache)
        return selected_cache

    @staticmethod
    def _trim_prefilled_cache_rows(
        prefilled_past_key_values,
        past_lengths: List[int],
        query_lengths: List[int],
        max_past_length: int,
    ):
        legacy_cache = LLMModel._to_legacy_cache(prefilled_past_key_values)
        request_caches = []
        effective_lengths = []

        for batch_idx, (past_length, query_length) in enumerate(
            zip(past_lengths, query_lengths)
        ):
            left_pad = max_past_length - int(past_length)
            query_start = max_past_length
            query_end = max_past_length + int(query_length)
            trimmed_layers = []

            for key, value in legacy_cache:
                request_key = key[batch_idx : batch_idx + 1]
                request_value = value[batch_idx : batch_idx + 1]
                past_key = request_key[:, :, left_pad:max_past_length, :]
                past_value = request_value[:, :, left_pad:max_past_length, :]
                query_key = request_key[:, :, query_start:query_end, :]
                query_value = request_value[:, :, query_start:query_end, :]
                trimmed_layers.append(
                    (
                        torch.cat([past_key, query_key], dim=2),
                        torch.cat([past_value, query_value], dim=2),
                    )
                )

            request_caches.append(tuple(trimmed_layers))
            effective_lengths.append(int(past_length) + int(query_length))

        return request_caches, effective_lengths

    @staticmethod
    def _pad_request_caches(request_caches):
        if len(request_caches) == 0:
            return None, 0

        num_layers = len(request_caches[0])
        max_length = max(cache[0][0].shape[2] for cache in request_caches)
        padded = []

        for layer_idx in range(num_layers):
            keys = []
            values = []
            for cache in request_caches:
                key, value = cache[layer_idx]
                pad_len = max_length - key.shape[2]
                if pad_len > 0:
                    pad_shape = (key.shape[0], key.shape[1], pad_len, key.shape[3])
                    key = torch.cat(
                        [
                            torch.zeros(pad_shape, dtype=key.dtype, device=key.device),
                            key,
                        ],
                        dim=2,
                    )
                    value = torch.cat(
                        [
                            torch.zeros(
                                pad_shape, dtype=value.dtype, device=value.device
                            ),
                            value,
                        ],
                        dim=2,
                    )
                keys.append(key)
                values.append(value)
            padded.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        return tuple(padded), max_length

    def _decode_with_grouped_caches(
        self,
        request_caches,
        effective_lengths: List[int],
        first_token_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> List[torch.Tensor]:
        num_requests = first_token_ids.shape[0]
        device = first_token_ids.device
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        generated_tokens = torch.full(
            (num_requests, max_new_tokens),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        generated_tokens[:, 0] = first_token_ids.squeeze(-1)
        generated_lengths = torch.ones(num_requests, dtype=torch.long, device=device)
        eos_token_id = self.tokenizer.eos_token_id

        active_request_indices = torch.nonzero(
            first_token_ids.squeeze(-1) != eos_token_id,
            as_tuple=False,
        ).squeeze(-1)
        if active_request_indices.numel() == 0:
            return [
                generated_tokens[idx, : int(generated_lengths[idx])].detach().cpu()
                for idx in range(num_requests)
            ]

        active_request_caches = [
            request_caches[int(idx)] for idx in active_request_indices.tolist()
        ]
        padded_caches, padded_cache_length = self._pad_request_caches(
            active_request_caches
        )
        active_caches = DynamicCache.from_legacy_cache(padded_caches)
        next_token_id = first_token_ids[active_request_indices].clone()
        active_lengths = torch.tensor(
            [effective_lengths[int(idx)] for idx in active_request_indices.tolist()],
            dtype=torch.long,
            device=device,
        )

        for _ in range(max_new_tokens - 1):
            if next_token_id.numel() == 0:
                break

            left_pad = padded_cache_length - active_lengths
            attention_positions = torch.arange(
                padded_cache_length + 1,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            attention_mask = (attention_positions >= left_pad.unsqueeze(1)).long()
            position_ids = active_lengths.unsqueeze(1)
            cache_position = torch.tensor(
                [padded_cache_length],
                dtype=torch.long,
                device=device,
            )

            outputs = self.model(
                input_ids=next_token_id,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                past_key_values=active_caches,
                return_dict=True,
            )

            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens[
                active_request_indices, generated_lengths[active_request_indices]
            ] = next_token_id.squeeze(-1)
            generated_lengths[active_request_indices] += 1
            active_lengths = active_lengths + 1
            padded_cache_length += 1
            active_caches = outputs.past_key_values

            active_local_mask = next_token_id.squeeze(-1) != eos_token_id
            if bool(torch.all(active_local_mask)):
                continue
            if not bool(torch.any(active_local_mask)):
                break

            active_request_indices = active_request_indices[active_local_mask]
            next_token_id = next_token_id[active_local_mask]
            active_lengths = active_lengths[active_local_mask]
            active_caches = self._select_cache_rows(active_caches, active_local_mask)

        return [
            generated_tokens[idx, : int(generated_lengths[idx])].detach().cpu()
            for idx in range(num_requests)
        ]

    def _decode_with_prefilled_cache(
        self,
        prefilled_past_key_values,
        effective_lengths: List[int],
        prefill_attention_mask: torch.Tensor,
        first_token_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> List[torch.Tensor]:
        num_requests = first_token_ids.shape[0]
        device = first_token_ids.device
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        generated_tokens = torch.full(
            (num_requests, max_new_tokens),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        generated_tokens[:, 0] = first_token_ids.squeeze(-1)
        generated_lengths = torch.ones(num_requests, dtype=torch.long, device=device)
        eos_token_id = self.tokenizer.eos_token_id

        active_mask = first_token_ids.squeeze(-1) != eos_token_id
        active_request_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        if active_request_indices.numel() == 0:
            return [
                generated_tokens[idx, : int(generated_lengths[idx])].detach().cpu()
                for idx in range(num_requests)
            ]

        active_caches = self._select_cache_rows(prefilled_past_key_values, active_mask)
        active_attention_mask = prefill_attention_mask[active_mask]
        next_token_id = first_token_ids[active_mask].clone()
        active_lengths = torch.tensor(
            [effective_lengths[int(idx)] for idx in active_request_indices.tolist()],
            dtype=torch.long,
            device=device,
        )
        padded_cache_length = int(prefilled_past_key_values[0][0].shape[2])

        for _ in range(max_new_tokens - 1):
            if next_token_id.numel() == 0:
                break

            active_attention_mask = torch.cat(
                [
                    active_attention_mask,
                    torch.ones(
                        (active_attention_mask.shape[0], 1),
                        dtype=active_attention_mask.dtype,
                        device=active_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            position_ids = active_lengths.unsqueeze(1)
            cache_position = torch.tensor(
                [padded_cache_length],
                dtype=torch.long,
                device=device,
            )

            outputs = self.model(
                input_ids=next_token_id,
                attention_mask=active_attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                past_key_values=active_caches,
                return_dict=True,
            )

            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens[
                active_request_indices, generated_lengths[active_request_indices]
            ] = next_token_id.squeeze(-1)
            generated_lengths[active_request_indices] += 1
            active_lengths = active_lengths + 1
            padded_cache_length += 1
            active_caches = outputs.past_key_values

            active_local_mask = next_token_id.squeeze(-1) != eos_token_id
            if bool(torch.all(active_local_mask)):
                continue
            if not bool(torch.any(active_local_mask)):
                break

            active_request_indices = active_request_indices[active_local_mask]
            next_token_id = next_token_id[active_local_mask]
            active_lengths = active_lengths[active_local_mask]
            active_attention_mask = active_attention_mask[active_local_mask]
            active_caches = self._select_cache_rows(active_caches, active_local_mask)

        return [
            generated_tokens[idx, : int(generated_lengths[idx])].detach().cpu()
            for idx in range(num_requests)
        ]

    def generate_response(
        self,
        inputs,
        past_kv_caches=None,
        past_lengths=None,
        max_new_tokens: int = 100,
    ) -> Tuple[MatKVTimeLog, List[str]]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_prefill = time.perf_counter()
        use_past_cache = past_kv_caches is not None

        if not use_past_cache:
            tokens = self.prompt_processor.tokenize_full_prompts(inputs)
            model_input_lengths = (
                tokens["attention_mask"].sum(dim=1).detach().cpu().tolist()
            )
        else:
            prefill_attention_mask = None
            query_inputs = [query for _, query in inputs]
            model_past_kv = past_kv_caches
            tokenizer_kwargs = dict(
                return_tensors="pt",
                padding=True,
                truncation=True,
                padding_side="right",
            )
            if self.use_front_bos_cache:
                tokenizer_kwargs["add_special_tokens"] = False
            tokenized_queries = self.tokenizer(
                query_inputs,
                **tokenizer_kwargs,
            ).to("cuda")

            input_ids = tokenized_queries["input_ids"]
            new_attention_mask = tokenized_queries["attention_mask"]
            query_lengths = new_attention_mask.sum(dim=1).tolist()
            max_past_length = model_past_kv[0][0].shape[2]
            if past_lengths is None:
                past_lengths = [max_past_length] * input_ids.shape[0]
            model_input_lengths = [
                int(past_length) + int(query_length)
                for past_length, query_length in zip(past_lengths, query_lengths)
            ]
            past_attention_rows = []
            for past_length in past_lengths:
                left_pad = max_past_length - past_length
                past_attention_rows.append([0] * left_pad + [1] * past_length)
            past_attention_mask = torch.tensor(
                past_attention_rows,
                dtype=new_attention_mask.dtype,
                device=new_attention_mask.device,
            )
            attention_mask = torch.cat([past_attention_mask, new_attention_mask], dim=1)
            prefill_attention_mask = attention_mask

            cache_position = torch.arange(
                int(max_past_length),
                int(max_past_length) + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
            position_ids = torch.stack(
                [
                    torch.arange(
                        int(past_length),
                        int(past_length) + input_ids.shape[1],
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                    for past_length in past_lengths
                ],
                dim=0,
            )

            # 최종 모델 입력 구성
            tokens = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "cache_position": cache_position,
            }

        # prefill
        prompt_length = tokens["input_ids"].shape[1]
        with torch.no_grad():
            if not use_past_cache:
                output_tokens = self.model.generate(
                    **tokens,
                    max_new_tokens=1,
                    use_cache=True,
                    past_key_values=past_kv_caches,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    return_legacy_cache=True,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
                next_token_id = output_tokens.sequences[:, -1].unsqueeze(-1)
                past_key_values = output_tokens.past_key_values
                attention_mask = (
                    output_tokens.sequences != self.tokenizer.pad_token_id
                ).long()
            else:
                if isinstance(model_past_kv, tuple):
                    model_past_kv = DynamicCache.from_legacy_cache(model_past_kv)

                output_tokens = self.model(
                    **tokens,
                    use_cache=True,
                    past_key_values=model_past_kv,
                    return_dict=True,
                )
                last_query_indices = torch.tensor(
                    [max(int(query_length) - 1, 0) for query_length in query_lengths],
                    device=input_ids.device,
                    dtype=torch.long,
                )
                batch_indices = torch.arange(
                    output_tokens.logits.shape[0], device=input_ids.device
                )
                next_token_logits = output_tokens.logits[
                    batch_indices, last_query_indices, :
                ]
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
                effective_lengths = [
                    int(past_length) + int(query_length)
                    for past_length, query_length in zip(past_lengths, query_lengths)
                ]
                past_key_values = output_tokens.past_key_values
        # print(past_key_values[0][0].shape)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_prefill = time.perf_counter()
        unit_prefill = end_prefill - start_prefill
        # print(f"prefill 1 request: {unit_prefill:6f} seconds")

        # decode
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_decode = time.perf_counter()
        with torch.no_grad():
            if not use_past_cache:
                outputs = self.model.generate(
                    input_ids=output_tokens.sequences,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens - 1,
                    use_cache=True,
                    past_key_values=past_key_values,
                    eos_token_id=self.tokenizer.pad_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    return_legacy_cache=True,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
                generated_answers = outputs.sequences[:, prompt_length:]
            else:
                generated_answers = self._decode_with_prefilled_cache(
                    prefilled_past_key_values=past_key_values,
                    effective_lengths=effective_lengths,
                    prefill_attention_mask=prefill_attention_mask,
                    first_token_ids=next_token_id,
                    max_new_tokens=max_new_tokens,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_decode = time.perf_counter()
        unit_decode = end_decode - start_decode

        generated_text = self.tokenizer.batch_decode(
            [answer.tolist() for answer in generated_answers],
            skip_special_tokens=True,
        )

        responses = [self.post_process_response(line) for line in generated_text]
        return (
            MatKVTimeLog(
                decode=unit_decode,
                prefill=unit_prefill,
                model_input_lengths=[int(length) for length in model_input_lengths],
            ),
            responses,
        )
