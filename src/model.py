"""LLM loading and full-prompt generation for evaluation."""

from dataclasses import dataclass

import os
import time
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import Tuple
from prompt import PromptProcessor


@dataclass
class InferenceTimeLog:
    """Measured prefill/decode time and effective model input lengths."""

    prefill: float
    decode: float
    model_input_lengths: Optional[List[int]] = None
    generated_token_counts: Optional[List[int]] = None
    decode_steps: int = 0


class LLMModel:
    """Wrap a Hugging Face causal LM with project prompt and generation logic."""

    SYSTEM_PROMPT = (
        "Answer the question using only the provided context.\n"
        "You may combine facts from different parts of the context, but do not use "
        "outside knowledge or guess missing facts.\n"
        "If the context does not provide enough information to answer the question, "
        'output exactly "Insufficient information".\n'
        "Otherwise, output only a short, direct answer.\n"
        "Do not include explanations or repeat the question."
    )

    def __init__(
        self,
        model_name: str,
        load_in_4bit: Optional[bool] = None,
        prompt_format: str = "raw_chunk_first",
    ):
        self.model_name = model_name
        self.prompt_format = prompt_format
        if load_in_4bit is None:
            load_in_4bit = os.environ.get(
                "MODEL_LOAD_IN_4BIT", "False"
            ).strip().lower() in {"true", "1", "yes", "y", "on"}
        self.load_in_4bit = bool(load_in_4bit)
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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prompt_processor = PromptProcessor(
            tokenizer=self.tokenizer,
            system_prompt=self.SYSTEM_PROMPT,
            prompt_format=self.prompt_format,
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

    def _generated_token_counts(self, generated_answers) -> List[int]:
        """Count generated IDs through the first EOS, including that EOS."""

        eos_token_id = self.tokenizer.eos_token_id
        counts = []
        for answer in generated_answers:
            token_ids = answer.tolist() if hasattr(answer, "tolist") else list(answer)
            if eos_token_id is not None and eos_token_id in token_ids:
                counts.append(token_ids.index(eos_token_id) + 1)
            else:
                counts.append(len(token_ids))
        return counts

    def generate_response(
        self,
        inputs,
        max_new_tokens: int = 100,
    ) -> Tuple[InferenceTimeLog, List[str]]:
        """Generate complete prompts with only the current request's memory cache."""

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_prefill = time.perf_counter()
        tokens = self.prompt_processor.tokenize_full_prompts(inputs)
        model_input_lengths = (
            tokens["attention_mask"].sum(dim=1).detach().cpu().tolist()
        )

        # prefill
        prompt_length = tokens["input_ids"].shape[1]
        with torch.no_grad():
            output_tokens = self.model.generate(
                **tokens,
                max_new_tokens=1,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                return_legacy_cache=True,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            past_key_values = output_tokens.past_key_values
            attention_mask = (
                output_tokens.sequences != self.tokenizer.pad_token_id
            ).long()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_prefill = time.perf_counter()
        unit_prefill = end_prefill - start_prefill

        # decode
        start_decode = time.perf_counter()
        with torch.no_grad():
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
            decode_steps = int(
                outputs.sequences.shape[1] - output_tokens.sequences.shape[1]
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_decode = time.perf_counter()
        unit_decode = end_decode - start_decode
        generated_answer_ids = (
            generated_answers.detach().cpu().tolist()
            if isinstance(generated_answers, torch.Tensor)
            else [answer.tolist() for answer in generated_answers]
        )
        generated_token_counts = self._generated_token_counts(generated_answer_ids)

        generated_text = self.tokenizer.batch_decode(
            generated_answer_ids,
            skip_special_tokens=True,
        )

        responses = [self.post_process_response(line) for line in generated_text]
        return (
            InferenceTimeLog(
                decode=unit_decode,
                prefill=unit_prefill,
                model_input_lengths=[int(length) for length in model_input_lengths],
                generated_token_counts=generated_token_counts,
                decode_steps=decode_steps,
            ),
            responses,
        )
