"""Load, rotate, concatenate, and pad materialized transformer KV caches."""

import torch
import os

# from deepspeed.ops.op_builder import GDSBuilder, AsyncIOBuilder
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, rotate_half
from utils import parse_json_query, file_read, restore_tensor_shape
from chunk import Chunk, RetrievableChunk, CacheableChunk

from typing import Dict, List, Optional, Tuple

DEFAULT_LLAMA31_ROPE_SCALING = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}

_ROTARY_EMBEDDING_CACHE: Dict[Tuple[str, int, float, int], LlamaRotaryEmbedding] = {}


# def load_kv_cache_gds(self, doc_id: str):
#     in_file = os.path.join(self.cache_dir, f"{doc_id}.pt")
#     file_sz = os.path.getsize(in_file)
#
#     file_sz = file_sz // 2
#
#     gds_buffer = self.gds_handle.new_pinned_device_tensor(file_sz,
#                                                           torch.empty(0, dtype=torch.float16, device='cuda',
#                                                                       requires_grad=False))
#     loaded_tensor = file_read(in_file, self.gds_handle, gds_buffer)
#
#     kv_cache = restore_tensor_shape(loaded_tensor, self.num_layers, self.num_kv_heads, self.dim)
#     return kv_cache
#
#
# def load_kv_cache_aio(self, doc_id: str):
#     in_file = os.path.join(self.cache_dir, f"{doc_id}.pt")
#     file_sz = os.path.getsize(in_file)
#
#     num_elements = file_sz // 2
#     bounce_buffer = torch.empty(num_elements, dtype=torch.float16).pin_memory()
#
#     loaded_tensor = file_read(in_file, self.aio_handle, bounce_buffer)
#
#     kv_cache = restore_tensor_shape(loaded_tensor, self.num_layers, self.num_kv_heads, self.dim)
#     return kv_cache


def load_caches_from_rchunks(
    cache_dir: str,
    docs: List[RetrievableChunk],
    apply_rotary_shift: bool = True,
    initial_position_offset: int = 0,
):
    all_caches = []
    total_cached_length = initial_position_offset
    for doc in docs:
        doc_caches = load_doc_caches(
            cache_dir,
            doc,
            total_cached_length,
            apply_rotary_shift=apply_rotary_shift,
        )
        all_caches.extend(doc_caches)
        total_cached_length += sum(cache[0][0].shape[2] for cache in doc_caches)
    return all_caches, total_cached_length - initial_position_offset


def load_caches_from_doc_batch(
    cache_dir: str,
    batch_chunks: List[List[RetrievableChunk]],
    apply_rotary_shift: bool = True,
    initial_position_offset: int = 0,
) -> Tuple[List[List[tuple]], List[int]]:
    batch_caches: List[List[tuple]] = []
    cached_lengths: List[int] = []
    for chunks in batch_chunks:
        request_caches, cached_length = load_caches_from_rchunks(
            cache_dir,
            chunks,
            apply_rotary_shift=apply_rotary_shift,
            initial_position_offset=initial_position_offset,
        )
        batch_caches.append(request_caches)
        cached_lengths.append(cached_length)

    return batch_caches, cached_lengths


def load_doc_caches(
    cache_dir: str,
    doc: Chunk,
    doc_base_idx=0,
    apply_rotary_shift: bool = True,
):
    cacheables = getattr(doc, "cacheables", None)
    if cacheables is not None:
        shifted_caches = []
        current_offset = doc_base_idx
        for cacheable in cacheables:
            cache = load_kv_cache(cache_dir, cacheable.id)
            if apply_rotary_shift:
                shifted_caches.append(
                    shift_rotary_cache(cache, position_offset=current_offset)
                )
            else:
                shifted_caches.append(
                    tuple((key.to("cuda"), value.to("cuda")) for key, value in cache)
                )
            current_offset += cache[0][0].shape[2]
        return shifted_caches
    return [load_kv_cache(cache_dir, doc.id)]


def load_kv_cache(
    cache_dir: str,
    doc_id: str,
):
    cache_file = os.path.join(cache_dir, f"{doc_id}.pt")
    return torch.load(cache_file, weights_only=True, map_location="cpu")


def _build_llama_rotary_embedding(
    head_dim: int,
    device: torch.device,
    base: float,
    max_position_embeddings: int,
):
    cache_key = (str(device), int(head_dim), float(base), int(max_position_embeddings))
    cached_rotary = _ROTARY_EMBEDDING_CACHE.get(cache_key)
    if cached_rotary is not None:
        return cached_rotary

    config = LlamaConfig(
        hidden_size=head_dim,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rope_theta=base,
        rope_scaling=DEFAULT_LLAMA31_ROPE_SCALING,
    )
    rotary_embedding = LlamaRotaryEmbedding(config=config, device=device)
    _ROTARY_EMBEDDING_CACHE[cache_key] = rotary_embedding
    return rotary_embedding


def _build_rotary_cos_sin(
    keys: torch.Tensor,
    position_offset: int,
    base: float = 500000.0,
):
    head_dim = keys.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"ROPE head_dim must be even, got {head_dim}")

    batch_size, _, seq_len, _ = keys.shape
    min_rope_positions = (
        DEFAULT_LLAMA31_ROPE_SCALING["original_max_position_embeddings"] + 1
    )
    max_position_embeddings = max(position_offset + seq_len, min_rope_positions)
    rotary_emb = _build_llama_rotary_embedding(
        head_dim=head_dim,
        device=keys.device,
        base=base,
        max_position_embeddings=max_position_embeddings,
    )
    position_ids = torch.full(
        (batch_size, seq_len),
        position_offset,
        dtype=torch.long,
        device=keys.device,
    )
    return rotary_emb(keys, position_ids)


def shift_rotary_cache(cache, position_offset: int, base: float = 500000.0):
    """Rebase cached Llama keys when their prompt position changes."""

    if position_offset == 0:
        return tuple((key.to("cuda"), value.to("cuda")) for key, value in cache)

    shared_cos = None
    shared_sin = None
    if len(cache) > 0:
        shared_cos, shared_sin = _build_rotary_cos_sin(
            cache[0][0].to("cuda"),
            position_offset=position_offset,
            base=base,
        )
        shared_cos = shared_cos.unsqueeze(0).unsqueeze(2)
        shared_sin = shared_sin.unsqueeze(0).unsqueeze(2)

    key_tensors = [key.to("cuda") for key, _ in cache]
    stacked_keys = torch.stack(key_tensors, dim=0)
    shifted_keys = (stacked_keys * shared_cos) + (
        rotate_half(stacked_keys) * shared_sin
    )

    shifted_cache = []
    for layer_idx, (_, value) in enumerate(cache):
        shifted_cache.append((shifted_keys[layer_idx], value.to("cuda")))
    return tuple(shifted_cache) if isinstance(cache, tuple) else shifted_cache


def concat_caches_single(caches):
    """
    concatenate the cache
    """
    if len(caches) == 0:
        return None

    num_layers = len(caches[0])
    concatenated = []
    for layer in range(num_layers):
        keys = torch.cat([cache[layer][0] for cache in caches], dim=2)
        values = torch.cat([cache[layer][1] for cache in caches], dim=2)
        concatenated.append((keys, values))
    return concatenated


def concat_caches(batch_caches):
    """
    batch_caches: List[List[List[Tuple[Tensor, Tensor]]]]
    - batch_size(4)개의 요청이 각각 top_k(3)개의 문서에 대해 가져온 KV 캐시 리스트
    - 즉, batch_caches[i]는 i번째 요청의 KV 캐시 리스트(top-3)

    반환값: Tuple[List[Tensor], List[Tensor]]
    - past_key_values에 올바르게 들어갈 수 있도록 변환
    """
    if len(batch_caches) == 0:
        print("concat_caches: No caches to concatenate")
        return None

    batch_size = len(batch_caches)  # 4
    num_layers = len(batch_caches[0][0])  # 16

    # batch_size만큼 KV 캐시를 각각 concat해서 저장할 리스트
    batch_keys_list = [[] for _ in range(num_layers)]
    batch_values_list = [[] for _ in range(num_layers)]

    for i in range(batch_size):  # 각 요청별 처리
        request_caches = batch_caches[i]  # i번째 요청의 top-3 문서 캐시 리스트
        concatenated_request = concat_caches_single(request_caches)

        for layer in range(num_layers):
            batch_keys_list[layer].append(concatenated_request[layer][0])  # Key 저장
            batch_values_list[layer].append(
                concatenated_request[layer][1]
            )  # Value 저장

    return (batch_keys_list, batch_values_list)


def pad_past_key_values(batch_keys_list, batch_values_list):
    """
    배치 내 요청마다 seq_len이 다를 경우, 최대 길이에 맞춰 padding 후 `past_key_values` 형태로 변환
    """
    num_layers = len(batch_keys_list)  # 총 레이어 개수 (16개)
    batch_size = len(batch_keys_list[0])  # 배치 크기 (4개 요청)

    # 각 레이어별로 최대 seq_len 찾기
    max_doc_length = max(k.shape[2] for k in batch_keys_list[0])
    past_key_values = []

    for layer_idx in range(num_layers):
        keys = []
        values = []
        # padding_counts_per_request = []

        for i in range(batch_size):
            past_k = batch_keys_list[layer_idx][i]
            past_v = batch_values_list[layer_idx][i]
            doc_length = past_k.shape[2]
            pad_len = max_doc_length - doc_length

            # if pad_len > 0:
            #     past_k = shift_rotary_cache_keys(past_k, position_offset=pad_len)

            # Zero-padding 적용
            if pad_len > 0:
                pad_shape = (
                    past_k.shape[0],
                    past_k.shape[1],
                    pad_len,
                    past_k.shape[3],
                )  # (1, num_heads, pad_len, head_dim)
                pad_tensor_k = torch.zeros(
                    pad_shape, dtype=past_k.dtype, device=past_k.device
                )
                pad_tensor_v = torch.zeros(
                    pad_shape, dtype=past_v.dtype, device=past_v.device
                )
                past_k = torch.cat([pad_tensor_k, past_k], dim=2)
                past_v = torch.cat([pad_tensor_v, past_v], dim=2)

            keys.append(past_k)
            values.append(past_v)

        past_key_values.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

    return tuple(past_key_values)
