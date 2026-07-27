"""Small JSON and legacy asynchronous cache-I/O helpers."""

import json


def parse_json_query(json_query: str):
    parsed = json.loads(json_query)
    return parsed["query"]


def file_read(inp_f, handle, gpu_buffer):
    handle.sync_pread(gpu_buffer, inp_f)
    return gpu_buffer.cuda()


def get_seq_len(flattened_tensor, num_layers, num_kv_heads, dim):
    """저장된 Flatten된 텐서에서 Token 수(seq_len) 자동 추출"""
    total_params = flattened_tensor.shape[0]  # 1D 텐서 크기
    per_layer_size = num_kv_heads * dim * 2  # Key + Value 크기

    seq_len = total_params // (num_layers * per_layer_size)  # 토큰 수 계산
    return seq_len


def restore_tensor_shape(flattened_tensor, num_layers, num_kv_heads, dim):
    restored_cache = []
    offset = 0
    seq_len = get_seq_len(flattened_tensor, num_layers, num_kv_heads, dim)
    each_layer = num_kv_heads * seq_len * dim
    for _ in range(num_layers):
        key = flattened_tensor[offset : offset + each_layer].reshape(
            1, num_kv_heads, seq_len, dim
        )
        offset += each_layer
        value = flattened_tensor[offset : offset + each_layer].reshape(
            1, num_kv_heads, seq_len, dim
        )
        offset += each_layer

        restored_cache.append((key, value))
        # print(restored_cache[0][0].shape)
    return tuple(restored_cache)
