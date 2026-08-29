import torch
from transformers import DynamicCache


def attention_head_dim(model):
    hidden_size = model.config.hidden_size
    attention_heads = model.config.num_attention_heads
    if hidden_size % attention_heads:
        raise RuntimeError(
            f"hidden size {hidden_size} is not divisible by {attention_heads} attention heads"
        )
    derived = hidden_size // attention_heads
    if hasattr(model.config, "head_dim") and model.config.head_dim != derived:
        raise RuntimeError(
            f"configured head dimension {model.config.head_dim} differs from {derived}"
        )
    return derived


def cache_pairs(cache):
    pairs = []
    for layer_index, layer in enumerate(cache.layers):
        if not layer.is_initialized:
            raise RuntimeError(f"cache layer {layer_index} is uninitialized")
        if layer.keys.ndim != 4 or layer.values.ndim != 4:
            raise RuntimeError(f"cache layer {layer_index} is not rank four")
        pairs.append((layer.keys, layer.values))
    if not pairs:
        raise RuntimeError("cache contains no layers")
    return pairs


def build_cache(config, pairs):
    cache = DynamicCache(config=config)
    for layer_index, (keys, values) in enumerate(pairs):
        if keys.shape != values.shape:
            raise RuntimeError(f"K/V shape mismatch at layer {layer_index}")
        cache.update(keys.contiguous(), values.contiguous(), layer_index)
    return cache


def clone_cache(cache, config):
    pairs = [(keys.clone(), values.clone()) for keys, values in cache_pairs(cache)]
    return build_cache(config, pairs)


def slice_cache(cache, length, config):
    pairs = []
    for keys, values in cache_pairs(cache):
        if not 0 < length <= keys.shape[2]:
            raise RuntimeError(f"invalid cache slice {length} for sequence length {keys.shape[2]}")
        pairs.append((keys[:, :, :length].clone(), values[:, :, :length].clone()))
    return build_cache(config, pairs)


def rotate_half(states):
    half = states.shape[-1] // 2
    if states.shape[-1] % 2:
        raise RuntimeError(f"RoPE head dimension must be even, found {states.shape[-1]}")
    first = states[..., :half]
    second = states[..., half:]
    return torch.cat((-second, first), dim=-1)


def rope_terms(model, batch, start, length, dtype, device):
    positions = torch.arange(start, start + length, device=device).unsqueeze(0).expand(batch, -1)
    head_dim = attention_head_dim(model)
    reference = torch.empty((batch, length, head_dim), dtype=dtype, device=device)
    cosines, sines = model.model.rotary_emb(reference, positions)
    return cosines.unsqueeze(1), sines.unsqueeze(1)


def remove_rope(keys, cosines, sines):
    keys_float = keys.float()
    cosines_float = cosines.float()
    sines_float = sines.float()
    scale = cosines_float.square() + sines_float.square()
    return (keys_float * cosines_float - rotate_half(keys_float) * sines_float) / scale


def apply_rope(keys, cosines, sines):
    keys_float = keys.float()
    return keys_float * cosines.float() + rotate_half(keys_float) * sines.float()


def cache_suffix_nmse(left, right, start, stop):
    left_pairs = cache_pairs(left)
    right_pairs = cache_pairs(right)
    if len(left_pairs) != len(right_pairs):
        raise RuntimeError(
            f"cache layer count differs: {len(left_pairs)} versus {len(right_pairs)}"
        )
    numerator = torch.zeros((), dtype=torch.float64, device=left_pairs[0][0].device)
    denominator = torch.zeros_like(numerator)
    for (left_keys, left_values), (right_keys, right_values) in zip(
        left_pairs, right_pairs, strict=True
    ):
        for left_state, right_state in (
            (left_keys[:, :, start:stop], right_keys[:, :, start:stop]),
            (left_values[:, :, start:stop], right_values[:, :, start:stop]),
        ):
            difference = left_state.double() - right_state.double()
            numerator += difference.square().sum()
            denominator += right_state.double().square().sum()
    if denominator == 0:
        raise RuntimeError("cache comparison denominator is zero")
    return (numerator / denominator).item()


def rope_roundtrip_nmse(model, keys):
    batch, heads, length, head_dim = keys.shape
    if heads != model.config.num_key_value_heads or head_dim != attention_head_dim(model):
        raise RuntimeError("cache shape does not match model RoPE configuration")
    cosines, sines = rope_terms(model, batch, 0, length, keys.dtype, keys.device)
    unrotated = remove_rope(keys, cosines, sines)
    reconstructed = apply_rope(unrotated, cosines, sines)
    numerator = (reconstructed.double() - keys.double()).square().sum()
    denominator = keys.double().square().sum()
    if denominator == 0:
        raise RuntimeError("RoPE round-trip denominator is zero")
    return (numerator / denominator).item()
