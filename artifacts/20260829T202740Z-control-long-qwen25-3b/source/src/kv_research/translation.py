import torch

from kv_research.cache_ops import (
    apply_rope,
    attention_head_dim,
    build_cache,
    cache_pairs,
    remove_rope,
    rope_terms,
)


def aligned_source_layers(source_layers, target_layers):
    if source_layers < 2 or target_layers < 2:
        raise RuntimeError("layer alignment requires at least two layers per model")
    return [
        round(index * (source_layers - 1) / (target_layers - 1)) for index in range(target_layers)
    ]


def source_layer_groups(source_layers, target_layers, layers_per_target):
    if not 1 <= layers_per_target <= source_layers:
        raise RuntimeError(
            f"source layers per target must be in [1, {source_layers}], found {layers_per_target}"
        )
    groups = []
    for anchor in aligned_source_layers(source_layers, target_layers):
        start = min(
            max(anchor - (layers_per_target - 1) // 2, 0),
            source_layers - layers_per_target,
        )
        groups.append(list(range(start, start + layers_per_target)))
    return groups


def kv_width(model):
    return model.config.num_key_value_heads * attention_head_dim(model)


def init_ridge_stats(source_model, target_model, layers_per_target):
    source_width = kv_width(source_model) * layers_per_target
    target_width = kv_width(target_model)
    target_layers = target_model.config.num_hidden_layers
    shape_gram = (target_layers, source_width + 1, source_width + 1)
    shape_cross = (target_layers, source_width + 1, target_width)
    device = next(target_model.parameters()).device
    return {
        "key_gram": torch.zeros(shape_gram, dtype=torch.float64, device=device),
        "key_cross": torch.zeros(shape_cross, dtype=torch.float64, device=device),
        "value_gram": torch.zeros(shape_gram, dtype=torch.float64, device=device),
        "value_cross": torch.zeros(shape_cross, dtype=torch.float64, device=device),
        "key_target_square": torch.zeros(target_layers, dtype=torch.float64, device=device),
        "value_target_square": torch.zeros(target_layers, dtype=torch.float64, device=device),
        "source_width": source_width,
        "source_layers_per_target": layers_per_target,
        "layer_groups": source_layer_groups(
            source_model.config.num_hidden_layers,
            target_model.config.num_hidden_layers,
            layers_per_target,
        ),
        "samples": 0,
    }


def flatten_cache_state(state):
    batch, heads, length, head_dim = state.shape
    return state.permute(0, 2, 1, 3).reshape(batch * length, heads * head_dim)


def add_affine_column(states):
    ones = torch.ones((states.shape[0], 1), dtype=states.dtype, device=states.device)
    return torch.cat((states, ones), dim=1)


@torch.inference_mode()
def accumulate_ridge_stats(stats, source_model, target_model, source_cache, target_cache):
    source_pairs = cache_pairs(source_cache)
    target_pairs = cache_pairs(target_cache)
    source_layers = source_model.config.num_hidden_layers
    target_layers = target_model.config.num_hidden_layers
    if len(source_pairs) != source_layers or len(target_pairs) != target_layers:
        raise RuntimeError("cache layer count does not match model configuration")
    layer_groups = stats["layer_groups"]
    sequence_length = source_pairs[0][0].shape[2]
    if target_pairs[0][0].shape[2] != sequence_length:
        raise RuntimeError("source and target training caches have different sequence lengths")

    source_cos, source_sin = rope_terms(
        source_model,
        source_pairs[0][0].shape[0],
        0,
        sequence_length,
        source_pairs[0][0].dtype,
        source_pairs[0][0].device,
    )
    target_cos, target_sin = rope_terms(
        target_model,
        target_pairs[0][0].shape[0],
        0,
        sequence_length,
        target_pairs[0][0].dtype,
        target_pairs[0][0].device,
    )
    source_key_layers = [
        flatten_cache_state(remove_rope(keys, source_cos, source_sin))
        for keys, values in source_pairs
    ]
    source_value_layers = [flatten_cache_state(values) for keys, values in source_pairs]

    for target_index, source_indices in enumerate(layer_groups):
        target_keys, target_values = target_pairs[target_index]
        target_unrotated = remove_rope(target_keys, target_cos, target_sin)

        source_key_rows = add_affine_column(
            torch.cat(
                [source_key_layers[source_index] for source_index in source_indices],
                dim=1,
            ).double()
        )
        target_key_rows = flatten_cache_state(target_unrotated).double()
        source_value_rows = add_affine_column(
            torch.cat(
                [source_value_layers[source_index] for source_index in source_indices],
                dim=1,
            ).double()
        )
        target_value_rows = flatten_cache_state(target_values).double()

        stats["key_gram"][target_index].addmm_(source_key_rows.T, source_key_rows)
        stats["key_cross"][target_index].addmm_(source_key_rows.T, target_key_rows)
        stats["value_gram"][target_index].addmm_(source_value_rows.T, source_value_rows)
        stats["value_cross"][target_index].addmm_(source_value_rows.T, target_value_rows)
        stats["key_target_square"][target_index] += target_key_rows.square().sum()
        stats["value_target_square"][target_index] += target_value_rows.square().sum()

    stats["samples"] += source_pairs[0][0].shape[0] * sequence_length


def solve_ridge(stats, ridge, source_model, target_model):
    samples = stats["samples"]
    source_width = stats["source_width"]
    if samples <= source_width:
        raise RuntimeError(f"ridge fit has {samples} samples for {source_width} source dimensions")
    if ridge <= 0:
        raise RuntimeError(f"ridge must be positive, found {ridge}")
    penalty = torch.eye(source_width + 1, dtype=torch.float64, device=stats["key_gram"].device)
    penalty[-1, -1] = 0
    key_maps = []
    value_maps = []
    diagnostics = []
    for layer_index in range(target_model.config.num_hidden_layers):
        key_system = stats["key_gram"][layer_index] / samples + ridge * penalty
        key_rhs = stats["key_cross"][layer_index] / samples
        value_system = stats["value_gram"][layer_index] / samples + ridge * penalty
        value_rhs = stats["value_cross"][layer_index] / samples
        key_map = torch.linalg.solve(key_system, key_rhs)
        value_map = torch.linalg.solve(value_system, value_rhs)
        key_maps.append(key_map.float().cpu())
        value_maps.append(value_map.float().cpu())
        key_square = stats["key_target_square"][layer_index]
        value_square = stats["value_target_square"][layer_index]
        key_sse = (
            key_square
            - 2 * torch.sum(key_map * stats["key_cross"][layer_index])
            + torch.sum(key_map * (stats["key_gram"][layer_index] @ key_map))
        )
        value_sse = (
            value_square
            - 2 * torch.sum(value_map * stats["value_cross"][layer_index])
            + torch.sum(value_map * (stats["value_gram"][layer_index] @ value_map))
        )
        diagnostics.append(
            {
                "layer": layer_index,
                "key_fit_nmse": max(0.0, (key_sse / key_square).item()),
                "value_fit_nmse": max(0.0, (value_sse / value_square).item()),
                "key_map_frobenius": torch.linalg.vector_norm(key_map).item(),
                "value_map_frobenius": torch.linalg.vector_norm(value_map).item(),
            }
        )
    return {
        "translator_schema": 2,
        "source_layers": source_model.config.num_hidden_layers,
        "target_layers": target_model.config.num_hidden_layers,
        "source_width_per_layer": kv_width(source_model),
        "source_width": source_width,
        "target_width": kv_width(target_model),
        "source_layers_per_target": stats["source_layers_per_target"],
        "layer_map": aligned_source_layers(
            source_model.config.num_hidden_layers, target_model.config.num_hidden_layers
        ),
        "layer_groups": stats["layer_groups"],
        "layer_group_policy": "contiguous_anchor_deeper_tie_v1",
        "key_maps": torch.stack(key_maps),
        "value_maps": torch.stack(value_maps),
        "ridge": ridge,
        "samples": samples,
        "diagnostics": diagnostics,
    }


def validate_translation_shape(translation, source_model, target_model):
    expected = {
        "translator_schema": 2,
        "source_layers": source_model.config.num_hidden_layers,
        "target_layers": target_model.config.num_hidden_layers,
        "source_width_per_layer": kv_width(source_model),
        "source_width": kv_width(source_model) * translation["source_layers_per_target"],
        "target_width": kv_width(target_model),
        "layer_group_policy": "contiguous_anchor_deeper_tie_v1",
    }
    for name, value in expected.items():
        if translation[name] != value:
            raise RuntimeError(f"translator {name} is {translation[name]}, expected {value}")
    expected_groups = source_layer_groups(
        source_model.config.num_hidden_layers,
        target_model.config.num_hidden_layers,
        translation["source_layers_per_target"],
    )
    if translation["layer_groups"] != expected_groups:
        raise RuntimeError("translator layer groups do not match the configured model pair")
    expected_map = aligned_source_layers(
        source_model.config.num_hidden_layers, target_model.config.num_hidden_layers
    )
    if translation["layer_map"] != expected_map:
        raise RuntimeError("translator layer map does not match the configured model pair")
    map_shape = (
        target_model.config.num_hidden_layers,
        translation["source_width"] + 1,
        kv_width(target_model),
    )
    for name in ("key_maps", "value_maps"):
        maps = translation[name]
        if tuple(maps.shape) != map_shape:
            raise RuntimeError(
                f"translator {name} shape is {tuple(maps.shape)}, expected {map_shape}"
            )
        if not maps.is_floating_point():
            raise RuntimeError(f"translator {name} is not floating point")


def identity_translation(model, layers_per_target):
    width_per_layer = kv_width(model)
    width = width_per_layer * layers_per_target
    layers = model.config.num_hidden_layers
    groups = source_layer_groups(layers, layers, layers_per_target)
    maps = torch.zeros((layers, width + 1, width_per_layer), dtype=torch.float32)
    identity = torch.eye(width_per_layer, dtype=torch.float32)
    for target_index, group in enumerate(groups):
        source_slot = group.index(target_index)
        start = source_slot * width_per_layer
        maps[target_index, start : start + width_per_layer] = identity
    return {
        "translator_schema": 2,
        "source_layers": layers,
        "target_layers": layers,
        "source_width_per_layer": width_per_layer,
        "source_width": width,
        "target_width": width_per_layer,
        "source_layers_per_target": layers_per_target,
        "layer_map": list(range(layers)),
        "layer_groups": groups,
        "layer_group_policy": "contiguous_anchor_deeper_tie_v1",
        "key_maps": maps.clone(),
        "value_maps": maps.clone(),
        "ridge": 0.0,
        "samples": 0,
    }


def translation_on_device(translation, device):
    runtime = dict(translation)
    runtime["key_maps"] = translation["key_maps"].to(device)
    runtime["value_maps"] = translation["value_maps"].to(device)
    if not torch.isfinite(runtime["key_maps"]).all():
        raise RuntimeError("key translation maps contain non-finite values")
    if not torch.isfinite(runtime["value_maps"]).all():
        raise RuntimeError("value translation maps contain non-finite values")
    return runtime


@torch.inference_mode()
def translate_cache(source_cache, source_model, target_model, translation, source_position_start=0):
    if source_position_start != 0:
        raise RuntimeError(
            "the pilot translator only supports prefix caches starting at position zero"
        )
    validate_translation_shape(translation, source_model, target_model)
    source_pairs = cache_pairs(source_cache)
    if len(source_pairs) != translation["source_layers"]:
        raise RuntimeError("source cache layer count does not match translator")
    batch = source_pairs[0][0].shape[0]
    length = source_pairs[0][0].shape[2]
    source_cos, source_sin = rope_terms(
        source_model,
        batch,
        0,
        length,
        source_pairs[0][0].dtype,
        source_pairs[0][0].device,
    )
    target_cos, target_sin = rope_terms(
        target_model,
        batch,
        0,
        length,
        source_pairs[0][0].dtype,
        source_pairs[0][0].device,
    )
    target_heads = target_model.config.num_key_value_heads
    target_head_dim = attention_head_dim(target_model)
    target_dtype = next(target_model.parameters()).dtype
    target_device = next(target_model.parameters()).device
    if translation["key_maps"].device != target_device:
        raise RuntimeError("key translation maps are not on the target device")
    if translation["value_maps"].device != target_device:
        raise RuntimeError("value translation maps are not on the target device")
    translated_pairs = []
    source_key_layers = [
        flatten_cache_state(remove_rope(keys, source_cos, source_sin))
        for keys, values in source_pairs
    ]
    source_value_layers = [flatten_cache_state(values) for keys, values in source_pairs]

    for target_index, source_indices in enumerate(translation["layer_groups"]):
        key_rows = add_affine_column(
            torch.cat(
                [source_key_layers[source_index] for source_index in source_indices],
                dim=1,
            ).float()
        )
        value_rows = add_affine_column(
            torch.cat(
                [source_value_layers[source_index] for source_index in source_indices],
                dim=1,
            ).float()
        )
        key_map = translation["key_maps"][target_index]
        value_map = translation["value_maps"][target_index]
        target_key_rows = key_rows @ key_map
        target_value_rows = value_rows @ value_map
        target_keys = target_key_rows.reshape(batch, length, target_heads, target_head_dim)
        target_values = target_value_rows.reshape(batch, length, target_heads, target_head_dim)
        target_keys = target_keys.permute(0, 2, 1, 3)
        target_values = target_values.permute(0, 2, 1, 3)
        target_keys = apply_rope(target_keys, target_cos, target_sin).to(target_dtype)
        target_values = target_values.to(target_dtype)
        translated_pairs.append((target_keys, target_values))
    return build_cache(target_model.config, translated_pairs)
