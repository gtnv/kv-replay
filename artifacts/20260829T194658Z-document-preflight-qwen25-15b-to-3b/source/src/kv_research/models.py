import hashlib
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id, revision, attention_backend):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation=attention_backend,
    )
    model.to("cuda")
    model.eval()
    return model, tokenizer


def tokenizer_hash(tokenizer):
    vocabulary = sorted(tokenizer.get_vocab().items())
    special = {
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
    }
    payload = json.dumps([vocabulary, special], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def assert_matching_tokenizers(source_tokenizer, target_tokenizer):
    source_hash = tokenizer_hash(source_tokenizer)
    target_hash = tokenizer_hash(target_tokenizer)
    if source_hash != target_hash:
        raise RuntimeError(
            f"tokenizers differ: source {source_hash[:12]}, target {target_hash[:12]}"
        )
    return source_hash


@torch.inference_mode()
def prefill(model, input_ids):
    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device)
    position_ids = positions.unsqueeze(0).expand(batch, -1)
    attention_mask = torch.ones((batch, length), dtype=torch.long, device=input_ids.device)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=positions,
        use_cache=True,
        logits_to_keep=1,
    )


@torch.inference_mode()
def continue_cache(model, cache, input_ids):
    batch, length = input_ids.shape
    prefix_length = cache.get_seq_length()
    positions = torch.arange(prefix_length, prefix_length + length, device=input_ids.device)
    position_ids = positions.unsqueeze(0).expand(batch, -1)
    attention_mask = torch.ones(
        (batch, prefix_length + length), dtype=torch.long, device=input_ids.device
    )
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=positions,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )


def fixed_control_tokens(tokenizer, length, device):
    text = (
        "A systems agent inspected a failed database migration. The transaction was rolled back, "
        "the schema version remained unchanged, and the next action must preserve user data. "
        "The observation includes a retry token, a timestamp, and an explicit instruction to avoid "
        "the destructive fallback. "
    )
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) < 8:
        raise RuntimeError("control text produced too few tokens")
    repeated = (token_ids * ((length // len(token_ids)) + 1))[:length]
    return torch.tensor(repeated, dtype=torch.long, device=device).unsqueeze(0)
