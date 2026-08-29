import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import HfApi

from kv_research.artifacts import (
    base_manifest,
    create_run_dir,
    sha256_file,
    snapshot_source,
    write_json,
    write_jsonl,
)
from kv_research.cache_ops import cache_pairs, cache_suffix_nmse, clone_cache, rope_roundtrip_nmse
from kv_research.metrics import control_summary, logit_metrics
from kv_research.models import continue_cache, fixed_control_tokens, load_model, prefill
from kv_research.translation import identity_translation, translate_cache, translation_on_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def run_control(config, model, tokenizer):
    device = next(model.parameters()).device
    identities = {
        layers_per_target: translation_on_device(
            identity_translation(model, layers_per_target), device
        )
        for layers_per_target in config["identity_layers_per_target"]
    }
    records = []
    for length in config["lengths"]:
        input_ids = fixed_control_tokens(tokenizer, length, device, config["control_text_variant"])
        model.set_attn_implementation(config["attention_backend"])
        full = prefill(model, input_ids)
        model.set_attn_implementation("eager")
        eager = prefill(model, input_ids)
        model.set_attn_implementation(config["attention_backend"])
        backend_metrics = logit_metrics(full.logits[:, -1], eager.logits[:, -1])

        for replay_length in config["replay_lengths"]:
            if replay_length >= length:
                raise RuntimeError(f"replay length {replay_length} is invalid for length {length}")
            prefix_length = length - replay_length
            prefix = prefill(model, input_ids[:, :prefix_length])
            roundtrip_nmse = rope_roundtrip_nmse(model, cache_pairs(prefix.past_key_values)[0][0])
            identity_prefixes = {}
            for layers_per_target, identity in identities.items():
                identity_source = clone_cache(prefix.past_key_values, model.config)
                identity_cache = translate_cache(identity_source, model, model, identity)
                identity_prefixes[layers_per_target] = (
                    identity_cache,
                    cache_suffix_nmse(
                        prefix.past_key_values,
                        identity_cache,
                        0,
                        prefix_length,
                    ),
                )
            rebuilt_cache = clone_cache(prefix.past_key_values, model.config)
            replay = continue_cache(
                model,
                prefix.past_key_values,
                input_ids[:, prefix_length:],
            )
            rebuilt = continue_cache(
                model,
                rebuilt_cache,
                input_ids[:, prefix_length:],
            )
            full_replay_metrics = logit_metrics(full.logits[:, -1], replay.logits[:, -1])
            rebuild_metrics = logit_metrics(replay.logits[:, -1], rebuilt.logits[:, -1])
            full_vs_replay_cache_nmse = cache_suffix_nmse(
                full.past_key_values,
                replay.past_key_values,
                prefix_length,
                length,
            )
            replay_vs_rebuilt_cache_nmse = cache_suffix_nmse(
                replay.past_key_values,
                rebuilt.past_key_values,
                0,
                length,
            )
            identity_records = {}
            for layers_per_target, (identity_cache, prefix_nmse) in identity_prefixes.items():
                identity_replay = continue_cache(
                    model,
                    identity_cache,
                    input_ids[:, prefix_length:],
                )
                identity_record = {
                    "prefix_nmse": prefix_nmse,
                    "continuation_nmse": cache_suffix_nmse(
                        replay.past_key_values,
                        identity_replay.past_key_values,
                        0,
                        length,
                    ),
                    "suffix_nmse": cache_suffix_nmse(
                        replay.past_key_values,
                        identity_replay.past_key_values,
                        prefix_length,
                        length,
                    ),
                    "logits": logit_metrics(replay.logits[:, -1], identity_replay.logits[:, -1]),
                }
                if config["identity_continuation_bound"] == "native_execution_dominance":
                    machine_limit = 64 * max(
                        prefix_nmse,
                        torch.finfo(torch.bfloat16).eps ** 4,
                    )
                    identity_record["effective_cache_nmse_limit"] = min(
                        config["native_cache_nmse_limit"],
                        max(
                            machine_limit,
                            replay_vs_rebuilt_cache_nmse,
                            full_vs_replay_cache_nmse,
                        ),
                    )
                    identity_record["effective_logit_js_limit"] = max(
                        torch.finfo(torch.bfloat16).eps ** 4,
                        full_replay_metrics["js"],
                        rebuild_metrics["js"],
                    )
                identity_records[str(layers_per_target)] = identity_record
            records.append(
                {
                    "length": length,
                    "replay_length": replay_length,
                    "full_vs_replay": full_replay_metrics,
                    "replay_vs_rebuilt": rebuild_metrics,
                    "full_vs_replay_cache_nmse": full_vs_replay_cache_nmse,
                    "replay_vs_rebuilt_cache_nmse": replay_vs_rebuilt_cache_nmse,
                    "identity_translation": identity_records,
                    "rope_roundtrip_nmse": roundtrip_nmse,
                    "sdpa_vs_eager": backend_metrics,
                }
            )
    return records


args = parse_args()
repo_root = Path(__file__).resolve().parents[1]
config_path = Path(args.config).resolve()
with open(config_path, encoding="utf-8") as stream:
    config = json.load(stream)
for key in ("control_text_variant", "identity_continuation_bound"):
    if key not in config:
        raise RuntimeError(f"control config must define {key}")
if config["identity_continuation_bound"] not in ("machine_floor", "native_execution_dominance"):
    raise RuntimeError("unknown identity continuation bound")

torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
run_dir = create_run_dir(repo_root, config["run_name"])
manifest = base_manifest(repo_root, config_path)
snapshot_source(repo_root, run_dir, manifest["source_sha256"])
for environment_name in ("bootstrap.json", "pip-freeze.txt"):
    environment_source = repo_root / "artifacts" / "environment" / environment_name
    if environment_source.is_file():
        environment_destination = run_dir / "environment" / environment_name
        environment_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(environment_source, environment_destination)
        manifest[f"{environment_name}_sha256"] = sha256_file(environment_destination)
write_json(run_dir / "manifest.partial.json", manifest)

resolved_revision = HfApi().model_info(config["model_id"], revision=config["revision"]).sha
manifest["model_id"] = config["model_id"]
manifest["model_revision_requested"] = config["revision"]
manifest["model_revision_resolved"] = resolved_revision
write_json(run_dir / "manifest.partial.json", manifest)
model, tokenizer = load_model(config["model_id"], resolved_revision, config["attention_backend"])
commit_hash = getattr(model.config, "_commit_hash", None)
if commit_hash != resolved_revision:
    raise RuntimeError(f"model loaded revision {commit_hash}, expected {resolved_revision}")
records = run_control(config, model, tokenizer)
summary = control_summary(
    records,
    config["native_cache_nmse_limit"],
    config["identity_continuation_bound"],
)
write_jsonl(run_dir / "records.jsonl", records)
write_json(run_dir / "summary.json", summary)
manifest["records_sha256"] = sha256_file(run_dir / "records.jsonl")
manifest["summary_sha256"] = sha256_file(run_dir / "summary.json")
manifest["status"] = "complete" if summary["passed"] else "failed_control"
manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
write_json(run_dir / "manifest.json", manifest)
(run_dir / "manifest.partial.json").unlink()
print(run_dir.relative_to(repo_root))
print(json.dumps(summary, indent=2, sort_keys=True))
if summary["failures"]:
    raise RuntimeError(f"control failed with {summary['failure_count']} violations")
