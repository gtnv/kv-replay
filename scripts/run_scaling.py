import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from kv_research.artifacts import (
    base_manifest,
    create_run_dir,
    sha256_file,
    snapshot_source,
    write_json,
    write_jsonl,
)
from kv_research.cache_ops import cache_suffix_nmse, slice_cache
from kv_research.control_evidence import verify_control_evidence
from kv_research.corpus import token_chunks
from kv_research.metrics import clopper_pearson_upper, jensen_shannon, logit_metrics
from kv_research.models import assert_matching_tokenizers, continue_cache, load_model, prefill
from kv_research.timing import measure_ms, paired_timing_summary, summarize_timing
from kv_research.translation import translate_cache, translation_on_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def path_record(path, length, example_index, token_sha256, window_token_sha256, measured):
    for row in measured:
        row.update(
            {
                "path": f"{path}:{length}",
                "replay_length": None,
                "sequence_length": length,
                "example_index": example_index,
                "token_sha256": token_sha256,
                "window_token_sha256": window_token_sha256,
            }
        )
    return measured


args = parse_args()
repo_root = Path(__file__).resolve().parents[1]
config_path = Path(args.config).resolve()
config = read_json(config_path)
for key in ("source_long_control_run", "target_long_control_run"):
    if key not in config:
        raise RuntimeError(f"scaling config must bind {key} before creating a run")
torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])

parent_dir = (repo_root / config["main_run"]).resolve()
if repo_root.resolve() not in parent_dir.parents:
    raise RuntimeError("main run path escapes repository")
parent_manifest = read_json(parent_dir / "manifest.json")
parent_summary = read_json(parent_dir / "summary.json")
if parent_manifest["status"] != "complete" or not parent_summary["primary_native_control_passed"]:
    raise RuntimeError("main run is not complete with a valid primary control")
for key, name in (
    ("records_sha256", "records.jsonl"),
    ("summary_sha256", "summary.json"),
    ("translator_sha256", "translator.pt"),
):
    if sha256_file(parent_dir / name) != parent_manifest[key]:
        raise RuntimeError(f"main run {name} hash mismatch")
parent_config_path = parent_dir / "source" / parent_manifest["config_relative"]
if sha256_file(parent_config_path) != parent_manifest["config_sha256"]:
    raise RuntimeError("main run config hash mismatch")
parent_config = read_json(parent_config_path)

run_dir = create_run_dir(repo_root, config["run_name"])
manifest = base_manifest(repo_root, config_path)
snapshot_source(repo_root, run_dir, manifest["source_sha256"])
for environment_name in ("bootstrap.json", "pip-freeze.txt"):
    source = repo_root / "artifacts" / "environment" / environment_name
    if source.is_file():
        destination = run_dir / "environment" / environment_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest[f"{environment_name}_sha256"] = sha256_file(destination)
manifest["parent_run"] = {
    "run": config["main_run"],
    "manifest_sha256": sha256_file(parent_dir / "manifest.json"),
    "summary_sha256": parent_manifest["summary_sha256"],
    "translator_sha256": parent_manifest["translator_sha256"],
}
translation_layers = parent_config["source_layers_per_target"]
manifest["long_controls"] = [
    verify_control_evidence(
        repo_root,
        config["source_long_control_run"],
        parent_manifest["source_model_id"],
        parent_manifest["source_revision_resolved"],
        "sdpa",
        translation_layers,
        max(config["sequence_lengths"]),
        [config["replay_length"], config["replay_length"] + 1],
        manifest["source_sha256"],
        manifest["environment"],
    ),
    verify_control_evidence(
        repo_root,
        config["target_long_control_run"],
        parent_manifest["target_model_id"],
        parent_manifest["target_revision_resolved"],
        "sdpa",
        translation_layers,
        max(config["sequence_lengths"]),
        [config["replay_length"], config["replay_length"] + 1],
        manifest["source_sha256"],
        manifest["environment"],
    ),
]
write_json(run_dir / "manifest.partial.json", manifest)

source_model, source_tokenizer = load_model(
    parent_manifest["source_model_id"],
    parent_manifest["source_revision_resolved"],
    "sdpa",
)
target_model, target_tokenizer = load_model(
    parent_manifest["target_model_id"],
    parent_manifest["target_revision_resolved"],
    "sdpa",
)
manifest["tokenizer_sha256"] = assert_matching_tokenizers(source_tokenizer, target_tokenizer)
translation = torch.load(parent_dir / "translator.pt", map_location="cpu", weights_only=True)
runtime_translation = translation_on_device(translation, next(target_model.parameters()).device)

maximum_length = max(config["sequence_lengths"])
chunks = list(
    token_chunks(
        source_tokenizer,
        parent_config["dataset_id"],
        parent_config["dataset_config"],
        parent_manifest["dataset_revision_resolved"],
        config["dataset_split"],
        parent_config["text_field"],
        maximum_length,
        config["sequences"],
        1,
        0,
        0,
    )
)
data = [
    {
        "dataset_split": chunk["dataset_split"],
        "document_index": chunk["document_index"],
        "document_start_row": chunk["document_start_row"],
        "document_sha256": chunk["document_sha256"],
        "source_rows": chunk["source_rows"],
        "token_sha256": chunk["token_sha256"],
        "window_token_sha256": {
            str(length): hashlib.sha256(chunk["input_ids"][-length:].numpy().tobytes()).hexdigest()
            for length in config["sequence_lengths"]
        },
    }
    for chunk in chunks
]
write_json(run_dir / "data.json", data)
manifest["data_sha256"] = sha256_file(run_dir / "data.json")

replay_length = config["replay_length"]
threshold = parent_summary["conditions"][str(replay_length)]["scores"]["commutator_js"][
    "full_transfer_top1_diff"
]["calibration"]["threshold"]
records = []
timing_records = []
for example_index, chunk in enumerate(chunks):
    length_order = config["sequence_lengths"]
    if example_index % 2:
        length_order = list(reversed(length_order))
    for length in length_order:
        input_ids = chunk["input_ids"][-length:].unsqueeze(0).to("cuda")
        window_token_sha256 = data[example_index]["window_token_sha256"][str(length)]
        context_ids = input_ids[:, :-1]
        anchor_ids = input_ids[:, -1:]
        prefix_length = context_ids.shape[1] - replay_length
        source_context = prefill(source_model, context_ids)

        def native_prefill(timed_ids=input_ids):
            return prefill(target_model, timed_ids)

        def full_action(
            source_cache=source_context.past_key_values,
            timed_anchor=anchor_ids,
        ):
            translated = translate_cache(
                source_cache, source_model, target_model, runtime_translation
            )
            return continue_cache(target_model, translated, timed_anchor)

        def replay_action(
            replay_tokens,
            source_cache=source_context.past_key_values,
            timed_ids=input_ids,
            timed_context_length=context_ids.shape[1],
        ):
            replay_prefix = timed_context_length - replay_tokens
            source_prefix = slice_cache(source_cache, replay_prefix, source_model.config)
            translated = translate_cache(
                source_prefix, source_model, target_model, runtime_translation
            )
            return continue_cache(target_model, translated, timed_ids[:, replay_prefix:])

        def gate_action(
            source_cache=source_context.past_key_values,
            timed_ids=input_ids,
            timed_anchor=anchor_ids,
            timed_prefix=prefix_length,
        ):
            translated = translate_cache(
                source_cache, source_model, target_model, runtime_translation
            )
            full = continue_cache(target_model, translated, timed_anchor)
            source_prefix = slice_cache(source_cache, timed_prefix, source_model.config)
            translated_prefix = translate_cache(
                source_prefix, source_model, target_model, runtime_translation
            )
            probe = continue_cache(target_model, translated_prefix, timed_ids[:, timed_prefix:])
            score = jensen_shannon(probe.logits[:, -1], full.logits[:, -1]).item()
            if score <= threshold:
                return full
            return prefill(target_model, timed_ids)

        operations = [
            ("native_target_prefill", native_prefill),
            ("full_transfer_action", full_action),
            (f"unconditional_replay_{replay_length}", lambda: replay_action(replay_length)),
            (
                f"unconditional_replay_{replay_length + 1}",
                lambda: replay_action(replay_length + 1),
            ),
            ("commutator_gate", gate_action),
        ]
        offset = (example_index + config["sequence_lengths"].index(length)) % len(operations)
        operations = operations[offset:] + operations[:offset]
        for path, operation in operations:
            timing_records.extend(
                path_record(
                    path,
                    length,
                    example_index,
                    chunk["token_sha256"],
                    window_token_sha256,
                    measure_ms(
                        operation,
                        config["timing_warmup"],
                        config["timing_repeats"],
                    ),
                )
            )

        native = native_prefill()
        full = full_action()
        replay64 = replay_action(replay_length)
        replay65 = replay_action(replay_length + 1)
        source_prefix = slice_cache(
            source_context.past_key_values, prefix_length, source_model.config
        )
        translated_prefix = translate_cache(
            source_prefix, source_model, target_model, runtime_translation
        )
        probe = continue_cache(target_model, translated_prefix, input_ids[:, prefix_length:])
        score = jensen_shannon(probe.logits[:, -1], full.logits[:, -1]).item()
        accepted = score <= threshold
        native_prefix = prefill(target_model, input_ids[:, :prefix_length])
        native_chunk = continue_cache(
            target_model, native_prefix.past_key_values, input_ids[:, prefix_length:]
        )
        records.append(
            {
                "length": length,
                "example_index": example_index,
                "token_sha256": chunk["token_sha256"],
                "window_token_sha256": window_token_sha256,
                "score": score,
                "accepted": accepted,
                "full_transfer": logit_metrics(native.logits[:, -1], full.logits[:, -1]),
                "replay_64": logit_metrics(native.logits[:, -1], replay64.logits[:, -1]),
                "replay_65": logit_metrics(native.logits[:, -1], replay65.logits[:, -1]),
                "native_chunk": {
                    **logit_metrics(native.logits[:, -1], native_chunk.logits[:, -1]),
                    "cache_nmse": cache_suffix_nmse(
                        native.past_key_values,
                        native_chunk.past_key_values,
                        prefix_length,
                        length,
                    ),
                },
            }
        )
    print(f"scaling document {example_index + 1}/{len(chunks)} complete", flush=True)

timing = summarize_timing(timing_records)
summary = {"threshold": threshold, "conditions": {}, "timing": timing}
for length in config["sequence_lengths"]:
    rows = [row for row in records if row["length"] == length]
    accepted = [row for row in rows if row["accepted"]]
    accepted_errors = sum(row["full_transfer"]["top1_diff"] for row in accepted)
    native_chunk_max = max(row["native_chunk"]["cache_nmse"] for row in rows)
    summary["conditions"][str(length)] = {
        "count": len(rows),
        "full_transfer_risk": sum(row["full_transfer"]["top1_diff"] for row in rows) / len(rows),
        "unconditional_replay_64_risk": sum(row["replay_64"]["top1_diff"] for row in rows)
        / len(rows),
        "unconditional_replay_65_risk": sum(row["replay_65"]["top1_diff"] for row in rows)
        / len(rows),
        "gate_coverage": len(accepted) / len(rows),
        "gate_conditional_risk": accepted_errors / len(accepted) if accepted else None,
        "gate_conditional_clopper_pearson_upper_95": clopper_pearson_upper(
            accepted_errors, len(accepted)
        )
        if accepted
        else None,
        "gate_population_risk_with_native_fallback": accepted_errors / len(rows),
        "native_chunk_top1_risk": sum(row["native_chunk"]["top1_diff"] for row in rows) / len(rows),
        "native_chunk_max_cache_nmse": native_chunk_max,
        "native_control_passed": native_chunk_max <= parent_config["native_cache_nmse_limit"],
        "native_cache_nmse_limit": parent_config["native_cache_nmse_limit"],
        "latency": paired_timing_summary(
            timing_records,
            f"commutator_gate:{length}",
            f"native_target_prefill:{length}",
            length,
            config["bootstrap_repetitions"],
            config["bootstrap_seed"] + length,
        ),
    }

all_native_controls_passed = all(
    condition["native_control_passed"] for condition in summary["conditions"].values()
)
summary["all_native_controls_passed"] = all_native_controls_passed
write_jsonl(run_dir / "records.jsonl", records)
write_jsonl(run_dir / "timing.jsonl", timing_records)
write_json(run_dir / "summary.json", summary)
manifest["records_sha256"] = sha256_file(run_dir / "records.jsonl")
manifest["timing_sha256"] = sha256_file(run_dir / "timing.jsonl")
manifest["summary_sha256"] = sha256_file(run_dir / "summary.json")
manifest["status"] = "complete" if all_native_controls_passed else "invalid_native_control"
manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
write_json(run_dir / "manifest.json", manifest)
(run_dir / "manifest.partial.json").unlink()
print(run_dir.relative_to(repo_root))
print(json.dumps(summary["conditions"], indent=2, sort_keys=True))
if not all_native_controls_passed:
    raise RuntimeError("one or more scaling lengths failed the native cache control")
