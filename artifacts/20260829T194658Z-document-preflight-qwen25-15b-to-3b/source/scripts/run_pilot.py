import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from functools import partial
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
from kv_research.cache_ops import cache_suffix_nmse, slice_cache
from kv_research.control_evidence import verify_control_evidence
from kv_research.corpus import token_chunks
from kv_research.metrics import (
    entropy,
    jensen_shannon,
    logit_metrics,
    negative_margin,
    negative_max_probability,
    pilot_summary,
    transferred_confidence,
)
from kv_research.models import assert_matching_tokenizers, continue_cache, load_model, prefill
from kv_research.timing import add_policy_timing, measure_ms, summarize_timing
from kv_research.translation import (
    accumulate_ridge_stats,
    init_ridge_stats,
    solve_ridge,
    translate_cache,
    translation_on_device,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def random_score(token_sha256, replay_length):
    digest = hashlib.sha256(f"{token_sha256}:{replay_length}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def chunk_provenance(chunk, partition):
    return {
        "partition": partition,
        "dataset_split": chunk["dataset_split"],
        "chunk_index": chunk["chunk_index"],
        "document_index": chunk["document_index"],
        "document_start_row": chunk["document_start_row"],
        "document_sha256": chunk["document_sha256"],
        "source_rows": chunk["source_rows"],
        "token_sha256": chunk["token_sha256"],
    }


def load_partition_chunks(config, tokenizer, dataset_revision):
    requests = {
        "translator": (
            config["translator_split"],
            config["translator_sequences"],
            config["partition_folds"]["translator"],
        ),
        "calibration": (
            config["calibration_split"],
            config["calibration_sequences"],
            config["partition_folds"]["calibration"],
        ),
        "test": (
            config["test_split"],
            config["test_sequences"],
            config["partition_folds"]["test"],
        ),
    }
    if len({request[2] for request in requests.values()}) != len(requests):
        raise RuntimeError("translator, calibration, and test must use distinct document folds")
    partitions = {}
    for partition, (split, count, fold) in requests.items():
        partitions[partition] = list(
            token_chunks(
                tokenizer,
                config["dataset_id"],
                config["dataset_config"],
                dataset_revision,
                split,
                config["text_field"],
                config["sequence_length"],
                count,
                config["document_fold_modulus"],
                fold,
                config["minimum_source_rows"][partition],
            )
        )
    names = list(partitions)
    for left_index, left_name in enumerate(names):
        left_tokens = {chunk["token_sha256"] for chunk in partitions[left_name]}
        left_rows = {
            (chunk["dataset_split"], row)
            for chunk in partitions[left_name]
            for row in chunk["source_rows"]
        }
        for right_name in names[left_index + 1 :]:
            right_tokens = {chunk["token_sha256"] for chunk in partitions[right_name]}
            right_rows = {
                (chunk["dataset_split"], row)
                for chunk in partitions[right_name]
                for row in chunk["source_rows"]
            }
            if left_tokens & right_tokens:
                raise RuntimeError(f"duplicate token chunks across {left_name} and {right_name}")
            if left_rows & right_rows:
                raise RuntimeError(f"source rows overlap across {left_name} and {right_name}")
    return partitions


@torch.inference_mode()
def fit_translation(config, chunks, source_model, target_model):
    stats = init_ridge_stats(source_model, target_model, config["source_layers_per_target"])
    for index, chunk in enumerate(chunks):
        input_ids = chunk["input_ids"].unsqueeze(0).to("cuda")
        source = prefill(source_model, input_ids)
        target = prefill(target_model, input_ids)
        accumulate_ridge_stats(
            stats, source_model, target_model, source.past_key_values, target.past_key_values
        )
        if (index + 1) % 4 == 0 or index + 1 == len(chunks):
            print(f"translator {index + 1}/{len(chunks)}", flush=True)
    return solve_ridge(stats, config["ridge"], source_model, target_model)


@torch.inference_mode()
def evaluate_chunks(chunks, partition, replay_lengths, source_model, target_model, translation):
    records = []
    for index, chunk in enumerate(chunks):
        input_ids = chunk["input_ids"].unsqueeze(0).to("cuda")
        if input_ids.shape[1] <= max(replay_lengths) + 1:
            raise RuntimeError(
                "sequence is too short for configured replay lengths and anchor token"
            )
        context_ids = input_ids[:, :-1]
        anchor_ids = input_ids[:, -1:]
        source_context = prefill(source_model, context_ids)
        native = prefill(target_model, input_ids)
        translated_context = translate_cache(
            source_context.past_key_values, source_model, target_model, translation
        )
        native_context = slice_cache(
            native.past_key_values, context_ids.shape[1], target_model.config
        )
        oracle_cache_nmse = cache_suffix_nmse(
            translated_context, native_context, 0, context_ids.shape[1]
        )
        full_transfer = continue_cache(target_model, translated_context, anchor_ids)
        native_full = logit_metrics(native.logits[:, -1], full_transfer.logits[:, -1])

        for replay_length in replay_lengths:
            prefix_length = context_ids.shape[1] - replay_length
            source_prefix = slice_cache(
                source_context.past_key_values, prefix_length, source_model.config
            )
            translated_prefix = translate_cache(
                source_prefix, source_model, target_model, translation
            )
            replay = continue_cache(target_model, translated_prefix, input_ids[:, prefix_length:])
            replay_full_js = jensen_shannon(
                replay.logits[:, -1], full_transfer.logits[:, -1]
            ).item()
            native_replay = logit_metrics(native.logits[:, -1], replay.logits[:, -1])
            native_prefix = prefill(target_model, input_ids[:, :prefix_length])
            native_chunked = continue_cache(
                target_model,
                native_prefix.past_key_values,
                input_ids[:, prefix_length:],
            )
            native_chunk = logit_metrics(native.logits[:, -1], native_chunked.logits[:, -1])
            confidence = transferred_confidence(full_transfer.logits[:, -1])
            records.append(
                {
                    "example": chunk_provenance(chunk, partition),
                    "online": {
                        "replay_length": replay_length,
                        "commutator_js": replay_full_js,
                        "commutator_cache_nmse": cache_suffix_nmse(
                            replay.past_key_values,
                            full_transfer.past_key_values,
                            prefix_length,
                            input_ids.shape[1],
                        ),
                        "transfer_entropy": entropy(full_transfer.logits[:, -1]),
                        **confidence,
                        "random_score": random_score(chunk["token_sha256"], replay_length),
                        "full_transfer_target_tokens": 1,
                        "probe_target_tokens": replay_length + 1,
                    },
                    "oracle": {
                        "native_top1": native_full["top1_left"],
                        "full_transfer_top1": native_full["top1_right"],
                        "full_transfer_top1_diff": native_full["top1_diff"],
                        "full_transfer_js": native_full["js"],
                        "native_margin": native_full["left_margin"],
                        "native_chunk_top1_diff": native_chunk["top1_diff"],
                        "native_chunk_js": native_chunk["js"],
                        "native_chunk_cache_nmse": cache_suffix_nmse(
                            native.past_key_values,
                            native_chunked.past_key_values,
                            prefix_length,
                            input_ids.shape[1],
                        ),
                        "replay_top1": native_replay["top1_right"],
                        "replay_top1_diff": native_replay["top1_diff"],
                        "replay_js": native_replay["js"],
                        "oracle_cache_nmse": oracle_cache_nmse,
                    },
                }
            )
        if (index + 1) % 16 == 0 or index + 1 == len(chunks):
            print(f"{partition} {index + 1}/{len(chunks)}", flush=True)
    return records


@torch.inference_mode()
def benchmark_paths(config, chunks, source_model, target_model, translation):
    records = []
    timed_chunks = chunks[: config["timing_examples"]]
    for example_index, chunk in enumerate(timed_chunks):
        input_ids = chunk["input_ids"].unsqueeze(0).to("cuda")
        context_ids = input_ids[:, :-1]
        anchor_ids = input_ids[:, -1:]
        source_context = prefill(source_model, context_ids)

        def native_target_prefill(timed_input_ids=input_ids):
            return prefill(target_model, timed_input_ids)

        def full_transfer_action(
            source_cache=source_context.past_key_values,
            timed_anchor_ids=anchor_ids,
        ):
            translated = translate_cache(source_cache, source_model, target_model, translation)
            return continue_cache(target_model, translated, timed_anchor_ids)

        for path, operation in (
            ("native_target_prefill", native_target_prefill),
            ("full_transfer_action", full_transfer_action),
        ):
            for record in measure_ms(operation, config["timing_warmup"], config["timing_repeats"]):
                record.update(
                    {
                        "path": path,
                        "replay_length": None,
                        "example_index": example_index,
                        "token_sha256": chunk["token_sha256"],
                    }
                )
                records.append(record)

        full_reference = full_transfer_action()

        def transfer_entropy_overhead(timed_full_reference=full_reference):
            logits = timed_full_reference.logits[:, -1].float().reshape(-1)
            log_prob = torch.log_softmax(logits, dim=0)
            return (-torch.sum(log_prob.exp() * log_prob)).item()

        def transfer_margin_overhead(timed_full_reference=full_reference):
            top = torch.topk(timed_full_reference.logits[:, -1].float().reshape(-1), 2).values
            return (-(top[0] - top[1])).item()

        def transfer_max_probability_overhead(timed_full_reference=full_reference):
            logits = timed_full_reference.logits[:, -1].float().reshape(-1)
            return (-torch.softmax(logits, dim=0).max()).item()

        for path, operation in (
            ("transfer_entropy_overhead", transfer_entropy_overhead),
            ("transfer_margin_overhead", transfer_margin_overhead),
            ("transfer_max_probability_overhead", transfer_max_probability_overhead),
        ):
            for record in measure_ms(operation, config["timing_warmup"], config["timing_repeats"]):
                record.update(
                    {
                        "path": path,
                        "replay_length": None,
                        "example_index": example_index,
                        "token_sha256": chunk["token_sha256"],
                    }
                )
                records.append(record)

        for replay_length in config["replay_lengths"]:
            prefix_length = context_ids.shape[1] - replay_length

            def probe_route(
                source_cache=source_context.past_key_values,
                timed_prefix_length=prefix_length,
                timed_input_ids=input_ids,
            ):
                source_prefix = slice_cache(source_cache, timed_prefix_length, source_model.config)
                translated = translate_cache(source_prefix, source_model, target_model, translation)
                return continue_cache(
                    target_model, translated, timed_input_ids[:, timed_prefix_length:]
                )

            for record in measure_ms(
                probe_route, config["timing_warmup"], config["timing_repeats"]
            ):
                record.update(
                    {
                        "path": "probe_route",
                        "replay_length": replay_length,
                        "example_index": example_index,
                        "token_sha256": chunk["token_sha256"],
                    }
                )
                records.append(record)

            probe_reference = probe_route()

            def commutator_js_overhead(
                timed_probe=probe_reference,
                timed_full=full_reference,
            ):
                return jensen_shannon(timed_probe.logits[:, -1], timed_full.logits[:, -1]).item()

            def commutator_cache_overhead(
                timed_probe=probe_reference,
                timed_full=full_reference,
                timed_prefix_length=prefix_length,
                timed_stop=input_ids.shape[1],
            ):
                return cache_suffix_nmse(
                    timed_probe.past_key_values,
                    timed_full.past_key_values,
                    timed_prefix_length,
                    timed_stop,
                )

            for path, operation in (
                ("commutator_js_overhead", commutator_js_overhead),
                ("commutator_cache_nmse_overhead", commutator_cache_overhead),
            ):
                for record in measure_ms(
                    operation, config["timing_warmup"], config["timing_repeats"]
                ):
                    record.update(
                        {
                            "path": path,
                            "replay_length": replay_length,
                            "example_index": example_index,
                            "token_sha256": chunk["token_sha256"],
                        }
                    )
                    records.append(record)
        print(f"timing {example_index + 1}/{len(timed_chunks)}", flush=True)
    return records


@torch.inference_mode()
def policy_action(
    score_name,
    label_name,
    threshold,
    token_sha256,
    replay_length,
    input_ids,
    source_cache,
    source_model,
    target_model,
    translation,
):
    context_length = input_ids.shape[1] - 1
    prefix_length = context_length - replay_length
    needs_full = score_name in (
        "commutator_js",
        "commutator_cache_nmse",
        "transfer_entropy",
        "transfer_negative_margin",
        "transfer_negative_max_probability",
    )
    needs_probe = score_name in ("commutator_js", "commutator_cache_nmse")
    full_result = None
    probe_result = None
    if needs_full:
        translated = translate_cache(source_cache, source_model, target_model, translation)
        full_result = continue_cache(target_model, translated, input_ids[:, -1:])
    if needs_probe:
        source_prefix = slice_cache(source_cache, prefix_length, source_model.config)
        translated_prefix = translate_cache(source_prefix, source_model, target_model, translation)
        probe_result = continue_cache(target_model, translated_prefix, input_ids[:, prefix_length:])
    if score_name == "commutator_js":
        score = jensen_shannon(probe_result.logits[:, -1], full_result.logits[:, -1]).item()
    elif score_name == "commutator_cache_nmse":
        score = cache_suffix_nmse(
            probe_result.past_key_values,
            full_result.past_key_values,
            prefix_length,
            input_ids.shape[1],
        )
    elif score_name == "transfer_entropy":
        score = entropy(full_result.logits[:, -1])
    elif score_name == "transfer_negative_margin":
        score = negative_margin(full_result.logits[:, -1])
    elif score_name == "transfer_negative_max_probability":
        score = negative_max_probability(full_result.logits[:, -1])
    elif score_name == "random_score":
        score = random_score(token_sha256, replay_length)
    else:
        raise RuntimeError(f"unknown policy score {score_name}")
    if score <= threshold:
        if label_name == "full_transfer_top1_diff":
            if full_result is None:
                translated = translate_cache(source_cache, source_model, target_model, translation)
                full_result = continue_cache(target_model, translated, input_ids[:, -1:])
            return full_result
        if label_name != "replay_top1_diff":
            raise RuntimeError(f"unknown policy label {label_name}")
        if probe_result is None:
            source_prefix = slice_cache(source_cache, prefix_length, source_model.config)
            translated_prefix = translate_cache(
                source_prefix, source_model, target_model, translation
            )
            probe_result = continue_cache(
                target_model, translated_prefix, input_ids[:, prefix_length:]
            )
        return probe_result
    return prefill(target_model, input_ids)


@torch.inference_mode()
def benchmark_policy_paths(
    config,
    chunks,
    test_records,
    summary,
    source_model,
    target_model,
    translation,
):
    records = []
    observed = {
        (row["example"]["token_sha256"], row["online"]["replay_length"]): row
        for row in test_records
    }
    for replay_length in config["replay_lengths"]:
        for score_name in summary["score_names"]:
            for label_name in ("full_transfer_top1_diff", "replay_top1_diff"):
                threshold = summary["conditions"][str(replay_length)]["scores"][score_name][
                    label_name
                ]["calibration"]["threshold"]
                accepted_chunks = [
                    chunk
                    for chunk in chunks
                    if observed[(chunk["token_sha256"], replay_length)]["online"][score_name]
                    <= threshold
                ][: config["timing_examples"]]
                rejected_chunks = [
                    chunk
                    for chunk in chunks
                    if observed[(chunk["token_sha256"], replay_length)]["online"][score_name]
                    > threshold
                ][: config["timing_examples"]]
                timed_chunks = [(chunk, True) for chunk in accepted_chunks]
                timed_chunks.extend((chunk, False) for chunk in rejected_chunks)
                for example_index, (chunk, accepted) in enumerate(timed_chunks):
                    input_ids = chunk["input_ids"].unsqueeze(0).to("cuda")
                    source_context = prefill(source_model, input_ids[:, :-1])
                    operation = partial(
                        policy_action,
                        score_name,
                        label_name,
                        threshold,
                        chunk["token_sha256"],
                        replay_length,
                        input_ids,
                        source_context.past_key_values,
                        source_model,
                        target_model,
                        translation,
                    )
                    path = f"policy__{score_name}__{label_name}"
                    for record in measure_ms(
                        operation,
                        config["timing_warmup"],
                        config["timing_repeats"],
                    ):
                        record.update(
                            {
                                "path": path,
                                "replay_length": replay_length,
                                "example_index": example_index,
                                "token_sha256": chunk["token_sha256"],
                                "accepted": accepted,
                            }
                        )
                        records.append(record)
                print(f"policy timing {score_name}/{label_name}/{replay_length}", flush=True)
    return records


args = parse_args()
repo_root = Path(__file__).resolve().parents[1]
config_path = Path(args.config).resolve()
with open(config_path, encoding="utf-8") as stream:
    config = json.load(stream)
for name in ("source_control_run", "target_control_run"):
    if name not in config or not config[name]:
        raise RuntimeError(f"pilot config requires {name}")
primary = config["primary_analysis"]
if primary["replay_length"] not in config["replay_lengths"]:
    raise RuntimeError("primary replay length is absent from replay lengths")
equal_compute_replay = primary["equal_compute_replay_length"]
if equal_compute_replay is not None and equal_compute_replay not in config["replay_lengths"]:
    raise RuntimeError("equal-compute replay length is absent from replay lengths")
if primary["score"] != "commutator_js":
    raise RuntimeError("the frozen primary score must be commutator_js")
if primary["label"] != "full_transfer_top1_diff":
    raise RuntimeError("the frozen primary label must be full_transfer_top1_diff")
if primary["baseline_score"] != "transfer_negative_margin":
    raise RuntimeError("the frozen primary baseline must be transfer_negative_margin")

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

api = HfApi()
source_revision = api.model_info(config["source_model_id"], revision=config["source_revision"]).sha
target_revision = api.model_info(config["target_model_id"], revision=config["target_revision"]).sha
dataset_revision = api.dataset_info(config["dataset_id"], revision=config["dataset_revision"]).sha
manifest.update(
    {
        "source_model_id": config["source_model_id"],
        "source_revision_requested": config["source_revision"],
        "source_revision_resolved": source_revision,
        "target_model_id": config["target_model_id"],
        "target_revision_requested": config["target_revision"],
        "target_revision_resolved": target_revision,
        "dataset_id": config["dataset_id"],
        "dataset_revision_requested": config["dataset_revision"],
        "dataset_revision_resolved": dataset_revision,
    }
)
manifest["controls"] = [
    verify_control_evidence(
        repo_root,
        config["source_control_run"],
        config["source_model_id"],
        source_revision,
        config["attention_backend"],
        config["source_layers_per_target"],
        config["sequence_length"],
        config["replay_lengths"],
        manifest["source_sha256"],
        manifest["environment"],
    ),
    verify_control_evidence(
        repo_root,
        config["target_control_run"],
        config["target_model_id"],
        target_revision,
        config["attention_backend"],
        config["source_layers_per_target"],
        config["sequence_length"],
        config["replay_lengths"],
        manifest["source_sha256"],
        manifest["environment"],
    ),
]
write_json(run_dir / "manifest.partial.json", manifest)

source_model, source_tokenizer = load_model(
    config["source_model_id"], source_revision, config["attention_backend"]
)
target_model, target_tokenizer = load_model(
    config["target_model_id"], target_revision, config["attention_backend"]
)
for name, model, resolved in (
    ("source", source_model, source_revision),
    ("target", target_model, target_revision),
):
    commit_hash = getattr(model.config, "_commit_hash", None)
    if commit_hash != resolved:
        raise RuntimeError(f"{name} model loaded revision {commit_hash}, expected {resolved}")
manifest["tokenizer_sha256"] = assert_matching_tokenizers(source_tokenizer, target_tokenizer)

partitions = load_partition_chunks(config, source_tokenizer, dataset_revision)
data_manifest = [
    chunk_provenance(chunk, partition)
    for partition, chunks in partitions.items()
    for chunk in chunks
]
write_json(run_dir / "data.json", data_manifest)
manifest["data_sha256"] = sha256_file(run_dir / "data.json")
write_json(run_dir / "manifest.partial.json", manifest)

translation = fit_translation(config, partitions["translator"], source_model, target_model)
translator_path = run_dir / "translator.pt"
torch.save(translation, translator_path)
translator_report = {
    name: value for name, value in translation.items() if name not in ("key_maps", "value_maps")
}
write_json(run_dir / "translator.json", translator_report)
manifest["translator_sha256"] = sha256_file(translator_path)
manifest["translator_report_sha256"] = sha256_file(run_dir / "translator.json")
write_json(run_dir / "manifest.partial.json", manifest)
runtime_translation = translation_on_device(translation, next(target_model.parameters()).device)

calibration_records = evaluate_chunks(
    partitions["calibration"],
    "calibration",
    config["replay_lengths"],
    source_model,
    target_model,
    runtime_translation,
)
test_records = evaluate_chunks(
    partitions["test"],
    "test",
    config["replay_lengths"],
    source_model,
    target_model,
    runtime_translation,
)
records = calibration_records + test_records
score_names = [
    "commutator_js",
    "commutator_cache_nmse",
    "transfer_negative_margin",
    "transfer_negative_max_probability",
    "transfer_entropy",
    "random_score",
]
summary = pilot_summary(
    calibration_records,
    test_records,
    config["replay_lengths"],
    score_names,
    risk_limit=config["risk_limit"],
    target_coverage=config["target_coverage"],
    bootstrap_repetitions=config["bootstrap_repetitions"],
    bootstrap_seed=config["bootstrap_seed"],
    native_cache_nmse_limit=config["native_cache_nmse_limit"],
)
summary["primary_analysis"] = config["primary_analysis"]
if not summary["native_controls_passed"]:
    write_jsonl(run_dir / "records.jsonl", records)
    write_json(run_dir / "summary.json", summary)
    manifest["records_sha256"] = sha256_file(run_dir / "records.jsonl")
    manifest["summary_sha256"] = sha256_file(run_dir / "summary.json")
    manifest["status"] = "invalid_native_control"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "manifest.partial.json").unlink()
    print(run_dir)
    raise RuntimeError("per-example native chunking control failed; timing was skipped")
timing_records = benchmark_paths(
    config, partitions["test"], source_model, target_model, runtime_translation
)
timing_records.extend(
    benchmark_policy_paths(
        config,
        partitions["test"],
        test_records,
        summary,
        source_model,
        target_model,
        runtime_translation,
    )
)
timing = summarize_timing(timing_records)
add_policy_timing(summary, timing, config["sequence_length"])

write_jsonl(run_dir / "records.jsonl", records)
write_jsonl(run_dir / "timing.jsonl", timing_records)
write_json(run_dir / "summary.json", summary)
manifest["records_sha256"] = sha256_file(run_dir / "records.jsonl")
manifest["timing_sha256"] = sha256_file(run_dir / "timing.jsonl")
manifest["summary_sha256"] = sha256_file(run_dir / "summary.json")
manifest["status"] = "complete"
manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
write_json(run_dir / "manifest.json", manifest)
(run_dir / "manifest.partial.json").unlink()
print(run_dir)
console_summary = {"native_controls_passed": summary["native_controls_passed"], "conditions": {}}
for replay_length, condition in summary["conditions"].items():
    commutator = condition["scores"]["commutator_js"]["full_transfer_top1_diff"]
    console_summary["conditions"][replay_length] = {
        "full_transfer_risk": condition["full_transfer_risk"],
        "unconditional_replay_risk": condition["replay_repair_risk"],
        "commutator_average_precision": commutator["test_ranking"]["harm_average_precision"],
        "fixed_coverage_test": commutator["test_at_calibrated_threshold"],
        "complete_policy_timing": commutator["test_policy_timing"]["complete_policy_measurement"],
    }
print(json.dumps(console_summary, indent=2, sort_keys=True))
