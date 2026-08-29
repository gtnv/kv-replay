import argparse
import json
from pathlib import Path

import torch

from kv_research.artifacts import read_jsonl, sha256_file
from kv_research.control_evidence import verify_control_evidence
from kv_research.metrics import control_summary, pilot_summary
from kv_research.timing import add_policy_timing, summarize_timing
from kv_research.translation import aligned_source_layers, source_layer_groups


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


args = parse_args()
run_dir = Path(args.run).resolve()
manifest = read_json(run_dir / "manifest.json")
saved_summary = read_json(run_dir / "summary.json")

for relative, expected in manifest["source_sha256"].items():
    snapshot = run_dir / "source" / relative
    if not snapshot.is_file() or sha256_file(snapshot) != expected:
        raise RuntimeError(f"source snapshot hash mismatch for {relative}")
config_relative = manifest["config_relative"]
if config_relative is None:
    raise RuntimeError("run config is outside the snapshotted repository")
config_path = run_dir / "source" / config_relative
if sha256_file(config_path) != manifest["config_sha256"]:
    raise RuntimeError("config hash does not match manifest")
config = read_json(config_path)

hashed_files = {
    "records_sha256": "records.jsonl",
    "summary_sha256": "summary.json",
    "data_sha256": "data.json",
    "timing_sha256": "timing.jsonl",
    "translator_sha256": "translator.pt",
    "translator_report_sha256": "translator.json",
    "bootstrap.json_sha256": "environment/bootstrap.json",
    "pip-freeze.txt_sha256": "environment/pip-freeze.txt",
}
pilot_config = "source_model_id" in config
required_hashes = {"records_sha256", "summary_sha256"}
if pilot_config:
    required_hashes.update({"data_sha256", "translator_sha256", "translator_report_sha256"})
    if manifest["status"] == "complete":
        required_hashes.add("timing_sha256")
    elif manifest["status"] != "invalid_native_control":
        raise RuntimeError(f"unexpected pilot status {manifest['status']}")
elif manifest["status"] != "complete":
    raise RuntimeError(f"unexpected control status {manifest['status']}")
for manifest_key in required_hashes:
    if manifest_key not in manifest:
        raise RuntimeError(f"manifest is missing required hash {manifest_key}")
    relative = hashed_files[manifest_key]
    if sha256_file(run_dir / relative) != manifest[manifest_key]:
        raise RuntimeError(f"{relative} hash does not match manifest")
for manifest_key in ("bootstrap.json_sha256", "pip-freeze.txt_sha256"):
    if manifest_key in manifest:
        relative = hashed_files[manifest_key]
        if sha256_file(run_dir / relative) != manifest[manifest_key]:
            raise RuntimeError(f"{relative} hash does not match manifest")

records = read_jsonl(run_dir / "records.jsonl")
if not records:
    raise RuntimeError("records.jsonl is empty")

if "online" in records[0]:
    controls = [
        verify_control_evidence(
            run_dir.parents[1],
            config["source_control_run"],
            config["source_model_id"],
            manifest["source_revision_resolved"],
            config["attention_backend"],
            config["source_layers_per_target"],
            manifest["source_sha256"],
            manifest["environment"],
        ),
        verify_control_evidence(
            run_dir.parents[1],
            config["target_control_run"],
            config["target_model_id"],
            manifest["target_revision_resolved"],
            config["attention_backend"],
            config["source_layers_per_target"],
            manifest["source_sha256"],
            manifest["environment"],
        ),
    ]
    if controls != manifest["controls"]:
        raise RuntimeError("pilot control evidence differs from manifest")
    data = read_json(run_dir / "data.json")
    expected_counts = {
        "translator": config["translator_sequences"],
        "calibration": config["calibration_sequences"],
        "test": config["test_sequences"],
    }
    expected_splits = {
        "translator": config["translator_split"],
        "calibration": config["calibration_split"],
        "test": config["test_split"],
    }
    data_keys = set()
    document_hashes = set()
    row_sets = {}
    for partition, count in expected_counts.items():
        rows = [row for row in data if row["partition"] == partition]
        if len(rows) != count:
            raise RuntimeError(f"{partition} data count is {len(rows)}, expected {count}")
        if {row["dataset_split"] for row in rows} != {expected_splits[partition]}:
            raise RuntimeError(f"{partition} uses an unexpected dataset split")
        row_sets[partition] = {
            (row["dataset_split"], source_row) for row in rows for source_row in row["source_rows"]
        }
        for row in rows:
            key = (partition, row["token_sha256"])
            if key in data_keys:
                raise RuntimeError(f"duplicate data key {key}")
            data_keys.add(key)
            document_hash = row["document_sha256"]
            if document_hash in document_hashes:
                raise RuntimeError(f"document reused across data partitions: {document_hash}")
            document_hashes.add(document_hash)
            observed_fold = int(document_hash[:16], 16) % config["document_fold_modulus"]
            if observed_fold != config["partition_folds"][partition]:
                raise RuntimeError(f"{partition} contains document from fold {observed_fold}")
            if row["document_start_row"] < config["minimum_source_rows"][partition]:
                raise RuntimeError(f"{partition} contains a document before its row cutoff")
        row_references = [
            (row["dataset_split"], source_row) for row in rows for source_row in row["source_rows"]
        ]
        if len(row_references) != len(set(row_references)):
            raise RuntimeError(f"source rows repeat within {partition}")
    partitions = list(expected_counts)
    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1 :]:
            if row_sets[left] & row_sets[right]:
                raise RuntimeError(f"source rows overlap across {left} and {right}")
    token_hashes = [row["token_sha256"] for row in data]
    if len(token_hashes) != len(set(token_hashes)):
        raise RuntimeError("token chunks overlap across data partitions")

    replay_lengths = config["replay_lengths"]
    expected_record_count = (config["calibration_sequences"] + config["test_sequences"]) * len(
        replay_lengths
    )
    if len(records) != expected_record_count:
        raise RuntimeError(f"pilot has {len(records)} records, expected {expected_record_count}")
    record_keys = set()
    for record in records:
        example = record["example"]
        key = (
            example["partition"],
            example["token_sha256"],
            record["online"]["replay_length"],
        )
        if key in record_keys:
            raise RuntimeError(f"duplicate evaluation record {key}")
        if (example["partition"], example["token_sha256"]) not in data_keys:
            raise RuntimeError(f"evaluation record is absent from data manifest: {key}")
        record_keys.add(key)
    expected_record_keys = {
        (row["partition"], row["token_sha256"], replay_length)
        for row in data
        if row["partition"] in ("calibration", "test")
        for replay_length in replay_lengths
    }
    if record_keys != expected_record_keys:
        raise RuntimeError("evaluation records are not the expected example/replay product")

    translator = torch.load(run_dir / "translator.pt", map_location="cpu", weights_only=True)
    translator_report = read_json(run_dir / "translator.json")
    report_from_tensor = {
        name: value for name, value in translator.items() if name not in ("key_maps", "value_maps")
    }
    if report_from_tensor != translator_report:
        raise RuntimeError("translator report differs from tensor artifact metadata")
    expected_map_shape = (
        translator["target_layers"],
        translator["source_width"] + 1,
        translator["target_width"],
    )
    for name in ("key_maps", "value_maps"):
        maps = translator[name]
        if tuple(maps.shape) != expected_map_shape:
            raise RuntimeError(f"{name} has unexpected shape {tuple(maps.shape)}")
        if not torch.isfinite(maps).all():
            raise RuntimeError(f"{name} contains non-finite values")
    if translator["translator_schema"] != 2:
        raise RuntimeError("translator schema is not 2")
    if translator["source_layers_per_target"] != config["source_layers_per_target"]:
        raise RuntimeError("translator layer count differs from config")
    if (
        translator["source_width"]
        != translator["source_width_per_layer"] * translator["source_layers_per_target"]
    ):
        raise RuntimeError("translator source feature width is inconsistent")
    expected_layer_map = aligned_source_layers(
        translator["source_layers"], translator["target_layers"]
    )
    expected_groups = source_layer_groups(
        translator["source_layers"],
        translator["target_layers"],
        translator["source_layers_per_target"],
    )
    if translator["layer_map"] != expected_layer_map:
        raise RuntimeError("translator layer map is inconsistent")
    if translator["layer_groups"] != expected_groups:
        raise RuntimeError("translator layer groups are inconsistent")
    if translator["layer_group_policy"] != "contiguous_anchor_deeper_tie_v1":
        raise RuntimeError("translator layer group policy is unknown")
    if len(translator["diagnostics"]) != translator["target_layers"]:
        raise RuntimeError("translator diagnostics do not cover every target layer")
    expected_samples = config["translator_sequences"] * config["sequence_length"]
    if translator["samples"] != expected_samples:
        raise RuntimeError(
            f"translator has {translator['samples']} samples, expected {expected_samples}"
        )
    if translator["samples"] <= translator["source_width"]:
        raise RuntimeError("translator regression is underdetermined")

    calibration = [row for row in records if row["example"]["partition"] == "calibration"]
    test = [row for row in records if row["example"]["partition"] == "test"]
    expected_score_names = [
        "commutator_js",
        "commutator_cache_nmse",
        "transfer_negative_margin",
        "transfer_negative_max_probability",
        "transfer_entropy",
        "random_score",
    ]
    if saved_summary["score_names"] != expected_score_names:
        raise RuntimeError("pilot score schema differs from the required baselines")
    recomputed = pilot_summary(
        calibration,
        test,
        replay_lengths,
        expected_score_names,
        risk_limit=config["risk_limit"],
        target_coverage=config["target_coverage"],
        bootstrap_repetitions=config["bootstrap_repetitions"],
        bootstrap_seed=config["bootstrap_seed"],
    )
    recomputed["primary_analysis"] = config["primary_analysis"]
    if manifest["status"] == "complete":
        timing_records = read_jsonl(run_dir / "timing.jsonl")
        test_data = [row for row in data if row["partition"] == "test"]
        timed_test_data = test_data[: config["timing_examples"]]
        expected_timing_keys = set()
        for path in (
            "native_target_prefill",
            "full_transfer_action",
            "transfer_entropy_overhead",
            "transfer_margin_overhead",
            "transfer_max_probability_overhead",
        ):
            for example_index, row in enumerate(timed_test_data):
                for iteration in range(config["timing_repeats"]):
                    expected_timing_keys.add(
                        (path, None, example_index, iteration, row["token_sha256"], None)
                    )
        for replay_length in replay_lengths:
            for path in (
                "probe_route",
                "commutator_js_overhead",
                "commutator_cache_nmse_overhead",
            ):
                for example_index, row in enumerate(timed_test_data):
                    for iteration in range(config["timing_repeats"]):
                        expected_timing_keys.add(
                            (
                                path,
                                replay_length,
                                example_index,
                                iteration,
                                row["token_sha256"],
                                None,
                            )
                        )
            replay_records = {
                row["example"]["token_sha256"]: row
                for row in test
                if row["online"]["replay_length"] == replay_length
            }
            for score_name in recomputed["score_names"]:
                for label_name in ("full_transfer_top1_diff", "replay_top1_diff"):
                    threshold = recomputed["conditions"][str(replay_length)]["scores"][score_name][
                        label_name
                    ]["calibration"]["threshold"]
                    accepted = [
                        row
                        for row in test_data
                        if replay_records[row["token_sha256"]]["online"][score_name] <= threshold
                    ][: config["timing_examples"]]
                    rejected = [
                        row
                        for row in test_data
                        if replay_records[row["token_sha256"]]["online"][score_name] > threshold
                    ][: config["timing_examples"]]
                    timed_policy_data = [(row, True) for row in accepted]
                    timed_policy_data.extend((row, False) for row in rejected)
                    path = f"policy__{score_name}__{label_name}"
                    for example_index, (row, accepted_value) in enumerate(timed_policy_data):
                        for iteration in range(config["timing_repeats"]):
                            expected_timing_keys.add(
                                (
                                    path,
                                    replay_length,
                                    example_index,
                                    iteration,
                                    row["token_sha256"],
                                    accepted_value,
                                )
                            )
        timing_keys = {
            (
                row["path"],
                row["replay_length"],
                row["example_index"],
                row["iteration"],
                row["token_sha256"],
                row.get("accepted"),
            )
            for row in timing_records
        }
        if len(timing_keys) != len(timing_records):
            raise RuntimeError("timing records contain duplicate path/example/iteration keys")
        if timing_keys != expected_timing_keys:
            raise RuntimeError("timing records differ from the exact expected path/sample product")
        add_policy_timing(
            recomputed,
            summarize_timing(timing_records),
            config["sequence_length"],
        )
    elif (run_dir / "timing.jsonl").exists():
        raise RuntimeError("invalid-native-control run unexpectedly contains timing")
else:
    expected_record_count = len(config["lengths"]) * len(config["replay_lengths"])
    if len(records) != expected_record_count:
        raise RuntimeError(f"control has {len(records)} records, expected {expected_record_count}")
    control_keys = {(row["length"], row["replay_length"]) for row in records}
    if len(control_keys) != len(records):
        raise RuntimeError("control records contain duplicate length/replay keys")
    expected_control_keys = {
        (length, replay_length)
        for length in config["lengths"]
        for replay_length in config["replay_lengths"]
        if replay_length < length
    }
    if control_keys != expected_control_keys:
        raise RuntimeError("control records are not the expected length/replay product")
    expected_identity_keys = {str(value) for value in config["identity_layers_per_target"]}
    for record in records:
        if set(record["identity_translation"]) != expected_identity_keys:
            raise RuntimeError("control identity regimes differ from config")
    recomputed = control_summary(records, config["native_cache_nmse_limit"])

if recomputed != saved_summary:
    raise RuntimeError("recomputed summary differs from saved summary")

print(
    json.dumps(
        {
            "run": str(run_dir),
            "records": len(records),
            "records_sha256": manifest["records_sha256"],
            "summary_sha256": manifest["summary_sha256"],
            "recomputed": True,
            "source_files_verified": len(manifest["source_sha256"]),
        },
        indent=2,
        sort_keys=True,
    )
)
