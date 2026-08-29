import argparse
import json
from pathlib import Path

from kv_research.artifacts import read_jsonl, sha256_file
from kv_research.control_evidence import verify_control_evidence
from kv_research.metrics import clopper_pearson_upper
from kv_research.timing import paired_timing_summary, summarize_timing


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
if manifest["status"] not in ("complete", "invalid_native_control"):
    raise RuntimeError(f"unexpected scaling status {manifest['status']}")
config = read_json(run_dir / "source" / manifest["config_relative"])
saved = read_json(run_dir / "summary.json")
if sha256_file(run_dir / "source" / manifest["config_relative"]) != manifest["config_sha256"]:
    raise RuntimeError("scaling config hash mismatch")
for relative, expected in manifest["source_sha256"].items():
    if sha256_file(run_dir / "source" / relative) != expected:
        raise RuntimeError(f"source snapshot mismatch: {relative}")
for key, name in (
    ("records_sha256", "records.jsonl"),
    ("timing_sha256", "timing.jsonl"),
    ("summary_sha256", "summary.json"),
    ("data_sha256", "data.json"),
):
    if sha256_file(run_dir / name) != manifest[key]:
        raise RuntimeError(f"artifact hash mismatch: {name}")
parent = run_dir.parents[1] / manifest["parent_run"]["run"]
if manifest["parent_run"]["run"] != config["main_run"]:
    raise RuntimeError("manifest parent differs from scaling config")
for key, name in (
    ("manifest_sha256", "manifest.json"),
    ("summary_sha256", "summary.json"),
    ("translator_sha256", "translator.pt"),
):
    if sha256_file(parent / name) != manifest["parent_run"][key]:
        raise RuntimeError(f"parent artifact hash mismatch: {name}")
parent_summary = read_json(parent / "summary.json")
parent_manifest = read_json(parent / "manifest.json")
if parent_manifest["status"] != "complete" or not parent_summary["primary_native_control_passed"]:
    raise RuntimeError("parent run is not complete with a valid primary control")
parent_config_path = parent / "source" / parent_manifest["config_relative"]
if sha256_file(parent_config_path) != parent_manifest["config_sha256"]:
    raise RuntimeError("parent config hash mismatch")
parent_config = read_json(parent_config_path)
expected_threshold = parent_summary["conditions"][str(config["replay_length"])]["scores"][
    "commutator_js"
]["full_transfer_top1_diff"]["calibration"]["threshold"]
if saved["threshold"] != expected_threshold:
    raise RuntimeError("scaling threshold differs from the frozen parent calibration")
controls = [
    verify_control_evidence(
        run_dir.parents[1],
        config["source_long_control_run"],
        parent_manifest["source_model_id"],
        parent_manifest["source_revision_resolved"],
        "sdpa",
        parent_config["source_layers_per_target"],
        max(config["sequence_lengths"]),
        [config["replay_length"], config["replay_length"] + 1],
        manifest["source_sha256"],
        manifest["environment"],
    ),
    verify_control_evidence(
        run_dir.parents[1],
        config["target_long_control_run"],
        parent_manifest["target_model_id"],
        parent_manifest["target_revision_resolved"],
        "sdpa",
        parent_config["source_layers_per_target"],
        max(config["sequence_lengths"]),
        [config["replay_length"], config["replay_length"] + 1],
        manifest["source_sha256"],
        manifest["environment"],
    ),
]
if controls != manifest["long_controls"]:
    raise RuntimeError("scaling long-control evidence differs from manifest")

records = read_jsonl(run_dir / "records.jsonl")
timing_records = read_jsonl(run_dir / "timing.jsonl")
data = read_json(run_dir / "data.json")
if len(data) != config["sequences"]:
    raise RuntimeError("scaling data count differs from config")
if len({row["document_sha256"] for row in data}) != len(data):
    raise RuntimeError("scaling data reuses a document")
if {row["dataset_split"] for row in data} != {config["dataset_split"]}:
    raise RuntimeError("scaling data uses the wrong split")
token_by_index = {index: row["token_sha256"] for index, row in enumerate(data)}
window_by_index = {index: row["window_token_sha256"] for index, row in enumerate(data)}
expected_record_keys = {
    (length, example_index)
    for length in config["sequence_lengths"]
    for example_index in range(config["sequences"])
}
record_keys = {(row["length"], row["example_index"]) for row in records}
if record_keys != expected_record_keys or len(record_keys) != len(records):
    raise RuntimeError("scaling records are not the expected length/example product")
for row in records:
    if row["token_sha256"] != token_by_index[row["example_index"]]:
        raise RuntimeError("scaling record token hash differs from data manifest")
    if row["window_token_sha256"] != window_by_index[row["example_index"]][str(row["length"])]:
        raise RuntimeError("scaling record window hash differs from data manifest")
    if row["accepted"] != (row["score"] <= expected_threshold):
        raise RuntimeError("scaling acceptance differs from frozen threshold")
expected_timing_keys = {
    (f"{path}:{length}", None, length, example_index, iteration)
    for path in (
        "native_target_prefill",
        "full_transfer_action",
        f"unconditional_replay_{config['replay_length']}",
        f"unconditional_replay_{config['replay_length'] + 1}",
        "commutator_gate",
    )
    for length in config["sequence_lengths"]
    for example_index in range(config["sequences"])
    for iteration in range(config["timing_repeats"])
}
timing_keys = {
    (
        row["path"],
        row["replay_length"],
        row["sequence_length"],
        row["example_index"],
        row["iteration"],
    )
    for row in timing_records
}
if timing_keys != expected_timing_keys or len(timing_keys) != len(timing_records):
    raise RuntimeError("scaling timing is not the expected path product")
for row in timing_records:
    if row["token_sha256"] != token_by_index[row["example_index"]]:
        raise RuntimeError("scaling timing token hash differs from data manifest")
    if (
        row["window_token_sha256"]
        != window_by_index[row["example_index"]][str(row["sequence_length"])]
    ):
        raise RuntimeError("scaling timing window hash differs from data manifest")

recomputed = {
    "threshold": saved["threshold"],
    "conditions": {},
    "timing": summarize_timing(timing_records),
}
for length in config["sequence_lengths"]:
    rows = [row for row in records if row["length"] == length]
    accepted = [row for row in rows if row["score"] <= expected_threshold]
    errors = sum(row["full_transfer"]["top1_diff"] for row in accepted)
    native_chunk_max = max(row["native_chunk"]["cache_nmse"] for row in rows)
    recomputed["conditions"][str(length)] = {
        "count": len(rows),
        "full_transfer_risk": sum(row["full_transfer"]["top1_diff"] for row in rows) / len(rows),
        "unconditional_replay_64_risk": sum(row["replay_64"]["top1_diff"] for row in rows)
        / len(rows),
        "unconditional_replay_65_risk": sum(row["replay_65"]["top1_diff"] for row in rows)
        / len(rows),
        "gate_coverage": len(accepted) / len(rows),
        "gate_conditional_risk": errors / len(accepted) if accepted else None,
        "gate_conditional_clopper_pearson_upper_95": clopper_pearson_upper(errors, len(accepted))
        if accepted
        else None,
        "gate_population_risk_with_native_fallback": errors / len(rows),
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
recomputed["all_native_controls_passed"] = all(
    condition["native_control_passed"] for condition in recomputed["conditions"].values()
)
expected_status = (
    "complete" if recomputed["all_native_controls_passed"] else "invalid_native_control"
)
if manifest["status"] != expected_status:
    raise RuntimeError("scaling status disagrees with native controls")
if recomputed != saved:
    raise RuntimeError("scaling summary does not recompute")
print(
    json.dumps(
        {
            "run": str(run_dir),
            "records": len(records),
            "timings": len(timing_records),
            "recomputed": True,
        },
        indent=2,
        sort_keys=True,
    )
)
