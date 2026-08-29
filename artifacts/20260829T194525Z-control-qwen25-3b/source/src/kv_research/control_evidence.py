import json

from kv_research.artifacts import read_jsonl, sha256_file
from kv_research.metrics import control_summary


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def verify_control_evidence(
    repo_root,
    run_relative,
    model_id,
    model_revision,
    attention_backend,
    source_layers_per_target,
    sequence_length,
    replay_lengths,
    pilot_source_hashes,
    pilot_environment,
):
    run_dir = (repo_root / run_relative).resolve()
    if repo_root.resolve() not in run_dir.parents:
        raise RuntimeError(f"control path escapes repository: {run_relative}")
    manifest = read_json(run_dir / "manifest.json")
    summary = read_json(run_dir / "summary.json")
    if manifest["status"] != "complete" or not summary["passed"]:
        raise RuntimeError(f"control is not complete and passed: {run_relative}")
    for key, name in (
        ("records_sha256", "records.jsonl"),
        ("summary_sha256", "summary.json"),
    ):
        if key not in manifest or sha256_file(run_dir / name) != manifest[key]:
            raise RuntimeError(f"control {name} hash mismatch: {run_relative}")
    if manifest["model_id"] != model_id or manifest["model_revision_resolved"] != model_revision:
        raise RuntimeError(f"control model does not match pilot: {run_relative}")
    config_relative = manifest["config_relative"]
    config_path = run_dir / "source" / config_relative
    if sha256_file(config_path) != manifest["config_sha256"]:
        raise RuntimeError(f"control config hash mismatch: {run_relative}")
    config = read_json(config_path)
    if config["attention_backend"] != attention_backend:
        raise RuntimeError(f"control attention backend does not match pilot: {run_relative}")
    required_identity_layers = {1, source_layers_per_target}
    if not required_identity_layers.issubset(set(config["identity_layers_per_target"])):
        raise RuntimeError(f"control lacks required identity regimes: {run_relative}")
    if sequence_length not in config["lengths"]:
        raise RuntimeError(f"control lacks pilot sequence length: {run_relative}")
    if not set(replay_lengths).issubset(set(config["replay_lengths"])):
        raise RuntimeError(f"control lacks pilot replay lengths: {run_relative}")
    for key in ("torch", "transformers", "cuda_runtime", "gpu_capability"):
        if manifest["environment"][key] != pilot_environment[key]:
            raise RuntimeError(f"control environment {key} does not match pilot: {run_relative}")
    critical_sources = (
        "scripts/run_control.py",
        "src/kv_research/cache_ops.py",
        "src/kv_research/metrics.py",
        "src/kv_research/models.py",
        "src/kv_research/translation.py",
    )
    for relative in critical_sources:
        if manifest["source_sha256"][relative] != pilot_source_hashes[relative]:
            raise RuntimeError(f"control source differs from pilot at {relative}")
        snapshot = run_dir / "source" / relative
        if sha256_file(snapshot) != manifest["source_sha256"][relative]:
            raise RuntimeError(f"control source snapshot hash mismatch at {relative}")
    records = read_jsonl(run_dir / "records.jsonl")
    record_keys = {(row["length"], row["replay_length"]) for row in records}
    expected_record_keys = {
        (length, replay_length)
        for length in config["lengths"]
        for replay_length in config["replay_lengths"]
    }
    if len(record_keys) != len(records) or record_keys != expected_record_keys:
        raise RuntimeError(f"control grid is incomplete or duplicated: {run_relative}")
    expected_identity_keys = {str(value) for value in config["identity_layers_per_target"]}
    for record in records:
        if set(record["identity_translation"]) != expected_identity_keys:
            raise RuntimeError(f"control record identity regimes differ: {run_relative}")
    recomputed = control_summary(records, config["native_cache_nmse_limit"])
    if recomputed != summary:
        raise RuntimeError(f"control summary does not recompute: {run_relative}")
    return {
        "run": run_relative,
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "records_sha256": manifest["records_sha256"],
        "summary_sha256": manifest["summary_sha256"],
        "model_revision_resolved": model_revision,
    }
