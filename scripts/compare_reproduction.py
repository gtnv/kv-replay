import argparse
import json
from pathlib import Path

import torch

from kv_research.artifacts import sha256_file, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--reproduction", required=True)
    parser.add_argument("--output", default="reproduction.json")
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if "timing" not in key}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def exact_value(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(exact_value(left[key], right[key]) for key in left)
        )
    if isinstance(left, list | tuple) or isinstance(right, list | tuple):
        return (
            isinstance(left, list | tuple)
            and isinstance(right, list | tuple)
            and len(left) == len(right)
            and all(exact_value(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


args = parse_args()
repo_root = Path(__file__).resolve().parents[1]
parent_dir = (repo_root / args.parent).resolve()
reproduction_dir = (repo_root / args.reproduction).resolve()
for run_dir in (parent_dir, reproduction_dir):
    if repo_root.resolve() not in run_dir.parents or not run_dir.is_dir():
        raise RuntimeError(f"run is absent or outside repository: {run_dir}")

parent_manifest = read_json(parent_dir / "manifest.json")
reproduction_manifest = read_json(reproduction_dir / "manifest.json")
parent_summary = read_json(parent_dir / "summary.json")
reproduction_summary = read_json(reproduction_dir / "summary.json")
reproduction_audit = read_json(reproduction_dir / "audit.json")

file_matches = {
    name: sha256_file(parent_dir / name) == sha256_file(reproduction_dir / name)
    for name in ("data.json", "translator.json", "records.jsonl")
}
parent_translation = torch.load(parent_dir / "translator.pt", map_location="cpu", weights_only=True)
reproduction_translation = torch.load(
    reproduction_dir / "translator.pt", map_location="cpu", weights_only=True
)
translator_tensor_match = exact_value(parent_translation, reproduction_translation)
summary_without_timing_match = without_timing(parent_summary) == without_timing(
    reproduction_summary
)
fresh_policy_ms = reproduction_summary["primary_result"]["complete_policy_timing"][
    "test_coverage_weighted_mean_wall_ms"
]
fresh_native_ms = reproduction_summary["timing"]["paths"]["native_target_prefill"]["mean_wall_ms"]
audit_match = reproduction_audit["recomputed"] is True and reproduction_audit[
    "manifest_sha256"
] == sha256_file(reproduction_dir / "manifest.json")
passed = (
    parent_manifest["status"] == "complete"
    and reproduction_manifest["status"] == "complete"
    and all(file_matches.values())
    and translator_tensor_match
    and summary_without_timing_match
    and fresh_policy_ms > fresh_native_ms
    and audit_match
)
result = {
    "parent": args.parent,
    "parent_manifest_sha256": sha256_file(parent_dir / "manifest.json"),
    "reproduction": args.reproduction,
    "reproduction_manifest_sha256": sha256_file(reproduction_dir / "manifest.json"),
    "file_matches": file_matches,
    "translator_file_sha256_match": sha256_file(parent_dir / "translator.pt")
    == sha256_file(reproduction_dir / "translator.pt"),
    "translator_tensor_match": translator_tensor_match,
    "summary_without_timing_match": summary_without_timing_match,
    "fresh_complete_policy_wall_ms": fresh_policy_ms,
    "fresh_native_wall_ms": fresh_native_ms,
    "fresh_policy_slower_than_native": fresh_policy_ms > fresh_native_ms,
    "reproduction_audit_match": audit_match,
    "passed": passed,
}
output_path = (reproduction_dir / args.output).resolve()
if reproduction_dir not in output_path.parents:
    raise RuntimeError("reproduction output path escapes run directory")
write_json(output_path, result)
print(json.dumps(result, indent=2, sort_keys=True))
if not passed:
    raise RuntimeError("fresh-process reproduction did not match the frozen parent")
