import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="research/ledger.jsonl")
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


args = parse_args()
repo_root = Path(__file__).resolve().parents[1]
ledger_path = (repo_root / args.ledger).resolve()
if repo_root.resolve() not in ledger_path.parents:
    raise RuntimeError("ledger path escapes repository")

entries = []
with open(ledger_path, encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, 1):
        if line.strip():
            entry = json.loads(line)
            entry["line_number"] = line_number
            entries.append(entry)

required = {
    "entry_id",
    "recorded_utc",
    "trial",
    "stage",
    "outcome",
    "next_action",
    "artifact",
    "root_file",
    "root_sha256",
    "evidence_sha256",
    "parents",
    "supersedes",
    "reason",
}
allowed_outcomes = {"success", "failure", "invalid", "superseded"}
allowed_actions = {"promote", "revise", "rerun", "stop", "none"}
allowed_stages = {
    "control",
    "preflight",
    "primary",
    "ablation",
    "reproduction",
    "scaling",
    "supersession",
}
known_ids = set()
superseded_ids = set()
for entry in entries:
    line_number = entry.pop("line_number")
    missing = required - set(entry)
    if missing:
        raise RuntimeError(f"ledger line {line_number} is missing {sorted(missing)}")
    entry_id = entry["entry_id"]
    if not isinstance(entry_id, str) or not entry_id or entry_id in known_ids:
        raise RuntimeError(f"ledger line {line_number} has an invalid or duplicate entry_id")
    if entry["outcome"] not in allowed_outcomes:
        raise RuntimeError(f"ledger line {line_number} has unknown outcome {entry['outcome']}")
    if entry["next_action"] not in allowed_actions:
        raise RuntimeError(f"ledger line {line_number} has unknown next_action")
    if entry["stage"] not in allowed_stages:
        raise RuntimeError(f"ledger line {line_number} has unknown stage {entry['stage']}")
    if not isinstance(entry["reason"], str) or not entry["reason"].strip():
        raise RuntimeError(f"ledger line {line_number} has no reason")
    for relation in ("parents", "supersedes"):
        if not isinstance(entry[relation], list):
            raise RuntimeError(f"ledger line {line_number} {relation} is not a list")
        unknown = set(entry[relation]) - known_ids
        if unknown:
            raise RuntimeError(
                f"ledger line {line_number} {relation} references later or missing entries "
                f"{sorted(unknown)}"
            )

    artifact_dir = (repo_root / entry["artifact"]).resolve()
    if repo_root.resolve() not in artifact_dir.parents or not artifact_dir.is_dir():
        raise RuntimeError(f"ledger line {line_number} artifact is absent or outside repository")
    root_path = (artifact_dir / entry["root_file"]).resolve()
    if artifact_dir not in root_path.parents or sha256_file(root_path) != entry["root_sha256"]:
        raise RuntimeError(f"ledger line {line_number} root artifact hash mismatch")
    root = read_json(root_path)

    evidence = entry["evidence_sha256"]
    if not isinstance(evidence, dict):
        raise RuntimeError(f"ledger line {line_number} evidence hashes are not an object")
    if not evidence and entry["outcome"] != "invalid":
        raise RuntimeError(f"ledger line {line_number} has no evidence hashes")
    manifest_keys = {
        "records.jsonl": "records_sha256",
        "summary.json": "summary_sha256",
        "timing.jsonl": "timing_sha256",
        "translator.pt": "translator_sha256",
        "translator.json": "translator_report_sha256",
        "data.json": "data_sha256",
    }
    for relative, expected in evidence.items():
        evidence_path = (artifact_dir / relative).resolve()
        if artifact_dir not in evidence_path.parents or sha256_file(evidence_path) != expected:
            raise RuntimeError(f"ledger line {line_number} evidence hash mismatch for {relative}")
        manifest_key = manifest_keys.get(relative)
        if manifest_key is not None and root.get(manifest_key) != expected:
            raise RuntimeError(
                f"ledger line {line_number} manifest does not bind evidence file {relative}"
            )

    if ("audit_file" in entry) != ("audit_sha256" in entry):
        raise RuntimeError(f"ledger line {line_number} has an incomplete audit binding")
    if "audit_file" in entry:
        audit_path = (artifact_dir / entry["audit_file"]).resolve()
        if (
            artifact_dir not in audit_path.parents
            or sha256_file(audit_path) != entry["audit_sha256"]
        ):
            raise RuntimeError(f"ledger line {line_number} audit hash mismatch")
        audit = read_json(audit_path)
        if "recomputed" not in audit or audit["recomputed"] is not True:
            raise RuntimeError(f"ledger line {line_number} audit did not recompute evidence")

    if entry["outcome"] in ("success", "failure"):
        for name in ("audit_file", "audit_sha256", "protocol_file", "protocol_sha256"):
            if name not in entry:
                raise RuntimeError(f"ledger line {line_number} valid result lacks {name}")
        if entry["root_file"] != "manifest.json" or root["status"] != "complete":
            raise RuntimeError(f"ledger line {line_number} valid result lacks a complete manifest")
        omitted = [
            relative
            for relative, manifest_key in manifest_keys.items()
            if manifest_key in root and relative not in evidence
        ]
        if omitted:
            raise RuntimeError(
                f"ledger line {line_number} omits manifest evidence files {sorted(omitted)}"
            )
        protocol_path = (artifact_dir / entry["protocol_file"]).resolve()
        if (
            artifact_dir not in protocol_path.parents
            or sha256_file(protocol_path) != entry["protocol_sha256"]
        ):
            raise RuntimeError(f"ledger line {line_number} frozen protocol hash mismatch")
        if not entry["protocol_file"].startswith("source/"):
            raise RuntimeError(f"ledger line {line_number} protocol is not an artifact snapshot")
        source_relative = entry["protocol_file"].removeprefix("source/")
        if root["source_sha256"][source_relative] != entry["protocol_sha256"]:
            raise RuntimeError(f"ledger line {line_number} manifest does not bind its protocol")
        config_path = artifact_dir / "source" / root["config_relative"]
        if sha256_file(config_path) != root["config_sha256"]:
            raise RuntimeError(f"ledger line {line_number} config snapshot hash mismatch")

    known_ids.add(entry_id)
    superseded_ids.update(entry["supersedes"])

active = [
    entry["entry_id"]
    for entry in entries
    if entry["entry_id"] not in superseded_ids and entry["outcome"] != "superseded"
]
print(
    json.dumps(
        {
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
            "entries": len(entries),
            "active_entries": active,
            "outcomes": {
                outcome: sum(entry["outcome"] == outcome for entry in entries)
                for outcome in sorted(allowed_outcomes)
            },
            "verified": True,
        },
        indent=2,
        sort_keys=True,
    )
)
