import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone

import torch
import transformers


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(repo_root):
    roots = ["src", "scripts", "configs", "research"]
    hashes = {}
    for root in roots:
        for path in sorted((repo_root / root).rglob("*")):
            ignored = "__pycache__" in path.parts or any(
                part.endswith(".egg-info") for part in path.parts
            )
            if path.is_file() and not ignored:
                hashes[str(path.relative_to(repo_root))] = sha256_file(path)
    for name in (".gitignore", "README.md", "pyproject.toml"):
        path = repo_root / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    return hashes


def snapshot_source(repo_root, run_dir, hashes):
    snapshot_dir = run_dir / "source"
    for relative, expected in hashes.items():
        source = repo_root / relative
        destination = snapshot_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != expected:
            raise RuntimeError(f"source snapshot changed while copying {relative}")


def create_run_dir(repo_root, run_name):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = repo_root / "artifacts" / f"{timestamp}-{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def environment_record():
    gpu = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu.name,
        "gpu_memory_bytes": gpu.total_memory,
        "gpu_capability": list(torch.cuda.get_device_capability()),
    }


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def base_manifest(repo_root, config_path):
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    try:
        config_relative = str(config_path.relative_to(repo_root))
    except ValueError:
        config_relative = None
    return {
        "config_relative": config_relative,
        "config_sha256": sha256_file(config_path),
        "git_revision": git_revision or None,
        "environment": environment_record(),
        "source_sha256": source_hashes(repo_root),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
