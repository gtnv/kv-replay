#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install ruff==0.12.11

mkdir -p artifacts/environment
python scripts/verify_environment.py | tee artifacts/environment/bootstrap.json
# The project source is already hashed into every run, so the machine-local
# editable-install path adds no evidence and must not enter public artifacts.
python -m pip freeze --exclude-editable > artifacts/environment/pip-freeze.txt
