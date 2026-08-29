# Cross-model KV state observability

This repository tests whether bounded target-side computation can identify source-model KV states that change a target model's behavior after translation.

The first scientific estimand is deliberately narrow:

> Does translate/replay disagreement predict target next-token disagreement beyond online confidence and an equal target-token recomputation budget?

The target model's full native prefill is an offline oracle used only for labels. Online scores may use the source state, translated state, target continuation from translated state, and bounded replay. They may not use the native target cache or native target action.

## One-command headless setup

```bash
bash scripts/bootstrap.sh
```

The script creates `.venv`, installs the pinned Python stack, verifies CUDA, and records the resolved environment. Activate later shells with:

```bash
source .venv/bin/activate
```

## Run order

```bash
python scripts/run_control.py --config configs/control.json
python scripts/run_control.py --config configs/control-target.json
python scripts/run_pilot.py --config configs/preflight.json
python scripts/run_pilot.py --config configs/pilot.json
python scripts/audit_evidence.py --run artifacts/<run-id>
```

Do not run the pilot until the source and target control configurations both pass.

## Initial scope

- Hugging Face Transformers `DynamicCache`, not vLLM.
- BF16 Qwen2.5 models with a shared tokenizer and compatible RoPE.
- Primary pair: Qwen2.5 1.5B Instruct to Qwen2.5 3B Instruct.
- Streaming affine-ridge maps for each target layer's unrotated K and V.
- Exact full-prefill versus translated-state next-token behavior.
- Latest-suffix probe, target-confidence baseline, oracle-only cache diagnostics, and matched suffix replay.
- Immutable JSON manifests and per-example numerical JSONL.

This pilot does not establish arbitrary cross-tokenizer transfer, multi-step agent success, or a safety guarantee under distribution shift.
