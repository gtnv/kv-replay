# KV-Replay

Can a target model cheaply tell when a translated KV cache will change its next
action, without rebuilding the whole target cache?

This repository contains a controlled negative result and the experiments that
produced it. For Qwen2.5 1.5B → 3B on 256-token WikiText documents, a 64-token
target replay does reveal some transfer errors, but it does not reliably beat a
simple confidence margin and its complete accept/fallback policy is much slower
than native target prefill. The result reproduced exactly in a fresh process.

The refined question is narrower and more useful:

> Does replay expose information about translation error that is not already in
> the target model's confidence, and at what context length can that extra
> information pay for itself?

## What happened

- Direct translated-cache reuse changed the target top-1 token on 93/256
  documents (36.3%).
- The replay score reached AP 0.714 versus 0.706 for negative margin. The paired
  lift was 0.0055 with a 95% interval from -0.1069 to 0.1122.
- The frozen gate accepted 42.2% of documents. Its accepted error rate was
  17.6%; the one-sided 95% upper bound was 24.7%, far from the 5% target.
- The complete gate took 62.3 ms versus 16.0 ms for native prefill at 256
  tokens.
- A stronger four-layer translator reduced harm relative to a one-layer map:
  it fixed 48 paired errors and introduced 11.
- Replay fixed 55 direct-transfer errors but introduced 12 new ones. This is why
  selecting between the two actions remains interesting even though the first
  gate failed.
- The full experiment reproduced byte-for-byte for data, translator, and all
  1,536 scientific records.

![paired action errors](research/figures/paired-errors.png)

The gate becomes relatively cheaper with longer context, but it is still
clearly slower through 2,048 tokens. At 4,096 tokens its mean is 1.21× native
and the paired interval crosses break-even, so the outcome there is
indeterminate.

![latency scaling](research/figures/latency-scaling.png)

## Read this first

- [`research/report.md`](research/report.md): intuitive mechanism, every main
  result, what failed, and the next ICML-oriented experiments.
- [`research/ledger.jsonl`](research/ledger.jsonl): append-only log of 16 passed,
  failed, invalid, and superseded runs with hashes.
- [`research/existing-record-analysis.json`](research/existing-record-analysis.json):
  direct numerical post-hoc analysis used to choose the next direction.
- [`research/trials/`](research/trials): frozen trial cards and control repairs.
- [`research/public-release-audit.md`](research/public-release-audit.md): public
  privacy audit and artifact-sanitization record.

## Repository map

```text
configs/                 frozen experiment inputs
src/kv_research/         cache, translation, metrics, timing, and evidence logic
scripts/                 setup, runs, independent audits, comparison, and figures
artifacts/               immutable raw rows, summaries, manifests, and audits
research/                report, ledger, protocol, trial cards, and figures
```

## Setup and exact reproduction

On the H100 host, the complete environment is one command:

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
```

The checked-in reproduction config points to checked-in fresh controls:

```bash
python scripts/run_pilot.py --config configs/reproduction.json
python scripts/audit_evidence.py --run artifacts/<new-reproduction-id> --output audit.json
python scripts/compare_reproduction.py \
  --parent artifacts/20260829T195327Z-document-pilot-qwen25-15b-to-3b \
  --reproduction artifacts/<new-reproduction-id>
```

Audit the experiment history separately:

```bash
python scripts/audit_ledger.py
```

Regenerate the figures only from the checked-in numerical artifacts:

```bash
python scripts/build_figures.py \
  --scaling artifacts/20260829T203053Z-context-scaling-qwen25-15b-to-3b/summary.json \
  --analysis research/existing-record-analysis.json \
  --output-dir research/figures
```

## Scope

This is one same-family, shared-tokenizer model direction, one corpus, greedy
next-token action equality, batch one, and one H100 PCIe. It establishes neither
cross-tokenizer state transfer nor end-to-end agent reliability. Its useful
contribution is a sharper question and a reproducible diagnosis: local replay
can observe transition-inconsistent cache error, but it is not a certificate of
safe reuse and its information must be tested conditionally against confidence.
