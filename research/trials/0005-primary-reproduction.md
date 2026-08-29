# Trial 0005: fresh-process primary reproduction

Status: passed exactly

## Parent

The result to reproduce is
`20260829T195327Z-document-pilot-qwen25-15b-to-3b`, the valid `k=4`, replay-64
primary run.

## Permitted differences

`configs/reproduction.json` differs from the parent protocol only in:

- source and target control artifact pointers, which now bind the final audited
  source tree;
- `run_name`, which identifies this as a reproduction.

No model, revision request, dataset, partition, document count, translator,
ridge, replay, threshold rule, statistic, seed, or timing parameter changes.

## Reproduction criteria

- The source and target controls independently audit as passed.
- `data.json`, `translator.json`, `records.jsonl`, translator tensor contents
  and metadata, and `summary.json` excluding timing-derived fields match the
  parent exactly. The `translator.pt` container hash is reported separately
  because binary archive metadata is not a scientific result.
- Raw timing need not byte-match, but the qualitative conclusion must remain:
  the complete commutator gate is slower than native target prefill.
- The fresh artifact passes `scripts/audit_evidence.py` from a new process.

Any deterministic scientific mismatch marks the reproduction failed and blocks
commit/push promotion.

## Result

Artifact `20260829T203748Z-document-pilot-reproduction-qwen25-15b-to-3b`
passed. Data, translator file and tensors, translator report, all 1,536 raw
records, and the summary with timing fields removed matched the parent exactly.
The independent audit recomputed the evidence. Fresh complete-policy latency
was 62.66 ms versus 15.75 ms native, preserving the systems conclusion.
