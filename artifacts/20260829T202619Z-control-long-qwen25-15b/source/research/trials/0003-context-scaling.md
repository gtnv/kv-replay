# Trial 0003: fixed-policy context scaling

Status: active exploratory systems trial

## Question

On one H100 PCIe, does complete 64-token commutator-gate mean wall latency cross native-target mean wall prefill latency as retained context grows?

## Frozen design

- Parent translator and unlabeled calibration threshold are hash-bound to the audited k=4 confirmatory run.
- Sixteen previously unused WikiText test articles each contribute one 4096-token window.
- Lengths 256, 1024, 2048, and 4096 use suffix-aligned nested windows, so the anchor and most recent 64 tokens are identical within each article while only older retained context changes.
- Source prefill is excluded because the source cache exists at handoff. Translation, both commutator routes, JS scoring, accepted action, and native fallback are included.
- Operation order is deterministically rotated across documents; length order alternates forward and reverse across documents. Each operation receives two warm-ups and three synchronized measurements.
- The primary latency estimand is complete-gate empirical mean wall latency divided by native empirical mean wall latency. CUDA time is a consistency check. A crossing claim requires adjacent tested lengths whose wall-ratio point estimates change sides of one, whose paired-bootstrap intervals lie wholly on their respective sides of one, and whose CUDA point estimates have the same directions.
- Per-document means are the independent timing units. A 2,000-resample paired document bootstrap gives the 95% interval for the latency ratio.

## Validity and scope

The parent cache NMSE limit of 0.001 is a hard fidelity gate at every length. Native top-1 path changes remain a reported action-noise floor. If a length fails, its latency remains diagnostic but its action-risk result is invalid.

The 16-document risk values are descriptive; they cannot establish 5% risk. Selecting long articles makes this a long-document cohort, not a representative WikiText estimate.

## Decision

- `crossover bracket`: adjacent wall-ratio estimates and their complete bootstrap intervals lie on opposite sides of one, with CUDA directions consistent.
- `no observed crossover`: every wall-ratio bootstrap interval lies wholly on the same side of one, with CUDA directions consistent.
- `indeterminate`: neither interval-aware condition is met, including any point-estimate crossing whose endpoint interval contains one.
- `invalid fidelity`: native cache NMSE exceeds 0.001 at any claimed fidelity length.
