# Trial 0003: fixed-policy context scaling

Status: complete, indeterminate crossover

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

## Result

Artifact `20260829T203053Z-context-scaling-qwen25-15b-to-3b` passed every
native cache gate and its independent audit. Complete-policy/native wall ratios
were 3.829 at 256, 2.462 at 1024, 1.653 at 2048, and 1.206 at 4096 tokens. The
4096 paired interval was `[0.957, 1.454]`, so no interval-aware crossover can be
claimed.

Backing out the 50% fallback cost gives point-estimate break-even coverages of
3.329, 1.962, 1.153, and 0.706. The first three are impossible because coverage
cannot exceed one. At 4096, the measured 50% coverage still falls below the
70.6% point break-even requirement.
