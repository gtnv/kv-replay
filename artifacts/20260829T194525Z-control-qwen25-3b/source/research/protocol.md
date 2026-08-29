# Pilot protocol

## State and labels

For a token history split into old prefix `p`, recent suffix `r`, and final anchor token `q`:

- Native route: target processes `p + r + q`.
- Full-transfer route: source processes `p + r`, its cache is translated, and target processes `q`.
- Replay route: source cache for `p` is translated and target processes `r + q`.

The detection label is whether the full-transfer top-1 token differs from the native target top-1 token. A separate routing label asks whether the replay route itself differs from native target top-1. Keeping these labels separate prevents a cheap repaired action from being credited with the fidelity of the naive transferred action. Also record Jensen-Shannon divergence, the native top-1 margin, and whether replay repairs the disagreement.

The online commutator scores compare full-transfer and replay routes. Oracle-only diagnostics compare either route with native target state or logits. Every recorded field is explicitly classified online or oracle-only.

## Data separation

- Each example is the first fixed-length window from one WikiText article; articles are never split into multiple examples.
- A pinned training stream is deterministically partitioned into disjoint translator, calibration, and test folds by article hash.
- Rows below the frozen cutoff are excluded so the independent-document confirmation does not reuse exploratory examples.
- Contributing article hashes, source rows, fold IDs, and token hashes are persisted and audited for overlap.
- The final test set is read once after score definitions are frozen.

The text-only pilot measures next-token compatibility. Structured tool actions and trajectory-grouped splits are the next stage, not a claim of this run.

The primary condition is frozen before confirmation: four contiguous depth-aligned source layers per target layer, a 64-token replay, commutator JS, direct-transfer top-1 mismatch, and transferred negative top-1 margin as the comparator. The original one-layer translator is a secondary quality/cost ablation. The primary threshold is the calibration-score median, selected without calibration labels. The untouched test fold estimates risk at approximately 50% coverage and receives exact one-sided 95% Clopper--Pearson and approximate Wilson bounds. Label-adaptive test risk/coverage curves are descriptive ranking diagnostics only.

## Hard controls

1. Full native prefill versus native prefix-cache plus suffix replay.
2. Native cache extraction/rebuild plus suffix replay.
3. Native K de-RoPE/re-RoPE round-trip.
4. Same-model identity-cache injection.
5. SDPA bulk backend versus eager correctness backend on fixed examples.
6. Native full prefill versus native chunked replay on every evaluation article and replay length.

The cache-state tolerance is frozen at NMSE 0.001 from independent controls. Native chunked top-1 changes are reported as an action-noise floor rather than removed with an oracle-only filter; exceeding the cache-state tolerance invalidates the run. The tolerance is not selected after observing translation results.

## Primary metrics

- Harm prevalence.
- Risk at 25%, 50%, and 75% accepted coverage.
- Area under the risk-coverage curve.
- Harm average precision.
- Test risk and exact one-sided 95% Clopper--Pearson bound at an unlabeled calibration-median threshold; Wilson is retained as an approximation.
- Test-label-selected coverage is reported only as a descriptive oracle diagnostic.
- Full-transfer and unconditional-replay top-1 disagreement, including a 65-token target-forward-token-matched repair baseline for the 64-token commutator.
- Translation, probe, repair, and fallback target-token equivalents.
- Raw synchronized wall and CUDA batch-1 latency after warm-up. Translation and cache management are included; the source prefix cache is assumed to exist at handoff.

## Stop/go decisions

- `invalid`: a hard control, split, manifest, or evidence check fails.
- `accept`: the mechanism beats the strongest compute-matched baseline and evidence reproduces.
- `revise`: the failure is interpretable and identifies one next intervention.
- `reject`: the mechanism is falsified or dominated at matched compute.
