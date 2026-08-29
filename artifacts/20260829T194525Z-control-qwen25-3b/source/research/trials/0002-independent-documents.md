# Trial 0002: independent-document reproduction

Status: active

## Claim

On independent articles, commutator logit disagreement ranks full-transfer next-token harm better than transferred entropy, random routing, and cache disagreement. A calibration-score median fixes the operating point without labels; the untouched test fold estimates its risk.

## Mechanism held fixed

Use the same Qwen2.5 1.5B-to-3B affine KV translator and replay lengths `{8, 32, 64}`. The intervention is evidence design, not a new verifier: one article per example, hash-disjoint folds, unseen rows, per-example native chunk controls, and complete branch timing.

## Primary comparisons

- Harm average precision and descriptive risk at 50% coverage versus transferred entropy and random score.
- Test risk and one-sided 95% Wilson bound at the unlabeled calibration-median threshold.
- Cost-risk comparison against unconditional replay, the strongest partial-recompute baseline.
- Complete accept/fallback wall and CUDA timing, not only summed component medians.

## Falsifier

Revise or reject if native chunking changes any test top-1, commutator ranking does not beat entropy, fixed-threshold risk remains far above 5%, unconditional replay dominates fidelity at lower cost, or translation/probe latency remains above native prefill.

## Budget

Use 32 translator articles, 128 calibration articles, and 256 untouched test articles of 256 tokens, all beginning after raw row 1000. Stop first on a 4/4/8 integration preflight.
