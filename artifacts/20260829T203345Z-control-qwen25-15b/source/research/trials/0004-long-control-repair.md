# Trial 0004: long-context identity control repair

Status: fresh-token rerun passed

## Original result

The source control `20260829T202015Z-control-long-qwen25-15b` passed the original
machine-floor rule. The target control
`20260829T202115Z-control-long-qwen25-3b` failed at length 1024, replay 64, for
both `k=1` and `k=4` identity translations.

The translated prefix NMSE was `4.34e-20`. After replay, whole-cache NMSE was
`4.12e-6` and logit JS was `6.61e-9`; top-1 did not change. The same native
full-versus-chunk path had suffix-cache NMSE `1.59e-4` and logit JS `8.91e-8`.
All other identity conditions stayed near the numerical floor.

The artifact remains failed. Its result falsifies the assumption that a
machine-floor prefix perturbation must remain at the machine floor after target
continuation.

## Revised composite rule

The rerun uses a second fixed token sequence that was not used to diagnose the
failure. It retains these independent gates:

- identity prefix NMSE stays below the BF16 round-trip bound;
- native fixed-control full-versus-chunk execution preserves top-1;
- native full-versus-chunk cache NMSE stays at or below `0.001`;
- native cache clone/rebuild remains exact;
- identity continuation preserves top-1 for both `k=1` and `k=4`;
- identity suffix-only NMSE is no greater than both `0.001` and the larger of
  the machine propagation floor, matched clone/rebuild NMSE, and native
  full-versus-chunk suffix NMSE;
- identity logit JS is no greater than the larger of the BF16 floor, native
  full-versus-chunk JS, and matched clone/rebuild JS.

The effective per-record cache and logit limits are saved with every identity
result. This is a no-worse-than-native-execution-variation rule, not a claim of
machine-precision equivalence.

## Decision rule

Both source and target must pass the fresh-token composite control before the
context-scaling run. Any top-1 change, prefix failure, clone/rebuild failure, or
excess over the native execution ceiling stops scaling.

## Fresh-token result

- Source: `20260829T202619Z-control-long-qwen25-15b`, passed. Maximum identity
  suffix NMSE was zero; maximum native chunk NMSE was `3.59e-6`.
- Target: `20260829T202740Z-control-long-qwen25-3b`, passed. Maximum identity
  suffix NMSE was zero; maximum native chunk NMSE was `4.37e-4`.

The revised ceiling did not admit a nonzero identity suffix error on the fresh
sequence. These two runs, not the failed diagnostic run, are bound into the
scaling configuration.
