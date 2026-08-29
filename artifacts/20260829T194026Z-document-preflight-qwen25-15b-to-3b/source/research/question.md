# Research question

## Primary question

When is cross-model state-translation error behaviorally observable from bounded target-native computation?

## Pilot question

Does disagreement between these routes predict whether a translated source cache changes the target's next token?

1. Translate the old source prefix, then let the target replay a recent suffix.
2. Let the source process the recent suffix, then translate the resulting state.

The comparison must beat online confidence and equal-compute recent recomputation. A score that only correlates with native target-cache MSE is not deployable because that cache requires the prefill being avoided.

## General hypothesis

Learned state-translation errors are structured. Behaviorally harmful error components often change how the target dynamics respond to a small diagnostic suffix. A bounded set of target-side probes should therefore rank per-handoff harm better than static online confidence. A single suffix will have blind spots, especially for errors tied to old history.

## Falsifiers

- Native cache reconstruction or chunked replay exceeds the measured backend numerical floor.
- The commutator score does not beat target confidence or equal-compute recent replay.
- Additional probe budget does not improve risk/coverage.
- Signal exists only for synthetic corruption, not natural learned-translation errors.
- Probe, translation, and fallback overhead erase the prefill savings.

