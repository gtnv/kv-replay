# Refined research question

## Question

Across controlled translator fidelities and model pairs, does the part of
KV-translation error exposed by a short target replay predict target action
changes **beyond what the target's own confidence already predicts**?

The practical follow-up is:

> At what retained context length does the extra safe coverage bought by replay
> exceed the cost of replay, translation, scoring, and fallback?

## Intuition

There are two ways to reach the same handoff point:

1. let the source process the recent observation, then translate its KV state;
2. translate the older source state, then let the target replay that observation.

If translation were perfectly compatible with the target's internal dynamics,
the two target predictions would agree. Their disagreement measures the part of
translation error that changes as the target processes the recent observation.
We call that the non-commuting, or transition-inconsistent, part.

The blind spot is equally important. A harmful error can travel consistently
through both routes. Then both predictions are wrong in the same way and the
replay score stays small. Replay is therefore a diagnostic signal, not a
certificate.

## Updated hypothesis

Let \(Y\) mark native-versus-transferred top-1 disagreement, \(C\) be
transferred negative margin, and \(R\) be replay-path disagreement.

The testable hypothesis is:

> A predictor using \(C+R\) has lower held-out log loss than the same predictor
> using \(C\) alone when natural translation error has a substantial
> transition-inconsistent component. The gain should vanish for commuting error
> modes and should be larger for the true recent suffix than for token-matched
> sham suffixes.

This replaces raw AP as the main estimand. AP changes with harm prevalence, so
it cannot cleanly compare weak and strong translators. Paired held-out log-loss
improvement directly asks whether replay adds information after confidence.

## What the completed pilot says

- The commutator alone did not reliably beat negative margin.
- A post-hoc margin-plus-commutator router improved test log loss by 0.043, but
  its refit-bootstrap interval `[-0.013, 0.089]` crossed zero.
- Replay repaired 55 direct errors and introduced 12. Both routes were wrong on
  38/256 documents.
- An oracle choosing between direct and replay could accept 85.2% with zero
  error by falling back on the rest. The candidate actions are therefore not
  the immediate bottleneck; learning the selector is.
- At 256 tokens, score computation alone was 3.33× native prefill, so no
  threshold could make this implementation profitable.

## ICML-level answer we still need

An ICML claim requires more than a working gate on one pair. It should explain
**which error geometry is observable**, show that the information is specific
to the real transition rather than generic perturbation sensitivity, reproduce
under stronger translators and multiple model/domain cells, and separate
prediction failure from systems-cost failure with a break-even law.

The next experiments in [`report.md`](report.md) are ordered to answer those
questions before expanding to expensive multi-pair agent evaluations.
