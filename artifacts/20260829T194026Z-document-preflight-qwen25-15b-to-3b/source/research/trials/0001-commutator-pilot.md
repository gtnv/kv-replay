# Trial 0001: commutator pilot

Status: revise

## Claim

Translate/replay disagreement ranks natural full-transfer next-token harm better than the transferred target's own entropy.

## Mechanism

The two routes should agree when the translator approximately intertwines source and target state updates. Harmful dynamical translation errors should increase their cache or logit disagreement.

## Changed variable

Replay length in `{8, 32, 64}` while the model pair, translator data, translator objective, histories, and anchor token remain fixed.

## Expected result

Commutator scores improve risk at 50% coverage and risk-coverage area relative to entropy. Improvement should grow from 8 to 32 tokens and then diminish.

## Falsifier

The score fails to beat entropy and the equal-compute replay-repair baseline, or longer replay does not improve ranking beyond uncertainty.

## Controls

- Native full prefill versus chunked replay.
- Extracted/rebuilt native target cache.
- K de-RoPE/re-RoPE.
- Same-model identity cache.
- Random score and oracle-only cache error.

## Budget

Use 32 translator chunks, 128 calibration chunks, and 256 test chunks of 256 tokens. With 128 calibration examples, a 50%-coverage threshold with zero observed errors can place the one-sided 95% Wilson upper bound below 5%; 64 calibration examples cannot. Run the 4/4/8 preflight first and treat it only as an integration check.

## Preflight checkpoint

Artifact `20260829T185839Z-preflight-qwen25-15b-to-3b` passed the independent evidence audit. It found 75% naive full-transfer disagreement on eight test chunks, while 32-token replay reduced disagreement to 12.5%. Those rates and rankings are too small to interpret statistically. Translation plus probe was slower than native 128-token prefill, so the 30% systems hypothesis is already doubtful at short context.

Decision before the main test: `accept` the integration path, not the research hypothesis. Run the frozen 32/128/256 condition to estimate calibrated risk and ranking. Treat any 256-token latency failure as a real result, then measure context-length crossover without changing the fidelity metric.

## Exploratory result

Artifact `20260829T190131Z-pilot-qwen25-15b-to-3b` passed its evidence audit. Naive transfer disagreed on 44.5% of 256 test chunks. At 64-token replay, commutator-JS harm average precision was 0.749 versus 0.660 for transferred entropy, and its descriptive 50%-coverage risk was 25.0% versus 29.7%. It did not approach 5% risk. Every label-adaptive calibration rule abstained, full transfer took 23.6 ms versus 15.4 ms native prefill, and the probe took 28.5 ms.

Decision: `revise`. The ranking signal is real enough to reproduce, but the operating-point and systems hypotheses failed at 256 tokens. The chunks were also article-dependent and the old threshold rule used calibration labels, so this artifact is exploratory next-token evidence only.
