# Trial 0002: independent-document reproduction

Status: active

## Claim

On independent articles, 64-token commutator JS ranks full-transfer next-token top-1 disagreement better than transferred negative margin. A calibration-score median fixes the operating point without labels; the untouched test fold estimates its conditional risk with a one-sided 95% Clopper-Pearson upper bound.

## Mechanism held fixed

The primary Qwen2.5 1.5B-to-3B translator concatenates a contiguous group of four source layers for each target layer and fits separate affine ridge maps for keys and values. The group is anchored by depth alignment, with the even-width tie assigned deeper. A paired secondary `k=1` run changes only the number of source layers per target; it is a translator-capacity ablation, not a second primary claim.

Candidate verifier replays are `{8, 32, 64}`. Replay 65 is reserved for the target-token-matched unconditional-replay control: the primary 64-token commutator computes one direct anchor plus 65 probe-route target tokens, while unconditional replay 65 computes 66 target tokens. Translation calls and total work differ, so this control is only target-token-matched.

The intervention is evidence design, not a new translator or verifier: one article per example, hash-disjoint translator/calibration/test folds, rows beginning at 1000, per-example native chunk controls, and complete accepted/rejected branch timing.

## Primary comparisons

- Primary: paired-bootstrap difference in harm average precision between commutator JS and transferred negative margin at replay 64 for `full_transfer_top1_diff`.
- Secondary: negative maximum probability, entropy, cache NMSE, random score, replays 8 and 32, and the replay-action label.
- Test risk and one-sided 95% Clopper-Pearson upper bound at the unlabeled calibration-median threshold.
- Cost-risk comparison against same-length unconditional replay 64 and target-token-matched unconditional replay 65.
- Complete accept/fallback wall and CUDA timing, not only summed component medians.
- Paired `k=4` primary versus `k=1` secondary translator-capacity results on the same documents and protocol.

## Falsifier

The hard native-chunk validity gate is maximum cache NMSE `<= 0.001`. Native full-prefill versus native chunked top-1 variation is reported as the action-noise floor and does not itself invalidate the run. Revise or reject if the NMSE gate fails, commutator JS does not beat transferred negative margin on the frozen primary comparison, the fixed-threshold Clopper-Pearson upper bound remains above 5%, either unconditional-replay control dominates the fidelity-cost frontier, or complete policy latency remains above native prefill.

## Budget

Use 32 translator articles, 128 calibration articles, and 256 untouched test articles of 256 tokens, all beginning after raw row 1000. Stop first on a validation-split integration preflight with 16 translator, 8 calibration, and 16 test articles of 128 tokens. The preflight uses replays 8 and 32 and contributes no confirmatory observations.
