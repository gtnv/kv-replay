# Detailed research report: local replay for selective cross-model KV reuse

Status: `[[active | accept | revise | reject | invalid]]`  
Primary run: `[[K4_RUN_ID]]`  
Fresh-process reproduction: `[[K4_REPRO_ID]]`  
Paired `k=1` ablation: `[[K1_RUN_ID]]`  
Last evidence audit: `[[AUDIT_TIMESTAMP_AND_STATUS]]`

This is an evidence report, not a paper. Bracketed fields must be filled only
from copied and independently audited numerical artifacts.

## 1. Research question

Can a short target-side replay, performed from a translated prefix cache,
predict whether direct cross-model KV transfer will change the target model's
greedy next-token decision, without constructing the target's native cache for
the full context?

The frozen confirmatory question is narrower:

> On hash-disjoint WikiText articles for Qwen2.5 1.5B Instruct to Qwen2.5 3B
> Instruct, does 64-token commutator JS rank direct-transfer top-1 disagreement
> better than transferred negative margin when the primary translator uses four
> contiguous source layers per target layer?

The systems question is whether a median-calibrated accept/fallback policy can
reach at least 50% coverage, a one-sided 95% Clopper-Pearson risk upper bound
at or below 5%, and at least 30% complete latency savings relative to native
target prefill.

The label is counterfactual next-token compatibility, not semantic correctness,
tool-call equality, or agent success.

## 2. Mechanism

Split a fixed token window into old prefix `p`, recent suffix `r`, and final
anchor token `q`.

- Native oracle `N`: target processes `p + r + q`.
- Direct transfer `D`: source processes `p + r`; translate its cache; target
  processes `q`.
- Target replay `R`: source processes `p`; translate its cache; target processes
  `r + q`.

The online score is

```text
commutator_js = JS(softmax(logits(D)), softmax(logits(R)))
```

The frozen offline harm label is

```text
full_transfer_top1_diff = top1(D) != top1(N)
```

`N` is used only after the fact to construct labels. Online acceptance code may
read `D`, `R`, and source/translated caches, but not native target state or any
oracle diagnostic. A lower score is accepted; a rejected example falls back to
fresh native target prefill.

Mechanistically, the cache commutator estimates an error-transport residual:

```text
F_target^r(M(C_source(p))) - M(F_source^r(C_source(p)))
    approximately J_F * e(p) - e(p + r)
```

where `M` is the translator and `e(x) = M(C_source(x)) - C_target(x)`. Direct
behavioral error instead depends on the target readout applied to `e(p + r)`.
The two quantities may correlate, but they are not equivalent.

Commutation is therefore not certification. For example, identity source and
target transitions commute with a translator that flips one state coordinate;
a target decision based on that coordinate can flip even though the commutator
is exactly zero. The experiment tests predictive ranking on one frozen
distribution, not a universal safety theorem.

## 3. Falsifiable hypotheses

### H1: primary ranking

At replay 64, commutator JS has higher harm average precision and lower
risk-coverage area than transferred negative margin for
`full_transfer_top1_diff`. The paired-bootstrap 95% interval for
`AP(commutator) - AP(negative margin)` should exclude zero.

### H2: operating point

The unlabeled calibration-median threshold achieves test coverage `>= 0.50`
and a one-sided 95% Clopper-Pearson conditional-risk upper bound `<= 0.05`.

### H3: replay budget

Ranking should improve from replay 8 to 32 to 64, with diminishing returns.
Flat or worse ranking falsifies a simple "more recent target computation reveals
more harmful mismatch" account.

### H4: systems utility

The complete coverage-weighted policy, including translation, both score
routes, cache work, and fallback, saves at least 30% wall time versus native
target prefill at batch one.

### H5: non-domination

The primary policy should improve the fidelity-latency frontier over
unconditional replay 64 and unconditional replay 65. Replay 65 is only
target-token-matched: both it and the replay-64 commutator use 66 target-token
forwards, but translation calls, memory work, and wall time differ.

### Secondary ablation question

Does repeating the same protocol with one depth-aligned source layer per target
materially change transfer harm, ranking, calibrated risk, or latency? This
paired `k=1` condition diagnoses translator-capacity dependence; it is not a
second primary hypothesis.

## 4. Frozen experimental design

### Models and translator

- Source: `Qwen/Qwen2.5-1.5B-Instruct` at `[[SOURCE_REVISION_SHA]]`.
- Target: `Qwen/Qwen2.5-3B-Instruct` at `[[TARGET_REVISION_SHA]]`.
- Shared tokenizer digest: `[[TOKENIZER_SHA256]]`.
- Precision: BF16 models/caches, FP64 ridge sufficient statistics, FP32 maps.
- Ridge penalty: `0.01` with separate key and value maps per target layer.
- Keys: remove source and target RoPE for fitting, then apply target RoPE after
  translation. Values are mapped directly.
- Primary `k=4`: concatenate four contiguous source layers around each
  depth-aligned anchor, resolving the even-width tie toward deeper layers.
- Secondary `k=1`: use only the depth-aligned anchor layer.

### Data and partitions

- Dataset: WikiText-103 raw at `[[DATASET_REVISION_SHA]]`.
- Main split: train; exclude raw rows below 1000.
- One first fixed-length window per article; no repeated windows from an article.
- Article SHA-256 modulo 3 assigns translator/calibration/test folds `0/1/2`.
- Main: 32 translator, 128 calibration, and 256 untouched test articles.
- Main window: 256 tokens = 255 context tokens + 1 anchor token.
- Integration preflight only: validation split, 128 tokens, 16/8/16 articles,
  replays 8 and 32. Preflight observations are never pooled with confirmation.
- Sampling seed: `1729`.

### Scores, labels, and replay budgets

- Candidate replays: 8, 32, and 64.
- Control replay: 65, reserved for target-token-matched unconditional replay.
- Frozen primary: commutator JS, replay 64, direct-transfer top-1 label.
- Frozen comparator: transferred negative top-1 margin.
- Secondary scores: transferred negative maximum probability, entropy,
  commutator cache NMSE, and deterministic random score.
- Secondary label: replay-route top-1 disagreement with native target.

### Calibration and statistics

- Threshold: calibration-score median, selected without calibration labels.
- Test reports: harm count/prevalence, harm AP, AURC, risk at 25/50/75%
  descriptive coverage, achieved calibrated coverage/risk, and exact one-sided
  95% Clopper-Pearson upper bound.
- Primary comparison: paired bootstrap AP difference, 2,000 repetitions,
  base seed `2718`.
- Test-label-selected operating points are oracle-only descriptions and cannot
  be reported as deployable thresholds.

### Timing boundary

- Hardware: one resident `[[GPU_IDENTITY]]`; batch one.
- Source cache and both models are resident at handoff.
- Four examples per accepted/rejected timing stratum, two warm-ups, five
  measured repetitions.
- Synchronize CUDA and retain raw wall and CUDA-event times.
- Measure complete accepted and rejected policy branches, then weight their
  means by full-test achieved coverage.
- Excluded: model loading, inter-node cache transport, queueing, batching, and
  serving-engine scheduling.

## 5. Controls and validity gates

### Numerical controls

1. Native full prefill versus native prefix-cache plus suffix replay.
2. Native cache extraction/rebuild plus suffix replay.
3. Key de-RoPE/re-RoPE round trip.
4. Same-model identity-cache translation and injection for relevant `k` values.
5. SDPA bulk execution versus eager-attention correctness execution.
6. Per-article native full prefill versus native chunked replay for every
   calibration/test article and replay length.

The hard native-chunk gate is maximum cache NMSE `<= 0.001`. Native full versus
native chunked top-1 variation is retained and reported as the action-noise
floor; it does not by itself invalidate the run.

### Evidence controls

- Resolved model and dataset revisions are immutable in the manifest.
- Translator, calibration, and test article hashes, token hashes, and source
  rows are disjoint.
- Online/oracle field separation is audited.
- Raw JSONL, timing JSONL, config, source snapshot, environment, translator,
  and summary digests are bound in the manifest.
- Summaries are independently recomputed from raw records.
- The fresh-process reproduction must preserve deterministic raw numerical
  records, subject only to explicitly documented hardware nondeterminism.

## 6. Results

### 6.1 Known exploratory evidence: not confirmatory

These outcomes motivated the independent-document protocol but cannot fill the
current primary fields.

- Preflight artifact `20260829T185839Z-preflight-qwen25-15b-to-3b` passed its
  evidence audit. On only eight test chunks, direct-transfer disagreement was
  75% and replay-32 disagreement was 12.5%. Translation plus probe was slower
  than native 128-token prefill. The sample is too small for a research claim.
- Exploratory artifact `20260829T190131Z-pilot-qwen25-15b-to-3b` passed its
  evidence audit. Direct transfer disagreed on 44.5% of 256 test chunks. At
  replay 64, commutator-JS harm AP was 0.749 versus 0.660 for entropy, and
  descriptive 50%-coverage risks were 25.0% versus 29.7%. Full transfer took
  23.6 ms, native prefill 15.4 ms, and the probe 28.5 ms.
- That run used article-dependent chunks, a different translator regime and
  comparator, and label-adaptive calibration. It supports only the decision to
  reproduce; it does not establish H1--H5.

### 6.2 Artifact identity and hard validity

| Field | Primary `k=4` | Reproduction | Secondary `k=1` |
|---|---:|---:|---:|
| Artifact ID | `[[K4_RUN_ID]]` | `[[K4_REPRO_ID]]` | `[[K1_RUN_ID]]` |
| Manifest digest | `[[K4_MANIFEST_SHA]]` | `[[K4_REPRO_MANIFEST_SHA]]` | `[[K1_MANIFEST_SHA]]` |
| Evidence audit | `[[K4_AUDIT]]` | `[[K4_REPRO_AUDIT]]` | `[[K1_AUDIT]]` |
| Split/overlap audit | `[[K4_SPLIT_AUDIT]]` | `[[K4_REPRO_SPLIT_AUDIT]]` | `[[K1_SPLIT_AUDIT]]` |
| Max native-chunk cache NMSE | `[[K4_MAX_NATIVE_NMSE]]` | `[[K4_REPRO_MAX_NATIVE_NMSE]]` | `[[K1_MAX_NATIVE_NMSE]]` |
| Native-chunk top-1 action-noise floor | `[[K4_NATIVE_TOP1_FLOOR]]` | `[[K4_REPRO_NATIVE_TOP1_FLOOR]]` | `[[K1_NATIVE_TOP1_FLOOR]]` |
| Hard-gate decision | `[[K4_VALIDITY]]` | `[[K4_REPRO_VALIDITY]]` | `[[K1_VALIDITY]]` |

If primary validity is not `valid`, all remaining primary numbers are diagnostic
only and the report decision must be `invalid`.

### 6.3 Frozen primary result

| Metric | Value |
|---|---:|
| Test articles | `[[TEST_COUNT]]` |
| Direct-transfer harm count / prevalence | `[[DIRECT_HARM_COUNT]] / [[DIRECT_HARM_PREVALENCE]]` |
| Commutator-JS AP at replay 64 | `[[R64_COMMUTATOR_AP]]` |
| Negative-margin AP at replay 64 | `[[R64_MARGIN_AP]]` |
| Paired AP difference | `[[R64_AP_DELTA]]` |
| Paired-bootstrap 95% interval | `[[R64_AP_DELTA_LOW]], [[R64_AP_DELTA_HIGH]]` |
| Calibration threshold | `[[R64_COMMUTATOR_THRESHOLD]]` |
| Accepted count / achieved coverage | `[[R64_ACCEPTED]] / [[R64_COVERAGE]]` |
| Accepted harm count / conditional risk | `[[R64_ACCEPTED_ERRORS]] / [[R64_RISK]]` |
| One-sided 95% Clopper-Pearson upper bound | `[[R64_CP_UPPER]]` |
| H1 decision | `[[H1_ACCEPT_OR_FAIL]]` |
| H2 decision | `[[H2_ACCEPT_OR_FAIL]]` |

### 6.4 Replay-budget and score ablations

| Replay | Comm. AP | Margin AP | Entropy AP | Random AP | Comm. AURC | Calibrated risk / CP upper |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | `[[R8_COMM_AP]]` | `[[R8_MARGIN_AP]]` | `[[R8_ENTROPY_AP]]` | `[[R8_RANDOM_AP]]` | `[[R8_COMM_AURC]]` | `[[R8_RISK]] / [[R8_CP_UPPER]]` |
| 32 | `[[R32_COMM_AP]]` | `[[R32_MARGIN_AP]]` | `[[R32_ENTROPY_AP]]` | `[[R32_RANDOM_AP]]` | `[[R32_COMM_AURC]]` | `[[R32_RISK]] / [[R32_CP_UPPER]]` |
| 64 | `[[R64_COMM_AP]]` | `[[R64_MARGIN_AP]]` | `[[R64_ENTROPY_AP]]` | `[[R64_RANDOM_AP]]` | `[[R64_COMM_AURC]]` | `[[R64_RISK]] / [[R64_CP_UPPER]]` |

Replay-budget trend: `[[REPLAY_TREND_DESCRIPTION]]`  
H3 decision: `[[H3_ACCEPT_OR_FAIL]]`

### 6.5 Fidelity and complete cost

| Path | Target-token forwards | Top-1 risk | Wall ms | CUDA ms | Wall savings vs native |
|---|---:|---:|---:|---:|---:|
| Native target prefill | 256 | 0 by definition | `[[NATIVE_WALL_MS]]` | `[[NATIVE_CUDA_MS]]` | 0 |
| Direct transfer | 1 | `[[DIRECT_RISK]]` | `[[DIRECT_WALL_MS]]` | `[[DIRECT_CUDA_MS]]` | `[[DIRECT_SAVINGS]]` |
| Unconditional replay 64 | 65 | `[[REPLAY64_RISK]]` | `[[REPLAY64_WALL_MS]]` | `[[REPLAY64_CUDA_MS]]` | `[[REPLAY64_SAVINGS]]` |
| Unconditional replay 65, target-token-matched | 66 | `[[REPLAY65_RISK]]` | `[[REPLAY65_WALL_MS]]` | `[[REPLAY65_CUDA_MS]]` | `[[REPLAY65_SAVINGS]]` |
| Primary commutator policy | 66 before fallback | `[[POLICY_RISK]]` | `[[POLICY_WEIGHTED_WALL_MS]]` | `[[POLICY_WEIGHTED_CUDA_MS]]` | `[[POLICY_SAVINGS]]` |

Accepted-branch mean: `[[ACCEPTED_BRANCH_WALL_MS]]` ms.  
Rejected-branch mean: `[[REJECTED_BRANCH_WALL_MS]]` ms.  
Timing strata observed: `[[TIMING_STRATA_COUNTS]]`.  
H4 decision: `[[H4_ACCEPT_OR_FAIL]]`.  
H5 decision: `[[H5_ACCEPT_OR_FAIL]]`.

### 6.6 Paired translator-capacity ablation

| Translator | Transfer harm | Comm. AP | Margin AP | Coverage | Risk / CP upper | Complete policy ms |
|---|---:|---:|---:|---:|---:|---:|
| Contiguous `k=4` primary | `[[K4_HARM]]` | `[[K4_COMM_AP]]` | `[[K4_MARGIN_AP]]` | `[[K4_COVERAGE]]` | `[[K4_RISK]] / [[K4_CP_UPPER]]` | `[[K4_POLICY_MS]]` |
| Depth-aligned `k=1` secondary | `[[K1_HARM]]` | `[[K1_COMM_AP]]` | `[[K1_MARGIN_AP]]` | `[[K1_COVERAGE]]` | `[[K1_RISK]] / [[K1_CP_UPPER]]` | `[[K1_POLICY_MS]]` |

Paired interpretation: `[[TRANSLATOR_CAPACITY_INTERPRETATION]]`

### 6.7 Error structure

- Harmful transfers accepted by the primary threshold: `[[FALSE_ACCEPT_COUNT]]`.
- Median native margin, false accepts versus true rejects:
  `[[FALSE_ACCEPT_NATIVE_MARGIN]]` versus `[[TRUE_REJECT_NATIVE_MARGIN]]`.
- Direct errors repaired by replay 64: `[[REPLAY_REPAIR_COUNT]]`.
- New replay errors introduced when direct transfer agreed: `[[REPLAY_INTRODUCED_COUNT]]`.
- Fraction of commutator JS above the matched native-chunk JS floor:
  `[[COMMUTATOR_ABOVE_NATIVE_FLOOR_FRACTION]]`.
- Dominant failure pattern: `[[ERROR_PATTERN_SUMMARY]]`.

## 7. Interpretation and decision

Final decision: `[[accept | revise | reject | invalid]]`

Decision rationale: `[[ONE_PARAGRAPH_TIED_TO_H1_THROUGH_H5_AND_VALIDITY]]`

Use these rules:

- `invalid`: cache NMSE exceeds 0.001, partitions overlap, online code reads an
  oracle field, required records are absent, or the evidence audit fails.
- `accept`: valid and reproduced H1, with a useful non-dominated operating
  point. State separately whether H2 and H4 meet their engineering targets.
- `revise`: the run is valid and informative, but ranking, calibrated risk, or
  cost misses target in a way that identifies one concrete next intervention.
- `reject`: valid evidence shows the commutator is uninformative or dominated,
  and the paired ablations do not isolate a plausible repair.

Interpret combinations conservatively:

- H1 succeeds, H2 fails: the score contains ranking information but is not a
  safe admission rule at the desired operating point.
- H1/H2 succeed, H4 fails: behavioral signal exists, but this implementation is
  uneconomic at 256 tokens.
- Replay 65 dominates: the verifier adds no value at its own target-token budget.
- `k=1` and `k=4` both fail similarly: failure is less likely to be explained
  only by the original single-layer translator, but stronger translators remain
  untested.
- Native action-noise floor is nonzero with NMSE within gate: report it and
  compare transfer effects against it; do not silently delete examples.

## 8. Gap and contribution

Prior cross-model cache work mainly asks whether a translator retains quality
on average for a model pair or improves aggregate prefill cost. Partial-reuse
systems study which tokens or layers to recompute when same-model cached state
is stale. Neither question directly supplies a per-handoff admission signal
when the target's native cache is unavailable.

This project contributes a narrower package:

1. Per-instance selective cross-model cache reuse as a measurable
   risk-coverage-cost problem.
2. A bounded target-native replay discrepancy that is online-observable without
   full target prefill.
3. A formal separation between commutation and certification, including an
   exact-commutation/action-flip counterexample.
4. An evidence design with document-disjoint fitting/calibration/testing,
   oracle separation, native numerical floors, target-token-matched replay,
   complete branch timing, and a paired translator-capacity ablation.

The contribution is the question, diagnostic, negative formal boundary, and
controlled evidence. It is not a new state-of-the-art translator and does not
establish agent-action safety.

## 9. Limitations

- One same-family model direction with a shared tokenizer and compatible RoPE.
- Affine fixed-depth translators only; `k=4` has higher input capacity than
  `k=1` but is not a
  learned nonlinear or predictively selected multi-layer mapper.
- WikiText next-token top-1 equality is a strict trajectory canary, not semantic
  correctness or structured action equivalence.
- Context length 256 may be below the latency crossover where reuse is useful.
- Calibration and test are document-disjoint but from the same corpus domain.
- The median threshold targets coverage, not future risk. Clopper-Pearson
  summarizes the held-out test sample; it is not distribution-free admission
  control.
- Batch-one resident-model timing excludes transfer, scheduling, and batching.
- Multiple secondary scores and replay budgets are exploratory; only the frozen
  replay-64 JS versus negative-margin comparison is primary.
- Local commutator probes can miss error modes shared by both paths by design.

## 10. Next experiments

Run only the branch justified by the audited outcome.

1. **If H1 succeeds but H2 fails:** freeze the current test set, create a new
   document split, and test a small multi-probe router using several suffix
   locations or perturbations. Do not tune a threshold on the old test labels.
2. **If `k=4` materially improves over `k=1`:** add one stronger translator
   with source layers selected by predictive fit or a small nonlinear mapper,
   then rerun the same verifier and negative-margin comparator.
3. **If H4 fails but H1/H2 succeed:** freeze policy parameters and sweep context
   lengths 512, 1K, 2K, and 4K to locate the measured latency crossover. Include
   complete branch timing at every length.
4. **If replay 65 dominates:** stop optimizing the verifier implementation and
   test whether a cheaper score can preserve ranking; otherwise reject this
   branch on systems grounds.
5. **If native action noise is material:** repeat the oracle and chunked paths
   under a deterministic attention configuration and report both floors. Do not
   change the 0.001 cache gate after seeing transfer results.
6. **If the text pilot remains promising:** move to a structured tool-action
   dataset, canonicalize tool name and arguments, and split by full trajectory.
7. **Before a general claim:** add multiple source-target directions, including
   a cross-family/shared-tokenizer-compatible pair, plus domain-shift evaluation.

Next selected experiment: `[[NEXT_EXPERIMENT_AND_FALSIFIER]]`

## 11. Artifact handoff checklist

- [ ] Copy only the primary, reproduction, `k=1`, source-control, and
  target-control artifact directories.
- [ ] Verify all manifest hashes after transfer.
- [ ] Run the independent evidence audit locally from raw JSONL.
- [ ] Fill this report from audited summary fields; retain counts beside rates.
- [ ] Preserve failed and invalid artifacts with their reason.
- [ ] Confirm every number in Sections 6--7 has an artifact path and JSON key.
- [ ] Record the final `accept`, `revise`, `reject`, or `invalid` decision before
  proposing another GPU run.
