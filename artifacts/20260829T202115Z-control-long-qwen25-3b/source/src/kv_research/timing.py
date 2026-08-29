import statistics
import time

import numpy as np
import torch


def measure_ms(operation, warmup, repeats):
    if warmup < 0 or repeats <= 0:
        raise RuntimeError("timing requires nonnegative warmup and positive repeats")
    warmup_index = 0
    while warmup_index < warmup:
        result = operation()
        torch.cuda.synchronize()
        del result
        warmup_index += 1

    records = []
    for iteration in range(repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter()
        result = operation()
        stop.record()
        torch.cuda.synchronize()
        records.append(
            {
                "iteration": iteration,
                "wall_ms": (time.perf_counter() - wall_start) * 1000,
                "cuda_ms": start.elapsed_time(stop),
            }
        )
        del result
    return records


def summarize_timing(records):
    groups = {}
    for record in records:
        key = record["path"]
        if record["replay_length"] is not None:
            key = f"{key}:{record['replay_length']}"
        groups.setdefault(key, []).append(record)
    summary = {}
    for key, rows in groups.items():
        summary[key] = {
            "samples": len(rows),
            "median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
            "median_cuda_ms": statistics.median(row["cuda_ms"] for row in rows),
            "mean_wall_ms": statistics.fmean(row["wall_ms"] for row in rows),
            "mean_cuda_ms": statistics.fmean(row["cuda_ms"] for row in rows),
            "min_wall_ms": min(row["wall_ms"] for row in rows),
            "max_wall_ms": max(row["wall_ms"] for row in rows),
        }
        if all("accepted" in row for row in rows):
            summary[key]["timing_sample_accepted_fraction"] = statistics.fmean(
                int(row["accepted"]) for row in rows
            )
            branches = {}
            for accepted, name in ((True, "accepted"), (False, "rejected")):
                branch_rows = [row for row in rows if row["accepted"] is accepted]
                if branch_rows:
                    branches[name] = {
                        "samples": len(branch_rows),
                        "median_wall_ms": statistics.median(row["wall_ms"] for row in branch_rows),
                        "median_cuda_ms": statistics.median(row["cuda_ms"] for row in branch_rows),
                        "mean_wall_ms": statistics.fmean(row["wall_ms"] for row in branch_rows),
                        "mean_cuda_ms": statistics.fmean(row["cuda_ms"] for row in branch_rows),
                    }
            summary[key]["branches"] = branches
    return summary


def paired_timing_summary(
    records,
    candidate_path,
    baseline_path,
    sequence_length,
    repetitions,
    seed,
):
    candidate = {}
    baseline = {}
    for row in records:
        if row.get("sequence_length") != sequence_length:
            continue
        if row["path"] == candidate_path:
            candidate.setdefault(row["example_index"], []).append(row)
        elif row["path"] == baseline_path:
            baseline.setdefault(row["example_index"], []).append(row)
    if set(candidate) != set(baseline) or not candidate:
        raise RuntimeError("paired timing paths do not cover the same examples")
    example_indices = sorted(candidate)
    candidate_wall = np.array(
        [statistics.fmean(row["wall_ms"] for row in candidate[index]) for index in example_indices]
    )
    baseline_wall = np.array(
        [statistics.fmean(row["wall_ms"] for row in baseline[index]) for index in example_indices]
    )
    candidate_cuda = np.array(
        [statistics.fmean(row["cuda_ms"] for row in candidate[index]) for index in example_indices]
    )
    baseline_cuda = np.array(
        [statistics.fmean(row["cuda_ms"] for row in baseline[index]) for index in example_indices]
    )
    generator = np.random.default_rng(seed)
    wall_ratios = []
    cuda_ratios = []
    completed = 0
    while completed < repetitions:
        indices = generator.integers(0, len(example_indices), len(example_indices))
        wall_ratios.append(candidate_wall[indices].mean() / baseline_wall[indices].mean())
        cuda_ratios.append(candidate_cuda[indices].mean() / baseline_cuda[indices].mean())
        completed += 1
    return {
        "independent_documents": len(example_indices),
        "repeats_per_document": len(candidate[example_indices[0]]),
        "candidate_mean_wall_ms": float(candidate_wall.mean()),
        "native_mean_wall_ms": float(baseline_wall.mean()),
        "mean_wall_ratio": float(candidate_wall.mean() / baseline_wall.mean()),
        "mean_wall_ratio_bootstrap_95": [
            float(np.percentile(wall_ratios, 2.5)),
            float(np.percentile(wall_ratios, 97.5)),
        ],
        "candidate_mean_cuda_ms": float(candidate_cuda.mean()),
        "native_mean_cuda_ms": float(baseline_cuda.mean()),
        "mean_cuda_ratio": float(candidate_cuda.mean() / baseline_cuda.mean()),
        "mean_cuda_ratio_bootstrap_95": [
            float(np.percentile(cuda_ratios, 2.5)),
            float(np.percentile(cuda_ratios, 97.5)),
        ],
        "bootstrap_repetitions": repetitions,
    }


def add_policy_timing(summary, timing, sequence_length):
    baseline = timing["native_target_prefill"]
    full_path = "full_transfer_action"
    entropy_path = "transfer_entropy_overhead"
    for replay_text, condition in summary["conditions"].items():
        replay_length = int(replay_text)
        probe_path = f"probe_route:{replay_length}"
        for baseline_name, path, target_tokens in (
            ("full_transfer", full_path, 1),
            ("unconditional_replay", probe_path, replay_length + 1),
        ):
            condition["baselines"][baseline_name]["timing"] = {
                "path": path,
                "median_wall_ms": timing[path]["median_wall_ms"],
                "wall_savings_vs_native_prefill": 1
                - timing[path]["median_wall_ms"] / baseline["median_wall_ms"],
                "median_cuda_ms": timing[path]["median_cuda_ms"],
                "cuda_savings_vs_native_prefill": 1
                - timing[path]["median_cuda_ms"] / baseline["median_cuda_ms"],
                "target_token_equivalents": target_tokens,
                "target_token_savings_vs_native_prefill": 1 - target_tokens / sequence_length,
            }
        for score_name, labels in condition["scores"].items():
            if score_name == "commutator_js":
                score_components = {
                    full_path,
                    probe_path,
                    f"commutator_js_overhead:{replay_length}",
                }
            elif score_name == "commutator_cache_nmse":
                score_components = {
                    full_path,
                    probe_path,
                    f"commutator_cache_nmse_overhead:{replay_length}",
                }
            elif score_name == "transfer_entropy":
                score_components = {full_path, entropy_path}
            elif score_name == "transfer_negative_margin":
                score_components = {full_path, "transfer_margin_overhead"}
            elif score_name == "transfer_negative_max_probability":
                score_components = {full_path, "transfer_max_probability_overhead"}
            elif score_name == "random_score":
                score_components = set()
            else:
                raise RuntimeError(f"unknown timing score {score_name}")
            for label_name, result in labels.items():
                candidate_path = full_path
                if label_name == "full_transfer_top1_diff":
                    candidate_tokens = 1
                elif label_name == "replay_top1_diff":
                    candidate_path = probe_path
                    candidate_tokens = replay_length + 1
                else:
                    raise RuntimeError(f"unknown policy label {label_name}")
                accepted_coverage = result["test_at_calibrated_threshold"]["coverage"]
                reject_rate = 1 - accepted_coverage
                candidate_missing = candidate_path not in score_components
                score_wall_ms = sum(timing[path]["median_wall_ms"] for path in score_components)
                score_cuda_ms = sum(timing[path]["median_cuda_ms"] for path in score_components)
                expected_wall_ms = (
                    score_wall_ms
                    + accepted_coverage
                    * int(candidate_missing)
                    * timing[candidate_path]["median_wall_ms"]
                    + reject_rate * baseline["median_wall_ms"]
                )
                expected_cuda_ms = (
                    score_cuda_ms
                    + accepted_coverage
                    * int(candidate_missing)
                    * timing[candidate_path]["median_cuda_ms"]
                    + reject_rate * baseline["median_cuda_ms"]
                )
                score_target_tokens = int(full_path in score_components) + (
                    replay_length + 1
                ) * int(probe_path in score_components)
                expected_target_tokens = (
                    score_target_tokens
                    + accepted_coverage * int(candidate_missing) * candidate_tokens
                    + reject_rate * sequence_length
                )
                result["test_policy_timing"] = {
                    "score_components": sorted(score_components),
                    "candidate_path": candidate_path,
                    "candidate_already_computed_by_score": not candidate_missing,
                    "accepted_coverage": accepted_coverage,
                    "component_model_expected_wall_ms": expected_wall_ms,
                    "wall_savings_vs_native_prefill": 1
                    - expected_wall_ms / baseline["median_wall_ms"],
                    "component_model_expected_cuda_ms": expected_cuda_ms,
                    "cuda_savings_vs_native_prefill": 1
                    - expected_cuda_ms / baseline["median_cuda_ms"],
                    "expected_target_token_equivalents_with_fallback": expected_target_tokens,
                    "target_token_savings_vs_native_prefill": 1
                    - expected_target_tokens / sequence_length,
                }
                policy_path = f"policy__{score_name}__{label_name}:{replay_length}"
                measured = timing[policy_path]
                branch_means = measured["branches"]
                if accepted_coverage > 0 and "accepted" not in branch_means:
                    raise RuntimeError(f"accepted timing stratum is missing for {policy_path}")
                if reject_rate > 0 and "rejected" not in branch_means:
                    raise RuntimeError(f"rejected timing stratum is missing for {policy_path}")
                measured_wall_ms = 0.0
                measured_cuda_ms = 0.0
                if accepted_coverage > 0:
                    measured_wall_ms += accepted_coverage * branch_means["accepted"]["mean_wall_ms"]
                    measured_cuda_ms += accepted_coverage * branch_means["accepted"]["mean_cuda_ms"]
                if reject_rate > 0:
                    measured_wall_ms += reject_rate * branch_means["rejected"]["mean_wall_ms"]
                    measured_cuda_ms += reject_rate * branch_means["rejected"]["mean_cuda_ms"]
                result["test_policy_timing"]["complete_policy_measurement"] = {
                    "path": policy_path,
                    "samples": measured["samples"],
                    "timing_sample_accepted_fraction": measured["timing_sample_accepted_fraction"],
                    "branches": branch_means,
                    "test_coverage_weighted_mean_wall_ms": measured_wall_ms,
                    "wall_savings_vs_native_prefill": 1
                    - measured_wall_ms / baseline["mean_wall_ms"],
                    "test_coverage_weighted_mean_cuda_ms": measured_cuda_ms,
                    "cuda_savings_vs_native_prefill": 1
                    - measured_cuda_ms / baseline["mean_cuda_ms"],
                }
    summary["timing"] = {
        "paths": timing,
        "assumption": "source cache exists at handoff; fallback is a fresh target prefill",
    }
    primary = summary["primary_analysis"]
    replay_length = str(primary["replay_length"])
    score_name = primary["score"]
    label_name = primary["label"]
    baseline_score = primary["baseline_score"]
    primary_condition = summary["conditions"][replay_length]
    gate = primary_condition["scores"][score_name][label_name]
    comparison_key = f"commutator_js_vs_{baseline_score}"
    result = {
        "replay_length": primary["replay_length"],
        "score": score_name,
        "label": label_name,
        "baseline_score": baseline_score,
        "ranking": gate["test_ranking"],
        "calibrated_test_point": gate["test_at_calibrated_threshold"],
        "paired_bootstrap": primary_condition["paired_bootstrap"][label_name][comparison_key],
        "complete_policy_timing": gate["test_policy_timing"]["complete_policy_measurement"],
        "same_length_unconditional_replay": primary_condition["baselines"]["unconditional_replay"],
    }
    equal_compute_replay = primary["equal_compute_replay_length"]
    if equal_compute_replay is not None:
        result["target_forward_matched_unconditional_replay"] = {
            "replay_length": equal_compute_replay,
            "reason": (
                "the commutator computes one full-transfer anchor plus replay_length + 1 "
                "probe tokens before any fallback"
            ),
            **summary["conditions"][str(equal_compute_replay)]["baselines"]["unconditional_replay"],
        }
    summary["primary_result"] = result
