import math

import numpy as np
import torch
from scipy.stats import beta
from sklearn.metrics import average_precision_score


def jensen_shannon(left_logits, right_logits):
    left = left_logits.float().reshape(-1)
    right = right_logits.float().reshape(-1)
    left_log_prob = torch.log_softmax(left, dim=0)
    right_log_prob = torch.log_softmax(right, dim=0)
    mixture_log_prob = torch.logaddexp(left_log_prob, right_log_prob) - math.log(2)
    left_kl = torch.sum(left_log_prob.exp() * (left_log_prob - mixture_log_prob))
    right_kl = torch.sum(right_log_prob.exp() * (right_log_prob - mixture_log_prob))
    return 0.5 * (left_kl + right_kl)


def logit_metrics(left_logits, right_logits):
    left = left_logits.float().reshape(-1)
    right = right_logits.float().reshape(-1)
    left_top = torch.topk(left, 2)
    return {
        "js": jensen_shannon(left, right).item(),
        "top1_left": int(torch.argmax(left).item()),
        "top1_right": int(torch.argmax(right).item()),
        "top1_diff": bool(torch.argmax(left) != torch.argmax(right)),
        "max_abs": torch.max(torch.abs(left - right)).item(),
        "left_margin": (left_top.values[0] - left_top.values[1]).item(),
    }


def entropy(logits):
    log_prob = torch.log_softmax(logits.float().reshape(-1), dim=0)
    return -torch.sum(log_prob.exp() * log_prob).item()


def negative_margin(logits):
    top = torch.topk(logits.float().reshape(-1), 2).values
    return -(top[0] - top[1]).item()


def negative_max_probability(logits):
    return -torch.softmax(logits.float().reshape(-1), dim=0).max().item()


def transferred_confidence(logits):
    return {
        "transfer_negative_margin": negative_margin(logits),
        "transfer_negative_max_probability": negative_max_probability(logits),
    }


def control_summary(records, native_cache_nmse_limit):
    bf16_roundtrip_limit = 64 * torch.finfo(torch.bfloat16).eps ** 4
    failures = []
    for record in records:
        length = record["length"]
        replay_length = record["replay_length"]
        if record["full_vs_replay"]["top1_diff"]:
            failures.append(f"native chunking changed top1 at {length}/{replay_length}")
        if record["full_vs_replay_cache_nmse"] > native_cache_nmse_limit:
            failures.append(
                f"native chunking cache NMSE exceeded limit at {length}/{replay_length}"
            )
        if record["replay_vs_rebuilt"]["top1_diff"]:
            failures.append(f"cache rebuild changed top1 at {length}/{replay_length}")
        if record["replay_vs_rebuilt_cache_nmse"] != 0:
            failures.append(f"cache rebuild was not exact at {length}/{replay_length}")
        if record["rope_roundtrip_nmse"] > bf16_roundtrip_limit:
            failures.append(f"RoPE round-trip exceeded BF16 bound at {length}")
        for layers_per_target, identity in record["identity_translation"].items():
            if identity["prefix_nmse"] > bf16_roundtrip_limit:
                failures.append(
                    "identity prefix exceeded BF16 bound at "
                    f"{length}/{replay_length}/k{layers_per_target}"
                )
            continuation_limit = 64 * max(
                identity["prefix_nmse"],
                torch.finfo(torch.bfloat16).eps ** 4,
            )
            if identity["continuation_nmse"] > continuation_limit:
                failures.append(
                    "identity continuation exceeded its numerical floor at "
                    f"{length}/{replay_length}/k{layers_per_target}"
                )
            if identity["logits"]["top1_diff"]:
                failures.append(
                    f"identity translation changed top1 at {length}/{replay_length}/"
                    f"k{layers_per_target}"
                )
        if record["sdpa_vs_eager"]["top1_diff"]:
            failures.append(f"attention backend changed top1 at length {length}")
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "bf16_rope_nmse_limit": bf16_roundtrip_limit,
        "native_cache_nmse_limit": native_cache_nmse_limit,
        "max_native_chunk_js": max(row["full_vs_replay"]["js"] for row in records),
        "max_native_chunk_cache_nmse": max(row["full_vs_replay_cache_nmse"] for row in records),
        "max_backend_js": max(row["sdpa_vs_eager"]["js"] for row in records),
        "max_rope_roundtrip_nmse": max(row["rope_roundtrip_nmse"] for row in records),
        "max_identity_prefix_nmse": max(
            identity["prefix_nmse"]
            for row in records
            for identity in row["identity_translation"].values()
        ),
        "max_identity_continuation_nmse": max(
            identity["continuation_nmse"]
            for row in records
            for identity in row["identity_translation"].values()
        ),
    }


def wilson_upper(errors, count, confidence_z=1.6448536269514722):
    if count <= 0:
        raise RuntimeError("Wilson bound requires at least one accepted example")
    rate = errors / count
    z2 = confidence_z * confidence_z
    denominator = 1 + z2 / count
    center = rate + z2 / (2 * count)
    radius = confidence_z * math.sqrt(rate * (1 - rate) / count + z2 / (4 * count * count))
    return (center + radius) / denominator


def clopper_pearson_upper(errors, count, confidence=0.95):
    if count <= 0:
        raise RuntimeError("Clopper-Pearson bound requires at least one accepted example")
    if not 0 <= errors <= count:
        raise RuntimeError(f"invalid error count {errors}/{count}")
    if errors == count:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, count - errors))


def ordered_labels(records, score_name, label_name):
    for record in records:
        if score_name not in record["online"]:
            raise RuntimeError(f"online score {score_name} is missing")
        if label_name not in record["oracle"]:
            raise RuntimeError(f"oracle label {label_name} is missing")
    ordered = sorted(records, key=lambda row: row["online"][score_name])
    labels = np.array([int(row["oracle"][label_name]) for row in ordered])
    return ordered, labels


def complete_tie_boundaries(ordered, score_name):
    return [
        index + 1
        for index, row in enumerate(ordered)
        if index + 1 == len(ordered)
        or row["online"][score_name] != ordered[index + 1]["online"][score_name]
    ]


def summarize_score(records, score_name, label_name, risk_limit=0.05):
    ordered, labels = ordered_labels(records, score_name, label_name)
    count = len(ordered)
    if count == 0:
        raise RuntimeError("cannot summarize an empty record set")
    cumulative = np.cumsum(labels)
    risk = cumulative / np.arange(1, count + 1)
    fixed = {}
    for coverage in (0.25, 0.5, 0.75):
        accepted = max(1, math.floor(count * coverage))
        fixed[str(coverage)] = {
            "accepted": accepted,
            "risk": float(risk[accepted - 1]),
            "wilson_upper_95": wilson_upper(int(cumulative[accepted - 1]), accepted),
            "clopper_pearson_upper_95": clopper_pearson_upper(
                int(cumulative[accepted - 1]), accepted
            ),
        }
    safe_accepted = 0
    for accepted in complete_tie_boundaries(ordered, score_name):
        if wilson_upper(int(cumulative[accepted - 1]), accepted) <= risk_limit:
            safe_accepted = accepted
    scores = np.array([row["online"][score_name] for row in ordered])
    positive_count = int(labels.sum())
    ranking_defined = 0 < positive_count < count
    average_precision = float(average_precision_score(labels, scores)) if ranking_defined else None
    return {
        "score": score_name,
        "label": label_name,
        "count": count,
        "harm_prevalence": float(labels.mean()),
        "risk_at_coverage": fixed,
        "aurc": float(risk.mean()),
        "harm_positive_count": positive_count,
        "ranking_defined": ranking_defined,
        "harm_average_precision": average_precision,
        "descriptive_test_coverage_at_5pct_wilson": safe_accepted / count,
        "descriptive_test_accepted_at_5pct_wilson": safe_accepted,
    }


def calibration_rule(records, score_name, target_coverage):
    if not 0 < target_coverage <= 1:
        raise RuntimeError(f"target coverage must be in (0, 1], found {target_coverage}")
    ordered = sorted(records, key=lambda row: row["online"][score_name])
    target_accepted = max(1, math.floor(len(ordered) * target_coverage))
    threshold = ordered[target_accepted - 1]["online"][score_name]
    accepted = sum(row["online"][score_name] <= threshold for row in records)
    return {
        "threshold": threshold,
        "target_coverage": target_coverage,
        "accepted": accepted,
        "coverage": accepted / len(ordered),
        "labels_used": False,
    }


def threshold_result(records, score_name, label_name, threshold):
    if threshold is None:
        return {
            "accepted": 0,
            "coverage": 0.0,
            "risk": None,
            "wilson_upper_95": None,
            "clopper_pearson_upper_95": None,
        }
    accepted = [row for row in records if row["online"][score_name] <= threshold]
    if not accepted:
        return {
            "accepted": 0,
            "coverage": 0.0,
            "risk": None,
            "wilson_upper_95": None,
            "clopper_pearson_upper_95": None,
        }
    errors = sum(int(row["oracle"][label_name]) for row in accepted)
    return {
        "accepted": len(accepted),
        "coverage": len(accepted) / len(records),
        "risk": errors / len(accepted),
        "wilson_upper_95": wilson_upper(errors, len(accepted)),
        "clopper_pearson_upper_95": clopper_pearson_upper(errors, len(accepted)),
    }


def paired_bootstrap_comparison(
    records,
    primary_score,
    baseline_score,
    label_name,
    primary_threshold,
    baseline_threshold,
    repetitions,
    seed,
):
    labels = np.array([int(row["oracle"][label_name]) for row in records])
    primary = np.array([row["online"][primary_score] for row in records])
    baseline = np.array([row["online"][baseline_score] for row in records])
    positive_count = int(labels.sum())
    if positive_count == 0 or positive_count == len(labels):
        return {
            "defined": False,
            "label": label_name,
            "positive_count": positive_count,
            "count": len(labels),
        }
    generator = np.random.default_rng(seed)
    ap_differences = []
    risk_differences = []
    repetition = 0
    while repetition < repetitions:
        indices = generator.integers(0, len(records), len(records))
        sampled_labels = labels[indices]
        if 0 < sampled_labels.sum() < len(sampled_labels):
            primary_ap = average_precision_score(sampled_labels, primary[indices])
            baseline_ap = average_precision_score(sampled_labels, baseline[indices])
            ap_differences.append(primary_ap - baseline_ap)
        primary_accepted = primary[indices] <= primary_threshold
        baseline_accepted = baseline[indices] <= baseline_threshold
        if primary_accepted.any() and baseline_accepted.any():
            primary_risk = sampled_labels[primary_accepted].mean()
            baseline_risk = sampled_labels[baseline_accepted].mean()
            risk_differences.append(primary_risk - baseline_risk)
        repetition += 1
    risk_summary = {
        "defined": bool(risk_differences),
        "defined_repetitions": len(risk_differences),
    }
    if risk_differences:
        risk_summary.update(
            {
                "mean": float(np.mean(risk_differences)),
                "lower_95": float(np.percentile(risk_differences, 2.5)),
                "upper_95": float(np.percentile(risk_differences, 97.5)),
            }
        )
    return {
        "defined": True,
        "primary_score": primary_score,
        "baseline_score": baseline_score,
        "label": label_name,
        "repetitions": repetitions,
        "ap_difference": {
            "mean": float(np.mean(ap_differences)),
            "lower_95": float(np.percentile(ap_differences, 2.5)),
            "upper_95": float(np.percentile(ap_differences, 97.5)),
        },
        "fixed_threshold_risk_difference": risk_summary,
    }


def pilot_summary(
    calibration_records,
    test_records,
    replay_lengths,
    score_names,
    risk_limit=0.05,
    target_coverage=0.5,
    bootstrap_repetitions=2000,
    bootstrap_seed=1729,
    native_cache_nmse_limit=0.001,
):
    conditions = {}
    for replay_length in replay_lengths:
        calibration = [
            row for row in calibration_records if row["online"]["replay_length"] == replay_length
        ]
        test = [row for row in test_records if row["online"]["replay_length"] == replay_length]
        if len(calibration) == 0 or len(test) == 0:
            raise RuntimeError(
                f"missing calibration or test rows for replay length {replay_length}"
            )
        score_results = {}
        for score_name in score_names:
            rule = calibration_rule(calibration, score_name, target_coverage)
            labels = {}
            for label_name in ("full_transfer_top1_diff", "replay_top1_diff"):
                labels[label_name] = {
                    "calibration": rule,
                    "test_ranking": summarize_score(
                        test, score_name, label_name, risk_limit=risk_limit
                    ),
                    "test_at_calibrated_threshold": threshold_result(
                        test, score_name, label_name, rule["threshold"]
                    ),
                }
            score_results[score_name] = labels
        bootstrap = {}
        for label_index, label_name in enumerate(("full_transfer_top1_diff", "replay_top1_diff")):
            bootstrap[label_name] = {}
            for baseline_index, baseline_score in enumerate(
                (
                    "transfer_negative_margin",
                    "transfer_negative_max_probability",
                    "transfer_entropy",
                )
            ):
                bootstrap[label_name][f"commutator_js_vs_{baseline_score}"] = (
                    paired_bootstrap_comparison(
                        test,
                        "commutator_js",
                        baseline_score,
                        label_name,
                        score_results["commutator_js"][label_name]["calibration"]["threshold"],
                        score_results[baseline_score][label_name]["calibration"]["threshold"],
                        bootstrap_repetitions,
                        bootstrap_seed + 100 * replay_length + 10 * label_index + baseline_index,
                    )
                )
        full_errors = sum(int(row["oracle"]["full_transfer_top1_diff"]) for row in test)
        replay_errors = sum(int(row["oracle"]["replay_top1_diff"]) for row in test)
        native_chunk_errors = sum(int(row["oracle"]["native_chunk_top1_diff"]) for row in test)
        calibration_native_chunk_errors = sum(
            int(row["oracle"]["native_chunk_top1_diff"]) for row in calibration
        )
        native_chunk_js = [row["oracle"]["native_chunk_js"] for row in test]
        native_chunk_cache = [row["oracle"]["native_chunk_cache_nmse"] for row in test]
        calibration_native_chunk_cache = [
            row["oracle"]["native_chunk_cache_nmse"] for row in calibration
        ]
        commutator_js = [row["online"]["commutator_js"] for row in test]
        commutator_cache = [row["online"]["commutator_cache_nmse"] for row in test]
        conditions[str(replay_length)] = {
            "calibration_count": len(calibration),
            "test_count": len(test),
            "full_transfer_risk": full_errors / len(test),
            "replay_repair_risk": replay_errors / len(test),
            "baselines": {
                "full_transfer": {
                    "errors": full_errors,
                    "risk": full_errors / len(test),
                    "wilson_upper_95": wilson_upper(full_errors, len(test)),
                    "clopper_pearson_upper_95": clopper_pearson_upper(full_errors, len(test)),
                },
                "unconditional_replay": {
                    "errors": replay_errors,
                    "risk": replay_errors / len(test),
                    "wilson_upper_95": wilson_upper(replay_errors, len(test)),
                    "clopper_pearson_upper_95": clopper_pearson_upper(replay_errors, len(test)),
                },
            },
            "native_chunk_control": {
                "passed": max(max(native_chunk_cache), max(calibration_native_chunk_cache))
                <= native_cache_nmse_limit,
                "interpretation": (
                    "top1 differences measure the native execution-path action-noise floor; "
                    "cache NMSE is the hard validity gate"
                ),
                "calibration_top1_errors": calibration_native_chunk_errors,
                "calibration_top1_error_rate": calibration_native_chunk_errors / len(calibration),
                "test_top1_errors": native_chunk_errors,
                "test_top1_error_rate": native_chunk_errors / len(test),
                "cache_nmse_limit": native_cache_nmse_limit,
                "max_js": max(native_chunk_js),
                "median_js": float(np.median(native_chunk_js)),
                "max_cache_nmse": max(native_chunk_cache),
                "calibration_max_cache_nmse": max(calibration_native_chunk_cache),
                "median_cache_nmse": float(np.median(native_chunk_cache)),
                "commutator_js_above_native_floor_fraction": float(
                    np.mean(np.array(commutator_js) > np.array(native_chunk_js))
                ),
                "commutator_cache_above_native_floor_fraction": float(
                    np.mean(np.array(commutator_cache) > np.array(native_chunk_cache))
                ),
            },
            "paired_bootstrap": bootstrap,
            "scores": score_results,
        }
    return {
        "conditions": conditions,
        "score_names": score_names,
        "risk_limit": risk_limit,
        "target_coverage": target_coverage,
        "native_controls_passed": all(
            condition["native_chunk_control"]["passed"] for condition in conditions.values()
        ),
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": bootstrap_seed,
    }
