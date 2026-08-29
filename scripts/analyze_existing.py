import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from kv_research.artifacts import read_jsonl, sha256_file, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--k1", required=True)
    parser.add_argument("--scaling", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=811)
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def rows_for(records, partition, replay_length):
    return [
        row
        for row in records
        if row["example"]["partition"] == partition
        and row["online"]["replay_length"] == replay_length
    ]


def features(rows, augmented):
    values = []
    for row in rows:
        feature = [row["online"]["transfer_negative_margin"]]
        if augmented:
            feature.append(np.log(max(row["online"]["commutator_js"], 0.0) + 1e-12))
        values.append(feature)
    return np.asarray(values, dtype=np.float64)


def labels(rows):
    return np.asarray([row["oracle"]["full_transfer_top1_diff"] for row in rows], dtype=np.int64)


def fit_probabilities(train_x, train_y, test_x, seed):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    model.fit((train_x - mean) / scale, train_y)
    return model.predict_proba((test_x - mean) / scale)[:, 1]


def conditional_information(calibration, test, repetitions, seed):
    calibration_y = labels(calibration)
    test_y = labels(test)
    baseline_calibration = features(calibration, False)
    augmented_calibration = features(calibration, True)
    baseline_test = features(test, False)
    augmented_test = features(test, True)
    baseline_probability = fit_probabilities(
        baseline_calibration, calibration_y, baseline_test, seed
    )
    augmented_probability = fit_probabilities(
        augmented_calibration, calibration_y, augmented_test, seed
    )
    generator = np.random.default_rng(seed)
    loss_improvements = []
    ap_improvements = []
    for bootstrap_index in range(repetitions):
        calibration_index = generator.integers(0, len(calibration), len(calibration))
        test_index = generator.integers(0, len(test), len(test))
        calibration_sample_y = calibration_y[calibration_index]
        if len(np.unique(calibration_sample_y)) != 2:
            raise RuntimeError("bootstrap calibration sample contains one class")
        baseline = fit_probabilities(
            baseline_calibration[calibration_index],
            calibration_sample_y,
            baseline_test[test_index],
            seed + bootstrap_index + 1,
        )
        augmented = fit_probabilities(
            augmented_calibration[calibration_index],
            calibration_sample_y,
            augmented_test[test_index],
            seed + bootstrap_index + 1,
        )
        sample_y = test_y[test_index]
        loss_improvements.append(
            log_loss(sample_y, baseline, labels=[0, 1])
            - log_loss(sample_y, augmented, labels=[0, 1])
        )
        ap_improvements.append(
            average_precision_score(sample_y, augmented)
            - average_precision_score(sample_y, baseline)
        )
    return {
        "calibration_documents": len(calibration),
        "test_documents": len(test),
        "model": "standardized logistic regression, L2 C=1",
        "baseline_features": ["transfer_negative_margin"],
        "augmented_features": [
            "transfer_negative_margin",
            "log(max(commutator_js, 0) + 1e-12)",
        ],
        "baseline_test_log_loss": log_loss(test_y, baseline_probability, labels=[0, 1]),
        "augmented_test_log_loss": log_loss(test_y, augmented_probability, labels=[0, 1]),
        "test_log_loss_improvement": log_loss(test_y, baseline_probability, labels=[0, 1])
        - log_loss(test_y, augmented_probability, labels=[0, 1]),
        "test_log_loss_improvement_bootstrap_95": [
            float(np.percentile(loss_improvements, 2.5)),
            float(np.percentile(loss_improvements, 97.5)),
        ],
        "baseline_test_brier": brier_score_loss(test_y, baseline_probability),
        "augmented_test_brier": brier_score_loss(test_y, augmented_probability),
        "baseline_test_average_precision": average_precision_score(test_y, baseline_probability),
        "augmented_test_average_precision": average_precision_score(test_y, augmented_probability),
        "test_average_precision_improvement_bootstrap_95": [
            float(np.percentile(ap_improvements, 2.5)),
            float(np.percentile(ap_improvements, 97.5)),
        ],
        "bootstrap_repetitions": repetitions,
        "bootstrap_refits_router": True,
    }


def error_overlap(rows, threshold):
    counts = {
        "both_correct": 0,
        "direct_wrong_replay_correct": 0,
        "direct_correct_replay_wrong": 0,
        "both_wrong": 0,
    }
    accepted = []
    false_accept_native_margins = []
    harmful_reject_native_margins = []
    commutator_above_native_floor = 0
    for row in rows:
        direct_wrong = row["oracle"]["full_transfer_top1_diff"]
        replay_wrong = row["oracle"]["replay_top1_diff"]
        if direct_wrong and replay_wrong:
            counts["both_wrong"] += 1
        elif direct_wrong:
            counts["direct_wrong_replay_correct"] += 1
        elif replay_wrong:
            counts["direct_correct_replay_wrong"] += 1
        else:
            counts["both_correct"] += 1
        is_accepted = row["online"]["commutator_js"] <= threshold
        if is_accepted:
            accepted.append(row)
            if direct_wrong:
                false_accept_native_margins.append(row["oracle"]["native_margin"])
        elif direct_wrong:
            harmful_reject_native_margins.append(row["oracle"]["native_margin"])
        if row["online"]["commutator_js"] > row["oracle"]["native_chunk_js"]:
            commutator_above_native_floor += 1
    accepted_direct_errors = sum(row["oracle"]["full_transfer_top1_diff"] for row in accepted)
    accepted_replay_errors = sum(row["oracle"]["replay_top1_diff"] for row in accepted)
    return {
        **counts,
        "documents": len(rows),
        "oracle_full_or_replay_selector_risk": counts["both_wrong"] / len(rows),
        "oracle_zero_error_coverage_with_native_fallback": 1 - counts["both_wrong"] / len(rows),
        "direct_errors_repaired_by_replay": counts["direct_wrong_replay_correct"],
        "replay_errors_introduced_after_direct_agreement": counts["direct_correct_replay_wrong"],
        "accepted": len(accepted),
        "accepted_direct_action_errors": accepted_direct_errors,
        "accepted_direct_action_risk": accepted_direct_errors / len(accepted),
        "accepted_replay_action_errors": accepted_replay_errors,
        "accepted_replay_action_risk": accepted_replay_errors / len(accepted),
        "median_native_margin_false_accepts": float(np.median(false_accept_native_margins)),
        "median_native_margin_harmful_rejects": float(np.median(harmful_reject_native_margins)),
        "commutator_above_native_chunk_js_fraction": commutator_above_native_floor / len(rows),
    }


def translator_pairing(k4_rows, k1_rows):
    k4 = {row["example"]["token_sha256"]: row for row in k4_rows}
    k1 = {row["example"]["token_sha256"]: row for row in k1_rows}
    if set(k4) != set(k1):
        raise RuntimeError("k=1 and k=4 test examples differ")
    counts = {"both_correct": 0, "k1_only_wrong": 0, "k4_only_wrong": 0, "both_wrong": 0}
    for token_sha256 in k4:
        k4_wrong = k4[token_sha256]["oracle"]["full_transfer_top1_diff"]
        k1_wrong = k1[token_sha256]["oracle"]["full_transfer_top1_diff"]
        if k1_wrong and k4_wrong:
            counts["both_wrong"] += 1
        elif k1_wrong:
            counts["k1_only_wrong"] += 1
        elif k4_wrong:
            counts["k4_only_wrong"] += 1
        else:
            counts["both_correct"] += 1
    return {**counts, "documents": len(k4)}


args = parse_args()
repo_root = Path(__file__).resolve().parents[1]
run_dirs = {
    "primary": (repo_root / args.primary).resolve(),
    "k1": (repo_root / args.k1).resolve(),
    "scaling": (repo_root / args.scaling).resolve(),
}
for name, run_dir in run_dirs.items():
    if repo_root.resolve() not in run_dir.parents or not run_dir.is_dir():
        raise RuntimeError(f"{name} run is absent or outside repository")

primary_records = read_jsonl(run_dirs["primary"] / "records.jsonl")
k1_records = read_jsonl(run_dirs["k1"] / "records.jsonl")
primary_summary = read_json(run_dirs["primary"] / "summary.json")
scaling_summary = read_json(run_dirs["scaling"] / "summary.json")
calibration = rows_for(primary_records, "calibration", 64)
test = rows_for(primary_records, "test", 64)
k1_test = rows_for(k1_records, "test", 64)
threshold = primary_summary["conditions"]["64"]["scores"]["commutator_js"][
    "full_transfer_top1_diff"
]["calibration"]["threshold"]

scaling_break_even = {}
for length, condition in scaling_summary["conditions"].items():
    latency = condition["latency"]
    fallback_rate = 1 - condition["gate_coverage"]
    break_even = latency["mean_wall_ratio"] - fallback_rate
    interval = [bound - fallback_rate for bound in latency["mean_wall_ratio_bootstrap_95"]]
    scaling_break_even[length] = {
        "actual_coverage": condition["gate_coverage"],
        "complete_policy_to_native_wall_ratio": latency["mean_wall_ratio"],
        "score_only_break_even_coverage": break_even,
        "score_only_break_even_coverage_bootstrap_95": interval,
        "point_estimate_can_beat_native": break_even < 1,
        "actual_coverage_exceeds_point_break_even": condition["gate_coverage"] > break_even,
    }

result = {
    "analysis_status": "post_hoc_direction_finding_not_confirmatory",
    "inputs": {
        name: {
            "run": str(run_dir.relative_to(repo_root)),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "records_sha256": sha256_file(run_dir / "records.jsonl"),
        }
        for name, run_dir in run_dirs.items()
    },
    "primary_replay_length": 64,
    "calibration_threshold": threshold,
    "full_vs_replay_error_overlap": error_overlap(test, threshold),
    "k1_vs_k4_direct_harm": translator_pairing(test, k1_test),
    "conditional_information": conditional_information(
        calibration,
        test,
        args.bootstrap_repetitions,
        args.seed,
    ),
    "context_break_even": scaling_break_even,
}
output_path = (repo_root / args.output).resolve()
if repo_root.resolve() not in output_path.parents:
    raise RuntimeError("analysis output path escapes repository")
write_json(output_path, result)
print(json.dumps(result, indent=2, sort_keys=True))
