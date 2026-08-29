import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


args = parse_args()
scaling = read_json(args.scaling)
analysis = read_json(args.analysis)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }
)

# Plot paired uncertainty, not only medians, because the 4096-token conclusion
# changes from "slower" to "indeterminate" once document variation is retained.
lengths = [256, 1024, 2048, 4096]
ratios = [scaling["conditions"][str(length)]["latency"]["mean_wall_ratio"] for length in lengths]
intervals = [
    scaling["conditions"][str(length)]["latency"]["mean_wall_ratio_bootstrap_95"]
    for length in lengths
]
lower = np.array([ratio - interval[0] for ratio, interval in zip(ratios, intervals, strict=True)])
upper = np.array([interval[1] - ratio for ratio, interval in zip(ratios, intervals, strict=True)])
figure, axis = plt.subplots(figsize=(7.2, 4.1))
axis.errorbar(
    lengths,
    ratios,
    yerr=np.vstack([lower, upper]),
    color="#2457C5",
    marker="o",
    markersize=6,
    linewidth=2,
    capsize=4,
)
axis.axhline(1, color="#252525", linestyle="--", linewidth=1)
axis.set_xscale("log", base=2)
axis.set_xticks(lengths, [str(length) for length in lengths])
axis.set_xlabel("retained context tokens")
axis.set_ylabel("complete gate / native wall time")
axis.set_title("bounded replay approaches break-even only at long context")
axis.text(270, 1.07, "native prefill", color="#252525")
axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
figure.tight_layout()
figure.savefig(
    output_dir / "latency-scaling.svg",
    format="svg",
    metadata={"Date": None, "Creator": "KV-Replay direct numerical artifact"},
)
figure.savefig(
    output_dir / "latency-scaling.png",
    dpi=180,
    metadata={"Software": "KV-Replay direct numerical artifact"},
)
plt.close(figure)

# These paired counts show why a selector is still worth studying: replay fixes
# many direct errors, but it also creates new errors and both paths can be wrong.
overlap = analysis["full_vs_replay_error_overlap"]
translator = analysis["k1_vs_k4_direct_harm"]
panels = [
    (
        "direct transfer vs 64-token replay",
        [
            overlap["both_correct"],
            overlap["direct_wrong_replay_correct"],
            overlap["direct_correct_replay_wrong"],
            overlap["both_wrong"],
        ],
        ["both correct", "replay fixes direct", "replay breaks direct", "both wrong"],
    ),
    (
        "k=1 vs k=4 translator",
        [
            translator["both_correct"],
            translator["k1_only_wrong"],
            translator["k4_only_wrong"],
            translator["both_wrong"],
        ],
        ["both correct", "k=4 fixes k=1", "k=4 adds error", "both wrong"],
    ),
]
colors = ["#B9D8C2", "#2457C5", "#E5A642", "#C94949"]
figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharex=True)
for axis, (title, counts, labels) in zip(axes, panels, strict=True):
    left = 0
    for count, label, color in zip(counts, labels, colors, strict=True):
        fraction = count / sum(counts)
        axis.barh([0], [fraction], left=left, height=0.45, color=color, label=label)
        if fraction >= 0.04:
            axis.text(
                left + fraction / 2,
                0,
                str(count),
                ha="center",
                va="center",
                color="white" if color != colors[0] else "#252525",
                fontweight="bold",
            )
        left += fraction
    axis.set_title(title)
    axis.set_yticks([])
    axis.set_xlim(0, 1)
    axis.set_xlabel("fraction of 256 paired test documents")
    axis.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25))
figure.suptitle("paired errors move rather than disappear", fontweight="bold")
figure.tight_layout()
figure.savefig(
    output_dir / "paired-errors.svg",
    format="svg",
    bbox_inches="tight",
    metadata={"Date": None, "Creator": "KV-Replay direct numerical artifact"},
)
figure.savefig(
    output_dir / "paired-errors.png",
    dpi=180,
    bbox_inches="tight",
    metadata={"Software": "KV-Replay direct numerical artifact"},
)
plt.close(figure)
