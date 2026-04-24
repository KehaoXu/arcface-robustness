#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

METRICS = [
    "accuracy",
    "tar",
    "far",
    "frr",
    "genuine_mean",
    "genuine_std",
    "impostor_mean",
    "impostor_std",
]
STEP_COLOR_MAP = {
    "none": "#0072B2",
    "2": "#E69F00",
    "4": "#CC79A7",
    "8": "#D55E00",
}
BASELINE_COLOR = "#444444"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize aggregated grid-search metrics for degradation and ResShift experiments."
    )
    parser.add_argument(
        "--input-csv",
        default="results/grid_metrics_summary.csv",
        help="Path to the aggregated grid metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/grid_figures",
        help="Directory where visualization PNG files will be written.",
    )
    parser.add_argument(
        "--include-step8",
        action="store_true",
        help="Include experiments with inference_steps=8. By default they are excluded.",
    )
    return parser.parse_args()


def import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required to generate figures. "
            "Please run this script in an environment where matplotlib is installed."
        ) from exc
    return plt


def load_rows(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))

    for row in rows:
        for metric in METRICS:
            row[metric] = float(row[metric])
        row["threshold"] = float(row["threshold"])
        row["valid_pairs"] = int(row["valid_pairs"])
        row["invalid_pairs"] = int(row["invalid_pairs"])
        row["downsample_scale_num"] = (
            float(row["downsample_scale"]) if row["downsample_scale"] else None
        )
        row["gaussian_sigma_num"] = float(row["gaussian_sigma"]) if row["gaussian_sigma"] else None
        row["d_prime"] = calculate_d_prime(row)
        row["genuine_cv"] = calculate_coefficient_of_variation(
            row["genuine_std"], row["genuine_mean"]
        )
        row["score_separation_margin"] = row["genuine_mean"] - row["impostor_mean"]
        row["js_distance"] = calculate_js_distance_from_results_dir(row["results_dir"])
        row.update(load_embedding_drift_metrics(row["results_dir"]))
    return rows


def calculate_coefficient_of_variation(std_value, mean_value):
    if mean_value == 0:
        return 0.0
    return std_value / mean_value


def calculate_d_prime(row):
    pooled_std = ((row["genuine_std"] ** 2 + row["impostor_std"] ** 2) / 2) ** 0.5
    if pooled_std == 0:
        return 0.0
    return (row["genuine_mean"] - row["impostor_mean"]) / pooled_std


def load_valid_scores_by_label(scores_csv_path):
    genuine_scores = []
    impostor_scores = []
    with Path(scores_csv_path).open("r", newline="", encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            if row["valid"] != "1":
                continue
            score = float(row["similarity_score"])
            if row["label"] == "1":
                genuine_scores.append(score)
            elif row["label"] == "0":
                impostor_scores.append(score)
    return genuine_scores, impostor_scores


def read_json(json_path):
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def load_embedding_drift_metrics(results_dir):
    drift_path = Path(results_dir) / "embedding_drift_summary.json"
    defaults = {
        "drift_valid_samples": 0,
        "drift_invalid_samples": 0,
        "cosine_drift_mean": 0.0,
        "cosine_drift_std": 0.0,
        "cosine_drift_median": 0.0,
        "cosine_drift_q1": 0.0,
        "cosine_drift_q3": 0.0,
        "euclidean_drift_mean": 0.0,
        "euclidean_drift_std": 0.0,
    }
    if not drift_path.exists():
        return defaults

    summary = read_json(drift_path)
    return {
        "drift_valid_samples": int(summary.get("valid_samples", 0)),
        "drift_invalid_samples": int(summary.get("invalid_samples", 0)),
        "cosine_drift_mean": float(summary.get("cosine_mean", 0.0)),
        "cosine_drift_std": float(summary.get("cosine_std", 0.0)),
        "cosine_drift_median": float(summary.get("cosine_median", 0.0)),
        "cosine_drift_q1": float(summary.get("cosine_q1", 0.0)),
        "cosine_drift_q3": float(summary.get("cosine_q3", 0.0)),
        "euclidean_drift_mean": float(summary.get("euclidean_mean", 0.0)),
        "euclidean_drift_std": float(summary.get("euclidean_std", 0.0)),
    }


def parse_float(value):
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return float(value)


def load_roc_points(roc_csv_path):
    with Path(roc_csv_path).open("r", newline="", encoding="utf-8") as file_obj:
        return [
            {
                "threshold": parse_float(row["threshold"]),
                "tar": float(row["tar"]),
                "far": float(row["far"]),
                "frr": float(row["frr"]),
            }
            for row in csv.DictReader(file_obj)
        ]


def find_closest_threshold_point(roc_points, threshold):
    finite_points = [point for point in roc_points if math.isfinite(point["threshold"])]
    if not finite_points:
        return None
    return min(finite_points, key=lambda point: abs(point["threshold"] - threshold))


def find_eer_point_from_roc_points(roc_points):
    finite_points = [point for point in roc_points if math.isfinite(point["threshold"])]
    if not finite_points:
        return None
    return min(finite_points, key=lambda point: abs(point["far"] - point["frr"]))


def build_probability_histogram(scores, bin_count, score_min, score_max):
    if not scores:
        return []
    if score_max <= score_min:
        return [1.0]

    counts = [0] * bin_count
    bin_width = (score_max - score_min) / bin_count
    for score in scores:
        index = int((score - score_min) / bin_width)
        if index >= bin_count:
            index = bin_count - 1
        elif index < 0:
            index = 0
        counts[index] += 1

    total = sum(counts)
    return [count / total for count in counts]


def calculate_js_distance(genuine_scores, impostor_scores, bin_count=100, score_range=(-1.0, 1.0)):
    if not genuine_scores or not impostor_scores:
        return 0.0

    score_min, score_max = score_range
    genuine_hist = build_probability_histogram(genuine_scores, bin_count, score_min, score_max)
    impostor_hist = build_probability_histogram(impostor_scores, bin_count, score_min, score_max)

    js_divergence = 0.0
    for genuine_prob, impostor_prob in zip(genuine_hist, impostor_hist):
        mixture_prob = 0.5 * (genuine_prob + impostor_prob)
        if genuine_prob > 0:
            js_divergence += 0.5 * genuine_prob * math.log2(genuine_prob / mixture_prob)
        if impostor_prob > 0:
            js_divergence += 0.5 * impostor_prob * math.log2(impostor_prob / mixture_prob)

    return math.sqrt(js_divergence)


def calculate_js_distance_from_results_dir(results_dir):
    scores_csv_path = Path(results_dir) / "pair_scores_with_predictions.csv"
    if not scores_csv_path.exists():
        return 0.0
    genuine_scores, impostor_scores = load_valid_scores_by_label(scores_csv_path)
    return calculate_js_distance(genuine_scores, impostor_scores)


def infer_condition_value(row):
    if row["condition_type"] == "downsample":
        return row["downsample_scale_num"]
    if row["condition_type"] == "noise":
        return row["gaussian_sigma_num"]
    return None


def format_condition_label(row):
    if row["experiment_id"] == "baseline":
        return "baseline"
    if row["condition_type"] == "downsample":
        return f"S {row['downsample_scale_num']:.2f}"
    if row["condition_type"] == "noise":
        return f"N {int(row['gaussian_sigma_num']):02d}"
    return row["experiment_id"]


def format_step_label(step):
    return "No ResShift" if step == "none" else f"ResShift steps={step}"


def format_experiment_short_label(row):
    if row["experiment_id"] == "baseline":
        return "Baseline"

    if row["condition_type"] == "downsample":
        base_label = f"S{row['downsample_scale_num']:.2f}"
    elif row["condition_type"] == "noise":
        base_label = f"N{int(row['gaussian_sigma_num']):02d}"
    else:
        base_label = row["experiment_id"]

    return base_label


def get_scatter_marker(row):
    if row["experiment_id"] == "baseline":
        return "X"
    if row["condition_type"] == "downsample":
        return "o"
    if row["condition_type"] == "noise":
        return "^"
    return "o"


def get_scatter_size(row):
    return 110 if row["experiment_id"] == "baseline" else 80


def group_by_condition(rows):
    grouped = {}
    for row in rows:
        key = (row["condition_type"], infer_condition_value(row))
        grouped.setdefault(key, []).append(row)
    for key in grouped:
        grouped[key].sort(key=lambda item: (item["inference_steps"] != "none", item["inference_steps"]))
    return grouped


def filter_rows(rows, include_step8=False):
    if include_step8:
        return rows
    return [row for row in rows if row["inference_steps"] != "8"]


def get_step_sort_key(step):
    return 0 if step == "none" else int(step)


def get_step_color(row):
    if row["experiment_id"] == "baseline":
        return BASELINE_COLOR
    return STEP_COLOR_MAP.get(row["inference_steps"], "#999999")


def build_step_legend_handles(plt, rows):
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=BASELINE_COLOR, label="Baseline"),
        plt.Rectangle((0, 0), 1, 1, color=STEP_COLOR_MAP["none"], label="No Resshift"),
        plt.Rectangle((0, 0), 1, 1, color=STEP_COLOR_MAP["2"], label="Resshift steps=2"),
        plt.Rectangle((0, 0), 1, 1, color=STEP_COLOR_MAP["4"], label="Resshift steps=4"),
    ]
    if any(row["inference_steps"] == "8" for row in rows):
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=STEP_COLOR_MAP["8"], label="Resshift steps=8")
        )
    return legend_handles


def plot_metric_ranking(rows, output_path):
    plt = import_matplotlib()
    sorted_rows = sorted(rows, key=lambda row: row["accuracy"], reverse=True)
    labels = [format_condition_label(row) for row in sorted_rows]
    values = [row["accuracy"] for row in sorted_rows]
    colors = [get_step_color(row) for row in sorted_rows]

    plt.figure(figsize=(14, 6))
    positions = list(range(len(sorted_rows)))
    plt.bar(positions, values, color=colors)
    plt.xticks(positions, labels, rotation=55, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(min(values) - 0.01, max(values) + 0.003)
    plt.title("Accuracy Ranking Across Grid Experiments")
    plt.legend(handles=build_step_legend_handles(plt, sorted_rows), frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tradeoff_scatter(rows, output_path):
    plt = import_matplotlib()
    plt.figure(figsize=(8, 6))
    annotation_offsets = [
        (4, 4),
        (5, 0),
        (4, -5),
        (6, 6),
        (6, -7),
        (-22, 4),
        (-24, -4),
        (0, 6),
        (0, -7),
        (8, 2),
        (-16, 7),
        (7, -10),
    ]
    sorted_rows = sorted(rows, key=lambda row: (row["far"], row["tar"], row["experiment_id"]))

    for index, row in enumerate(sorted_rows):
        color = get_step_color(row)
        marker = get_scatter_marker(row)
        size = get_scatter_size(row)
        plt.scatter(row["far"], row["tar"], s=size, c=color, marker=marker, alpha=0.9)
        offset_x, offset_y = annotation_offsets[index % len(annotation_offsets)]
        plt.annotate(
            format_experiment_short_label(row),
            (row["far"], row["tar"]),
            fontsize=8,
            alpha=0.9,
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.65},
        )

    plt.xlabel("FAR")
    plt.ylabel("TAR")
    plt.title("Verification Trade-off by Experiment")
    plt.grid(alpha=0.25, linestyle="--")
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            markerfacecolor=BASELINE_COLOR,
            markeredgecolor=BASELINE_COLOR,
            markersize=9,
            label="Baseline",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=STEP_COLOR_MAP["none"],
            markeredgecolor=STEP_COLOR_MAP["none"],
            markersize=8,
            label="No ResShift",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=STEP_COLOR_MAP["2"],
            markeredgecolor=STEP_COLOR_MAP["2"],
            markersize=8,
            label="ResShift steps=2",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=STEP_COLOR_MAP["4"],
            markeredgecolor=STEP_COLOR_MAP["4"],
            markersize=8,
            label="ResShift steps=4",
        ),
    ]
    if any(row["inference_steps"] == "8" for row in sorted_rows):
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=STEP_COLOR_MAP["8"],
                markeredgecolor=STEP_COLOR_MAP["8"],
                markersize=8,
                label="ResShift step=8",
            )
        )
    legend_handles.extend(
        [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="#555555",
                markersize=8,
                label="Downsample",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="^",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="#555555",
                markersize=8,
                label="Noise",
            ),
        ]
    )
    plt.legend(handles=legend_handles, frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_condition_trends(rows, output_path, condition_type, metric_specs, figure_suffix):
    plt = import_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=False)
    style_map = {
        "none": (STEP_COLOR_MAP["none"], "o"),
        "2": (STEP_COLOR_MAP["2"], "s"),
        "4": (STEP_COLOR_MAP["4"], "^"),
        "8": (STEP_COLOR_MAP["8"], "D"),
    }

    for row in rows:
        row["condition_value"] = infer_condition_value(row)

    condition_rows = sorted(
        [row for row in rows if row["condition_type"] == condition_type],
        key=lambda row: (row["condition_value"], row["inference_steps"]),
    )
    if not condition_rows:
        plt.close(fig)
        return

    baseline_row = next((row for row in rows if row["experiment_id"] == "baseline"), None)
    condition_values = sorted({row["condition_value"] for row in condition_rows})

    for axis, (metric, title) in zip(axes, metric_specs):
        if baseline_row is not None:
            axis.axhline(
                baseline_row[metric],
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=1.5,
                label="baseline",
                alpha=0.8,
            )
        for step in ["none", "2", "4", "8"]:
            step_rows = [row for row in condition_rows if row["inference_steps"] == step]
            if not step_rows:
                continue
            xs = [row["condition_value"] for row in step_rows]
            ys = [row[metric] for row in step_rows]
            color, marker = style_map[step]
            axis.plot(xs, ys, color=color, marker=marker, linewidth=2, label=format_step_label(step))
        axis.set_title(title)
        axis.set_xlabel("Downsample Scale" if condition_type == "downsample" else "Gaussian Sigma")
        axis.grid(alpha=0.25, linestyle="--")
        if condition_type == "downsample":
            axis.set_xticks(condition_values)
            axis.set_xticklabels([f"{value:.2f}" for value in condition_values])
        else:
            axis.set_xticks(condition_values)
            axis.set_xticklabels([f"{int(value)}" for value in condition_values])

    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.92))
    title_name = "Downsample" if condition_type == "downsample" else "Noise"
    fig.suptitle(f"{title_name} {figure_suffix} Across Severity and ResShift Steps", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_verification_condition_trends(rows, output_path, condition_type):
    plot_condition_trends(
        rows,
        output_path,
        condition_type,
        [
            ("accuracy", "Accuracy"),
            ("tar", "TAR"),
            ("far", "FAR"),
        ],
        "Verification Metrics",
    )


def plot_genuine_condition_trends(rows, output_path, condition_type):
    plot_condition_trends(
        rows,
        output_path,
        condition_type,
        [
            ("genuine_mean", "Genuine Mean"),
            ("genuine_std", "Genuine Std"),
            ("genuine_cv", "Genuine Coefficient of Variation (CV)"),
        ],
        "Genuine Score Statistics",
    )


def plot_within_condition_comparisons(rows, output_path):
    plt = import_matplotlib()
    baseline_row = next((row for row in rows if row["experiment_id"] == "baseline"), None)
    grouped = group_by_condition([row for row in rows if row["experiment_id"] != "baseline"])
    ordered_keys = sorted(grouped, key=lambda item: (item[0], item[1]))
    if not ordered_keys:
        return

    fig, axes = plt.subplots(len(ordered_keys), 3, figsize=(14, 3.4 * len(ordered_keys)), sharex=False)
    if len(ordered_keys) == 1:
        axes = [axes]

    metric_specs = [
        ("accuracy", "Accuracy"),
        ("tar", "TAR"),
        ("far", "FAR"),
    ]
    color_map = {
        "none": STEP_COLOR_MAP["none"],
        "2": STEP_COLOR_MAP["2"],
        "4": STEP_COLOR_MAP["4"],
        "8": STEP_COLOR_MAP["8"],
    }

    def format_compact_step_label(step):
        return "No Res" if step == "none" else f"Res {step}"

    for row_axes, key in zip(axes, ordered_keys):
        condition_rows = sorted(grouped[key], key=lambda row: get_step_sort_key(row["inference_steps"]))
        step_labels = [format_compact_step_label(row["inference_steps"]) for row in condition_rows]
        colors = [color_map.get(row["inference_steps"], "#999999") for row in condition_rows]
        if baseline_row is not None:
            step_labels = ["Baseline", *step_labels]
            colors = [BASELINE_COLOR, *colors]
        title_prefix = format_condition_label(condition_rows[0])

        for axis, (metric, metric_title) in zip(row_axes, metric_specs):
            values = [row[metric] for row in condition_rows]
            if baseline_row is not None:
                values = [baseline_row[metric], *values]
            positions = list(range(len(values)))
            axis.bar(positions, values, color=colors)
            if baseline_row is not None:
                axis.axhline(
                    baseline_row[metric],
                    color=BASELINE_COLOR,
                    linewidth=1,
                    linestyle="--",
                    alpha=0.75,
                )
            axis.set_xticks(positions, step_labels)
            axis.set_title(f"{title_prefix} | {metric_title}")
            axis.grid(alpha=0.2, linestyle="--", axis="y")
            if metric != "far":
                axis.set_ylim(min(values) - 0.01, max(values) + 0.01)

    fig.suptitle(
        "Controlled Comparison Within Each Condition: Only Hyperparameters Vary",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_vs_none(rows, output_path):
    plt = import_matplotlib()
    grouped = group_by_condition([row for row in rows if row["experiment_id"] != "baseline"])
    labels = []
    delta_acc = []
    delta_far = []

    for key in sorted(grouped):
        per_condition = {row["inference_steps"]: row for row in grouped[key]}
        reference = per_condition.get("none")
        if reference is None:
            continue
        for step in ["2", "4", "8"]:
            current = per_condition.get(step)
            if current is None:
                continue
            labels.append(f"{format_condition_label(reference)}\nRes{step}")
            delta_acc.append(current["accuracy"] - reference["accuracy"])
            delta_far.append(current["far"] - reference["far"])

    x_axis = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(14, 6))
    plt.axhline(0.0, color="#444444", linewidth=1)
    left_positions = [value - width / 2 for value in x_axis]
    right_positions = [value + width / 2 for value in x_axis]
    plt.bar(left_positions, delta_acc, width=width, color="#457b9d", label="Delta Accuracy")
    plt.bar(right_positions, delta_far, width=width, color="#e76f51", label="Delta FAR")
    plt.xticks(x_axis, labels, rotation=55, ha="right")
    plt.ylabel("Change Relative to No ResShift")
    plt.title("ResShift Impact Relative to Each Condition's No-ResShift Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_separation_margin(rows, output_path):
    plot_split_condition_metric_trends(
        rows,
        output_path,
        "score_separation_margin",
        "Genuine Mean - Impostor Mean",
        "Score Separation Margin Across Severity and ResShift Steps",
    )


def plot_d_prime_ranking(rows, output_path):
    plot_split_condition_metric_trends(
        rows,
        output_path,
        "d_prime",
        "d-prime",
        "Standardized Genuine-Impostor Separation Across Severity and ResShift Steps",
    )


def plot_js_distance_ranking(rows, output_path):
    plot_split_condition_metric_trends(
        rows,
        output_path,
        "js_distance",
        "Jensen-Shannon Distance",
        "Distribution Distance Across Severity and ResShift Steps",
    )


def plot_split_condition_metric_trends(rows, output_path, metric, ylabel, figure_title):
    plt = import_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    condition_specs = [
        ("downsample", "Downsample Scale", "Downsample"),
        ("noise", "Gaussian Sigma", "Noise"),
    ]
    marker_map = {
        "none": "o",
        "2": "s",
        "4": "^",
        "8": "D",
    }
    baseline_row = next((row for row in rows if row["experiment_id"] == "baseline"), None)

    for axis, (condition_type, xlabel, title) in zip(axes, condition_specs):
        condition_rows = [
            row for row in rows if row["condition_type"] == condition_type and row["experiment_id"] != "baseline"
        ]
        condition_values = sorted({infer_condition_value(row) for row in condition_rows})
        if baseline_row is not None:
            axis.axhline(
                baseline_row[metric],
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=1.4,
                alpha=0.8,
                label="Baseline",
            )

        for step in ["none", "2", "4", "8"]:
            step_rows = sorted(
                [row for row in condition_rows if row["inference_steps"] == step],
                key=lambda row: infer_condition_value(row),
            )
            if not step_rows:
                continue
            xs = [infer_condition_value(row) for row in step_rows]
            ys = [row[metric] for row in step_rows]
            axis.plot(
                xs,
                ys,
                color=STEP_COLOR_MAP[step],
                marker=marker_map[step],
                linewidth=2,
                label=format_step_label(step),
            )

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_xticks(condition_values)
        if condition_type == "downsample":
            axis.set_xticklabels([f"{value:.2f}" for value in condition_values])
        else:
            axis.set_xticklabels([f"{int(value)}" for value in condition_values])
        axis.grid(alpha=0.25, linestyle="--")

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(figure_title, y=0.99)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distance_summary(rows, output_path, condition_type):
    plt = import_matplotlib()
    metric_specs = [
        ("score_separation_margin", "Mean Distance", "Genuine Mean - Impostor Mean"),
        ("d_prime", "d'", "d-prime"),
        ("js_distance", "JS Distance", "Jensen-Shannon Distance"),
    ]
    marker_map = {
        "none": "o",
        "2": "s",
        "4": "^",
        "8": "D",
    }
    condition_rows = [
        row for row in rows if row["condition_type"] == condition_type and row["experiment_id"] != "baseline"
    ]
    if not condition_rows:
        return

    baseline_row = next((row for row in rows if row["experiment_id"] == "baseline"), None)
    condition_values = sorted({infer_condition_value(row) for row in condition_rows})
    xlabel = "Downsample Scale" if condition_type == "downsample" else "Gaussian Sigma"
    title_name = "Downsample" if condition_type == "downsample" else "Noise"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharex=False)
    for axis, (metric, title, ylabel) in zip(axes, metric_specs):
        if baseline_row is not None:
            axis.axhline(
                baseline_row[metric],
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=1.4,
                alpha=0.8,
                label="Baseline",
            )

        for step in ["none", "2", "4", "8"]:
            step_rows = sorted(
                [row for row in condition_rows if row["inference_steps"] == step],
                key=lambda row: infer_condition_value(row),
            )
            if not step_rows:
                continue
            xs = [infer_condition_value(row) for row in step_rows]
            ys = [row[metric] for row in step_rows]
            axis.plot(
                xs,
                ys,
                color=STEP_COLOR_MAP[step],
                marker=marker_map[step],
                linewidth=2,
                label=format_step_label(step),
            )

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xticks(condition_values)
        if condition_type == "downsample":
            axis.set_xticklabels([f"{value:.2f}" for value in condition_values])
        else:
            axis.set_xticklabels([f"{int(value)}" for value in condition_values])
        axis.grid(alpha=0.25, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{title_name} Score-Distance Metrics Across Severity and ResShift Steps", y=0.99)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.92))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_roc_from_csv(baseline_dir, output_path):
    plt = import_matplotlib()
    baseline_dir = Path(baseline_dir)
    roc_points_path = baseline_dir / "roc_points.csv"
    threshold_summary_path = baseline_dir / "threshold_summary.json"
    if not roc_points_path.exists() or not threshold_summary_path.exists():
        return

    roc_points = load_roc_points(roc_points_path)
    threshold_summary = read_json(threshold_summary_path)
    threshold = threshold_summary["eer_threshold"]
    threshold_point = find_closest_threshold_point(roc_points, threshold)
    eer_point = find_eer_point_from_roc_points(roc_points)
    auc = threshold_summary["auc"]

    fars = [point["far"] for point in roc_points]
    tars = [point["tar"] for point in roc_points]

    plt.figure(figsize=(7, 6))
    plt.plot(fars, tars, color=STEP_COLOR_MAP["none"], linewidth=2.2, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)

    if threshold_point is not None:
        plt.scatter(
            threshold_point["far"],
            threshold_point["tar"],
            color=STEP_COLOR_MAP["8"],
            marker="o",
            s=70,
            zorder=3,
            label=f"Threshold={threshold:.4f}",
        )

    plt.xlabel("FAR")
    plt.ylabel("TAR")
    plt.title("Baseline ROC Curve")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def format_compact_step_label(step):
    return "No Res" if step == "none" else f"Res {step}"


def format_drift_point_label(row):
    base_label = format_condition_label(row).replace(" ", "")
    if row["experiment_id"] == "baseline":
        return "baseline"
    if row["inference_steps"] == "none":
        return base_label
    return f"{base_label} {format_compact_step_label(row['inference_steps'])}"


def plot_embedding_drift_ranking(rows, output_path):
    plt = import_matplotlib()
    sorted_rows = sorted(rows, key=lambda row: row["cosine_drift_mean"])
    labels = [format_condition_label(row) for row in sorted_rows]
    values = [row["cosine_drift_mean"] for row in sorted_rows]
    colors = [get_step_color(row) for row in sorted_rows]

    plt.figure(figsize=(14, 6))
    positions = list(range(len(sorted_rows)))
    plt.bar(positions, values, color=colors)
    plt.xticks(positions, labels, rotation=55, ha="right")
    plt.ylabel("Cosine Drift Mean")
    plt.title("Embedding Drift Ranking Across Experiments")
    plt.legend(handles=build_step_legend_handles(plt, sorted_rows), frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_embedding_drift_vs_metrics(rows, output_path):
    plt = import_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    metric_specs = [
        ("accuracy", "Accuracy", "Accuracy"),
        ("far", "FAR", "FAR"),
    ]

    for axis, (metric, ylabel, title) in zip(axes, metric_specs):
        for row in rows:
            axis.scatter(
                row["cosine_drift_mean"],
                row[metric],
                color=get_step_color(row),
                marker=get_scatter_marker(row),
                s=get_scatter_size(row),
                alpha=0.9,
            )
            axis.annotate(
                format_drift_point_label(row),
                (row["cosine_drift_mean"], row[metric]),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
                alpha=0.85,
            )
        axis.set_xlabel("Cosine Drift Mean")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Embedding Drift vs {title}")
        axis.grid(alpha=0.25, linestyle="--")

    legend_handles = build_step_legend_handles(plt, rows)
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_drift_trends(rows, output_path):
    plt = import_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    condition_specs = [
        ("downsample", "Downsample Scale", "Downsample Embedding Drift"),
        ("noise", "Gaussian Sigma", "Noise Embedding Drift"),
    ]
    marker_map = {
        "none": "o",
        "2": "s",
        "4": "^",
        "8": "D",
    }
    baseline_row = next((row for row in rows if row["experiment_id"] == "baseline"), None)

    for axis, (condition_type, xlabel, title) in zip(axes, condition_specs):
        condition_rows = [
            row for row in rows if row["condition_type"] == condition_type and row["experiment_id"] != "baseline"
        ]
        condition_values = sorted({infer_condition_value(row) for row in condition_rows})
        if baseline_row is not None:
            axis.axhline(
                baseline_row["cosine_drift_mean"],
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=1.4,
                alpha=0.8,
                label="Baseline",
            )

        for step in ["none", "2", "4", "8"]:
            step_rows = sorted(
                [row for row in condition_rows if row["inference_steps"] == step],
                key=lambda row: infer_condition_value(row),
            )
            if not step_rows:
                continue
            xs = [infer_condition_value(row) for row in step_rows]
            ys = [row["cosine_drift_mean"] for row in step_rows]
            axis.plot(
                xs,
                ys,
                color=STEP_COLOR_MAP[step],
                marker=marker_map[step],
                linewidth=2,
                label=format_step_label(step),
            )

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_xticks(condition_values)
        if condition_type == "downsample":
            axis.set_xticklabels([f"{value:.2f}" for value in condition_values])
        else:
            axis.set_xticklabels([f"{int(value)}" for value in condition_values])
        axis.grid(alpha=0.25, linestyle="--")

    axes[0].set_ylabel("Cosine Drift Mean")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Embedding Drift Trends Across Severity and ResShift Steps", y=0.99)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rows = filter_rows(load_rows(args.input_csv), include_step8=args.include_step8)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_row = next((row for row in rows if row["experiment_id"] == "baseline"), None)

    plot_metric_ranking(rows, output_dir / "01_accuracy_ranking.png")
    plot_tradeoff_scatter(rows, output_dir / "02_tar_far_scatter.png")
    plot_verification_condition_trends(
        rows, output_dir / "03a_downsample_verification_trends.png", "downsample"
    )
    plot_verification_condition_trends(rows, output_dir / "03b_noise_verification_trends.png", "noise")
    plot_genuine_condition_trends(rows, output_dir / "03c_downsample_genuine_trends.png", "downsample")
    plot_genuine_condition_trends(rows, output_dir / "03d_noise_genuine_trends.png", "noise")
    plot_within_condition_comparisons(rows, output_dir / "04_within_condition_comparisons.png")
    plot_delta_vs_none(rows, output_dir / "05_delta_vs_none.png")
    plot_distance_summary(rows, output_dir / "06_downsample_distance_summary.png", "downsample")
    plot_distance_summary(rows, output_dir / "07_noise_distance_summary.png", "noise")
    if baseline_row is not None:
        plot_baseline_roc_from_csv(baseline_row["results_dir"], output_dir / "09_baseline_roc.png")
    plot_embedding_drift_ranking(rows, output_dir / "10_embedding_drift_ranking.png")
    plot_embedding_drift_vs_metrics(rows, output_dir / "11_embedding_drift_vs_metrics.png")
    plot_embedding_drift_trends(rows, output_dir / "12_embedding_drift_trends.png")

    if args.include_step8:
        print(f"Wrote 12 figures to {output_dir} with step=8 included")
    else:
        print(f"Wrote 12 figures to {output_dir} with step=8 excluded")


if __name__ == "__main__":
    main()
