from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from face_pipeline.io_utils import load_csv_records


def plot_baseline_roc(roc_points_path, output_path):
    roc_points = load_csv_records(roc_points_path)
    far_values = [float(point["far"]) for point in roc_points]
    tar_values = [float(point["tar"]) for point in roc_points]

    plt.figure(figsize=(6, 6))
    plt.plot(far_values, tar_values, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False Accept Rate")
    plt.ylabel("True Accept Rate")
    plt.title("Baseline ROC")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_condition_metrics(metrics_summary_csv_path, output_path):
    rows = load_csv_records(metrics_summary_csv_path)
    conditions = [row["condition"] for row in rows]
    accuracy_values = [float(row["accuracy"]) for row in rows]
    tar_values = [float(row["tar"]) for row in rows]
    far_values = [float(row["far"]) for row in rows]

    x_axis = np.arange(len(conditions))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x_axis - width, accuracy_values, width=width, label="Accuracy")
    plt.bar(x_axis, tar_values, width=width, label="TAR")
    plt.bar(x_axis + width, far_values, width=width, label="FAR")
    plt.xticks(x_axis, conditions, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Fixed-Threshold Metrics by Condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_score_distributions(condition_result_dirs, output_path):
    plt.figure(figsize=(10, 5))
    plotted_positions = []
    plotted_labels = []
    boxplot_data = []

    position = 1
    for condition_name, condition_result_dir in condition_result_dirs.items():
        pair_rows = load_csv_records(Path(condition_result_dir) / "pair_scores_with_predictions.csv")
        genuine_scores = [
            float(row["similarity_score"])
            for row in pair_rows
            if row["valid"] == "1" and row["label"] == "1"
        ]
        impostor_scores = [
            float(row["similarity_score"])
            for row in pair_rows
            if row["valid"] == "1" and row["label"] == "0"
        ]
        if genuine_scores:
            boxplot_data.append(genuine_scores)
            plotted_positions.append(position)
            plotted_labels.append(f"{condition_name}\nG")
            position += 1
        if impostor_scores:
            boxplot_data.append(impostor_scores)
            plotted_positions.append(position)
            plotted_labels.append(f"{condition_name}\nI")
            position += 1
        position += 1

    plt.boxplot(boxplot_data, positions=plotted_positions, tick_labels=plotted_labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Cosine Similarity")
    plt.title("Genuine / Impostor Score Distributions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_embedding_drift(condition_result_dirs, output_path):
    drift_data = []
    condition_names = []
    for condition_name, condition_result_dir in condition_result_dirs.items():
        if condition_name == "original":
            continue
        drift_rows = load_csv_records(Path(condition_result_dir) / "embedding_drift.csv")
        cosine_drifts = [
            float(row["embedding_drift_cosine"])
            for row in drift_rows
            if row["valid"] == "1"
        ]
        if cosine_drifts:
            drift_data.append(cosine_drifts)
            condition_names.append(condition_name)

    plt.figure(figsize=(8, 5))
    plt.boxplot(drift_data, tick_labels=condition_names)
    plt.ylabel("1 - Cosine Similarity")
    plt.title("Embedding Drift Relative to Original")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_figures(baseline_dir, aggregate_dir, condition_result_dirs, figure_dir):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_baseline_roc(Path(baseline_dir) / "roc_points.csv", figure_dir / "baseline_roc.png")
    plot_condition_metrics(Path(aggregate_dir) / "metrics_summary.csv", figure_dir / "condition_metrics.png")
    plot_score_distributions(condition_result_dirs, figure_dir / "score_distributions.png")
    plot_embedding_drift(condition_result_dirs, figure_dir / "embedding_drift.png")

