import argparse
import csv
import json
from pathlib import Path

METRIC_FIELDNAMES = [
    "accuracy",
    "tar",
    "far",
    "frr",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "genuine_count",
    "impostor_count",
    "genuine_mean",
    "genuine_std",
    "genuine_median",
    "impostor_mean",
    "impostor_std",
    "impostor_median",
    "threshold",
    "valid_pairs",
    "invalid_pairs",
]


def load_csv_records(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def read_json(json_path):
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def write_csv_records(csv_path, records, fieldnames):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(json_path, payload):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate all grid experiment metrics into a single CSV.")
    parser.add_argument("--grid-csv", default="results/hyperparameter_grid.csv")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-csv", default="results/grid_metrics_summary.csv")
    parser.add_argument("--missing-csv", default="results/grid_metrics_missing.csv")
    parser.add_argument("--summary-json", default="results/grid_metrics_summary.json")
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Append results/baseline as an extra row in the output summary.",
    )
    return parser.parse_args()


def _format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def _build_output_row(experiment_row, metrics_summary, results_dir):
    output_row = dict(experiment_row)
    output_row["results_dir"] = str(results_dir)
    for fieldname in METRIC_FIELDNAMES:
        output_row[fieldname] = _format_metric_value(metrics_summary[fieldname])
    return output_row


def _build_missing_row(experiment_row, results_dir, reason):
    output_row = dict(experiment_row)
    output_row["results_dir"] = str(results_dir)
    output_row["reason"] = reason
    return output_row


def aggregate_grid_results(grid_csv_path, results_root, include_baseline=False):
    grid_rows = load_csv_records(grid_csv_path)
    results_root = Path(results_root)

    aggregated_rows = []
    missing_rows = []

    for grid_row in grid_rows:
        experiment_id = grid_row["experiment_id"]
        results_dir = results_root / experiment_id
        metrics_summary_path = results_dir / "metrics_summary.json"

        if not results_dir.exists():
            missing_rows.append(_build_missing_row(grid_row, results_dir, "missing_results_dir"))
            continue

        if not metrics_summary_path.exists():
            missing_rows.append(_build_missing_row(grid_row, results_dir, "missing_metrics_summary"))
            continue

        aggregated_rows.append(_build_output_row(grid_row, read_json(metrics_summary_path), results_dir))

    if include_baseline:
        baseline_dir = results_root / "baseline"
        baseline_metrics_path = baseline_dir / "metrics_summary.json"
        if baseline_metrics_path.exists():
            baseline_row = {
                "experiment_id": "baseline",
                "condition_type": "original",
                "downsample_scale": "",
                "gaussian_sigma": "",
                "inference_steps": "none",
                "notes": "baseline/original condition",
            }
            aggregated_rows.append(_build_output_row(baseline_row, read_json(baseline_metrics_path), baseline_dir))
        else:
            missing_rows.append(
                {
                    "experiment_id": "baseline",
                    "condition_type": "original",
                    "downsample_scale": "",
                    "gaussian_sigma": "",
                    "inference_steps": "none",
                    "notes": "baseline/original condition",
                    "results_dir": str(baseline_dir),
                    "reason": "missing_metrics_summary",
                }
            )

    aggregated_rows.sort(key=lambda row: row["experiment_id"])
    missing_rows.sort(key=lambda row: row["experiment_id"])
    return aggregated_rows, missing_rows


def main():
    args = parse_args()
    aggregated_rows, missing_rows = aggregate_grid_results(
        grid_csv_path=args.grid_csv,
        results_root=args.results_root,
        include_baseline=args.include_baseline,
    )

    output_fieldnames = [
        "experiment_id",
        "condition_type",
        "downsample_scale",
        "gaussian_sigma",
        "inference_steps",
        "notes",
        "results_dir",
        *METRIC_FIELDNAMES,
    ]
    write_csv_records(args.output_csv, aggregated_rows, output_fieldnames)
    write_csv_records(
        args.missing_csv,
        missing_rows,
        [
            "experiment_id",
            "condition_type",
            "downsample_scale",
            "gaussian_sigma",
            "inference_steps",
            "notes",
            "results_dir",
            "reason",
        ],
    )
    write_json(
        args.summary_json,
        {
            "grid_csv": args.grid_csv,
            "results_root": args.results_root,
            "output_csv": args.output_csv,
            "missing_csv": args.missing_csv,
            "experiments_in_grid": len(load_csv_records(args.grid_csv)),
            "aggregated_experiments": len(aggregated_rows),
            "missing_experiments": len(missing_rows),
            "included_baseline": args.include_baseline,
        },
    )
    print(f"Aggregated {len(aggregated_rows)} experiment rows into {args.output_csv}")
    print(f"Recorded {len(missing_rows)} missing experiment rows into {args.missing_csv}")


if __name__ == "__main__":
    main()
