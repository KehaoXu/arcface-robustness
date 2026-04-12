import argparse
from pathlib import Path

from evaluation import aggregate_condition_results
from visualization import generate_all_figures


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate condition results and generate figures.")
    parser.add_argument("--baseline-dir", default="results/baseline")
    parser.add_argument("--aggregate-dir", default="results/final")
    parser.add_argument("--original-dir", default="results/baseline")
    parser.add_argument("--downsample-dir", default="results/downsample")
    parser.add_argument("--noise-dir", default="results/noise")
    parser.add_argument("--downsample-resshift-dir", default="results/downsample_resshift")
    parser.add_argument("--noise-resshift-dir", default="results/noise_resshift")
    return parser.parse_args()


def main():
    args = parse_args()
    condition_result_dirs = {
        "original": args.original_dir,
        "downsample": args.downsample_dir,
        "noise": args.noise_dir,
        "downsample_resshift": args.downsample_resshift_dir,
        "noise_resshift": args.noise_resshift_dir,
    }
    aggregate_dir = Path(args.aggregate_dir)
    aggregate_condition_results(condition_result_dirs=condition_result_dirs, output_dir=aggregate_dir)
    generate_all_figures(
        baseline_dir=args.baseline_dir,
        aggregate_dir=aggregate_dir,
        condition_result_dirs=condition_result_dirs,
        figure_dir=aggregate_dir / "figures",
    )
    print(f"Aggregate outputs written to {aggregate_dir}")


if __name__ == "__main__":
    main()
