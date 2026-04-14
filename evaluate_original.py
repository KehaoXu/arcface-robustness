import argparse

from face_pipeline.workflows import run_original_baseline_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the original baseline using existing sample_index and pair_manifest.")
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--baseline-dir", default="results/baseline")
    parser.add_argument("--model-name", default="buffalo_l")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_output_dir = run_original_baseline_evaluation(
        original_dir=args.original_dir,
        baseline_dir=args.baseline_dir,
        model_name=args.model_name,
    )
    print(f"Original baseline artifacts written to {baseline_output_dir}")


if __name__ == "__main__":
    main()
