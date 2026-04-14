import argparse

from face_pipeline.workflows import run_condition_evaluation_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one condition using the fixed baseline threshold.")
    parser.add_argument("--condition-name", required=True)
    parser.add_argument("--condition-dir", required=True)
    parser.add_argument("--baseline-dir", default="results/baseline")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="buffalo_l")
    return parser.parse_args()


def main():
    args = parse_args()
    condition_output_dir = run_condition_evaluation_pipeline(
        condition_name=args.condition_name,
        condition_dir=args.condition_dir,
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )
    print(f"Condition artifacts written to {condition_output_dir}")


if __name__ == "__main__":
    main()

