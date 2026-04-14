import argparse

from face_pipeline.workflows import run_baseline_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Build the original LFW baseline protocol.")
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--min-images-per-identity", type=int, default=10)
    parser.add_argument("--max-genuine-pairs-per-identity", type=int, default=100)
    parser.add_argument("--model-name", default="buffalo_l")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_output_dir = run_baseline_pipeline(
        original_dir=args.original_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        min_images_per_identity=args.min_images_per_identity,
        max_genuine_pairs_per_identity=args.max_genuine_pairs_per_identity,
    )
    print(f"Baseline artifacts written to {baseline_output_dir}")


if __name__ == "__main__":
    main()

