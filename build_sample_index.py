import argparse

from face_pipeline.workflows import build_baseline_sample_index


def parse_args():
    parser = argparse.ArgumentParser(description="Build sample_index.csv for the baseline protocol.")
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--min-images-per-identity", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    sample_index_path, summary_path = build_baseline_sample_index(
        original_dir=args.original_dir,
        output_dir=args.output_dir,
        min_images_per_identity=args.min_images_per_identity,
    )
    print(
        {
            "sample_index_path": str(sample_index_path),
            "identity_filter_summary_path": str(summary_path),
        }
    )


if __name__ == "__main__":
    main()
