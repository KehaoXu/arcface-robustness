import argparse

from face_pipeline.config import DEFAULT_ORIGINAL_DIR
from face_pipeline.workflows import (
    build_baseline_pair_manifest,
    build_baseline_sample_index,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build sample_index.csv and pair_manifest.csv for the baseline protocol."
    )
    parser.add_argument("--original-dir", default=str(DEFAULT_ORIGINAL_DIR))
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--min-images-per-identity", type=int, default=10)
    parser.add_argument("--max-genuine-pairs-per-identity", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    sample_index_path, summary_path = build_baseline_sample_index(
        original_dir=args.original_dir,
        output_dir=args.output_dir,
        min_images_per_identity=args.min_images_per_identity,
    )
    pair_manifest_path = build_baseline_pair_manifest(
        sample_index_path=sample_index_path,
        output_dir=args.output_dir,
        max_genuine_pairs_per_identity=args.max_genuine_pairs_per_identity,
    )
    print(
        {
            "sample_index_path": str(sample_index_path),
            "identity_filter_summary_path": str(summary_path),
            "pair_manifest_path": str(pair_manifest_path),
        }
    )


if __name__ == "__main__":
    main()
