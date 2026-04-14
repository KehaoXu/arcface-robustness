import argparse

from face_pipeline.workflows import build_baseline_pair_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Build pair_manifest.csv from an existing sample_index.csv.")
    parser.add_argument("--sample-index-path", required=True)
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--max-genuine-pairs-per-identity", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    pair_manifest_path = build_baseline_pair_manifest(
        sample_index_path=args.sample_index_path,
        output_dir=args.output_dir,
        max_genuine_pairs_per_identity=args.max_genuine_pairs_per_identity,
    )
    print({"pair_manifest_path": str(pair_manifest_path)})


if __name__ == "__main__":
    main()
