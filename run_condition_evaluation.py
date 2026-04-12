import argparse
from pathlib import Path

from embedding_extractor import FaceEmbedder
from evaluation import (
    compute_embedding_drift,
    compute_pair_scores,
    evaluate_condition_with_fixed_threshold,
    summarize_embedding_success,
)


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
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face_embedder = FaceEmbedder(model_name=args.model_name)
    embedding_metadata_path, embedding_matrix_path = face_embedder.extract_embeddings_for_condition(
        sample_index_path=baseline_dir / "sample_index.csv",
        condition_dir=args.condition_dir,
        output_dir=output_dir,
        condition_name=args.condition_name,
    )
    summarize_embedding_success(
        embedding_metadata_path=embedding_metadata_path,
        output_path=output_dir / "embedding_success_summary.json",
    )
    pair_scores_path = compute_pair_scores(
        pair_manifest_path=baseline_dir / "pair_manifest.csv",
        embedding_metadata_path=embedding_metadata_path,
        embedding_matrix_path=embedding_matrix_path,
        output_dir=output_dir,
    )
    evaluate_condition_with_fixed_threshold(
        pair_scores_path=pair_scores_path,
        threshold_summary_path=baseline_dir / "threshold_summary.json",
        output_dir=output_dir,
    )
    compute_embedding_drift(
        original_embedding_metadata_path=baseline_dir / "embedding_results.csv",
        original_embedding_matrix_path=baseline_dir / "embedding_vectors.npy",
        condition_embedding_metadata_path=embedding_metadata_path,
        condition_embedding_matrix_path=embedding_matrix_path,
        output_dir=output_dir,
    )
    print(f"Condition artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
