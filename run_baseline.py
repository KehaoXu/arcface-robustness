import argparse
from pathlib import Path

from embedding_extractor import FaceEmbedder, load_csv_records
from evaluation import (
    build_pair_manifest,
    build_sample_index,
    compute_embedding_drift,
    compute_pair_scores,
    compute_roc_auc_eer,
    evaluate_condition_with_fixed_threshold,
    summarize_embedding_success,
    write_roc_outputs,
)


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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_index_path, _ = build_sample_index(
        original_dir=args.original_dir,
        output_dir=output_dir,
        min_images_per_identity=args.min_images_per_identity,
    )
    pair_manifest_path = build_pair_manifest(
        sample_index_path=sample_index_path,
        output_dir=output_dir,
        max_genuine_pairs_per_identity=args.max_genuine_pairs_per_identity,
    )

    face_embedder = FaceEmbedder(model_name=args.model_name)
    original_embedding_metadata_path, original_embedding_matrix_path = face_embedder.extract_embeddings_for_condition(
        sample_index_path=sample_index_path,
        condition_dir=args.original_dir,
        output_dir=output_dir,
        condition_name="original",
    )
    summarize_embedding_success(
        embedding_metadata_path=original_embedding_metadata_path,
        output_path=output_dir / "embedding_success_summary.json",
    )

    pair_scores_path = compute_pair_scores(
        pair_manifest_path=pair_manifest_path,
        embedding_metadata_path=original_embedding_metadata_path,
        embedding_matrix_path=original_embedding_matrix_path,
        output_dir=output_dir,
    )
    pair_score_records = load_csv_records(pair_scores_path)
    roc_summary = compute_roc_auc_eer(pair_score_records)
    write_roc_outputs(roc_summary, output_dir)
    evaluate_condition_with_fixed_threshold(
        pair_scores_path=pair_scores_path,
        threshold_summary_path=output_dir / "threshold_summary.json",
        output_dir=output_dir,
    )
    compute_embedding_drift(
        original_embedding_metadata_path=original_embedding_metadata_path,
        original_embedding_matrix_path=original_embedding_matrix_path,
        condition_embedding_metadata_path=original_embedding_metadata_path,
        condition_embedding_matrix_path=original_embedding_matrix_path,
        output_dir=output_dir,
    )
    print(f"Baseline artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
