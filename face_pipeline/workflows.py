from dataclasses import dataclass
from pathlib import Path

from face_pipeline.data_prep import build_sample_index
from face_pipeline.embeddings import FaceEmbedder
from face_pipeline.io_utils import load_csv_records
from face_pipeline.plots import generate_all_figures
from face_pipeline.verification import (
    aggregate_condition_results,
    build_pair_manifest,
    compute_embedding_drift,
    compute_pair_scores,
    compute_roc_auc_eer,
    evaluate_condition_with_fixed_threshold,
    summarize_embedding_success,
    write_roc_outputs,
)


@dataclass(frozen=True)
class EvaluationArtifacts:
    embedding_metadata_path: Path
    embedding_matrix_path: Path
    pair_scores_path: Path


def _extract_and_score_condition(
    *,
    embedder,
    sample_index_path,
    condition_dir,
    output_dir,
    condition_name,
    pair_manifest_path,
    eligible_only=True,
):
    embedding_metadata_path, embedding_matrix_path = embedder.extract_embeddings_for_condition(
        sample_index_path=sample_index_path,
        condition_dir=condition_dir,
        output_dir=output_dir,
        condition_name=condition_name,
        eligible_only=eligible_only,
    )
    summarize_embedding_success(
        embedding_metadata_path=embedding_metadata_path,
        output_path=Path(output_dir) / "embedding_success_summary.json",
    )
    pair_scores_path = compute_pair_scores(
        pair_manifest_path=pair_manifest_path,
        embedding_metadata_path=embedding_metadata_path,
        embedding_matrix_path=embedding_matrix_path,
        output_dir=output_dir,
    )
    return EvaluationArtifacts(
        embedding_metadata_path=Path(embedding_metadata_path),
        embedding_matrix_path=Path(embedding_matrix_path),
        pair_scores_path=Path(pair_scores_path),
    )


def build_baseline_sample_index(
    *,
    original_dir,
    output_dir,
    min_images_per_identity,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return build_sample_index(
        original_dir=original_dir,
        output_dir=output_dir,
        min_images_per_identity=min_images_per_identity,
    )


def build_baseline_pair_manifest(
    *,
    sample_index_path,
    output_dir,
    max_genuine_pairs_per_identity,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return build_pair_manifest(
        sample_index_path=sample_index_path,
        output_dir=output_dir,
        max_genuine_pairs_per_identity=max_genuine_pairs_per_identity,
    )


def run_original_baseline_evaluation(
    *,
    original_dir,
    baseline_dir,
    model_name,
):
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    face_embedder = FaceEmebdder(model_name=model_name)
    baseline_artifacts = _extract_and_score_condition(
        embedder=face_embedder,
        sample_index_path=baseline_dir / "sample_index.csv",
        condition_dir=original_dir,
        output_dir=baseline_dir,
        condition_name="original",
        pair_manifest_path=baseline_dir / "pair_manifest.csv",
        eligible_only=True,
    )
    roc_summary = compute_roc_auc_eer(load_csv_records(baseline_artifacts.pair_scores_path))
    write_roc_outputs(roc_summary, baseline_dir)
    evaluate_condition_with_fixed_threshold(
        pair_scores_path=baseline_artifacts.pair_scores_path,
        threshold_summary_path=baseline_dir / "threshold_summary.json",
        output_dir=baseline_dir,
    )
    compute_embedding_drift(
        original_embedding_metadata_path=baseline_artifacts.embedding_metadata_path,
        original_embedding_matrix_path=baseline_artifacts.embedding_matrix_path,
        condition_embedding_metadata_path=baseline_artifacts.embedding_metadata_path,
        condition_embedding_matrix_path=baseline_artifacts.embedding_matrix_path,
        output_dir=baseline_dir,
    )
    return baseline_dir


def run_baseline_pipeline(
    original_dir,
    output_dir,
    model_name,
    min_images_per_identity,
    max_genuine_pairs_per_identity,
):
    output_dir = Path(output_dir)
    build_baseline_sample_index(
        original_dir=original_dir,
        output_dir=output_dir,
        min_images_per_identity=min_images_per_identity,
    )
    build_baseline_pair_manifest(
        sample_index_path=output_dir / "sample_index.csv",
        output_dir=output_dir,
        max_genuine_pairs_per_identity=max_genuine_pairs_per_identity,
    )
    return run_original_baseline_evaluation(
        original_dir=original_dir,
        baseline_dir=output_dir,
        model_name=model_name,
    )


def run_condition_evaluation_pipeline(condition_name, condition_dir, baseline_dir, output_dir, model_name):
    baseline_dir = Path(baseline_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face_embedder = FaceEmbedder(model_name=model_name)
    condition_artifacts = _extract_and_score_condition(
        embedder=face_embedder,
        sample_index_path=baseline_dir / "sample_index.csv",
        condition_dir=condition_dir,
        output_dir=output_dir,
        condition_name=condition_name,
        pair_manifest_path=baseline_dir / "pair_manifest.csv",
        eligible_only=True,
    )
    evaluate_condition_with_fixed_threshold(
        pair_scores_path=condition_artifacts.pair_scores_path,
        threshold_summary_path=baseline_dir / "threshold_summary.json",
        output_dir=output_dir,
    )
    compute_embedding_drift(
        original_embedding_metadata_path=baseline_dir / "embedding_results.csv",
        original_embedding_matrix_path=baseline_dir / "embedding_vectors.npy",
        condition_embedding_metadata_path=condition_artifacts.embedding_metadata_path,
        condition_embedding_matrix_path=condition_artifacts.embedding_matrix_path,
        output_dir=output_dir,
    )
    return output_dir


def run_full_evaluation_pipeline(baseline_dir, aggregate_dir, condition_result_dirs):
    aggregate_dir = Path(aggregate_dir)
    aggregate_condition_results(condition_result_dirs=condition_result_dirs, output_dir=aggregate_dir)
    generate_all_figures(
        baseline_dir=baseline_dir,
        aggregate_dir=aggregate_dir,
        condition_result_dirs=condition_result_dirs,
        figure_dir=aggregate_dir / "figures",
    )
    return aggregate_dir
