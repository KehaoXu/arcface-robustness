import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from face_pipeline.config import (
    DEFAULT_MAX_GENUINE_PAIRS_PER_IDENTITY,
    DEFAULT_RANDOM_SEED,
)
from face_pipeline.io_utils import load_csv_records, load_embedding_lookup, read_json, write_csv_records, write_json
from face_pipeline.metrics import (
    compute_classification_metrics,
    compute_roc_auc_eer,
    cosine_similarity_score,
    summarize_score_distribution,
)


def _load_eligible_samples_by_identity(sample_index_path):
    eligible_samples = defaultdict(list)
    for sample_record in load_csv_records(sample_index_path):
        if sample_record["include_in_protocol"] == "1":
            eligible_samples[sample_record["identity"]].append(sample_record["image_id"])
    return eligible_samples


def _sample_genuine_pairs(eligible_samples_by_identity, max_genuine_pairs_per_identity, rng):
    genuine_pairs = []
    for identity_name in sorted(eligible_samples_by_identity):
        image_ids = sorted(eligible_samples_by_identity[identity_name])
        all_pairs = list(itertools.combinations(image_ids, 2))
        if len(all_pairs) > max_genuine_pairs_per_identity:
            sampled_indices = rng.choice(len(all_pairs), size=max_genuine_pairs_per_identity, replace=False)
            selected_pairs = [all_pairs[index] for index in sorted(sampled_indices)]
        else:
            selected_pairs = all_pairs

        genuine_pairs.extend(
            {
                "left_image_id": left_image_id,
                "right_image_id": right_image_id,
                "left_identity": identity_name,
                "right_identity": identity_name,
                "pair_type": "genuine",
                "label": "1",
            }
            for left_image_id, right_image_id in selected_pairs
        )
    return genuine_pairs


def _sample_impostor_pairs(eligible_samples_by_identity, target_count, rng):
    identity_names = sorted(eligible_samples_by_identity)
    sampled_pairs = set()
    max_attempts = max(target_count * 20, 1000)
    attempts = 0

    while len(sampled_pairs) < target_count and attempts < max_attempts:
        left_identity, right_identity = rng.choice(identity_names, size=2, replace=False)
        left_image_id = str(rng.choice(eligible_samples_by_identity[left_identity]))
        right_image_id = str(rng.choice(eligible_samples_by_identity[right_identity]))
        sampled_pairs.add(tuple(sorted((left_image_id, right_image_id))))
        attempts += 1

    return [
        {
            "left_image_id": left_image_id,
            "right_image_id": right_image_id,
            "left_identity": left_image_id.split("/")[0],
            "right_identity": right_image_id.split("/")[0],
            "pair_type": "impostor",
            "label": "0",
        }
        for left_image_id, right_image_id in sorted(sampled_pairs)
    ]


def _assign_pair_ids(pair_records):
    return [
        {
            "pair_id": f"pair_{index:08d}",
            **pair_record,
        }
        for index, pair_record in enumerate(pair_records)
    ]


def build_pair_manifest(
    sample_index_path,
    output_dir,
    max_genuine_pairs_per_identity=DEFAULT_MAX_GENUINE_PAIRS_PER_IDENTITY,
    random_seed=DEFAULT_RANDOM_SEED,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible_samples_by_identity = _load_eligible_samples_by_identity(sample_index_path)
    if len(eligible_samples_by_identity) < 2:
        raise ValueError("Need at least two eligible identities to build the pair manifest.")

    rng = np.random.default_rng(random_seed)
    genuine_pair_records = _sample_genuine_pairs(
        eligible_samples_by_identity,
        max_genuine_pairs_per_identity,
        rng,
    )
    impostor_pair_records = _sample_impostor_pairs(
        eligible_samples_by_identity,
        len(genuine_pair_records),
        rng,
    )
    pair_records = _assign_pair_ids(genuine_pair_records + impostor_pair_records)

    pair_manifest_path = output_dir / "pair_manifest.csv"
    write_csv_records(pair_manifest_path, pair_records, fieldnames=list(pair_records[0].keys()))
    write_json(
        output_dir / "pair_manifest_summary.json",
        {
            "genuine_pairs": len(genuine_pair_records),
            "impostor_pairs": len(impostor_pair_records),
            "total_pairs": len(pair_records),
            "random_seed": random_seed,
            "max_genuine_pairs_per_identity": max_genuine_pairs_per_identity,
            "eligible_identities": len(eligible_samples_by_identity),
        },
    )
    return pair_manifest_path


def compute_pair_scores(pair_manifest_path, embedding_metadata_path, embedding_matrix_path, output_dir):
    pair_records = load_csv_records(pair_manifest_path)
    embedding_metadata_records, embedding_lookup = load_embedding_lookup(embedding_metadata_path, embedding_matrix_path)
    invalid_image_ids = {
        record["image_id"]: record["error_message"]
        for record in embedding_metadata_records
        if record["success"] != "1"
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_score_records = []
    for pair_record in pair_records:
        left_image_id = pair_record["left_image_id"]
        right_image_id = pair_record["right_image_id"]
        output_record = {
            **pair_record,
            "valid": "0",
            "similarity_score": "",
            "prediction": "",
            "error_reason": "",
        }
        if left_image_id not in embedding_lookup:
            output_record["error_reason"] = invalid_image_ids.get(left_image_id, "Missing left embedding")
        elif right_image_id not in embedding_lookup:
            output_record["error_reason"] = invalid_image_ids.get(right_image_id, "Missing right embedding")
        else:
            similarity_score = cosine_similarity_score(
                embedding_lookup[left_image_id],
                embedding_lookup[right_image_id],
            )
            output_record["valid"] = "1"
            output_record["similarity_score"] = f"{similarity_score:.10f}"
        pair_score_records.append(output_record)

    pair_scores_path = output_dir / "pair_scores.csv"
    write_csv_records(pair_scores_path, pair_score_records, fieldnames=list(pair_score_records[0].keys()))
    return pair_scores_path


def write_roc_outputs(roc_summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_summary_path = write_json(
        output_dir / "threshold_summary.json",
        {
            "auc": roc_summary["auc"],
            "eer": roc_summary["eer"],
            "eer_threshold": roc_summary["eer_threshold"],
            "valid_pairs": roc_summary["valid_pairs"],
            "genuine_pairs": roc_summary["genuine_pairs"],
            "impostor_pairs": roc_summary["impostor_pairs"],
        },
    )

    roc_points_path = output_dir / "roc_points.csv"
    write_csv_records(
        roc_points_path,
        [
            {
                "threshold": f"{point['threshold']:.10f}",
                "tar": f"{point['tar']:.10f}",
                "far": f"{point['far']:.10f}",
                "frr": f"{point['frr']:.10f}",
            }
            for point in roc_summary["roc_points"]
        ],
        fieldnames=["threshold", "tar", "far", "frr"],
    )
    return threshold_summary_path, roc_points_path


def evaluate_condition_with_fixed_threshold(pair_scores_path, threshold_summary_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_score_records = load_csv_records(pair_scores_path)
    threshold = float(read_json(threshold_summary_path)["eer_threshold"])

    updated_records = []
    valid_labels = []
    valid_predictions = []
    genuine_scores = []
    impostor_scores = []

    for pair_score_record in pair_score_records:
        updated_record = dict(pair_score_record)
        if updated_record["valid"] == "1":
            similarity_score = float(updated_record["similarity_score"])
            prediction = 1 if similarity_score >= threshold else 0
            updated_record["prediction"] = str(prediction)
            valid_labels.append(int(updated_record["label"]))
            valid_predictions.append(prediction)
            if updated_record["label"] == "1":
                genuine_scores.append(similarity_score)
            else:
                impostor_scores.append(similarity_score)
        updated_records.append(updated_record)

    evaluated_pair_scores_path = output_dir / "pair_scores_with_predictions.csv"
    write_csv_records(evaluated_pair_scores_path, updated_records, fieldnames=list(updated_records[0].keys()))

    metrics_summary = compute_classification_metrics(valid_labels, valid_predictions)
    metrics_summary.update(summarize_score_distribution(genuine_scores=genuine_scores, impostor_scores=impostor_scores))
    metrics_summary["threshold"] = threshold
    metrics_summary["valid_pairs"] = len(valid_labels)
    metrics_summary["invalid_pairs"] = len(updated_records) - len(valid_labels)

    metrics_summary_path = write_json(output_dir / "metrics_summary.json", metrics_summary)
    return evaluated_pair_scores_path, metrics_summary_path


def compute_embedding_drift(
    original_embedding_metadata_path,
    original_embedding_matrix_path,
    condition_embedding_metadata_path,
    condition_embedding_matrix_path,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, original_lookup = load_embedding_lookup(original_embedding_metadata_path, original_embedding_matrix_path)
    condition_metadata_records, condition_lookup = load_embedding_lookup(
        condition_embedding_metadata_path,
        condition_embedding_matrix_path,
    )

    drift_records = []
    euclidean_distances = []
    cosine_drifts = []
    for condition_metadata_record in condition_metadata_records:
        image_id = condition_metadata_record["image_id"]
        drift_record = {
            "image_id": image_id,
            "identity": condition_metadata_record["identity"],
            "condition": condition_metadata_record["condition"],
            "valid": "0",
            "embedding_drift_cosine": "",
            "embedding_drift_euclidean": "",
            "error_reason": "",
        }
        if image_id not in original_lookup:
            drift_record["error_reason"] = "Missing original embedding"
        elif image_id not in condition_lookup:
            drift_record["error_reason"] = condition_metadata_record["error_message"] or "Missing condition embedding"
        else:
            original_vector = original_lookup[image_id]
            condition_vector = condition_lookup[image_id]
            cosine_drift = 1.0 - cosine_similarity_score(original_vector, condition_vector)
            euclidean_distance = float(np.linalg.norm(original_vector - condition_vector))
            drift_record["valid"] = "1"
            drift_record["embedding_drift_cosine"] = f"{cosine_drift:.10f}"
            drift_record["embedding_drift_euclidean"] = f"{euclidean_distance:.10f}"
            cosine_drifts.append(cosine_drift)
            euclidean_distances.append(euclidean_distance)
        drift_records.append(drift_record)

    drift_records_path = output_dir / "embedding_drift.csv"
    write_csv_records(drift_records_path, drift_records, fieldnames=list(drift_records[0].keys()))
    drift_summary_path = write_json(
        output_dir / "embedding_drift_summary.json",
        {
            "valid_samples": len(cosine_drifts),
            "invalid_samples": len(drift_records) - len(cosine_drifts),
            "cosine_mean": float(np.mean(cosine_drifts)) if cosine_drifts else 0.0,
            "cosine_std": float(np.std(cosine_drifts)) if cosine_drifts else 0.0,
            "cosine_median": float(np.median(cosine_drifts)) if cosine_drifts else 0.0,
            "cosine_q1": float(np.quantile(cosine_drifts, 0.25)) if cosine_drifts else 0.0,
            "cosine_q3": float(np.quantile(cosine_drifts, 0.75)) if cosine_drifts else 0.0,
            "euclidean_mean": float(np.mean(euclidean_distances)) if euclidean_distances else 0.0,
            "euclidean_std": float(np.std(euclidean_distances)) if euclidean_distances else 0.0,
        },
    )
    return drift_records_path, drift_summary_path


def summarize_embedding_success(embedding_metadata_path, output_path):
    embedding_metadata_records = load_csv_records(embedding_metadata_path)
    success_count = sum(record["success"] == "1" for record in embedding_metadata_records)
    return write_json(
        output_path,
        {
            "total_samples": len(embedding_metadata_records),
            "successful_embeddings": success_count,
            "failed_embeddings": len(embedding_metadata_records) - success_count,
            "success_rate": success_count / len(embedding_metadata_records) if embedding_metadata_records else 0.0,
        },
    )


def _load_condition_payload(condition_name, condition_result_dir):
    condition_result_dir = Path(condition_result_dir)
    return {
        "condition": condition_name,
        "pair_records": load_csv_records(condition_result_dir / "pair_scores_with_predictions.csv"),
        "drift_records": load_csv_records(condition_result_dir / "embedding_drift.csv"),
        "metrics_summary": read_json(condition_result_dir / "metrics_summary.json"),
        "drift_summary": read_json(condition_result_dir / "embedding_drift_summary.json"),
        "success_summary": read_json(condition_result_dir / "embedding_success_summary.json"),
    }


def _build_metrics_row(condition_name, metrics_summary, success_summary, drift_cosine_mean):
    return {
        "condition": condition_name,
        "accuracy": f"{metrics_summary['accuracy']:.10f}",
        "tar": f"{metrics_summary['tar']:.10f}",
        "far": f"{metrics_summary['far']:.10f}",
        "frr": f"{metrics_summary['frr']:.10f}",
        "valid_pairs": str(metrics_summary["valid_pairs"]),
        "invalid_pairs": str(metrics_summary["invalid_pairs"]),
        "embedding_success_rate": f"{success_summary['success_rate']:.10f}",
        "drift_cosine_mean": f"{drift_cosine_mean:.10f}",
    }


def aggregate_condition_results(condition_result_dirs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_valid_sets = {}
    image_valid_sets = {}
    full_condition_rows = []
    condition_payloads = {}

    for condition_name, condition_result_dir in condition_result_dirs.items():
        payload = _load_condition_payload(condition_name, condition_result_dir)
        pair_valid_sets[condition_name] = {record["pair_id"] for record in payload["pair_records"] if record["valid"] == "1"}
        image_valid_sets[condition_name] = {record["image_id"] for record in payload["drift_records"] if record["valid"] == "1"}
        condition_payloads[condition_name] = payload
        full_condition_rows.append(
            _build_metrics_row(
                condition_name,
                payload["metrics_summary"],
                payload["success_summary"],
                payload["drift_summary"]["cosine_mean"],
            )
        )

    common_valid_pair_ids = set.intersection(*pair_valid_sets.values()) if pair_valid_sets else set()
    common_valid_image_ids = set.intersection(*image_valid_sets.values()) if image_valid_sets else set()
    write_json(
        output_dir / "common_subset_summary.json",
        {
            "conditions": list(condition_result_dirs.keys()),
            "common_valid_pair_count": len(common_valid_pair_ids),
            "common_valid_image_count": len(common_valid_image_ids),
        },
    )

    common_condition_rows = []
    distribution_rows = []
    for condition_name, payload in condition_payloads.items():
        common_pair_records = [
            record
            for record in payload["pair_records"]
            if record["pair_id"] in common_valid_pair_ids and record["valid"] == "1"
        ]
        common_labels = [int(record["label"]) for record in common_pair_records]
        common_predictions = [int(record["prediction"]) for record in common_pair_records]
        genuine_scores = [float(record["similarity_score"]) for record in common_pair_records if record["label"] == "1"]
        impostor_scores = [float(record["similarity_score"]) for record in common_pair_records if record["label"] == "0"]
        common_metrics = compute_classification_metrics(common_labels, common_predictions)
        common_metrics["valid_pairs"] = len(common_pair_records)
        common_metrics["invalid_pairs"] = payload["metrics_summary"]["invalid_pairs"]
        common_distribution = summarize_score_distribution(genuine_scores, impostor_scores)
        common_drift_values = [
            float(record["embedding_drift_cosine"])
            for record in payload["drift_records"]
            if record["image_id"] in common_valid_image_ids and record["valid"] == "1"
        ]

        common_condition_rows.append(
            _build_metrics_row(
                condition_name,
                common_metrics,
                payload["success_summary"],
                float(np.mean(common_drift_values)) if common_drift_values else 0.0,
            )
        )
        distribution_rows.append(
            {
                "condition": condition_name,
                "genuine_mean": f"{common_distribution['genuine_mean']:.10f}",
                "genuine_std": f"{common_distribution['genuine_std']:.10f}",
                "impostor_mean": f"{common_distribution['impostor_mean']:.10f}",
                "impostor_std": f"{common_distribution['impostor_std']:.10f}",
            }
        )

    write_csv_records(
        output_dir / "metrics_summary.csv",
        common_condition_rows,
        fieldnames=list(common_condition_rows[0].keys()) if common_condition_rows else ["condition"],
    )
    write_csv_records(
        output_dir / "metrics_summary_all_valid.csv",
        full_condition_rows,
        fieldnames=list(full_condition_rows[0].keys()) if full_condition_rows else ["condition"],
    )
    write_csv_records(
        output_dir / "distribution_summary.csv",
        distribution_rows,
        fieldnames=list(distribution_rows[0].keys()) if distribution_rows else ["condition"],
    )
    return output_dir / "metrics_summary.csv"
