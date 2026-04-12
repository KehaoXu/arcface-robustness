import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from embedding_extractor import load_csv_records, load_embedding_lookup, write_csv_records
from experiment_config import (
    DEFAULT_MAX_GENUINE_PAIRS_PER_IDENTITY,
    DEFAULT_MIN_IMAGES_PER_IDENTITY,
    DEFAULT_RANDOM_SEED,
    IMAGE_EXTENSIONS,
)


def cosine_similarity_score(vector_a, vector_b):
    vector_a = vector_a / np.linalg.norm(vector_a)
    vector_b = vector_b / np.linalg.norm(vector_b)
    return float(np.dot(vector_a, vector_b))


def build_sample_index(original_dir, output_dir, min_images_per_identity=DEFAULT_MIN_IMAGES_PER_IDENTITY):
    original_dir = Path(original_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_records = []
    identity_to_paths = defaultdict(list)
    for image_path in sorted(original_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        identity_name = image_path.parent.name
        relative_path = image_path.relative_to(original_dir).as_posix()
        image_id = str(Path(relative_path).with_suffix("")).replace("\\", "/")
        identity_to_paths[identity_name].append(relative_path)
        sample_records.append(
            {
                "image_id": image_id,
                "identity": identity_name,
                "relative_path": relative_path,
                "file_stem": Path(relative_path).stem,
            }
        )

    if not sample_records:
        raise FileNotFoundError(f"No supported images found under {original_dir}")

    identity_counts = {identity_name: len(paths) for identity_name, paths in identity_to_paths.items()}
    included_identity_names = {
        identity_name
        for identity_name, count in identity_counts.items()
        if count >= min_images_per_identity
    }

    for sample_record in sample_records:
        sample_record["include_in_protocol"] = "1" if sample_record["identity"] in included_identity_names else "0"

    sample_index_path = output_dir / "sample_index.csv"
    write_csv_records(sample_index_path, sample_records, fieldnames=list(sample_records[0].keys()))

    filter_summary = {
        "total_images": len(sample_records),
        "total_identities": len(identity_counts),
        "min_images_per_identity": min_images_per_identity,
        "eligible_identities": len(included_identity_names),
        "eligible_images": sum(identity_counts[name] for name in included_identity_names),
        "excluded_identities": len(identity_counts) - len(included_identity_names),
        "identity_image_counts": dict(sorted(identity_counts.items())),
    }
    identity_filter_summary_path = output_dir / "identity_filter_summary.json"
    identity_filter_summary_path.write_text(json.dumps(filter_summary, indent=2), encoding="utf-8")
    return sample_index_path, identity_filter_summary_path


def build_pair_manifest(
    sample_index_path,
    output_dir,
    max_genuine_pairs_per_identity=DEFAULT_MAX_GENUINE_PAIRS_PER_IDENTITY,
    random_seed=DEFAULT_RANDOM_SEED,
):
    sample_records = load_csv_records(sample_index_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible_samples_by_identity = defaultdict(list)
    for sample_record in sample_records:
        if sample_record["include_in_protocol"] == "1":
            eligible_samples_by_identity[sample_record["identity"]].append(sample_record["image_id"])

    if len(eligible_samples_by_identity) < 2:
        raise ValueError("Need at least two eligible identities to build the pair manifest.")

    rng = np.random.default_rng(random_seed)
    pair_records = []
    pair_id = 0
    total_genuine_pairs = 0

    for identity_name in sorted(eligible_samples_by_identity):
        image_ids = sorted(eligible_samples_by_identity[identity_name])
        all_pairs = list(itertools.combinations(image_ids, 2))
        if len(all_pairs) > max_genuine_pairs_per_identity:
            sampled_indices = rng.choice(
                len(all_pairs),
                size=max_genuine_pairs_per_identity,
                replace=False,
            )
            selected_pairs = [all_pairs[index] for index in sorted(sampled_indices)]
        else:
            selected_pairs = all_pairs

        total_genuine_pairs += len(selected_pairs)
        for left_image_id, right_image_id in selected_pairs:
            pair_records.append(
                {
                    "pair_id": f"pair_{pair_id:08d}",
                    "left_image_id": left_image_id,
                    "right_image_id": right_image_id,
                    "left_identity": identity_name,
                    "right_identity": identity_name,
                    "pair_type": "genuine",
                    "label": "1",
                }
            )
            pair_id += 1

    identity_names = sorted(eligible_samples_by_identity)
    impostor_pairs = set()
    max_attempts = max(total_genuine_pairs * 20, 1000)
    attempts = 0
    while len(impostor_pairs) < total_genuine_pairs and attempts < max_attempts:
        left_identity, right_identity = rng.choice(identity_names, size=2, replace=False)
        left_image_id = str(rng.choice(eligible_samples_by_identity[left_identity]))
        right_image_id = str(rng.choice(eligible_samples_by_identity[right_identity]))
        ordered_pair = tuple(sorted((left_image_id, right_image_id)))
        impostor_pairs.add(ordered_pair)
        attempts += 1

    for left_image_id, right_image_id in sorted(impostor_pairs):
        pair_records.append(
            {
                "pair_id": f"pair_{pair_id:08d}",
                "left_image_id": left_image_id,
                "right_image_id": right_image_id,
                "left_identity": left_image_id.split("/")[0],
                "right_identity": right_image_id.split("/")[0],
                "pair_type": "impostor",
                "label": "0",
            }
        )
        pair_id += 1

    pair_manifest_path = output_dir / "pair_manifest.csv"
    write_csv_records(pair_manifest_path, pair_records, fieldnames=list(pair_records[0].keys()))

    pair_summary = {
        "genuine_pairs": total_genuine_pairs,
        "impostor_pairs": len(impostor_pairs),
        "total_pairs": len(pair_records),
        "random_seed": random_seed,
        "max_genuine_pairs_per_identity": max_genuine_pairs_per_identity,
        "eligible_identities": len(identity_names),
    }
    (output_dir / "pair_manifest_summary.json").write_text(json.dumps(pair_summary, indent=2), encoding="utf-8")
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


def compute_roc_auc_eer(pair_score_records):
    valid_records = [record for record in pair_score_records if record["valid"] == "1"]
    if not valid_records:
        raise ValueError("No valid pair scores available for ROC calculation.")

    labels = np.array([int(record["label"]) for record in valid_records], dtype=np.int32)
    scores = np.array([float(record["similarity_score"]) for record in valid_records], dtype=np.float64)
    unique_thresholds = np.sort(np.unique(scores))[::-1]
    thresholds = np.concatenate(([np.inf], unique_thresholds, [-np.inf]))

    roc_points = []
    for threshold in thresholds:
        predictions = scores >= threshold
        true_positive = np.sum((predictions == 1) & (labels == 1))
        false_positive = np.sum((predictions == 1) & (labels == 0))
        true_negative = np.sum((predictions == 0) & (labels == 0))
        false_negative = np.sum((predictions == 0) & (labels == 1))

        tar = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        far = false_positive / (false_positive + true_negative) if (false_positive + true_negative) else 0.0
        frr = false_negative / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        roc_points.append(
            {
                "threshold": float(threshold),
                "tar": float(tar),
                "far": float(far),
                "frr": float(frr),
            }
        )

    roc_points_sorted = sorted(roc_points, key=lambda point: point["far"])
    auc = float(np.trapz([point["tar"] for point in roc_points_sorted], [point["far"] for point in roc_points_sorted]))
    best_point = min(roc_points, key=lambda point: abs(point["far"] - point["frr"]))
    eer = (best_point["far"] + best_point["frr"]) / 2.0

    return {
        "auc": auc,
        "eer": float(eer),
        "eer_threshold": float(best_point["threshold"]),
        "roc_points": roc_points_sorted,
        "valid_pairs": len(valid_records),
        "genuine_pairs": int(np.sum(labels == 1)),
        "impostor_pairs": int(np.sum(labels == 0)),
    }


def write_roc_outputs(roc_summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_summary = {
        "auc": roc_summary["auc"],
        "eer": roc_summary["eer"],
        "eer_threshold": roc_summary["eer_threshold"],
        "valid_pairs": roc_summary["valid_pairs"],
        "genuine_pairs": roc_summary["genuine_pairs"],
        "impostor_pairs": roc_summary["impostor_pairs"],
    }
    threshold_summary_path = output_dir / "threshold_summary.json"
    threshold_summary_path.write_text(json.dumps(threshold_summary, indent=2), encoding="utf-8")

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
    threshold_summary = json.loads(Path(threshold_summary_path).read_text(encoding="utf-8"))
    threshold = float(threshold_summary["eer_threshold"])

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
    metrics_summary.update(
        summarize_score_distribution(
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
        )
    )
    metrics_summary["threshold"] = threshold
    metrics_summary["valid_pairs"] = len(valid_labels)
    metrics_summary["invalid_pairs"] = len(updated_records) - len(valid_labels)

    metrics_summary_path = output_dir / "metrics_summary.json"
    metrics_summary_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    return evaluated_pair_scores_path, metrics_summary_path


def compute_classification_metrics(labels, predictions):
    if not labels:
        return {
            "accuracy": 0.0,
            "tar": 0.0,
            "far": 0.0,
            "frr": 0.0,
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
        }

    labels_array = np.array(labels, dtype=np.int32)
    predictions_array = np.array(predictions, dtype=np.int32)
    true_positive = int(np.sum((labels_array == 1) & (predictions_array == 1)))
    false_positive = int(np.sum((labels_array == 0) & (predictions_array == 1)))
    true_negative = int(np.sum((labels_array == 0) & (predictions_array == 0)))
    false_negative = int(np.sum((labels_array == 1) & (predictions_array == 0)))
    total = len(labels)

    return {
        "accuracy": float((true_positive + true_negative) / total) if total else 0.0,
        "tar": float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) else 0.0,
        "far": float(false_positive / (false_positive + true_negative)) if (false_positive + true_negative) else 0.0,
        "frr": float(false_negative / (true_positive + false_negative)) if (true_positive + false_negative) else 0.0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def summarize_score_distribution(genuine_scores, impostor_scores):
    return {
        "genuine_count": len(genuine_scores),
        "impostor_count": len(impostor_scores),
        "genuine_mean": float(np.mean(genuine_scores)) if genuine_scores else 0.0,
        "genuine_std": float(np.std(genuine_scores)) if genuine_scores else 0.0,
        "genuine_median": float(np.median(genuine_scores)) if genuine_scores else 0.0,
        "impostor_mean": float(np.mean(impostor_scores)) if impostor_scores else 0.0,
        "impostor_std": float(np.std(impostor_scores)) if impostor_scores else 0.0,
        "impostor_median": float(np.median(impostor_scores)) if impostor_scores else 0.0,
    }


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
    drift_summary = {
        "valid_samples": len(cosine_drifts),
        "invalid_samples": len(drift_records) - len(cosine_drifts),
        "cosine_mean": float(np.mean(cosine_drifts)) if cosine_drifts else 0.0,
        "cosine_std": float(np.std(cosine_drifts)) if cosine_drifts else 0.0,
        "cosine_median": float(np.median(cosine_drifts)) if cosine_drifts else 0.0,
        "cosine_q1": float(np.quantile(cosine_drifts, 0.25)) if cosine_drifts else 0.0,
        "cosine_q3": float(np.quantile(cosine_drifts, 0.75)) if cosine_drifts else 0.0,
        "euclidean_mean": float(np.mean(euclidean_distances)) if euclidean_distances else 0.0,
        "euclidean_std": float(np.std(euclidean_distances)) if euclidean_distances else 0.0,
    }
    drift_summary_path = output_dir / "embedding_drift_summary.json"
    drift_summary_path.write_text(json.dumps(drift_summary, indent=2), encoding="utf-8")
    return drift_records_path, drift_summary_path


def summarize_embedding_success(embedding_metadata_path, output_path):
    embedding_metadata_records = load_csv_records(embedding_metadata_path)
    success_count = sum(record["success"] == "1" for record in embedding_metadata_records)
    summary = {
        "total_samples": len(embedding_metadata_records),
        "successful_embeddings": success_count,
        "failed_embeddings": len(embedding_metadata_records) - success_count,
        "success_rate": success_count / len(embedding_metadata_records) if embedding_metadata_records else 0.0,
    }
    output_path = Path(output_path)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def aggregate_condition_results(condition_result_dirs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_valid_sets = {}
    image_valid_sets = {}
    full_condition_rows = []
    condition_payloads = {}

    for condition_name, condition_result_dir in condition_result_dirs.items():
        condition_result_dir = Path(condition_result_dir)
        pair_records = load_csv_records(condition_result_dir / "pair_scores_with_predictions.csv")
        drift_records = load_csv_records(condition_result_dir / "embedding_drift.csv")
        metrics_summary = json.loads((condition_result_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        drift_summary = json.loads((condition_result_dir / "embedding_drift_summary.json").read_text(encoding="utf-8"))
        success_summary = json.loads((condition_result_dir / "embedding_success_summary.json").read_text(encoding="utf-8"))

        pair_valid_sets[condition_name] = {record["pair_id"] for record in pair_records if record["valid"] == "1"}
        image_valid_sets[condition_name] = {record["image_id"] for record in drift_records if record["valid"] == "1"}

        condition_payloads[condition_name] = {
            "pair_records": pair_records,
            "drift_records": drift_records,
            "metrics_summary": metrics_summary,
            "drift_summary": drift_summary,
            "success_summary": success_summary,
        }
        full_condition_rows.append(
            {
                "condition": condition_name,
                "accuracy": f"{metrics_summary['accuracy']:.10f}",
                "tar": f"{metrics_summary['tar']:.10f}",
                "far": f"{metrics_summary['far']:.10f}",
                "frr": f"{metrics_summary['frr']:.10f}",
                "valid_pairs": str(metrics_summary["valid_pairs"]),
                "invalid_pairs": str(metrics_summary["invalid_pairs"]),
                "embedding_success_rate": f"{success_summary['success_rate']:.10f}",
                "drift_cosine_mean": f"{drift_summary['cosine_mean']:.10f}",
            }
        )

    common_valid_pair_ids = set.intersection(*pair_valid_sets.values()) if pair_valid_sets else set()
    common_valid_image_ids = set.intersection(*image_valid_sets.values()) if image_valid_sets else set()

    common_subset_summary = {
        "conditions": list(condition_result_dirs.keys()),
        "common_valid_pair_count": len(common_valid_pair_ids),
        "common_valid_image_count": len(common_valid_image_ids),
    }
    (output_dir / "common_subset_summary.json").write_text(json.dumps(common_subset_summary, indent=2), encoding="utf-8")

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
        genuine_scores = [
            float(record["similarity_score"])
            for record in common_pair_records
            if record["label"] == "1"
        ]
        impostor_scores = [
            float(record["similarity_score"])
            for record in common_pair_records
            if record["label"] == "0"
        ]
        common_metrics = compute_classification_metrics(common_labels, common_predictions)
        common_distribution = summarize_score_distribution(genuine_scores, impostor_scores)
        common_drift_values = [
            float(record["embedding_drift_cosine"])
            for record in payload["drift_records"]
            if record["image_id"] in common_valid_image_ids and record["valid"] == "1"
        ]

        common_condition_rows.append(
            {
                "condition": condition_name,
                "accuracy": f"{common_metrics['accuracy']:.10f}",
                "tar": f"{common_metrics['tar']:.10f}",
                "far": f"{common_metrics['far']:.10f}",
                "frr": f"{common_metrics['frr']:.10f}",
                "valid_pairs": str(len(common_pair_records)),
                "invalid_pairs": str(payload["metrics_summary"]["invalid_pairs"]),
                "embedding_success_rate": f"{payload['success_summary']['success_rate']:.10f}",
                "drift_cosine_mean": f"{(float(np.mean(common_drift_values)) if common_drift_values else 0.0):.10f}",
            }
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
