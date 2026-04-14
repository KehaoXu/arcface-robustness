import numpy as np


def cosine_similarity_score(vector_a, vector_b):
    vector_a = vector_a / np.linalg.norm(vector_a)
    vector_b = vector_b / np.linalg.norm(vector_b)
    return float(np.dot(vector_a, vector_b))


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
