import csv
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

from experiment_config import DEFAULT_DETECTION_SIZE, DEFAULT_MODEL_NAME, DEFAULT_PROVIDERS


class FaceEmbedder:
    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        providers=None,
        det_size=DEFAULT_DETECTION_SIZE,
        ctx_id=0,
    ):
        provider_names = providers or DEFAULT_PROVIDERS
        self.app = FaceAnalysis(name=model_name, providers=provider_names)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def extract_embedding(self, image_path):
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")

        detected_faces = self.app.get(image_bgr)
        if not detected_faces:
            raise ValueError(f"No face detected: {image_path}")

        return detected_faces[0].embedding.astype(np.float32)

    def extract_embeddings_for_condition(
        self,
        sample_index_path,
        condition_dir,
        output_dir,
        condition_name,
    ):
        sample_records = load_csv_records(sample_index_path)
        condition_dir = Path(condition_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        embedding_matrix = []
        embedding_records = []

        for sample_record in tqdm(sample_records, desc=f"Embedding {condition_name}"):
            relative_path = Path(sample_record["relative_path"])
            image_path = condition_dir / relative_path
            embedding_record = {
                "image_id": sample_record["image_id"],
                "identity": sample_record["identity"],
                "condition": condition_name,
                "relative_path": sample_record["relative_path"],
                "success": "0",
                "error_message": "",
                "embedding_index": "",
            }

            try:
                embedding_vector = self.extract_embedding(image_path)
                embedding_record["success"] = "1"
                embedding_record["embedding_index"] = str(len(embedding_matrix))
                embedding_matrix.append(embedding_vector)
            except Exception as exc:
                embedding_record["error_message"] = str(exc)

            embedding_records.append(embedding_record)

        if embedding_matrix:
            embedding_array = np.stack(embedding_matrix).astype(np.float32)
        else:
            embedding_array = np.empty((0, 0), dtype=np.float32)

        metadata_path = output_dir / "embedding_results.csv"
        matrix_path = output_dir / "embedding_vectors.npy"
        write_csv_records(metadata_path, embedding_records, fieldnames=list(embedding_records[0].keys()))
        np.save(matrix_path, embedding_array)

        return metadata_path, matrix_path


def load_csv_records(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv_records(csv_path, records, fieldnames):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_embedding_lookup(metadata_path, matrix_path):
    metadata_records = load_csv_records(metadata_path)
    embedding_array = np.load(matrix_path, allow_pickle=False)
    embedding_lookup = {}

    for metadata_record in metadata_records:
        if metadata_record["success"] != "1":
            continue
        embedding_index = int(metadata_record["embedding_index"])
        embedding_lookup[metadata_record["image_id"]] = embedding_array[embedding_index]

    return metadata_records, embedding_lookup
