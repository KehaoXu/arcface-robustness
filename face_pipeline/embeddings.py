from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from tqdm import tqdm

from face_pipeline.config import DEFAULT_DETECTION_SIZE, DEFAULT_MODEL_NAME, DEFAULT_PROVIDERS
from face_pipeline.io_utils import load_csv_records, load_embedding_lookup, write_csv_records


CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


def _resolve_provider_names(provider_names):
    requested_providers = list(provider_names or DEFAULT_PROVIDERS)
    available_providers = list(ort.get_available_providers())
    available_provider_set = set(available_providers)

    resolved_providers = [
        provider_name
        for provider_name in requested_providers
        if provider_name in available_provider_set
    ]

    if not resolved_providers and CPU_PROVIDER in available_provider_set:
        resolved_providers = [CPU_PROVIDER]

    if not resolved_providers:
        raise RuntimeError(
            "Embedding extraction requires at least one supported ONNX Runtime provider, but none of the "
            "requested providers are available. "
            f"Requested providers: {requested_providers}. "
            f"Available providers: {available_providers}"
        )

    return requested_providers, resolved_providers, available_providers


def _iter_model_sessions(face_app):
    for model in getattr(face_app, "models", {}).values():
        session = getattr(model, "session", None)
        if session is not None:
            yield session


def _get_session_providers(face_app):
    session_providers = []
    for session in _iter_model_sessions(face_app):
        providers = list(session.get_providers())
        session_providers.extend(providers)
    return session_providers


class FaceEmbedder:
    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        providers=None,
        det_size=DEFAULT_DETECTION_SIZE,
        ctx_id=0,
    ):
        (
            requested_provider_names,
            provider_names,
            available_provider_names,
        ) = _resolve_provider_names(providers)
        self.app = FaceAnalysis(name=model_name, providers=provider_names)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.session_providers = _get_session_providers(self.app)
        if not self.session_providers:
            raise RuntimeError("FaceAnalysis initialized, but no ONNX Runtime sessions were created.")

        print(
            "FaceEmbedder initialized with providers "
            f"{provider_names}. Requested: {requested_provider_names}. "
            f"Available: {available_provider_names}. "
            f"Model session providers: {self.session_providers}"
        )

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
        eligible_only=False,
    ):
        sample_records = load_csv_records(sample_index_path)
        if eligible_only:
            sample_records = [
                sample_record
                for sample_record in sample_records
                if sample_record.get("include_in_protocol") == "1"
            ]
        if not sample_records:
            raise ValueError(f"No sample records available for condition {condition_name}.")
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


__all__ = [
    "FaceEmbedder",
    "load_csv_records",
    "load_embedding_lookup",
    "write_csv_records",
]
