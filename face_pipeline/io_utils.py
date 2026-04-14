import csv
import json
from pathlib import Path

import numpy as np


def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv_records(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv_records(csv_path, records, fieldnames):
    csv_path = Path(csv_path)
    ensure_directory(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def read_json(json_path):
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def write_json(json_path, payload):
    json_path = Path(json_path)
    ensure_directory(json_path.parent)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


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
