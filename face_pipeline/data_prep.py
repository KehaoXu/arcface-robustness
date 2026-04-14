from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from face_pipeline.config import DEFAULT_MIN_IMAGES_PER_IDENTITY, IMAGE_EXTENSIONS
from face_pipeline.io_utils import write_csv_records, write_json


def _iter_supported_images(root_dir):
    root_dir = Path(root_dir)
    for image_path in sorted(root_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def _build_sample_record(original_dir, image_path):
    relative_path = image_path.relative_to(original_dir).as_posix()
    image_id = str(Path(relative_path).with_suffix("")).replace("\\", "/")
    return {
        "image_id": image_id,
        "identity": image_path.parent.name,
        "relative_path": relative_path,
        "file_stem": Path(relative_path).stem,
    }


def resize_lfw_images(input_dir, output_dir, output_size=(512, 512), output_suffix=".png"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    image_paths = sorted(input_dir.rglob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPG files found under {input_dir}")

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for image_path in tqdm(image_paths, desc="Preparing LFW"):
        relative_path = image_path.relative_to(input_dir).with_suffix(output_suffix)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            skipped_count += 1
            continue

        try:
            image_rgb = Image.open(image_path).convert("RGB")
            image_rgb = image_rgb.resize(output_size, Image.LANCZOS)
            image_rgb.save(output_path)
            success_count += 1
        except Exception:
            failed_count += 1

    return {
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "output_dir": str(output_dir),
    }


def build_sample_index(original_dir, output_dir, min_images_per_identity=DEFAULT_MIN_IMAGES_PER_IDENTITY):
    original_dir = Path(original_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_records = []
    identity_to_paths = defaultdict(list)
    for image_path in _iter_supported_images(original_dir):
        sample_record = _build_sample_record(original_dir, image_path)
        identity_to_paths[sample_record["identity"]].append(sample_record["relative_path"])
        sample_records.append(sample_record)

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
    write_json(identity_filter_summary_path, filter_summary)
    return sample_index_path, identity_filter_summary_path
