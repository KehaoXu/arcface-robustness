import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from face_pipeline.config import IMAGE_EXTENSIONS
from face_pipeline.conditions.transforms import apply_downsample, apply_gaussian_noise
from face_pipeline.io_utils import load_csv_records


def save_image(image_rgb, output_path, jpeg_quality):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image_rgb.save(output_path, quality=jpeg_quality)
    else:
        image_rgb.save(output_path)


def load_rgb_image(image_path):
    return Image.open(image_path).convert("RGB")


def collect_image_paths(input_dir):
    input_dir = Path(input_dir)
    return [image_path for image_path in sorted(input_dir.rglob("*")) if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS]


def collect_image_paths_from_sample_index(original_dir, sample_index_path, eligible_only=False):
    original_dir = Path(original_dir)
    sample_records = load_csv_records(sample_index_path)

    image_paths = []
    for sample_record in sample_records:
        if eligible_only and sample_record.get("include_in_protocol") != "1":
            continue
        image_path = original_dir / sample_record["relative_path"]
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(image_path)

    return sorted(image_paths)


CONDITION_TRANSFORMS = {
    "downsample": lambda image_rgb, task: apply_downsample(
        image_rgb=image_rgb,
        downsample_scale=task["downsample_scale"],
        blur_radius=task["blur_radius"],
    ),
    "noise": lambda image_rgb, task: apply_gaussian_noise(
        image_rgb=image_rgb,
        gaussian_sigma=task["gaussian_sigma"],
    ),
}


def _resolve_output_path(output_dir, original_dir, image_path):
    return Path(output_dir) / image_path.relative_to(original_dir)


def _process_condition_worker(task_record):
    condition_name = task_record["condition_name"]
    try:
        transform = CONDITION_TRANSFORMS[condition_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported degradation mode: {condition_name}") from exc

    degraded_image = transform(load_rgb_image(task_record["input_path"]), task_record)
    save_image(degraded_image, task_record["output_path"], task_record["jpeg_quality"])
    return str(task_record["output_path"])


def _build_condition_tasks(
    *,
    condition_name,
    image_paths,
    original_dir,
    output_dir,
    downsample_scale,
    blur_radius,
    gaussian_sigma,
    jpeg_quality,
    skip_existing,
):
    condition_tasks = []
    for image_path in image_paths:
        output_path = _resolve_output_path(output_dir, original_dir, image_path)
        if skip_existing and output_path.exists():
            continue

        task_record = {
            "condition_name": condition_name,
            "input_path": str(image_path),
            "output_path": str(output_path),
            "jpeg_quality": jpeg_quality,
        }
        if condition_name == "downsample":
            task_record["downsample_scale"] = downsample_scale
            task_record["blur_radius"] = blur_radius
        elif condition_name == "noise":
            task_record["gaussian_sigma"] = gaussian_sigma
        else:
            raise ValueError(f"Unsupported degradation mode: {condition_name}")

        condition_tasks.append(task_record)
    return condition_tasks


def generate_condition_images(
    *,
    condition_name,
    original_dir,
    output_dir,
    downsample_scale=0.5,
    blur_radius=1.5,
    gaussian_sigma=22.0,
    jpeg_quality=75,
    max_workers=None,
    skip_existing=False,
    sample_index_path=None,
    eligible_only=False,
):
    original_dir = Path(original_dir)
    output_dir = Path(output_dir)
    if sample_index_path is None:
        image_paths = collect_image_paths(original_dir)
    else:
        image_paths = collect_image_paths_from_sample_index(
            original_dir=original_dir,
            sample_index_path=sample_index_path,
            eligible_only=eligible_only,
        )

    if not image_paths:
        raise FileNotFoundError(f"No supported images found under {original_dir}")

    condition_tasks = _build_condition_tasks(
        condition_name=condition_name,
        image_paths=image_paths,
        original_dir=original_dir,
        output_dir=output_dir,
        downsample_scale=downsample_scale,
        blur_radius=blur_radius,
        gaussian_sigma=gaussian_sigma,
        jpeg_quality=jpeg_quality,
        skip_existing=skip_existing,
    )

    if not condition_tasks:
        return {
            "condition_name": condition_name,
            "total_images": len(image_paths),
            "scheduled_tasks": 0,
            "output_dir": str(output_dir),
            "sample_index_path": str(sample_index_path) if sample_index_path is not None else "",
            "eligible_only": eligible_only,
        }

    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(_process_condition_worker, condition_tasks),
                total=len(condition_tasks),
                desc=f"Generating {condition_name}",
            )
        )

    return {
        "condition_name": condition_name,
        "total_images": len(image_paths),
        "scheduled_tasks": len(condition_tasks),
        "output_dir": str(output_dir),
        "sample_index_path": str(sample_index_path) if sample_index_path is not None else "",
        "eligible_only": eligible_only,
    }


def generate_degraded_conditions(*args, **kwargs):
    raise RuntimeError(
        "generate_degraded_conditions has been split into single-condition runs. "
        "Use generate_condition_images(..., condition_name='downsample') or "
        "generate_condition_images(..., condition_name='noise'), or the new CLI scripts."
    )
