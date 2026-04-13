import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from experiment_config import IMAGE_EXTENSIONS


def _save_image(image_rgb, output_path, jpeg_quality):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image_rgb.save(output_path, quality=jpeg_quality)
    else:
        image_rgb.save(output_path)


def _process_condition_worker(args):
    (
        input_path_str,
        output_path_str,
        mode_name,
        downsample_scale,
        blur_radius,
        gaussian_sigma,
        jpeg_quality,
    ) = args

    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    image_rgb = Image.open(input_path).convert("RGB")

    if mode_name == "downsample":
        width, height = image_rgb.size
        resized_width = max(1, int(width * downsample_scale))
        resized_height = max(1, int(height * downsample_scale))
        image_rgb = image_rgb.resize((resized_width, resized_height), resample=Image.BILINEAR)
        image_rgb = image_rgb.resize((width, height), resample=Image.BILINEAR)
        if blur_radius > 0:
            image_rgb = image_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    elif mode_name == "noise":
        image_array = np.asarray(image_rgb).astype(np.float32)
        image_array += np.random.normal(0, gaussian_sigma, image_array.shape).astype(np.float32)
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        image_rgb = Image.fromarray(image_array)
    else:
        raise ValueError(f"Unsupported degradation mode: {mode_name}")

    _save_image(image_rgb, output_path, jpeg_quality)
    return str(output_path)


def _collect_image_paths(input_dir):
    input_dir = Path(input_dir)
    return [
        image_path
        for image_path in sorted(input_dir.rglob("*"))
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _load_sample_records(sample_index_path):
    sample_index_path = Path(sample_index_path)
    with sample_index_path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def _collect_image_paths_from_sample_index(original_dir, sample_index_path, eligible_only=False):
    original_dir = Path(original_dir)
    sample_records = _load_sample_records(sample_index_path)

    image_paths = []
    for sample_record in sample_records:
        if eligible_only and sample_record.get("include_in_protocol") != "1":
            continue
        image_path = original_dir / sample_record["relative_path"]
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(image_path)

    return sorted(image_paths)


def generate_degraded_conditions(
    original_dir,
    downsample_dir,
    noise_dir,
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
    downsample_dir = Path(downsample_dir)
    noise_dir = Path(noise_dir)
    if sample_index_path is None:
        image_paths = _collect_image_paths(original_dir)
    else:
        image_paths = _collect_image_paths_from_sample_index(
            original_dir=original_dir,
            sample_index_path=sample_index_path,
            eligible_only=eligible_only,
        )

    if not image_paths:
        raise FileNotFoundError(f"No supported images found under {original_dir}")

    task_args = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(original_dir)
        downsample_output_path = downsample_dir / relative_path
        noise_output_path = noise_dir / relative_path

        if not skip_existing or not downsample_output_path.exists():
            task_args.append(
                (
                    str(image_path),
                    str(downsample_output_path),
                    "downsample",
                    downsample_scale,
                    blur_radius,
                    gaussian_sigma,
                    jpeg_quality,
                )
            )
        if not skip_existing or not noise_output_path.exists():
            task_args.append(
                (
                    str(image_path),
                    str(noise_output_path),
                    "noise",
                    downsample_scale,
                    blur_radius,
                    gaussian_sigma,
                    jpeg_quality,
                )
            )

    if not task_args:
        return {
            "total_images": len(image_paths),
            "scheduled_tasks": 0,
            "downsample_dir": str(downsample_dir),
            "noise_dir": str(noise_dir),
        }

    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(_process_condition_worker, task_args), total=len(task_args), desc="Generating conditions"))

    return {
        "total_images": len(image_paths),
        "scheduled_tasks": len(task_args),
        "downsample_dir": str(downsample_dir),
        "noise_dir": str(noise_dir),
        "sample_index_path": str(sample_index_path) if sample_index_path is not None else "",
        "eligible_only": eligible_only,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate downsampled and noisy image conditions for ArcFace evaluation.")
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--downsample-dir", default="dataset/downsample")
    parser.add_argument("--noise-dir", default="dataset/noise")
    parser.add_argument("--downsample-scale", type=float, default=0.5)
    parser.add_argument("--blur-radius", type=float, default=0)
    parser.add_argument("--gaussian-sigma", type=float, default=22.0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sample-index-path")
    parser.add_argument("--eligible-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    degradation_summary = generate_degraded_conditions(
        original_dir=args.original_dir,
        downsample_dir=args.downsample_dir,
        noise_dir=args.noise_dir,
        downsample_scale=args.downsample_scale,
        blur_radius=args.blur_radius,
        gaussian_sigma=args.gaussian_sigma,
        jpeg_quality=args.jpeg_quality,
        max_workers=args.max_workers,
        skip_existing=args.skip_existing,
        sample_index_path=args.sample_index_path,
        eligible_only=args.eligible_only,
    )
    print({"degradation_summary": degradation_summary})


if __name__ == "__main__":
    main()
