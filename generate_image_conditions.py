import argparse
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
):
    original_dir = Path(original_dir)
    downsample_dir = Path(downsample_dir)
    noise_dir = Path(noise_dir)
    image_paths = _collect_image_paths(original_dir)

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
    }


def _find_matching_restored_image(source_dir, relative_path):
    source_dir = Path(source_dir)
    relative_path = Path(relative_path)
    exact_path = source_dir / relative_path
    if exact_path.exists():
        return exact_path

    same_stem_candidates = []
    same_parent_dir = source_dir / relative_path.parent
    if same_parent_dir.exists():
        for extension in IMAGE_EXTENSIONS:
            candidate_path = same_parent_dir / f"{relative_path.stem}{extension}"
            if candidate_path.exists():
                same_stem_candidates.append(candidate_path)
    if same_stem_candidates:
        return sorted(same_stem_candidates)[0]

    recursive_candidates = list(source_dir.rglob(f"{relative_path.stem}.*"))
    recursive_candidates = [
        candidate_path
        for candidate_path in recursive_candidates
        if candidate_path.is_file() and candidate_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if recursive_candidates:
        return sorted(recursive_candidates)[0]

    return None


def normalize_restored_condition(reference_dir, restored_source_dir, output_dir, jpeg_quality=95, skip_existing=False):
    reference_dir = Path(reference_dir)
    restored_source_dir = Path(restored_source_dir)
    output_dir = Path(output_dir)
    reference_image_paths = _collect_image_paths(reference_dir)

    if not reference_image_paths:
        raise FileNotFoundError(f"No supported images found under {reference_dir}")

    copied_count = 0
    missing_count = 0
    for reference_image_path in tqdm(reference_image_paths, desc=f"Normalizing {output_dir.name}"):
        relative_path = reference_image_path.relative_to(reference_dir)
        output_path = output_dir / relative_path
        if skip_existing and output_path.exists():
            copied_count += 1
            continue

        restored_image_path = _find_matching_restored_image(restored_source_dir, relative_path)
        if restored_image_path is None:
            missing_count += 1
            continue

        image_rgb = Image.open(restored_image_path).convert("RGB")
        _save_image(image_rgb, output_path, jpeg_quality)
        copied_count += 1

    return {
        "reference_images": len(reference_image_paths),
        "copied_images": copied_count,
        "missing_images": missing_count,
        "output_dir": str(output_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and normalize image-quality conditions for ArcFace evaluation.")
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--downsample-dir", default="dataset/downsample")
    parser.add_argument("--noise-dir", default="dataset/noise")
    parser.add_argument("--downsample-scale", type=float, default=0.5)
    parser.add_argument("--blur-radius", type=float, default=1.5)
    parser.add_argument("--gaussian-sigma", type=float, default=22.0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--downsample-resshift-source-dir")
    parser.add_argument("--noise-resshift-source-dir")
    parser.add_argument("--downsample-resshift-dir", default="dataset/downsample_resshift")
    parser.add_argument("--noise-resshift-dir", default="dataset/noise_resshift")
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
    )
    print({"degradation_summary": degradation_summary})

    if args.downsample_resshift_source_dir:
        downsample_resshift_summary = normalize_restored_condition(
            reference_dir=args.original_dir,
            restored_source_dir=args.downsample_resshift_source_dir,
            output_dir=args.downsample_resshift_dir,
            skip_existing=args.skip_existing,
        )
        print({"downsample_resshift_summary": downsample_resshift_summary})

    if args.noise_resshift_source_dir:
        noise_resshift_summary = normalize_restored_condition(
            reference_dir=args.original_dir,
            restored_source_dir=args.noise_resshift_source_dir,
            output_dir=args.noise_resshift_dir,
            skip_existing=args.skip_existing,
        )
        print({"noise_resshift_summary": noise_resshift_summary})


if __name__ == "__main__":
    main()
