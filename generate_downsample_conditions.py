import argparse

from face_pipeline.conditions import generate_condition_images


def parse_args():
    parser = argparse.ArgumentParser(description="Generate downsampled image conditions for ArcFace evaluation.")
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--output-dir", default="dataset/downsample")
    parser.add_argument("--downsample-scale", type=float, default=0.5)
    parser.add_argument("--blur-radius", type=float, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sample-index-path")
    parser.add_argument("--eligible-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    condition_summary = generate_condition_images(
        condition_name="downsample",
        original_dir=args.original_dir,
        output_dir=args.output_dir,
        downsample_scale=args.downsample_scale,
        blur_radius=args.blur_radius,
        jpeg_quality=args.jpeg_quality,
        max_workers=args.max_workers,
        skip_existing=args.skip_existing,
        sample_index_path=args.sample_index_path,
        eligible_only=args.eligible_only,
    )
    print({"condition_summary": condition_summary})


if __name__ == "__main__":
    main()
