import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


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


def parse_args():
    parser = argparse.ArgumentParser(description="Resize and normalize LFW images.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--output-suffix", default=".png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = resize_lfw_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_size=(args.width, args.height),
        output_suffix=args.output_suffix,
    )
    print(summary)
