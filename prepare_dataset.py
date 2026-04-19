import argparse

from face_pipeline.config import DEFAULT_ORIGINAL_DIR
from face_pipeline.data_prep import resize_lfw_images


def parse_args():
    parser = argparse.ArgumentParser(description="Resize and normalize LFW images.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_ORIGINAL_DIR))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--output-suffix", default=".png")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = resize_lfw_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_size=(args.width, args.height),
        output_suffix=args.output_suffix,
    )
    print(summary)


if __name__ == "__main__":
    main()
