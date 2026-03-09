import sys
from PIL import Image
from pathlib import Path

LFW_ROOT    = "lfw/lfw-deepfunneled/lfw-deepfunneled"  # LFW dataset root directory
RESIZED_DIR = "lfw_resized"                             # Directory to save resized images

def resize_images(src_root, dst_root, size=(512, 512)):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if not src_root.exists():
        print(f"[ERROR] Dataset path does not exist: {src_root}")
        sys.exit(1)

    img_paths = list(src_root.rglob("*.jpg"))

    if not img_paths:
        print(f"[ERROR] No images found in: {src_root}")
        sys.exit(1)

    print(f"[Step 1] Found {len(img_paths)} images, resizing to {size}...")

    success, skip, fail = 0, 0, 0
    for i, img_path in enumerate(img_paths):
        rel_path = img_path.relative_to(src_root)
        dst_path = dst_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already processed
        if dst_path.exists():
            skip += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size, Image.LANCZOS)
            img.save(dst_path)
            success += 1
        except Exception as e:
            print(f"  [WARN] Failed to process {img_path}: {e}")
            fail += 1

        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{len(img_paths)}  success:{success}  skipped:{skip}  failed:{fail}")

    print(f"[Step 1] Done! success:{success}  skipped(already exists):{skip}  failed:{fail}")
    print(f"         Resized images saved to: {dst_root}\n")

if __name__ == "__main__":
    resize_images(LFW_ROOT, RESIZED_DIR, size=(512, 512))