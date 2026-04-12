from pathlib import Path


CONDITION_NAMES = [
    "original",
    "downsample",
    "noise",
    "downsample_resshift",
    "noise_resshift",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_IMAGES_PER_IDENTITY = 10
DEFAULT_MAX_GENUINE_PAIRS_PER_IDENTITY = 100
DEFAULT_DETECTION_SIZE = (640, 640)
DEFAULT_MODEL_NAME = "buffalo_l"
DEFAULT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
DEFAULT_RESULTS_DIR = Path("results")
