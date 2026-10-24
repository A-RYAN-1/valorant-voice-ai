"""The only module that reads environment variables. Everything else imports these constants."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, ROOT / default)).expanduser().resolve()


# Local-only folders, git-ignored. They hold the user's own extracted game audio and the weights.
DATA_DIR = _path("VV_DATA_DIR", "data")
MODEL_DIR = _path("VV_MODEL_DIR", "models")
OUTPUT_DIR = _path("VV_OUTPUT_DIR", "outputs")

RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
MANIFEST_DIR = DATA_DIR / "manifests"
REFERENCE_DIR = DATA_DIR / "reference"

# Fine-tuned checkpoint folder (config.json, model.pth, vocab.json); empty means the base XTTS-v2.
CHECKPOINT_DIR = os.environ.get("VV_CHECKPOINT_DIR", "")

DEVICE = os.environ.get("VV_DEVICE", "")  # "" picks cuda when available, otherwise cpu

# Request limits for the API.
MAX_CHARS = int(os.environ.get("VV_MAX_CHARS", "300"))
RATE_LIMIT_PER_MINUTE = int(os.environ.get("VV_RATE_LIMIT_PER_MINUTE", "20"))

# Dataset rules.
SAMPLE_RATE = 22050  # the rate XTTS-v2 expects for training audio
MIN_SECONDS = float(os.environ.get("VV_MIN_SECONDS", "1.5"))
MAX_SECONDS = 12.0
SPLIT_SEED = int(os.environ.get("VV_SPLIT_SEED", "13"))
VAL_FRACTION = 0.05
TEST_FRACTION = 0.05
