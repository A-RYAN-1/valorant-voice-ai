from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"

SAGE_VOICE_PATH = BASE_DIR / "models" / "sage" / "reference.wav"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
