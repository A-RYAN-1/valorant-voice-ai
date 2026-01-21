from uuid import uuid4
from app.core.config import OUTPUT_DIR

def generate_audio_path():
    return OUTPUT_DIR / f"{uuid4().hex}.wav"
