from fastapi import APIRouter
from app.services.tts_engine import TTSEngine
from app.services.audio_utils import generate_audio_path

router = APIRouter()
tts_engine = TTSEngine()

@router.post("/generate")
def generate_voice(text: str):
    output_path = generate_audio_path()
    tts_engine.generate(text, str(output_path))

    return {
        "status": "success",
        "audio_path": str(output_path)
    }
