from TTS.api import TTS
from app.core.config import MODEL_NAME, SAGE_VOICE_PATH

class TTSEngine:
    def __init__(self):
        self.tts = TTS(model_name=MODEL_NAME, progress_bar=False)

    def generate(self, text: str, output_path: str):
        self.tts.tts_to_file(
            text=text,
            speaker_wav=str(SAGE_VOICE_PATH),
            language="en",
            file_path=output_path
        )
