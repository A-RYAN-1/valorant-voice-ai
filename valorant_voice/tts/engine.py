"""XTTS-v2 inference: the base model or a fine-tuned checkpoint, conditioned per agent."""

import threading
from pathlib import Path

import numpy as np

from .. import config
from ..agents import AGENTS, require_agent
from .references import reference_paths

BASE_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
OUTPUT_RATE = 24000


class XttsEngine:
    def __init__(self, checkpoint_dir: str = config.CHECKPOINT_DIR, device: str = config.DEVICE):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self._model = None
        self._latents = {}
        self._lock = threading.Lock()  # one synthesis at a time on one GPU

    def _load(self):
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        folder = self.checkpoint_dir or Path(ModelManager().download_model(BASE_MODEL)[1]).parent
        model_config = XttsConfig()
        model_config.load_json(str(Path(folder) / "config.json"))
        model = Xtts(model_config)
        model.load_checkpoint(model_config, checkpoint_dir=str(folder), eval=True)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self.device)

    def available_agents(self) -> list[str]:
        return [slug for slug in AGENTS if reference_paths(slug)]

    def synthesize(self, agent: str, text: str) -> tuple[np.ndarray, int]:
        require_agent(agent)
        paths = reference_paths(agent)
        if not paths:
            raise ValueError(f"No reference clips for {agent}; run `python -m valorant_voice references`")
        with self._lock:
            if self._model is None:
                self._load()
            if agent not in self._latents:
                self._latents[agent] = self._model.get_conditioning_latents(audio_path=[str(p) for p in paths])
            gpt_latent, speaker_embedding = self._latents[agent]
            output = self._model.inference(text, "en", gpt_latent, speaker_embedding, temperature=0.7)
        return np.asarray(output["wav"], dtype="float32"), OUTPUT_RATE
