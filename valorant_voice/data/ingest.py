"""Step 1: turn the fetched raw lines into clean 22.05 kHz mono clips."""

from collections import Counter
from pathlib import Path

import soundfile as sf

from .. import config
from ..agents import require_agent
from . import audio

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}


def clean_clip(samples, rate):
    """Returns (clip, None) when usable, or (None, reason) when the clip is dropped."""
    mono = audio.to_mono(samples)
    mono = audio.resample(mono, rate, config.SAMPLE_RATE)
    mono = audio.trim_silence(mono, config.SAMPLE_RATE)
    length = audio.duration(mono, config.SAMPLE_RATE)
    if length < config.MIN_SECONDS:
        return None, "too_short"
    if length > config.MAX_SECONDS:
        return None, "too_long"
    return audio.normalise_loudness(mono), None


def ingest_agent(agent: str, raw_dir: Path = config.RAW_DIR, clean_dir: Path = config.CLEAN_DIR) -> Counter:
    """Cleans every clip in raw_dir/<agent> into clean_dir/<agent>. Returns counts per outcome."""
    require_agent(agent)
    source = raw_dir / agent
    target = clean_dir / agent
    target.mkdir(parents=True, exist_ok=True)
    outcome = Counter()
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        try:
            samples, rate = sf.read(path, dtype="float32")
        except (RuntimeError, sf.LibsndfileError):
            outcome["unreadable"] += 1
            continue
        clip, reason = clean_clip(samples, rate)
        if clip is None:
            outcome[reason] += 1
            continue
        sf.write(target / f"{path.stem}.wav", clip, config.SAMPLE_RATE, subtype="PCM_16")
        outcome["kept"] += 1
    return outcome
