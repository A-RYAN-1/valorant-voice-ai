"""Step 1: turn the fetched raw lines into clean 22.05 kHz mono clips."""

from collections import Counter
from pathlib import Path

import soundfile as sf

from .. import config
from ..agents import require_agent
from . import audio

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}


LENGTH_EDGES = (0.5, 1.0, 1.5, 3.0, 6.0, config.MAX_SECONDS)


def length_bucket(seconds: float) -> str:
    for edge in LENGTH_EDGES:
        if seconds < edge:
            return f"len_under_{edge:g}s"
    return f"len_over_{LENGTH_EDGES[-1]:g}s"


def speech_only(samples, rate):
    mono = audio.resample(audio.to_mono(samples), rate, config.SAMPLE_RATE)
    return audio.trim_silence(mono, config.SAMPLE_RATE)


def accept_length(speech):
    """Returns (clip, None) when usable, or (None, reason) when the clip is dropped."""
    length = audio.duration(speech, config.SAMPLE_RATE)
    if length < config.MIN_SECONDS:
        return None, "too_short"
    if length > config.MAX_SECONDS:
        return None, "too_long"
    return audio.normalise_loudness(speech), None


def clean_clip(samples, rate):
    return accept_length(speech_only(samples, rate))


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
        speech = speech_only(samples, rate)
        outcome[length_bucket(audio.duration(speech, config.SAMPLE_RATE))] += 1
        clip, reason = accept_length(speech)
        if clip is None:
            outcome[reason] += 1
            continue
        sf.write(target / f"{path.stem}.wav", clip, config.SAMPLE_RATE, subtype="PCM_16")
        outcome["kept"] += 1
    return outcome
