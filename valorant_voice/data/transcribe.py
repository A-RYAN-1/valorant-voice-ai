"""Whisper steps: transcribe clips that have no text, or verify the text a source gave."""

import csv
from collections import Counter
from pathlib import Path

from .. import config
from ..agents import require_agent
from ..evaluation.metrics import word_error_rate

MIN_LANGUAGE_PROBABILITY = 0.8
MIN_AVG_LOGPROB = -1.0
SEPARATOR = "|"
MAX_VERIFY_WER = 0.3


def normalise_text(text: str) -> str:
    return " ".join(text.split())


def accept(language: str, language_probability: float, avg_logprob: float, text: str) -> bool:
    return (
        language == "en"
        and language_probability >= MIN_LANGUAGE_PROBABILITY
        and avg_logprob >= MIN_AVG_LOGPROB
        and len(text) >= 3
    )


def transcribe_agent(agent: str, model_size: str = "small.en", clean_dir: Path = config.CLEAN_DIR) -> int:
    """Writes clean/<agent>/transcripts.csv as `id|text`. Returns how many lines were kept."""
    from faster_whisper import WhisperModel  # heavy and optional, so imported here

    require_agent(agent)
    model = WhisperModel(model_size, device=config.DEVICE or "auto")
    folder = clean_dir / agent
    kept = []
    for path in sorted(folder.glob("*.wav")):
        segments, info = model.transcribe(str(path), beam_size=5, language="en")
        segments = list(segments)
        if not segments:
            continue
        text = normalise_text(" ".join(segment.text for segment in segments))
        average = sum(segment.avg_logprob for segment in segments) / len(segments)
        if accept(info.language, info.language_probability, average, text):
            kept.append((path.stem, text))
    with open(folder / "transcripts.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=SEPARATOR)
        writer.writerows(kept)
    return len(kept)


def matches_audio(source_text: str, heard_text: str, max_wer: float = MAX_VERIFY_WER) -> bool:
    return word_error_rate(source_text, heard_text) <= max_wer


def verify_agent(agent: str, model_size: str = "small.en", clean_dir: Path = config.CLEAN_DIR) -> Counter:
    """Keeps only the lines in transcripts.csv whose text Whisper hears in the clip."""
    from faster_whisper import WhisperModel

    require_agent(agent)
    path = clean_dir / agent / "transcripts.csv"
    with open(path, encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter=SEPARATOR) if len(row) == 2]
    model = WhisperModel(model_size, device=config.DEVICE or "auto")
    outcome = Counter()
    kept = []
    for clip_id, text in rows:
        segments, _ = model.transcribe(str(path.parent / f"{clip_id}.wav"), beam_size=5, language="en")
        heard = " ".join(segment.text for segment in segments)
        if matches_audio(text, heard):
            kept.append((clip_id, text))
        else:
            outcome["mismatch"] += 1
    with open(path, "w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter=SEPARATOR).writerows(kept)
    outcome["kept"] = len(kept)
    return outcome
