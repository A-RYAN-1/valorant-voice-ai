"""Picks the reference clips that condition the model on one agent's voice."""

import shutil
from pathlib import Path

import soundfile as sf

from .. import config
from ..data.manifest import read_transcripts
from ..data.split import assign_split

REFERENCES_PER_AGENT = 3
MIN_REFERENCE_SECONDS = 6.0
MAX_REFERENCE_SECONDS = 12.0


def choose_references(clips: list[tuple[str, float]], count: int = REFERENCES_PER_AGENT) -> list[str]:
    """From (clip_id, seconds) pairs, takes the `count` clips closest to 9 s inside the 6 to 12 s window.

    Only train-split clips are passed in, so a test line is never the voice sample for itself.
    """
    usable = [(abs(seconds - 9.0), clip_id) for clip_id, seconds in clips
              if MIN_REFERENCE_SECONDS <= seconds <= MAX_REFERENCE_SECONDS]
    return [clip_id for _, clip_id in sorted(usable)[:count]]


def build_references(agent: str, clean_dir: Path = config.CLEAN_DIR, out_dir: Path = config.REFERENCE_DIR) -> list[str]:
    """Copies the chosen clips to reference/<agent>/. Returns their ids (empty when none qualify)."""
    folder = clean_dir / agent
    candidates = []
    for clip_id, _ in read_transcripts(folder):
        if assign_split(f"{agent}/{clip_id}") != "train":
            continue
        info = sf.info(folder / f"{clip_id}.wav")
        candidates.append((clip_id, info.duration))
    chosen = choose_references(candidates)
    target = out_dir / agent
    if target.exists():
        shutil.rmtree(target)
    if chosen:
        target.mkdir(parents=True)
        for clip_id in chosen:
            shutil.copy(folder / f"{clip_id}.wav", target / f"{clip_id}.wav")
    return chosen


def reference_paths(agent: str, out_dir: Path = config.REFERENCE_DIR) -> list[Path]:
    return sorted((out_dir / agent).glob("*.wav"))
