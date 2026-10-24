"""Step 2: pair each cleaned clip with its source transcript and drop empty and repeated lines."""

import csv
from collections import Counter
from pathlib import Path

from .. import config
from ..agents import require_agent
from .fetch import SOURCE_TRANSCRIPTS
from .text import text_key
from .transcribe import SEPARATOR


def prepare_agent(agent: str, raw_dir: Path = config.RAW_DIR, clean_dir: Path = config.CLEAN_DIR) -> Counter:
    """Writes clean/<agent>/transcripts.csv as `id|text`. Only clips that survived ingest are paired,
    and a text said by the agent more than once keeps its first clip, so no line is trained twice."""
    require_agent(agent)
    folder = clean_dir / agent
    with open(raw_dir / agent / SOURCE_TRANSCRIPTS, encoding="utf-8", newline="") as handle:
        source = [row for row in csv.reader(handle, delimiter=SEPARATOR) if len(row) == 2]
    outcome = Counter()
    seen = set()
    kept = []
    for clip_id, text in sorted(source):
        if not (folder / f"{clip_id}.wav").exists():
            continue  # dropped by ingest, already counted there
        key = text_key(text)
        if not key:
            outcome["no_text"] += 1
        elif key in seen:
            outcome["duplicate"] += 1
        else:
            seen.add(key)
            kept.append((clip_id, text))
    with open(folder / "transcripts.csv", "w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter=SEPARATOR).writerows(kept)
    outcome["kept"] = len(kept)
    return outcome
