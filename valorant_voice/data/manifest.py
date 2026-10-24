"""Step 4: one train, val and test manifest across all agents, in the format the XTTS trainer reads.

A line goes to a split by its text alone, so the same words never train one agent and test another."""

import csv
from collections import Counter
from pathlib import Path

from .. import config
from .split import assign_split
from .text import text_key
from .transcribe import SEPARATOR

HEADER = ["audio_file", "text", "speaker_name"]


def read_transcripts(agent_dir: Path):
    path = agent_dir / "transcripts.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return [(row[0], row[1]) for row in csv.reader(handle, delimiter=SEPARATOR) if len(row) == 2]


def build_manifests(clean_dir: Path = config.CLEAN_DIR, out_dir: Path = config.MANIFEST_DIR) -> Counter:
    """Writes train.csv, val.csv and test.csv. Returns the number of lines per split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": [], "test": []}
    for agent_dir in sorted(p for p in clean_dir.iterdir() if p.is_dir()):
        for clip_id, text in read_transcripts(agent_dir):
            wav = agent_dir / f"{clip_id}.wav"
            rows[assign_split(text_key(text))].append(
                [wav.relative_to(clean_dir).as_posix(), text, agent_dir.name]
            )
    counts = Counter()
    for split, split_rows in rows.items():
        with open(out_dir / f"{split}.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=SEPARATOR)
            writer.writerow(HEADER)
            writer.writerows(split_rows)
        counts[split] = len(split_rows)
    return counts
