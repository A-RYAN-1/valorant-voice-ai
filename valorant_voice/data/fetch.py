"""Step 0: download one agent's voice lines from the public Hugging Face dataset.

Runs on Kaggle (internet on); nothing is stored in the repository. The dataset has no licence on
its card and the audio is Riot Games', so see the README, Legal, before using the output.
"""

import csv
import re
from pathlib import Path

from .. import config
from ..agents import require_agent
from .text import clean_text
from .transcribe import SEPARATOR

DATASET = "NgThVinh/ValorantAgentVoiceLines"
NO_PUBLIC_DATA = frozenset({"miks", "veto", "waylay"})  # agents the dataset does not contain
SOURCE_TRANSCRIPTS = "source_transcripts.csv"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def write_rows(agent: str, rows, raw_dir: Path = config.RAW_DIR) -> int:
    """Writes each row's audio bytes to raw_dir/<agent>/<id>.<ext> and the cleaned text beside it."""
    target = raw_dir / require_agent(agent)
    target.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        audio = row["audio_file"]
        if not audio or not audio.get("bytes"):
            continue
        clip_id = safe_name(row["audio_name"])
        suffix = Path(audio.get("path") or "").suffix.lower() or ".wav"
        (target / f"{clip_id}{suffix}").write_bytes(audio["bytes"])
        lines.append((clip_id, clean_text(row["transcript"])))
    with open(target / SOURCE_TRANSCRIPTS, "w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter=SEPARATOR).writerows(lines)
    return len(lines)


def fetch_agent(agent: str, raw_dir: Path = config.RAW_DIR) -> int:
    """Returns how many lines were written. Raises ValueError for an agent without public data."""
    if require_agent(agent) in NO_PUBLIC_DATA:
        raise ValueError(f"The dataset has no lines for {agent}")
    import pyarrow.parquet as pq  # optional dependencies, needed only here
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, f"{agent}/train-00000-of-00001.parquet", repo_type="dataset")
    return write_rows(agent, pq.read_table(path).to_pylist(), raw_dir)
