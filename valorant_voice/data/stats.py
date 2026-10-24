"""Per-agent counts of what every data step kept and dropped, and the audit table built from them."""

import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from .. import config
from .manifest import read_transcripts
from .split import assign_split
from .text import text_key

STATS_FILE = "stats.json"
DROP_COLUMNS = ("unreadable", "too_short", "too_long", "no_text", "duplicate", "mismatch")


def record(agent_dir: Path, stage: str, counts) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / STATS_FILE
    stats = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    stats[stage] = dict(counts)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")


def audit_agent(agent_dir: Path) -> dict:
    """One row: lines fetched, drops by reason, lines kept, minutes of audio and the split sizes."""
    stats = json.loads((agent_dir / STATS_FILE).read_text(encoding="utf-8")) if (agent_dir / STATS_FILE).exists() else {}
    kept = read_transcripts(agent_dir)
    seconds = sum(sf.info(agent_dir / f"{clip_id}.wav").duration for clip_id, _ in kept)
    splits = Counter(assign_split(text_key(text)) for _, text in kept)
    drops = Counter()
    for stage in stats.values():
        drops.update({reason: n for reason, n in stage.items() if reason in DROP_COLUMNS})
    return {
        "agent": agent_dir.name,
        "fetched": stats.get("fetch", {}).get("fetched", 0),
        "kept": len(kept),
        "minutes": round(seconds / 60, 1),
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        **{reason: drops[reason] for reason in DROP_COLUMNS},
    }


def audit_table(clean_dir: Path = config.CLEAN_DIR) -> str:
    rows = [audit_agent(p) for p in sorted(clean_dir.iterdir()) if p.is_dir()]
    columns = ["agent", "fetched", "kept", "minutes", "train", "val", "test", *DROP_COLUMNS]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows]
    total = {c: sum(row[c] for row in rows) for c in columns[1:]}
    lines.append("| **total** | " + " | ".join(str(round(total[c], 1)) for c in columns[1:]) + " |")
    return "\n".join(lines)
