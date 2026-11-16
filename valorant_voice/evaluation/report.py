"""Turns an evaluation result file into per-agent quality data and a markdown table."""

import json
from pathlib import Path

from .. import config
from ..agents import AGENTS

THIN_TEST_SET = 5  # fewer held-out lines than this and the agent's numbers are mostly noise


def load_quality(path: Path | None = None) -> dict[str, dict]:
    """Per-agent measured numbers keyed by agent id. Empty when the result file is missing."""
    path = Path(path or config.QUALITY_FILE)
    if not path.is_file():
        return {}
    result = json.loads(path.read_text(encoding="utf-8"))
    return {
        slug: {**numbers, "thin_test_set": numbers["lines"] < THIN_TEST_SET}
        for slug, numbers in result["agents"].items()
        if slug in AGENTS
    }


def agent_table(path: Path | None = None) -> str:
    """Markdown table of every measured agent, lowest speaker similarity first."""
    path = Path(path or config.QUALITY_FILE)
    result = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(load_quality(path).items(), key=lambda item: item[1]["speaker_similarity"])
    lines = [
        "| agent | held-out lines | speaker similarity | word error rate | real-time factor |",
        "|---|---|---|---|---|",
    ]
    for slug, n in rows:
        flag = " (thin)" if n["thin_test_set"] else ""
        lines.append(
            f"| {AGENTS[slug]} | {n['lines']}{flag} | {n['speaker_similarity']:.3f} | "
            f"{n['word_error_rate']:.3f} | {n['real_time_factor']:.2f} |"
        )
    mean = result["mean"]
    lines.append(
        f"| **all {len(rows)}** | {sum(n['lines'] for _, n in rows)} | {mean['speaker_similarity']:.3f} | "
        f"{mean['word_error_rate']:.3f} | {mean['real_time_factor']:.2f} |"
    )
    return "\n".join(lines)
