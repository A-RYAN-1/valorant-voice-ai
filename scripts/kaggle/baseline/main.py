"""Kaggle kernel: build the dataset, then measure zero-shot XTTS-v2 on the held-out test lines.

Audio and transcripts stay in /kaggle/tmp and are discarded with the session; only the audit table,
the result files and the step log are written to /kaggle/working (the kernel output).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/A-RYAN-1/valorant-voice-ai.git"
TMP = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")
CODE = TMP / "repo"
DATA = TMP / "data"
DEPENDENCIES = ["soundfile", "scipy", "faster-whisper", "huggingface_hub", "pyarrow", "speechbrain", "coqui-tts[codec]>=0.27"]
LINES_PER_AGENT = "5"

log = []


def run(name, command, cwd=None, env=None):
    result = subprocess.run(command, check=False, cwd=cwd, env=env, capture_output=True, text=True)
    tail = (result.stdout + result.stderr)[-3000:]
    log.append(f"## {name} (exit {result.returncode})\n{tail}\n")
    print(f"{name}: exit {result.returncode}", flush=True)
    return result


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    run("clone", ["git", "clone", "--depth", "1", REPO, str(CODE)])
    run("install", [sys.executable, "-m", "pip", "install", "-q", *DEPENDENCIES])
    env = {**os.environ, "PYTHONPATH": str(CODE), "VV_DATA_DIR": str(DATA), "VV_DEVICE": "cuda", "COQUI_TOS_AGREED": "1"}
    steps = [("fetch",), ("ingest",), ("prepare",), ("verify",), ("manifests",), ("audit",), ("references",),
             ("evaluate", "--lines", LINES_PER_AGENT)]
    for step, *arguments in steps:
        result = run(step, [sys.executable, "-m", "valorant_voice", step, *arguments], cwd=CODE, env=env)
        if result.returncode != 0:
            break
        if step == "audit":
            (OUT / "audit.md").write_text(result.stdout, encoding="utf-8")
    stats = OUT / "stats"
    stats.mkdir(exist_ok=True)
    for path in (DATA / "clean").glob("*/stats.json"):
        shutil.copy(path, stats / f"{path.parent.name}.json")
    results = OUT / "results"
    results.mkdir(exist_ok=True)
    for path in (CODE / "results").glob("*.json"):
        shutil.copy(path, results / path.name)
    (OUT / "log.md").write_text("\n".join(log), encoding="utf-8")
    manifests = DATA / "manifests"
    if manifests.exists():
        counts = {p.stem: len(p.read_text(encoding="utf-8").splitlines()) - 1 for p in manifests.glob("*.csv")}
        (OUT / "manifest_counts.json").write_text(json.dumps(counts), encoding="utf-8")


main()
