"""Kaggle kernel: run the whole dataset pipeline on the public voice-line dataset and keep only the numbers.

Audio and transcripts stay in /kaggle/tmp and are discarded with the session; only the audit table,
the per-agent stats and the step log are written to /kaggle/working (the kernel output).
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
DEPENDENCIES = ["soundfile", "scipy", "faster-whisper", "huggingface_hub", "pyarrow"]

log = []


def run(name, command, cwd=None, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    tail = (result.stdout + result.stderr)[-3000:]
    log.append(f"## {name} (exit {result.returncode})\n{tail}\n")
    print(f"{name}: exit {result.returncode}", flush=True)
    return result


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    run("clone", ["git", "clone", "--depth", "1", REPO, str(CODE)])
    run("install", [sys.executable, "-m", "pip", "install", "-q", *DEPENDENCIES])
    env = {**os.environ, "PYTHONPATH": str(CODE), "VV_DATA_DIR": str(DATA), "VV_DEVICE": "cuda"}
    for step in ("fetch", "ingest", "prepare", "verify", "manifests", "audit"):
        result = run(step, [sys.executable, "-m", "valorant_voice", step], cwd=CODE, env=env)
        if result.returncode != 0:
            break
        if step == "audit":
            (OUT / "audit.md").write_text(result.stdout, encoding="utf-8")
    stats = OUT / "stats"
    stats.mkdir(exist_ok=True)
    for path in (DATA / "clean").glob("*/stats.json"):
        shutil.copy(path, stats / f"{path.parent.name}.json")
    (OUT / "log.md").write_text("\n".join(log), encoding="utf-8")
    manifests = DATA / "manifests"
    if manifests.exists():
        counts = {p.stem: max(0, sum(1 for _ in open(p, encoding="utf-8")) - 1) for p in manifests.glob("*.csv")}
        (OUT / "manifest_counts.json").write_text(json.dumps(counts), encoding="utf-8")


main()
