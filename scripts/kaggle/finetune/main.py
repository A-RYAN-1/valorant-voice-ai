"""Kaggle kernel: build the dataset, fine-tune XTTS-v2 on the pooled train lines, then measure it on the held-out test lines.

The trained weights stay in /kaggle/tmp and are discarded with the session: nothing but metrics leaves.

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
CHECKPOINTS = TMP / "checkpoints"
TRAIN_ARGUMENTS = ["--epochs", "6", "--batch-size", "3", "--grad-accum", "5", "--output", str(CHECKPOINTS)]

log = []


def run(name, command, cwd=None, env=None, keep=3000):
    result = subprocess.run(command, check=False, cwd=cwd, env=env, capture_output=True, text=True)
    tail = (result.stdout + result.stderr)[-keep:]
    log.append(f"## {name} (exit {result.returncode})\n{tail}\n")
    print(f"{name}: exit {result.returncode}", flush=True)
    return result


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    run("clone", ["git", "clone", "--depth", "1", REPO, str(CODE)])
    run("install", [sys.executable, "-m", "pip", "install", "-q", *DEPENDENCIES])
    env = {**os.environ, "PYTHONPATH": str(CODE), "VV_DATA_DIR": str(DATA), "VV_DEVICE": "cuda", "COQUI_TOS_AGREED": "1"}
    steps = [["fetch"], ["ingest"], ["prepare"], ["verify"], ["manifests"], ["audit"], ["references"]]
    for step, *arguments in steps:
        result = run(step, [sys.executable, "-m", "valorant_voice", step, *arguments], cwd=CODE, env=env)
        if result.returncode != 0:
            break
        if step == "audit":
            (OUT / "audit.md").write_text(result.stdout, encoding="utf-8")
    else:
        train = run("finetune", [sys.executable, "-m", "valorant_voice.train.finetune", *TRAIN_ARGUMENTS],
                    cwd=CODE, env=env, keep=20000)
        if train.returncode == 0:
            tuned = {**env, "VV_CHECKPOINT_DIR": str(CHECKPOINTS / "model")}
            run("evaluate", [sys.executable, "-m", "valorant_voice", "evaluate", "--lines", LINES_PER_AGENT,
                             "--label", "xtts-v2-pooled-finetune"], cwd=CODE, env=tuned)
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
