"""Measures a model on the held-out test lines: speaker similarity, word error rate, speed.

Every number is computed on lines that were never trained on (data/manifests/test.csv) and is
written with the checkpoint and manifest hash it came from, so two runs can be compared honestly.
"""

import csv
import hashlib
import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean

import numpy as np
import soundfile as sf

from .. import config
from ..data.transcribe import SEPARATOR
from ..tts.engine import XttsEngine
from . import metrics

RESULTS_DIR = config.ROOT / "results"


def _test_rows(lines_per_agent: int):
    by_agent = defaultdict(list)
    with open(config.MANIFEST_DIR / "test.csv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=SEPARATOR):
            by_agent[row["speaker_name"]].append(row)
    return {agent: sorted(rows, key=lambda r: r["audio_file"])[:lines_per_agent] for agent, rows in by_agent.items()}


def _embedder():
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def embed(samples: np.ndarray, rate: int) -> np.ndarray:
        import torchaudio

        tensor = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
        tensor = torchaudio.functional.resample(tensor, rate, 16000)
        return model.encode_batch(tensor).squeeze().detach().cpu().numpy()

    return embed


def run_evaluation(lines_per_agent: int = 10, engine=None, label: str = "") -> Path:
    from faster_whisper import WhisperModel

    engine = engine or XttsEngine()
    embed = _embedder()
    whisper = WhisperModel("small.en", device=config.DEVICE or "auto")
    test_rows = _test_rows(lines_per_agent)
    first_agent, first_rows = next(iter(test_rows.items()))
    engine.synthesize(first_agent, first_rows[0]["text"])  # warm-up: loads the model, so it is not timed
    per_agent = {}
    for agent, rows in test_rows.items():
        similarity, wer, speed = [], [], []
        for row in rows:
            started = time.perf_counter()
            samples, rate = engine.synthesize(agent, row["text"])
            speed.append(metrics.real_time_factor(time.perf_counter() - started, len(samples) / rate))
            real, real_rate = sf.read(config.CLEAN_DIR / row["audio_file"], dtype="float32")
            similarity.append(metrics.cosine_similarity(embed(samples, rate), embed(real, real_rate)))
            heard, _ = whisper.transcribe(samples.astype("float32"), language="en")
            wer.append(metrics.word_error_rate(row["text"], " ".join(s.text for s in heard)))
        per_agent[agent] = {
            "lines": len(rows),
            "speaker_similarity": round(mean(similarity), 4),
            "word_error_rate": round(mean(wer), 4),
            "real_time_factor": round(mean(speed), 3),
        }

    result = {
        "date": date.today().isoformat(),
        "label": label or (Path(engine.checkpoint_dir).name if engine.checkpoint_dir else "xtts-v2-zero-shot"),
        "checkpoint": engine.checkpoint_dir or "base xtts_v2",
        "manifest_sha1": hashlib.sha1((config.MANIFEST_DIR / "test.csv").read_bytes()).hexdigest(),
        "agents": per_agent,
        "mean": {
            key: round(mean(a[key] for a in per_agent.values()), 4)
            for key in ("speaker_similarity", "word_error_rate", "real_time_factor")
        },
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{result['date']}-{result['label']}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out
