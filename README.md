<h1 align="center">Valorant Voice AI</h1>

<p align="center">
  Type a line, pick a Valorant agent, hear it in that agent's voice.<br>
  A speech pipeline: clean the lines you extracted from your own game, fine-tune XTTS-v2 on all agents, measure it against the zero-shot baseline, and serve it through a small API.
</p>

<p align="center">
  <a href="#how-it-works">How it works</a>
  &middot;
  <a href="#run-it-locally">Run locally</a>
  &middot;
  <a href="#legal-and-ethics">Legal and ethics</a>
  &middot;
  <a href="#status-and-roadmap">Roadmap</a>
</p>

<p align="center"><sub>Last updated: 2026-10-21 &middot; Runs locally only; not deployed yet</sub></p>

---

## Contents

- [About](#about)
- [Try it live](#try-it-live)
- [Features](#features)
- [Why it exists](#why-it-exists)
- [How it works](#how-it-works)
- [Legal and ethics](#legal-and-ethics)
- [Engineering quality](#engineering-quality)
- [Run it locally](#run-it-locally)
- [Environment variables](#environment-variables)
- [Deployment](#deployment)
- [Status and roadmap](#status-and-roadmap)
- [Credits](#credits)

## About

Most voice-cloning demos are a pretrained model and a reference clip, with no way to tell whether the
result is any good. This project treats the voices as a machine-learning problem with numbers: a
dataset step that decides which lines are trained on, a fine-tune on the pooled voices of every agent,
and an evaluation on held-out lines that compares the fine-tuned model with the untouched baseline on
speaker similarity, word error rate and speed.

The product on top is deliberately small: `POST /v1/speak` with an agent and a text returns a WAV file,
and a one-page web UI does the same from a browser.

## Try it live

Not deployed yet, on purpose: the voices are Riot Games' property, so there is no public demo and no
weights are published (see [Legal and ethics](#legal-and-ethics)). To hear it, run it on your own
machine with audio you extracted from your own copy of the game.

## Features

- **Agent list is fixed**: the 29 playable agents. There is no upload of arbitrary reference audio, so
  the service cannot be used to clone a real person.
- **Dataset pipeline**: downloads the public voice-line dataset of 26 agents on Kaggle, resamples to
  22.05 kHz mono, trims silence, normalises loudness, drops clips outside 1.5 to 12 seconds, drops empty
  and repeated lines, and keeps only lines whose text Whisper hears in the clip. A per-agent audit table
  shows what was kept and dropped, and why.
- **Leak-free split**: a line's train, validation or test assignment is a hash of its text, so test lines
  are never trained on, a repeated line such as "Okay." never sits in both train and test, and nothing
  moves when more data is added.
- **API**: typed JSON request, text length cap, per-client rate limit, every response labelled
  `X-Generated-Audio: ai`.
- **Evaluation harness**: speaker similarity (ECAPA-TDNN cosine), word error rate (Whisper hears the
  output), and real-time factor, written with the checkpoint and manifest hash.

## Why it exists

Speech ML is mostly data work and honest measurement. The goal is one project that shows the whole
loop end to end on a problem people care about, with every claim backed by a result file that names the
model and the data it came from.

## How it works

```
Hugging Face dataset   NgThVinh/ValorantAgentVoiceLines  (fetched on Kaggle, never committed)
  -> fetch             wav and source text per agent      python -m valorant_voice fetch
  -> ingest            clean 22.05 kHz mono clips         python -m valorant_voice ingest
  -> prepare           pair with text, drop repeats       python -m valorant_voice prepare
  -> verify            keep lines Whisper hears           python -m valorant_voice verify
  -> manifests         train / val / test, text-hash split  python -m valorant_voice manifests
  -> audit             kept, minutes and drops per agent  python -m valorant_voice audit
  -> references        3 voice samples per agent          python -m valorant_voice references
  -> fine-tune         XTTS-v2 on all agents (GPU)        valorant_voice/train/finetune.py
  -> evaluate          vs the zero-shot baseline          python -m valorant_voice evaluate
  -> serve             FastAPI and one web page           python -m valorant_voice serve
```

**One model for every agent.** XTTS-v2 conditions on a few seconds of reference audio. The fine-tune runs
once on the pooled lines of all agents, so the model learns the studio character of the game audio, and
each agent is selected at inference by that agent's reference clips. The baseline is the same model
without the fine-tune.

**Stack:** Python 3.12, PyTorch, coqui-tts (XTTS-v2), faster-whisper, speechbrain (ECAPA-TDNN), FastAPI.

## Results so far

**Dataset (measured 2026-10-24, Kaggle):** 12,164 lines fetched for 26 agents; 2,611 lines (148 minutes) kept after the length filter (1.5 to 12 s), repeat removal and the Whisper text check; 2,346 train, 125 val, 140 test. Per-agent table and the length histogram: [docs/dataset-audit.md](docs/dataset-audit.md). 
**Zero-shot XTTS-v2 baseline (measured 2026-10-24, Kaggle T4, `results/2026-10-24-xtts-v2-zero-shot.json`):** over 26 agents and 2 to 5 held-out test lines each (117 lines): speaker similarity 0.401 (ECAPA-TDNN cosine, higher is better), word error rate 0.072 (lower is better), real-time factor 0.67 mean and 0.46 typical (the mean includes the model load on the first line, 5.9 for astra; the harness now times after a warm-up). This is the row every fine-tune has to beat.

## Legal and ethics

The voice lines are Riot Games' copyright and the performances belong to the voice actors. This is an
unofficial, non-commercial fan and learning project, not affiliated with Riot Games. The rules it follows:

1. **No game audio and no trained weights are ever committed or hosted.** `data/`, `models/` and
   `outputs/` are git-ignored. The voice lines come from the public Hugging Face dataset
   `NgThVinh/ValorantAgentVoiceLines`, downloaded at run time on Kaggle. Its card declares no licence and
   the audio is Riot's, so a public copy is not permission: nothing here redistributes it.
2. **No public demo.**
3. **Fixed list of fictional characters.** Nothing accepts an uploaded voice.
4. **Output is labelled** with the `X-Generated-Audio: ai` header and a footer on the page.
5. **Model licence:** XTTS-v2 is released under the Coqui Public Model License, which is non-commercial.
   Using it for this project is within that licence; selling anything built on it is not.

## Engineering quality

- `pytest` covers the audio helpers, ingest rules, the split (no clip in two splits, stable when data
  grows), the manifests, reference selection, the metrics and the API (limits, unknown agent, rate limit).
  The tests use generated tones, so they need no game audio and no GPU.
- Every setting comes from environment variables read in one module, `valorant_voice/config.py`.
- Results are JSON files in `results/` that record the checkpoint and the SHA-1 of the test manifest.
- Not yet: CI, the fine-tune and its results table (see the roadmap).

## Run it locally

Prerequisites: [mise](https://mise.jdx.dev) (it installs Python 3.12 from `mise.toml`), an NVIDIA GPU
for reasonable speed (CPU works and is slow), and the FFmpeg shared libraries on PATH, which PyTorch
2.9 and later need to read audio (Windows: `winget install Gyan.FFmpeg.Shared`; Debian or Ubuntu:
`apt install ffmpeg`).

```bash
mise install
mise exec -- python -m venv .venv
. .venv/Scripts/activate          # Windows Git Bash; on Linux/macOS: . .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[tts,data,evaluate,dev]"
pytest                             # no game audio needed
```

**The data.** The dataset covers 26 of the 29 agents (Miks, Veto and Waylay are not in it). The steps
below download about 2 GB, so run them on a Kaggle notebook or a machine you do not mind filling; a
single agent is `--agent sage`.

```bash
python -m valorant_voice fetch
python -m valorant_voice ingest
python -m valorant_voice prepare
python -m valorant_voice verify
python -m valorant_voice manifests
python -m valorant_voice audit
python -m valorant_voice references
python -m valorant_voice speak sage "Healing up, stay close."   # writes outputs/sage.wav
python -m valorant_voice serve                                  # http://127.0.0.1:8000
```

The first `speak` downloads the XTTS-v2 base model (about 1.8 GB) and, by running it, you accept its
non-commercial licence.

## Environment variables

All optional; the defaults work from a fresh clone. The full list with comments is `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `VV_DATA_DIR`, `VV_MODEL_DIR`, `VV_OUTPUT_DIR` | next to the code | local-only folders for audio, weights, output |
| `VV_CHECKPOINT_DIR` | empty | fine-tuned checkpoint folder; empty uses the base model |
| `VV_DEVICE` | empty | `cuda` or `cpu`; empty picks cuda when available |
| `VV_MAX_CHARS` | 300 | longest text the API accepts |
| `VV_RATE_LIMIT_PER_MINUTE` | 20 | requests per client per minute |
| `VV_SPLIT_SEED` | 13 | seed of the train, val and test split |

## Deployment

Not deployed yet, by design (see [Legal and ethics](#legal-and-ethics)). There is no host, no Vercel,
Render or Hugging Face Space for this project. Training of the fine-tune runs on a free Kaggle or Colab
GPU session when the 6 GB laptop GPU is not enough; the checkpoint stays on the owner's machine.

## Status and roadmap

Rewritten 2026-10-21 from a 40-line wrapper around one reference clip. What exists: the data pipeline,
the engine, the API and page, the evaluation harness and the tests. What does not exist yet: any result on agent voices.

Measured 2026-10-21 on the RTX 4050 laptop GPU with the base XTTS-v2 and a stand-in reference clip (a smoke
test of the engine and the API, not a quality result): after the first call, which loads the model and
computes the voice conditioning in about 28 s, a 2.4 s line is generated in 0.9 s (real-time factor 0.38),
using 1.8 GB of GPU memory.

- [x] Runnable package on Python 3.12 with tests
- [ ] Baseline numbers: zero-shot XTTS-v2 on the held-out lines of every agent
- [ ] Fine-tune on all agents and the results table, baseline against fine-tuned
- [ ] Inaudible watermark on generated audio
- [ ] CI (lint and the unit tests), Dockerfile for the API

## Credits

Built by Aryan Deshmukh. Models: XTTS-v2 by Coqui (maintained by Idiap as `coqui-tts`), Whisper by
OpenAI through faster-whisper, ECAPA-TDNN by SpeechBrain. Valorant is a trademark of Riot Games.
