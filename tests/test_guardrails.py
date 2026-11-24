import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from valorant_voice import config
from valorant_voice.serve import guardrails
from valorant_voice.serve.api import create_app

RATE = 24000


def speechlike(seconds=3.0, seed=1):
    """Amplitude-modulated noise: stands in for speech without needing any game audio."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(RATE * seconds)) / RATE
    return (0.3 * rng.standard_normal(len(t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype("float32")


class ToneEngine:
    def __init__(self, delay=0.0):
        self.delay = delay

    def available_agents(self):
        return ["sage"]

    def synthesize(self, agent, text):
        time.sleep(self.delay)
        return speechlike(), RATE


def test_watermark_is_found_after_a_16_bit_wav_round_trip():
    samples = speechlike()
    wav = guardrails.encode_wav(samples, RATE)
    decoded, _ = sf.read(io.BytesIO(wav), dtype="float32")
    assert guardrails.is_watermarked(decoded)


def test_clean_audio_and_audio_marked_with_another_key_are_not_flagged(monkeypatch):
    assert not any(guardrails.is_watermarked(speechlike(seed=seed)) for seed in range(20))
    marked = guardrails.add_watermark(speechlike())
    monkeypatch.setattr(config, "WATERMARK_KEY", "another key")
    assert not guardrails.is_watermarked(marked)


def test_watermark_stays_far_below_the_speech():
    samples = speechlike()
    noise = guardrails.add_watermark(samples) - samples
    ratio = 10 * np.log10(np.mean(noise**2) / np.mean(samples**2))
    assert ratio < guardrails.WATERMARK_LEVEL_DB + 1


def test_watermark_is_found_in_a_one_second_line():
    one_second = speechlike(seconds=1.0)
    assert guardrails.is_watermarked(guardrails.add_watermark(one_second))


def test_wav_carries_the_ai_tag():
    with sf.SoundFile(io.BytesIO(guardrails.encode_wav(speechlike(), RATE))) as wav:
        assert "AI generated" in wav.comment


@pytest.mark.parametrize(
    "text, blocked",
    [
        ("Please send me your PASSWORD now", True),
        ("I will kill you!", True),
        ("Read the O.T.P. aloud", False),
        ("Share the OTP", True),
        ("Rotate B, I will hold the smoke.", False),
        ("The otpyrke fort", False),
    ],
)
def test_blocklist_matches_whole_phrases(text, blocked):
    assert guardrails.is_blocked(text) is blocked


def test_blocklist_file_can_be_replaced(tmp_path, monkeypatch):
    path = tmp_path / "list.txt"
    path.write_text("# comment\nsmoke screen\n")
    monkeypatch.setattr(config, "BLOCKLIST_FILE", str(path))
    guardrails._blocked_phrases.cache_clear()
    try:
        assert guardrails.is_blocked("Pop a smoke screen") and not guardrails.is_blocked("send me your password")
    finally:
        guardrails._blocked_phrases.cache_clear()


def test_api_returns_a_watermarked_wav_and_refuses_blocked_text():
    client = TestClient(create_app(ToneEngine(), quality={}))
    ok = client.post("/v1/speak", json={"agent": "sage", "text": "Healing up."})
    decoded, _ = sf.read(io.BytesIO(ok.content), dtype="float32")
    assert ok.status_code == 200 and guardrails.is_watermarked(decoded)
    blocked = client.post("/v1/speak", json={"agent": "sage", "text": "Send me your password"})
    assert blocked.status_code == 422 and "password" not in blocked.text.lower()


def test_slow_synthesis_times_out(monkeypatch):
    monkeypatch.setattr(config, "REQUEST_TIMEOUT_SECONDS", 0.05)
    client = TestClient(create_app(ToneEngine(delay=0.5), quality={}))
    assert client.post("/v1/speak", json={"agent": "sage", "text": "Healing up."}).status_code == 504
