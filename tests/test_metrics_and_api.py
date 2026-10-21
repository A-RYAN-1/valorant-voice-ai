import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from valorant_voice import agents, config
from valorant_voice.data.transcribe import accept, normalise_text
from valorant_voice.evaluation import metrics
from valorant_voice.serve.api import RateLimiter, create_app


def test_word_error_rate():
    assert metrics.word_error_rate("rotate to B site", "rotate to B site") == 0
    assert metrics.word_error_rate("rotate to B site", "rotate to A site") == 0.25
    assert metrics.word_error_rate("one two", "") == 1.0
    assert metrics.word_error_rate("Hold the smoke.", "hold the SMOKE") == 0


def test_cosine_similarity():
    assert metrics.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1)
    assert metrics.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0)


def test_transcript_filters():
    assert normalise_text("  hold   the\nsmoke ") == "hold the smoke"
    assert accept("en", 0.95, -0.3, "hold the smoke")
    assert not accept("es", 0.99, -0.2, "mantén el humo")
    assert not accept("en", 0.5, -0.2, "hold the smoke")
    assert not accept("en", 0.95, -2.0, "hold the smoke")


def test_agent_slugs():
    assert agents.AGENTS["kayo"] == "KAY/O"
    assert agents.require_agent("sage") == "sage"
    with pytest.raises(ValueError):
        agents.require_agent("not-an-agent")


class FakeEngine:
    def available_agents(self):
        return ["sage"]

    def synthesize(self, agent, text):
        return np.zeros(2205, dtype="float32"), 22050


@pytest.fixture
def client():
    return TestClient(create_app(FakeEngine()))


def test_agents_lists_all_and_marks_ready(client):
    body = client.get("/v1/agents").json()
    assert len(body) == len(agents.AGENTS)
    assert {entry["id"]: entry["ready"] for entry in body}["sage"] is True
    assert {entry["id"]: entry["ready"] for entry in body}["jett"] is False


def test_speak_returns_labelled_wav(client):
    response = client.post("/v1/speak", json={"agent": "sage", "text": "Healing up."})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-generated-audio"] == "ai"
    samples, rate = sf.read(io.BytesIO(response.content))
    assert rate == 22050 and len(samples) == 2205


@pytest.mark.parametrize(
    "payload, status",
    [
        ({"agent": "sage", "text": ""}, 422),
        ({"agent": "sage", "text": "x" * (config.MAX_CHARS + 1)}, 422),
        ({"agent": "nobody", "text": "hello"}, 404),
        ({"agent": "jett", "text": "hello"}, 409),
        ({"text": "hello"}, 422),
    ],
)
def test_speak_rejects_bad_requests(client, payload, status):
    assert client.post("/v1/speak", json=payload).status_code == status


def test_rate_limiter_blocks_after_the_limit():
    limiter = RateLimiter(limit=2)
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")


def test_index_page_is_served(client):
    assert "Valorant Voice AI" in client.get("/").text
