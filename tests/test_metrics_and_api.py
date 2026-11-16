import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from valorant_voice import agents, config
from valorant_voice.data.transcribe import accept, normalise_text
from valorant_voice.evaluation import metrics
from valorant_voice.evaluation.report import agent_table, load_quality
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


def _entry(similarity, lines=5):
    return {"lines": lines, "speaker_similarity": similarity, "word_error_rate": 0.1, "real_time_factor": 0.5}


def test_agents_carry_measured_quality_for_measured_agents():
    app = create_app(FakeEngine(), quality={"sage": _entry(0.4)})
    body = {entry["id"]: entry for entry in TestClient(app).get("/v1/agents").json()}
    assert body["sage"]["quality"]["speaker_similarity"] == 0.4
    assert body["sage"]["strategy"] == "zero-shot"
    assert body["jett"]["quality"] is None


def test_load_quality_flags_thin_test_sets_and_skips_unknown_agents(tmp_path):
    path = tmp_path / "result.json"
    agents_block = {"sage": _entry(0.4, lines=4), "jett": _entry(0.4, lines=6), "nobody": _entry(0.4, lines=9)}
    path.write_text(json.dumps({"agents": agents_block}))
    quality = load_quality(path)
    assert set(quality) == {"sage", "jett"}
    assert quality["sage"]["thin_test_set"] is True
    assert quality["jett"]["thin_test_set"] is False
    assert load_quality(tmp_path / "missing.json") == {}


def test_agent_table_lists_lowest_first_and_totals(tmp_path):
    path = tmp_path / "result.json"
    mean = {"speaker_similarity": 0.4, "word_error_rate": 0.1, "real_time_factor": 0.5}
    path.write_text(json.dumps({"agents": {"sage": _entry(0.5), "jett": _entry(0.3)}, "mean": mean}))
    rows = agent_table(path).splitlines()
    assert rows[2].startswith("| Jett") and rows[3].startswith("| Sage")
    assert rows[-1].startswith("| **all 2** | 10 |")
