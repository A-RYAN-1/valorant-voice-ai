"""The HTTP API: list agents, turn text into speech as an agent."""

import io
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .. import config
from ..agents import AGENTS

STATIC_DIR = Path(__file__).parent / "static"


class Engine(Protocol):
    def available_agents(self) -> list[str]: ...

    def synthesize(self, agent: str, text: str) -> tuple[np.ndarray, int]: ...


class SpeakRequest(BaseModel):
    agent: str = Field(description="Agent id from GET /v1/agents, for example 'sage'")
    text: str = Field(min_length=1, max_length=config.MAX_CHARS)


class RateLimiter:
    """At most `limit` requests per client in any 60 second window."""

    def __init__(self, limit: int):
        self.limit = limit
        self.hits = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        window = self.hits[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.limit:
            return False
        window.append(now)
        return True


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(
        title="Valorant Voice AI",
        description="Unofficial, non-commercial fan project. Not affiliated with Riot Games.",
        version="0.1.0",
    )
    limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE)

    @app.get("/health")
    def health():
        return {"status": "ok", "agents_ready": len(engine.available_agents())}

    @app.get("/v1/agents")
    def agents():
        ready = set(engine.available_agents())
        return [{"id": slug, "name": name, "ready": slug in ready} for slug, name in AGENTS.items()]

    @app.post("/v1/speak", response_class=Response, responses={200: {"content": {"audio/wav": {}}}})
    def speak(body: SpeakRequest, request: Request):
        if body.agent not in AGENTS:
            raise HTTPException(status_code=404, detail=f"Unknown agent '{body.agent}'")
        if body.agent not in engine.available_agents():
            raise HTTPException(status_code=409, detail=f"No voice data is loaded for {AGENTS[body.agent]}")
        if not limiter.allow(request.client.host if request.client else "unknown"):
            raise HTTPException(status_code=429, detail="Too many requests, try again in a minute")
        samples, rate = engine.synthesize(body.agent, body.text)
        buffer = io.BytesIO()
        sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
        return Response(buffer.getvalue(), media_type="audio/wav", headers={"X-Generated-Audio": "ai"})

    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(STATIC_DIR / "index.html")

    return app
