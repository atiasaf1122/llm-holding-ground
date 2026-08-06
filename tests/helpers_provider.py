"""Fixtures shared by the provider tests.

Every one of them runs against ``httpx.MockTransport``: no daemon, no network and
no GPU, because the card on the development machine is busy and a suite that
needs it is a suite that does not get run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from council.agents.ollama import OllamaProvider
from council.config import Settings
from council.domain.signal import Signal

Handler = Callable[[httpx.Request], httpx.Response]

SIGNAL_SCHEMA: dict[str, Any] = Signal.model_json_schema()
SIGNAL_OUTPUT: dict[str, Any] = {"exposure": 0.4, "confidence": 0.7, "rationale": "trend intact"}


def make_settings(**overrides: Any) -> Settings:
    """Settings with every value these tests assert on pinned explicitly."""
    base: dict[str, Any] = {
        "ollama_base_url": "http://ollama.test",
        "temperature": 0.0,
        "context_tokens": 4096,
        "keep_alive": "30m",
        "seed": 20260101,
        "max_output_tokens": 320,
        "max_retries": 2,
        "concurrency": 4,
        "request_timeout_seconds": 5.0,
    }
    return Settings(**{**base, **overrides})


def make_provider(handler: Handler, *, model: str = "qwen3:8b", **overrides: Any) -> OllamaProvider:
    return OllamaProvider(
        model=model,
        settings=make_settings(**overrides),
        transport=httpx.MockTransport(handler),
    )


def chat_envelope(
    content: str, *, done_reason: str | None = "stop", prompt_eval_count: int = 512
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "eval_count": 42,
        "prompt_eval_count": prompt_eval_count,
    }
    if done_reason is not None:
        envelope["done_reason"] = done_reason
    return envelope


def chat_handler(
    content: str, *, done_reason: str | None = "stop", prompt_eval_count: int = 512
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=chat_envelope(
                content, done_reason=done_reason, prompt_eval_count=prompt_eval_count
            ),
        )

    return handler


def daemon_handler(
    *, version: str = "0.31.2", models: tuple[str, ...] = ("qwen3:8b", "gemma4:latest")
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": version})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": name} for name in models]})
        raise AssertionError(f"unexpected request to {request.url.path}")

    return handler
