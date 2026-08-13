"""LLM adapters. The pipeline only needs a callable (prompt: str) -> str,
so there is no SDK lock-in. These adapters are conveniences.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

LLM = Callable[[str], str]


def anthropic_llm(model: str = "claude-opus-5", max_tokens: int = 16000,
                  api_key: str | None = None) -> LLM:
    """Anthropic Messages API via stdlib urllib.

    Raw HTTP rather than the `anthropic` SDK because this package is
    zero-dependency by design; if you already depend on the SDK, pass your own
    `(str) -> str` callable instead — the pipeline does not care which you use.

    max_tokens defaults to 16000, not a small number: synthesis returns a JSON
    claim set whose length scales with the evidence table, and a truncated
    response is unparseable JSON, which the pipeline correctly treats as a
    failed round and fails closed on. Budget for the output rather than
    discovering the cap as an empty answer.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def call(prompt: str) -> str:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode(),
            headers={"content-type": "application/json",
                     "x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return "".join(b.get("text", "") for b in data.get("content", []))
    return call


def ollama_llm(model: str = "llama3.1", host: str = "http://localhost:11434",
               format_json: bool = True) -> LLM:
    """Local models via Ollama — the DIEX-style fully local path."""
    def call(prompt: str) -> str:
        body = {"model": model, "prompt": prompt, "stream": False}
        if format_json:
            body["format"] = "json"
        req = urllib.request.Request(f"{host}/api/generate",
                                     data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read()).get("response", "")
    return call


class MockLLM:
    """Scripted responses for tests and demos."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("MockLLM exhausted")
        return self.responses.pop(0)
