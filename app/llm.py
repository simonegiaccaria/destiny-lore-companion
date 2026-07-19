"""One OpenAI-compatible client for both backends (Ollama local / API).

Swapping backend = editing LLM_BASE_URL / LLM_MODEL / LLM_API_KEY in .env.
Keep it this way: the local-vs-API decision is empirical (reference
questions in eval/), not architectural.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
PROMPT = (ROOT / "app" / "prompts" / "ghost.md").read_text()


def _client() -> tuple[OpenAI, str]:
    return (
        OpenAI(base_url=os.environ.get("LLM_BASE_URL",
                                       "http://localhost:11434/v1"),
               api_key=os.environ.get("LLM_API_KEY", "ollama")),
        os.environ.get("LLM_MODEL", "llama3.1:8b-instruct-q5_K_M"),
    )


def build_context(hits) -> str:
    """Chunks -> context block. Metadata travels with every fragment so the
    model can date sources (retcon rule) and see spoiler flags."""
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h.meta
        header = (f"[{i}] source={m.get('source_type')} "
                  f"| title={m.get('title')} "
                  f"| book={m.get('book') or '-'} "
                  f"| release={m.get('release')} ({m.get('release_date') or '?'})"
                  f"| vaulted={m.get('vaulted')}"
                  + (f" | SPOILS={','.join(h.spoils)}" if h.spoils else ""))
        blocks.append(header + "\n" + h.text)
    return "\n\n---\n\n".join(blocks)


def answer_stream(question: str, hits, history: list[dict] | None = None
                  ) -> Iterator[str]:
    client, model = _client()
    messages = [{"role": "system", "content": PROMPT}]
    messages += (history or [])[-8:]          # short rolling memory
    messages.append({
        "role": "user",
        "content": (f"ARCHIVE FRAGMENTS:\n{build_context(hits)}\n\n"
                    f"GUARDIAN'S QUESTION:\n{question}"),
    })
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, temperature=0.6)
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta
