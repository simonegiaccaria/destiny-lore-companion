"""One OpenAI-compatible client for both backends (Ollama local / API).

Swapping backend = editing LLM_BASE_URL / LLM_MODEL / LLM_API_KEY in .env.
Keep it this way: the local-vs-API decision is empirical (reference
questions in eval/), not architectural.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI

ROOT = Path(__file__).resolve().parents[1]
PROMPT = (ROOT / "app" / "prompts" / "ghost.md").read_text(encoding="utf-8")

# Low by default: this task rewards discipline (stay on the record, admit
# gaps) over invention. Raise it only if answers become dry AND grounding
# still holds on the negative-control questions in eval/.
DEFAULT_TEMPERATURE = 0.2

# How each source type is presented to the model. Plain words, not field
# names: the model mirrors whatever vocabulary it is shown.
SOURCE_LABELS = {
    "lore_entry": "Lore entry",
    "flavor_text": "Item inscription",
    "grimoire": "Grimoire card",
    "transcript": "Transcript",
}

# The boundary between supplied records and the model's own memory must be
# EXPLICIT in the input, even though the persona hides it in the output.
# A softer label ("what you recall") was tested and made the model invent
# whole documents: telling it to act as if it remembers made it remember
# things that were never supplied.
CONTEXT_HEADER = (
    "ARCHIVE RECORDS — the complete and only material you may draw on for "
    "this answer. Anything not written below is unknown to you, however "
    "familiar it seems. Never quote, name or describe a document that is "
    "not below."
)


def _client() -> tuple[OpenAI, str]:
    return (
        OpenAI(base_url=os.environ.get("LLM_BASE_URL",
                                       "http://127.0.0.1:11434/v1"),
               api_key=os.environ.get("LLM_API_KEY", "ollama")),
        os.environ.get("LLM_MODEL", "llama3.1:8b-instruct-q5_K_M"),
    )


def _known(value) -> bool:
    """True if a metadata value carries information worth showing."""
    return value not in (None, "", "unknown", "?", "-", 0)


def build_context(hits) -> str:
    """Chunks -> context block for the model.

    Only metadata the model can USE goes in, and only when it is known:
    - source kind + title + book: needed for the source hierarchy and for
      precise citations on request;
    - release + date: needed for the retcon rule — omitted when unknown,
      otherwise the model narrates "release unknown" as if it were lore;
    - spoils: needed for the pre-spoiler warning.
    Deliberately NOT included: `vaulted` (gating logic, not story) and
    fragment numbers (they invite "Fragment [4]" in the answer).
    """
    blocks = []
    for h in hits:
        m = h.meta
        label = SOURCE_LABELS.get(m.get("source_type"), "Record")
        parts = [f"{label}: {m.get('title') or 'Untitled'}"]
        if _known(m.get("book")):
            parts.append(f"from the book \"{m['book']}\"")
        if _known(m.get("release")):
            rel = str(m["release"]).replace("_", " ").title()
            date = m.get("release_date")
            parts.append(f"entered the record with {rel}"
                         + (f" ({date})" if _known(date) else ""))
        header = " — ".join(parts)
        if h.spoils:
            spoiled = ", ".join(s.replace("_", " ").title() for s in h.spoils)
            header += f"\n[Foreshadows: {spoiled}]"
        blocks.append(header + "\n" + h.text)
    return "\n\n---\n\n".join(blocks)


def answer_stream(question: str, hits, history: list[dict] | None = None
                  ) -> Iterator[str]:
    """Stream the answer. Errors are yielded as text, not raised: the HTTP
    response has already started with 200 OK by the time this generator
    runs, so an exception here reaches the client as a silently truncated
    stream — an empty chat box with no diagnosis. Yield instead."""
    client, model = _client()
    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    temperature = float(os.environ.get("LLM_TEMPERATURE", DEFAULT_TEMPERATURE))
    messages = [{"role": "system", "content": PROMPT}]
    messages += (history or [])[-8:]          # short rolling memory
    messages.append({
        "role": "user",
        "content": (f"{CONTEXT_HEADER}\n\n{build_context(hits)}\n\n"
                    f"=== END OF ARCHIVE RECORDS ===\n\n"
                    f"GUARDIAN'S QUESTION:\n{question}"),
    })

    try:
        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True,
            temperature=temperature)
    except APIConnectionError:
        yield (f"[No connection to the model backend at {base_url}.\n"
               f"If this is Ollama, start it with `ollama serve`.]")
        return
    except APIStatusError as e:
        yield (f"[Model backend rejected the request: {e.status_code}.\n"
               f"Check that model '{model}' exists (`ollama list`).]")
        return
    except Exception as e:                    # last resort: never fail silent
        yield f"[Model call failed: {type(e).__name__}: {e}]"
        return

    try:
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"\n\n[Stream interrupted: {type(e).__name__}: {e}]"