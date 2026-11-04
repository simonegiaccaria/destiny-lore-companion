"""One OpenAI-compatible client for both backends (Ollama local / API).

Swapping backend = editing LLM_BASE_URL / LLM_MODEL / LLM_API_KEY in .env.
Keep it this way: the local-vs-API decision is empirical (reference
questions in eval/), not architectural.
"""

from __future__ import annotations

import os
import re
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

# End-of-message reminder. Measured on the T=0.2 x3 baseline: Italian
# questions answered in English on 3 of 5 questions (6/15 repeats OK),
# lists in 41/90 answers, frame breaks in 35/90. The rules are already in
# the system prompt; the reminder repeats the three most-violated ones
# right after the question, where the model reads last. Whether position
# matters for this model is the hypothesis being tested, not a known fact.
# ANSWER_REMINDER in .env: "on" (default) or "off" (for A/B runs).
# Same word lists as eval/run_eval.py, so the app and the check agree.
_IT_WORDS = re.compile(r"\b(il|lo|la|gli|le|di|che|è|non|una|per|sono|"
                       r"della|del|nel|con|come|anche|questo|questa)\b",
                       re.IGNORECASE)
_EN_WORDS = re.compile(r"\b(the|and|of|is|that|to|was|with|for|this|"
                       r"are|his|her|in|as|it)\b", re.IGNORECASE)
LANGUAGE_NAMES = {"it": "Italian", "en": "English"}
REMINDER = ("(Answer in {language}. Speak as the Ghost, from inside the "
            "world, in flowing prose: no numbered lists, no bold headings. "
            "Use only the archive records above; where they are silent, "
            "say so.)")
REMINDER_UNKNOWN_LANGUAGE = "the language of the Guardian's question"

# Follow-up rewriting. Retrieval only sees one string; a follow-up like
# "yes" or "and her?" carries no content of its own, so without this step
# retrieval returns near-random records (observed: "si" -> Laminak Li,
# Failsafe, Soteria). The rewrite is used for RETRIEVAL ONLY; the Ghost
# still answers the original message with the full history.
# QUERY_REWRITE in .env: "followups" (default), "always", "off".
#
# Known model limit: an Italian follow-up naming a NEW subject ("e la donna
# di cui parla in The Recluse?") was rewritten back to the previous topic
# twice, even with the explicit rule below. Hence the safety net in
# rewrite_query(): the original words always stay in the final query.
REWRITE_PROMPT = """You turn a Guardian's latest message into a standalone \
search query for an archive of Destiny lore.

Work in this order:
1. Translate the latest message into English, word by word in meaning.
2. Decide what the latest message asks about. THE LATEST MESSAGE DECIDES \
THE TOPIC. If it names or describes something (a person, a place, "the \
woman in X", "his brother"), the query is about THAT thing — never swap it \
for an earlier topic of the conversation.
3. Use the conversation only to fill in what the latest message leaves \
unsaid: who "he", "she", "it", "they" is; what a bare "yes" or "tell me \
more" agrees to.

Output rules:
- ONE line: the query itself, in English, naming the people, places and \
things involved.
- No label ("Search:", "Query:"), no quotes, no preamble, no explanation.
- Do not answer the question. Do not add facts that are not in the \
message or the conversation.
- If the latest message is already self-contained, output its English \
translation and nothing else."""

REWRITE_MAX_CHARS = 300
_LABEL_PREFIX = re.compile(r"^\s*(search(\s+query)?|query|rewritten(\s+query)?)"
                           r"\s*[:\-–—]\s*", re.IGNORECASE)

# Quote verification. After the answer is complete, every quoted passage
# of meaningful length is looked up in the retrieved records. A quote that
# is not there verbatim gets a visible warning under the answer: it cannot
# stop an invented quote, but it stops one from passing as evidence.
# Observed case: a fabricated "She was a Recluse, through and through…"
# presented as "the relevant passage".
#
# Pairing matters: quote marks are matched left to right in pairs, of ANY
# length, and only afterwards are short spans (titles like "II: Freedom")
# discarded. Requiring a minimum length inside the regex made it pair the
# closing mark of one title with the opening mark of the next, and flag
# the model's own prose in between as a "quote". A span never crosses a
# line break, so one unbalanced mark cannot swallow a paragraph.
_QUOTE = re.compile(r'"([^"\n]*)"|“([^”\n]*)”|«([^»\n]*)»')
_ELLIPSIS = re.compile(r"\.\.\.|…")
MIN_QUOTE_NORM_CHARS = 20
MAX_QUOTE_WARNINGS = 3


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


def _norm(text: str) -> str:
    """Lowercase, letters/digits only: makes quote matching robust to
    punctuation, curly quotes and whitespace differences."""
    return re.sub(r"\W+", "", (text or "").lower())


def _clip(text: str, limit: int, keep_end: bool) -> str:
    """Flatten whitespace and cut to `limit` chars. Ghost replies keep their
    END, because offers ("would you like to hear about…?") come last."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return ("…" + text[-limit:]) if keep_end else (text[:limit] + "…")


def _detect_language(text: str) -> str | None:
    """'it', 'en' or None (too short / tied), by function-word counts."""
    it = len(_IT_WORDS.findall(text or ""))
    en = len(_EN_WORDS.findall(text or ""))
    if it == en:
        return None
    return "it" if it > en else "en"


def question_language(question: str, history: list[dict] | None) -> str | None:
    """Language of the question; a bare follow-up ("si", "ok") inherits the
    language of the Guardian's most recent message that has one."""
    lang = _detect_language(question)
    if lang:
        return lang
    for m in reversed(history or []):
        if m.get("role") == "user":
            lang = _detect_language(m.get("content", ""))
            if lang:
                return lang
    return None


def answer_reminder(question: str, history: list[dict] | None) -> str:
    """Reminder appended after the question; empty if disabled."""
    if os.environ.get("ANSWER_REMINDER", "on").strip().lower() == "off":
        return ""
    lang = question_language(question, history)
    language = LANGUAGE_NAMES.get(lang, REMINDER_UNKNOWN_LANGUAGE)
    return "\n\n" + REMINDER.format(language=language)


def _fallback_query(question: str, history: list[dict] | None) -> str:
    """Deterministic backup: previous Guardian message + current one."""
    prev = next((m.get("content", "") for m in reversed(history or [])
                 if m.get("role") == "user"), "")
    return f"{prev} {question}".strip()


def _clean_rewrite(raw: str) -> str:
    """First line only, without labels or wrapping quotes."""
    if not raw:
        return ""
    line = raw.strip().splitlines()[0].strip()
    line = _LABEL_PREFIX.sub("", line)
    return line.strip().strip("\"'“”‘’").strip()


def rewrite_query(question: str, history: list[dict] | None) -> str:
    """Return the string to use for retrieval (see REWRITE_PROMPT).

    The logged value is the FINAL query — the exact string to pass to
    tools/show_hits.py when grading an answer.
    """
    mode = os.environ.get("QUERY_REWRITE", "followups").strip().lower()
    if mode == "off" or (mode == "followups" and not history):
        return question

    lines = []
    for m in (history or [])[-4:]:
        ghost = m.get("role") != "user"
        who = "Ghost" if ghost else "Guardian"
        lines.append(f"{who}: {_clip(m.get('content', ''), 600, keep_end=ghost)}")
    convo = "\n".join(lines) if lines else "(none)"

    client, model = _client()
    raw = ""
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=60,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": (f"CONVERSATION:\n{convo}\n\n"
                                             f"LATEST MESSAGE:\n{question}")},
            ])
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[retrieval query] rewrite failed ({type(e).__name__}); "
              f"using fallback", flush=True)

    out = _clean_rewrite(raw)
    if not out or len(out) > REWRITE_MAX_CHARS:
        out = _fallback_query(question, history)     # already contains question
    elif _norm(question) not in _norm(out):
        # Safety net: keep the Guardian's own words in the query, so a
        # rewrite that drops the new subject cannot hide it from retrieval.
        out = f"{out} {question.strip()}"

    print(f"[retrieval query] {question!r} -> {out!r}", flush=True)
    return out


def unverified_quotes(answer: str, hits) -> list[str]:
    """Quoted passages in `answer` that do not appear verbatim (modulo
    punctuation) in any retrieved record. Quotes joined by an ellipsis are
    checked piece by piece; spans too short to be a quotation (titles,
    single names) are ignored."""
    corpus = _norm(" ".join(h.text for h in hits))
    missing: list[str] = []
    for match in _QUOTE.finditer(answer):
        quote = next(g for g in match.groups() if g is not None)
        pieces = [p for p in _ELLIPSIS.split(quote)
                  if len(_norm(p)) >= MIN_QUOTE_NORM_CHARS]
        if pieces and any(_norm(p) not in corpus for p in pieces):
            missing.append(" ".join(quote.split()))
    return missing


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
    stream — an empty chat box with no diagnosis. Yield instead.

    After the stream completes, quoted passages are checked against the
    retrieved records and unverified ones are flagged under the answer.
    """
    client, model = _client()
    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    temperature = float(os.environ.get("LLM_TEMPERATURE", DEFAULT_TEMPERATURE))
    messages = [{"role": "system", "content": PROMPT}]
    messages += (history or [])[-8:]          # short rolling memory
    messages.append({
        "role": "user",
        "content": (f"{CONTEXT_HEADER}\n\n{build_context(hits)}\n\n"
                    f"=== END OF ARCHIVE RECORDS ===\n\n"
                    f"GUARDIAN'S QUESTION:\n{question}"
                    f"{answer_reminder(question, history)}"),
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

    answer = []
    try:
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                answer.append(delta)
                yield delta
    except Exception as e:
        yield f"\n\n[Stream interrupted: {type(e).__name__}: {e}]"
        return

    missing = unverified_quotes("".join(answer), hits)
    for q in missing:
        print(f"[quote check] NOT FOUND: {q!r}", flush=True)
    if missing:
        shown = missing[:MAX_QUOTE_WARNINGS]
        lines = "\n".join(f"  “{q[:120]}{'…' if len(q) > 120 else ''}”"
                          for q in shown)
        extra = (f"\n  (+{len(missing) - len(shown)} more)"
                 if len(missing) > len(shown) else "")
        yield ("\n\n[⚠ Quoted text not found verbatim in the archive records "
               "used for this answer — a translation or an invention:\n"
               f"{lines}{extra}]")