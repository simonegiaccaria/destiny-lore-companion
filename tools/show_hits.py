"""Show what the retriever actually returns for a question — no LLM.

Usage:
    .venv\\Scripts\\python tools\\show_hits.py "What is the Traveler?"
    .venv\\Scripts\\python tools\\show_hits.py --full "What is the Traveler?"

Diagnostic tool: separates retrieval failures from generation failures.
If the right fragments are here but the answer ignored them, the problem
is the prompt/model. If they are not here, the problem is retrieval.

When GRADING an answer, always use --full: judging a claim against a
truncated chunk (or against memory) has already produced four wrong
verdicts in this project — the detail was past the cut every time.

Read `sim` (raw cosine similarity), not `rrf`: RRF is rank-based and says
nothing about how good a match actually is. Roughly, sim < 0.4 means the
corpus has nothing relevant — an honest "the archive is silent" is then
the CORRECT answer, not a lucky one.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.retrieval import Retriever  # noqa: E402

PREVIEW_CHARS = 300


def main() -> None:
    args = sys.argv[1:]
    full = "--full" in args
    args = [a for a in args if a != "--full"]
    if not args:
        sys.exit('Usage: python tools/show_hits.py [--full] "your question"')
    question = " ".join(args)

    print(f"QUESTION: {question}\n" + "=" * 70)
    hits = Retriever().retrieve(question, k=8, unlocked_records=None)

    if not hits:
        print("NO HITS — retrieval returned nothing.")
        return

    for i, h in enumerate(hits, 1):
        m = h.meta
        sim = "n/a" if h.sim is None else f"{h.sim:.3f}"
        print(f"\n[{i}] sim={sim}  rrf={h.score:.4f}  id={h.id}")
        print(f"    type={m.get('source_type')}  title={m.get('title')}")
        print(f"    book={m.get('book') or '-'}")
        print(f"    release={m.get('release')} ({m.get('release_date') or '?'})"
              f"  vaulted={m.get('vaulted')}")
        if h.spoils:
            print(f"    SPOILS={','.join(h.spoils)}")
        if full:
            print("    text:")
            for line in h.text.splitlines():
                print(f"      {line}")
        else:
            text = h.text.replace("\n", " ")
            cut = len(text) > PREVIEW_CHARS
            print(f"    text: {text[:PREVIEW_CHARS]}"
                  f"{'...  [TRUNCATED — use --full to grade]' if cut else ''}")

    mix = Counter(h.meta.get("source_type", "?") for h in hits)
    print("\n" + "=" * 70)
    print("source mix: " + ", ".join(f"{t}={n}" for t, n in mix.most_common()))


if __name__ == "__main__":
    main()