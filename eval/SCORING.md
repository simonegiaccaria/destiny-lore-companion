# Scoring — keep it brutal and repeatable

## Grading procedure (read first)

Never grade an answer against memory of Destiny lore, and never against a
chunk you *think* was retrieved. Five wrong verdicts in this project came
from doing exactly that — twice the detail was past a truncation, once two
different lore entries shared the same title.

For every answer:
1. Take the retrieval query from the uvicorn terminal line
   `[retrieval query] '<message>' -> '<query>'`. For a first message in a
   fresh chat (no history) there is no such line: the query is the
   message itself.
2. Run `tools/show_hits.py --full "<that exact query>"`. Retrieval is
   deterministic: this shows exactly the records the model had.
3. Check every claim in the answer against that output. A claim is
   grounded only if its content is in one of those records; an
   attribution is correct only if it names the record that holds it
   (mind duplicate titles — check the id).

## Axes

Score every question 0-3 on each axis. Track per-run totals in a copy of
`runs/TEMPLATE.csv` so local-vs-API comparisons are numbers, not vibes.

**Grounding (0-3)** 3 = every claim traceable to a retrieved record;
0 = invented. Any single hallucinated "fact" caps the question at 1.

**Attribution (0-3)** only when the answer names a source (asked or not):
3 = every named source holds what is attributed to it; 0 = sources named
that are not in the retrieved records at all.

**Coverage (0-3)** 3 = used the relevant retrieved records, including the
non-obvious ones; for [X] questions, missing a whole era/source = max 2.

**Honesty (0-3)** for [N]/[C]: 3 = admits gaps / presents all accounts
with attribution; 0 = confident single answer where none exists.

**Language (0-3)** 3 = fluent reply in the question's language, with
official terminology; 0 = reply in another language.

**Persona (0-3)** Ghost voice in prose; no lists, no retrieval-speak
("the fragments say…" = max 1), no citations unless asked.

## Exit bar

Phase-1 exit: median Grounding ≥ 2.5, no zeros anywhere, and all three
[N] questions at Honesty 3. If the local model can't clear the bar,
re-run identical questions via the API backend before redesigning
anything.