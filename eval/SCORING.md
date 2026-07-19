# Scoring — keep it brutal and repeatable

Score every question 0-3 on each axis. Track per-run totals in a copy of
`runs/TEMPLATE.csv` so local-vs-API comparisons are numbers, not vibes.

**Grounding (0-3)** 3 = every claim traceable to real lore; 0 = invented.
Any single hallucinated "fact" caps the whole question at Grounding 1.

**Coverage (0-3)** 3 = found the relevant sources incl. non-obvious ones;
for [X] questions, missing a whole era/source = max 2.

**Honesty (0-3)** for [N]/[C]: 3 = admits gaps / presents all accounts with
attribution; 0 = confident single answer where none exists.

**Language (0-3)** for [IT]: 3 = fluent Italian with official terminology;
0 = English reply or invented name translations.

**Persona (0-3)** Ghost voice, no retrieval-speak ("the fragments say…" = max 1),
citations only on request.

Phase-1 exit bar: median Grounding ≥ 2.5, no zeros anywhere, and
all three [N] questions at Honesty 3. If the local model can't clear the
bar, re-run identical questions via API backend before redesigning anything.
