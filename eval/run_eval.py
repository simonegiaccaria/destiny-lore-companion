"""Run the reference benchmark end to end and write a gradable report.

For every question in eval/reference_questions.md this runs exactly what
the app runs — follow-up rewriting, hybrid retrieval, Ghost answer, quote
check — optionally several times, and records everything in ONE place:

  data/eval/<run_id>.md     report: automatic-check summary, then per
                            question the query, every retrieved record IN
                            FULL, every answer. Contains Bungie text, so
                            it lives under data/ (git-ignored).
  eval/runs/<run_id>.csv    one row per question per repeat: automatic
                            checks filled in, the manual axes of
                            eval/SCORING.md left blank. No lore text, so
                            it is safe to commit and compare over time.

Why repeats: answers are NOT reproducible on this setup, not even at
temperature 0 — with identical records, 7 of 8 answers diverged between
two runs (same opening, then a fork). A single run cannot tell whether a
change helped. Each question is therefore run N times and the report
counts how OFTEN each failure happens.

Automatic checks (regex, deterministic — necessary, not sufficient):
  quote_warnings  quoted passages not found verbatim in the records
  frame_break     talks about the game from outside ("Destiny universe",
                  "Bungie", "you've provided", "based on the provided…")
  list_format     2+ numbered or bulleted lines (the Ghost speaks in prose)
  language_ok     reply language matches the question ([IT] -> Italian)
  admits_gap      [N] questions only: says the archive does not hold it
Grounding and attribution still need a human: grade them against the
records printed right above each answer (see eval/SCORING.md).

Usage:
    .venv\\Scripts\\python eval\\run_eval.py                       all, once
    .venv\\Scripts\\python eval\\run_eval.py --repeat 3            all, 3x
    .venv\\Scripts\\python eval\\run_eval.py --only 1,24-26        a subset
    .venv\\Scripts\\python eval\\run_eval.py --temperature 0       override
    .venv\\Scripts\\python eval\\run_eval.py --k 5                 top-k

Each repeat of a question is a fresh conversation, except a follow-up
("Follow-up to 29: ...") which runs inside the same repeat of its parent;
selecting a follow-up pulls its parent in automatically.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import os
import re
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.llm import (DEFAULT_TEMPERATURE, answer_stream,  # noqa: E402
                     rewrite_query, unverified_quotes)
from app.retrieval import Retriever                       # noqa: E402

QUESTIONS = ROOT / "eval" / "reference_questions.md"
REPORT_DIR = ROOT / "data" / "eval"
SCORES_DIR = ROOT / "eval" / "runs"

# Must match the text answer_stream() puts in front of its quote warning,
# so the answer and the warning can be shown separately in the report.
WARNING_MARKER = "\n\n[⚠ Quoted text not found"
BACKEND_ERRORS = ("[No connection to the model backend",
                  "[Model backend rejected", "[Model call failed")

AXES = ["grounding", "attribution", "coverage", "honesty",
        "language", "persona"]

_ITEM = re.compile(r"^(\d+)\.\s+(.*)$")               # numbered, column 0
_TAGS = re.compile(r"^((?:\[[A-Z]+\]\s*)+)")
_FOLLOWUP = re.compile(r'^Follow-up to (\d+):\s*["“](.+?)["”]', re.IGNORECASE)

# ---- automatic checks ------------------------------------------------------
# Each pattern encodes a failure actually observed in benchmark answers.
_FRAME_BREAK = re.compile(
    r"\b(destiny (universe|franchise|series|2)|bungie|video ?game|in[- ]game|"
    r"the game|you(?:'ve| have) (provided|shared)|"
    r"(provided|shared) (lore|records?|entries|archive|fragments)|"
    r"based on the (provided|available|given)|"
    r"universo di destiny|sviluppat[oa] da bungie|gioco di ruolo)\b",
    re.IGNORECASE)
_LIST_LINE = re.compile(r"^\s*(\d+\.|[-*•])\s+\S", re.MULTILINE)
_ADMITS_GAP = re.compile(
    r"(archive is silent|is silent on|silent on (that|this)|"
    r"(archive|records?) (do(es)?n'?t|do(es)? not|never) "
    r"(say|state|describe|mention|hold|cover|record|tell)|"
    r"no (direct |single )?(lore )?(entry|book|record|account)|"
    r"there is no|there isn'?t|nothing in the (archive|records?)|"
    r"l'archivio tace|non (c'è|esiste)|nessuna (voce|fonte)|"
    r"non (dice|descrive|riporta|menziona))", re.IGNORECASE)
_IT_WORDS = re.compile(r"\b(il|lo|la|gli|le|di|che|è|non|una|per|sono|"
                       r"della|del|nel|con|come|anche|questo|questa)\b",
                       re.IGNORECASE)
_EN_WORDS = re.compile(r"\b(the|and|of|is|that|to|was|with|for|this|"
                       r"are|his|her|in|as|it)\b", re.IGNORECASE)


def auto_checks(answer: str, missing: list[str], tags: str) -> dict:
    it = len(_IT_WORDS.findall(answer))
    en = len(_EN_WORDS.findall(answer))
    lang = "it" if it > en else "en"
    want = "it" if "[IT]" in tags else "en"
    return {
        "quote_warnings": len(missing),
        "frame_break": int(bool(_FRAME_BREAK.search(answer))),
        "list_format": int(len(_LIST_LINE.findall(answer)) >= 2),
        "language_ok": int(lang == want),
        # only meaningful for negative controls; None elsewhere
        "admits_gap": (int(bool(_ADMITS_GAP.search(answer)))
                       if "[N]" in tags else None),
    }


AUTO_COLS = ["quote_warnings", "frame_break", "list_format",
             "language_ok", "admits_gap"]


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------

def parse_questions(path: Path) -> list[dict]:
    """Parse eval/reference_questions.md.

    A question is a line starting at column 0 with "<n>." plus any indented
    continuation lines. Leading tags like [L][IT] are kept apart; text in
    *italics* is the grader's note, not part of the question.
    """
    items: list[dict] = []
    section = ""
    cur: dict | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if raw.startswith("## "):
            section = stripped[3:].strip()
            cur = None
            continue
        m = _ITEM.match(raw)
        if m:
            cur = {"num": int(m.group(1)), "section": section,
                   "lines": [m.group(2).strip()]}
            items.append(cur)
        elif cur is not None and raw[:1].isspace() and stripped:
            cur["lines"].append(stripped)
        else:
            cur = None

    out = []
    for it in items:
        text = " ".join(it["lines"])
        star = text.find("*")
        body = (text[:star] if star >= 0 else text).strip()
        note = text[star:].replace("*", "").strip().strip("()").strip() \
            if star >= 0 else ""
        tm = _TAGS.match(body)
        tags = "".join(tm.group(1).split()) if tm else ""
        if tm:
            body = body[tm.end():].strip()
        fm = _FOLLOWUP.match(body)
        out.append({
            "num": it["num"],
            "section": it["section"],
            "tags": tags,
            "question": fm.group(2).strip() if fm else body,
            "parent": int(fm.group(1)) if fm else None,
            "note": note,
        })
    return out


def parse_selection(spec: str | None, available: set[int]) -> list[int]:
    """'1,24-26' -> [1, 24, 25, 26]. None -> every available question."""
    if not spec:
        return sorted(available)
    nums: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            nums.update(range(int(a), int(b) + 1))
        else:
            nums.add(int(part))
    unknown = nums - available
    if unknown:
        sys.exit(f"Unknown question numbers: {sorted(unknown)}")
    return sorted(nums)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def _safe(name: str) -> str:
    """Filename-safe (Windows forbids ':' and friends)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "model"


def _rate(values: list) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "–"
    return f"{sum(vals)}/{len(vals)}"


def _stability(answers: list[str]) -> str:
    """Mean pairwise similarity of the repeats (1.00 = identical)."""
    if len(answers) < 2:
        return "–"
    sims = [difflib.SequenceMatcher(None, a, b).ratio()
            for a, b in combinations(answers, 2)]
    return f"{sum(sims) / len(sims):.2f}"


def render_summary(meta: dict, order: list[int], by_num: dict,
                   results: dict) -> list[str]:
    reps = meta["repeat"]
    out = ["## Automatic checks", "",
           "Counts are *failures or passes over repeats*, e.g. 2/3. "
           "frame break, list format: lower is better. language ok, "
           "admits gap: higher is better. stability: 1.00 = repeats "
           "identical.", "",
           "| Q | tags | quote warnings | frame break | list format | "
           "language ok | admits gap | stability |",
           "|---|---|---|---|---|---|---|---|"]
    totals = {c: [] for c in AUTO_COLS}
    for n in order:
        runs = [results[(n, r)] for r in range(1, reps + 1)
                if (n, r) in results]
        if not runs:
            continue
        cols = {c: [x["auto"][c] for x in runs] for c in AUTO_COLS}
        for c in AUTO_COLS:
            totals[c] += cols[c]
        qw = sum(cols["quote_warnings"])
        out.append(f"| Q{n} | {by_num[n]['tags']} | {qw} | "
                   f"{_rate(cols['frame_break'])} | "
                   f"{_rate(cols['list_format'])} | "
                   f"{_rate(cols['language_ok'])} | "
                   f"{_rate(cols['admits_gap'])} | "
                   f"{_stability([x['answer'] for x in runs])} |")
    out.append(f"| **all** | | **{sum(totals['quote_warnings'])}** | "
               f"**{_rate(totals['frame_break'])}** | "
               f"**{_rate(totals['list_format'])}** | "
               f"**{_rate(totals['language_ok'])}** | "
               f"**{_rate(totals['admits_gap'])}** | |")
    return out + [""]


def _records_block(hits) -> list[str]:
    out = []
    for i, h in enumerate(hits, 1):
        m = h.meta
        sim = "n/a" if h.sim is None else f"{h.sim:.3f}"
        book = m.get("book") or "-"
        out += [f"**[{i}] `{h.id}` — {m.get('title') or 'Untitled'}**  ",
                f"type={m.get('source_type')} · book={book} · sim={sim}",
                "", "```text", h.text, "```", ""]
    return out


def render_report(meta: dict, order: list[int], by_num: dict,
                  results: dict) -> str:
    reps = meta["repeat"]
    done = len(results)
    out = [
        f"# Benchmark run {meta['run_id']}",
        "",
        "| setting | value |",
        "|---|---|",
        f"| chat model | `{meta['model']}` |",
        f"| temperature | {meta['temperature']} |",
        f"| query rewrite | {meta['rewrite']} |",
        f"| top-k | {meta['k']} |",
        f"| repeats per question | {reps} |",
        f"| embedding model | `{meta['embedding']}` |",
        f"| started | {meta['started']} |",
        f"| answers done | {done} / {len(order) * reps} |",
        "",
    ]
    out += render_summary(meta, order, by_num, results)
    out += ["Grade each answer ONLY against the records printed right above "
            "it (eval/SCORING.md). Scores go in "
            f"`eval/runs/{meta['run_id']}.csv`.", ""]

    for n in order:
        runs = [(r, results[(n, r)]) for r in range(1, reps + 1)
                if (n, r) in results]
        if not runs:
            continue
        q = by_num[n]
        out += ["---", "", f"## Q{n} {q['tags']}".rstrip(), "",
                f"*{q['section']}*", ""]
        if q["parent"]:
            out += [f"**Follow-up to Q{q['parent']}** (same conversation, "
                    f"same repeat)", ""]
        out += [f"**Question:** {q['question']}", ""]
        if q["note"]:
            out += [f"**Grader's note:** {q['note']}", ""]

        first_query = runs[0][1]["query"]
        for r, res in runs:
            if r == runs[0][0] or res["query"] != first_query:
                label = "" if reps == 1 else f" (repeat {r})"
                out += [f"**Retrieval query{label}:** `{res['query']}`", "",
                        f"### Records retrieved{label} ({len(res['hits'])})",
                        ""]
                out += _records_block(res["hits"])
            label = "" if reps == 1 else f" — repeat {r}"
            a = res["auto"]
            flags = ", ".join(f"{c}={a[c]}" for c in AUTO_COLS
                              if a[c] is not None)
            out += [f"### Answer{label}", "",
                    f"*{res['seconds']:.1f}s · {flags}*", "",
                    "```text", res["answer"].strip(), "```", ""]
            if res["missing"]:
                out += ["⚠ Quotes not found in the records above:", ""]
                out += [f"- “{qt}”" for qt in res["missing"]]
                out += [""]
        out += ["### Score", "",
                " · ".join(f"{a} _" for a in AXES), ""]
    return "\n".join(out) + "\n"


def write_scores(path: Path, meta: dict, order: list[int], by_num: dict,
                 results: dict) -> None:
    # utf-8-sig: Excel on Windows reads accented text correctly only with BOM
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "model", "temperature", "question", "repeat",
                    "tags", *AXES, *AUTO_COLS, "seconds", "notes"])
        for n in order:
            for r in range(1, meta["repeat"] + 1):
                res = results.get((n, r))
                if res is None:
                    continue
                a = res["auto"]
                w.writerow([meta["run_id"], meta["model"],
                            meta["temperature"], n, r, by_num[n]["tags"],
                            *([""] * len(AXES)),
                            *["" if a[c] is None else a[c] for c in AUTO_COLS],
                            f"{res['seconds']:.1f}", ""])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):          # Windows: never crash
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="questions to run, e.g. 1,24-26")
    ap.add_argument("--k", type=int, default=8, help="records per answer")
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per question (answers are not reproducible)")
    ap.add_argument("--temperature", type=float,
                    help="override LLM_TEMPERATURE for this run only")
    args = ap.parse_args()
    if args.repeat < 1:
        sys.exit("--repeat must be at least 1")
    if args.temperature is not None:
        # app.llm reads the variable at call time, so this takes effect.
        os.environ["LLM_TEMPERATURE"] = str(args.temperature)

    questions = parse_questions(QUESTIONS)
    if not questions:
        sys.exit(f"No questions parsed from {QUESTIONS}. Numbered questions "
                 f"must start at column 0, like '1.  [L] What is ...?'")
    by_num = {q["num"]: q for q in questions}

    order = parse_selection(args.only, set(by_num))
    for n in list(order):                        # pull in missing parents
        p = by_num[n]["parent"]
        while p is not None:
            if p not in by_num:
                sys.exit(f"Q{n} is a follow-up to Q{p}, which does not exist.")
            if p not in order:
                order.append(p)
                print(f"Q{n} is a follow-up: adding its parent Q{p}.")
            p = by_num[p]["parent"]
    order.sort()

    model = os.environ.get("LLM_MODEL", "llama3.1:8b-instruct-q5_K_M")
    temperature = os.environ.get("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE))
    run_id = (f"{datetime.now():%Y%m%d-%H%M}_{_safe(model)}"
              f"_T{_safe(temperature)}_x{args.repeat}")
    meta = {
        "run_id": run_id,
        "model": model,
        "temperature": temperature,
        "rewrite": os.environ.get("QUERY_REWRITE", "followups"),
        "k": args.k,
        "repeat": args.repeat,
        "embedding": os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        "started": f"{datetime.now():%Y-%m-%d %H:%M}",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{run_id}.md"
    scores_path = SCORES_DIR / f"{run_id}.csv"

    total_answers = len(order) * args.repeat
    print(f"Parsed {len(questions)} questions; running {len(order)} x "
          f"{args.repeat} with model {model}, T={temperature}, k={args.k}.")
    print("Loading retriever...")
    retriever = Retriever()
    hit_cache: dict[str, list] = {}           # retrieval is deterministic

    results: dict[tuple[int, int], dict] = {}
    t_start = time.perf_counter()
    done = 0
    for r in range(1, args.repeat + 1):
        for n in order:
            q = by_num[n]
            history = (results[(q["parent"], r)]["history_out"]
                       if q["parent"] else [])

            t0 = time.perf_counter()
            query = rewrite_query(q["question"], history)
            if query not in hit_cache:
                hit_cache[query] = retriever.retrieve(
                    query, k=args.k, unlocked_records=None)
            hits = hit_cache[query]
            streamed = "".join(answer_stream(q["question"], hits, history))
            seconds = time.perf_counter() - t0

            if streamed.startswith(BACKEND_ERRORS):
                sys.exit(f"Q{n}: model backend error, run aborted.\n{streamed}")

            answer = streamed.split(WARNING_MARKER, 1)[0]
            missing = unverified_quotes(answer, hits)
            results[(n, r)] = {
                "query": query,
                "hits": hits,
                "answer": answer,
                "missing": missing,
                "auto": auto_checks(answer, missing, q["tags"]),
                "seconds": seconds,
                # Same history the browser keeps: the streamed text, warning
                # included — so a follow-up here behaves as it does in the app.
                "history_out": history + [
                    {"role": "user", "content": q["question"]},
                    {"role": "assistant", "content": streamed},
                ],
            }

            # Rewrite both files after every answer: a crash or Ctrl+C
            # midway keeps everything done so far.
            report_path.write_text(
                render_report(meta, order, by_num, results), encoding="utf-8")
            write_scores(scores_path, meta, order, by_num, results)

            done += 1
            a = results[(n, r)]["auto"]
            bad = [c for c in ("frame_break", "list_format") if a[c]] \
                + (["language"] if not a["language_ok"] else []) \
                + (["no gap admitted"] if a["admits_gap"] == 0 else []) \
                + ([f"{a['quote_warnings']} quote warning(s)"]
                   if a["quote_warnings"] else [])
            print(f"[{done}/{total_answers}] Q{n} {q['tags']} rep {r} — "
                  f"{seconds:.1f}s" + (f" | {', '.join(bad)}" if bad else ""))

    total = time.perf_counter() - t_start
    print(f"\nDone in {total / 60:.1f} min.")
    print(f"Report (read this):  {report_path.relative_to(ROOT)}")
    print(f"Scores (fill these): {scores_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()