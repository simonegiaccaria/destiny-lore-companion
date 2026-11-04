"""Manifest -> chunks (JSONL) + localized lookup + book structure.

Outputs
-------
data/chunks/chunks.jsonl        one chunk per line, schema below
data/chunks/books.json          book -> ordered pages (drives the wiki)
data/chunks/review_splits.txt   split candidates, APPLIED or not (review!)
data/chunks/review_dedup.txt    duplicate lore groups, with match type (review!)
data/chunks/review_flavor.txt   flavour texts dropped, and why (review!)
data/lookup/strings.sqlite      (hash, lang, kind, name, text) localizations

Chunk schema (the load-bearing wall — see README rule #2):
{
  "id": "lore-<hash>" | "lore-<hash>#s<n>" | "flavor-<hash>",
  "text": "...",
  "title": "...", "book": "...", "page_order": 3,
  "source_type": "lore_entry" | "flavor_text",
  "release": "witch_queen" | "unknown",
  "release_date": "2022-02-22" | null,
  "vaulted": true | false | null,          # null = unknown release
  "unlock_record_hash": 1234567890 | null,
  "alt_unlock_records": [...],             # records of collapsed duplicates
  "aliases": ["Cinder Pinion Bond", ...],  # other names for the same text
  "section": 0, "section_count": 2,        # lore entries split in parts
  "spoils": ["lightfall"],
  "lang": "en"
}

Corpus-hygiene steps (all verified problems, all logged for review):

1. Lore duplicates. The same lore text is attached to several items (class
   variants of one armour piece, Adept reissues). Groups are collapsed on
   an exact normalized key, or on a tail key (last TAIL_KEY_CHARS
   normalized chars) for texts that differ only at the start. A tail
   match is never allowed between two entries that are both in a book.

2. Mixed-voice lore entries. Splitting on wide blank gaps is EDITORIAL:
   only entries allowed in config/corpus.yaml are split.

3. Flavour text already inside a lore entry. An item's flavour line is
   usually also the first line of that item's lore entry, so retrieval
   returned both and wasted a context slot (6 of 30 benchmark questions).
   A flavour text is dropped when it equals a line, or two consecutive
   lines, of a lore entry; its item name is added to that entry's aliases.

4. Gameplay text. Some flavour fields hold objectives, not lore ("Complete
   the dungeon ... with a fireteam of two players or fewer"); the model
   turned one into a plan by Mara Sov, another into something Osiris did.
   Dropped when the text has an objective line AND names a gameplay term
   (rules and lists below).

Release attribution is still best-effort and currently near-empty: the
manifest barely carries seasonHash on items (see tools/inspect_seasons.py).
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_CHUNKS = ROOT / "data" / "chunks"
OUT_LOOKUP = ROOT / "data" / "lookup"

SECTION_BREAK = re.compile(r"\n[ \t]*\n[ \t]*\n+")   # 3+ newlines
MIN_SECTION_WORDS = 30

# Near-duplicate detection: entries sharing their last N normalized chars.
TAIL_KEY_CHARS = 300

# Gameplay-text filter. A flavour text is gameplay when it names a gameplay
# term AND has an objective line:
#  - a line starting with a gameplay verb followed by an objective-style
#    complement ("Complete the…", "Defeat 50…"), so lore that merely starts
#    with the same word ("Defeat is not the end…") is kept; or
#  - a bulleted line ("- Complete Blind Well activities…") starting with a
#    gameplay verb: bulleted objective lists are never lore. This catches
#    event armour like "During Solstice 2021: - Complete …", which the
#    first rule missed and the model then narrated as something Osiris did.
# Deliberately conservative: a missed objective leaves noise in, a false
# match would delete lore. Every drop is listed in review_flavor.txt.
_VERBS = (r"(complete|defeat|earn|acquire|obtain|collect|reach|win|finish|"
          r"kill|purchase|redeem|unlock|play|participate|pick up|equip|use|"
          r"rank up|claim|turn in)")
GAMEPLAY_VERB = re.compile(
    r"^\s*" + _VERBS + r"\s+(the|a|an|any|all|your|\d+)\b", re.IGNORECASE)
GAMEPLAY_BULLET = re.compile(r"^\s*[-•*]\s+" + _VERBS + r"\b", re.IGNORECASE)
GAMEPLAY_TERM = re.compile(
    r"\b(dungeon|raid|strike|crucible|gambit|nightfall|fireteam|activity|"
    r"activities|bounty|bounties|playlist|triumph|seal|iron banner|trials|"
    r"vanguard ops|match(es)?|precision|multikills?|players?|rank|"
    r"engram|quest|combatants|\d+\s+times)\b", re.IGNORECASE)

MANIFEST: Path | None = None  # resolved in main()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _manifest_dir() -> Path:
    """Resolve the current manifest snapshot via the version pointer file
    (cross-platform replacement for a symlink)."""
    pointer = ROOT / "data" / "manifest" / "current.txt"
    if not pointer.exists():
        sys.exit("No manifest snapshot. Run pipeline/download_manifest.py first.")
    return ROOT / "data" / "manifest" / pointer.read_text(encoding="utf-8").strip()


def load_table(lang: str, table: str) -> dict:
    path = MANIFEST / lang / f"{table}.json"
    if not path.exists():
        sys.exit(f"Missing {path}. Run pipeline/download_manifest.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


TAG_RE = re.compile(r"<[^>]+>")


def clean(text: str | None) -> str:
    """Strip residual HTML/entities, normalize unicode and inline whitespace.
    Newlines are preserved: section detection and line matching need them."""
    if not text:
        return ""
    text = html.unescape(text)
    text = TAG_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def norm_key(text: str) -> str:
    """Normalization key for exact dedup."""
    return re.sub(r"\W+", "", text.lower())


def tail_key(text: str) -> str | None:
    """Key for near-duplicates that differ only at the start. None for
    short texts, where a shared tail is not evidence of duplication."""
    k = norm_key(text)
    return k[-TAIL_KEY_CHARS:] if len(k) >= TAIL_KEY_CHARS else None


def line_keys(text: str) -> list[str]:
    """Normalized keys of every non-empty line and of every pair of
    consecutive lines (a flavour text may span a line break)."""
    lines = [norm_key(l) for l in text.splitlines()]
    lines = [l for l in lines if l]
    return lines + [a + b for a, b in zip(lines, lines[1:])]


def is_gameplay_text(text: str) -> bool:
    """See the GAMEPLAY_* comment above."""
    if not GAMEPLAY_TERM.search(text):
        return False
    return any(GAMEPLAY_VERB.match(line) or GAMEPLAY_BULLET.match(line)
               for line in text.splitlines())


def candidate_sections(body: str) -> list[str]:
    """Detect sections separated by wide blank gaps. Pure detection: whether
    to APPLY the split is an editorial decision (config/corpus.yaml)."""
    raw = [p.strip() for p in SECTION_BREAK.split(body) if p.strip()]
    if len(raw) < 2:
        return [body]
    merged: list[str] = []
    for part in raw:
        if merged and len(part.split()) < MIN_SECTION_WORDS:
            merged[-1] = merged[-1] + "\n\n" + part      # short tail: join back
        else:
            merged.append(part)
    if len(merged) >= 2 and len(merged[0].split()) < MIN_SECTION_WORDS:
        merged[1] = merged[0] + "\n\n" + merged[1]       # short head: join forward
        merged.pop(0)
    return merged if len(merged) >= 2 else [body]


def load_releases() -> tuple[dict, dict]:
    cfg = yaml.safe_load(
        (ROOT / "config" / "releases.yaml").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in cfg["releases"]}
    overrides = {o["lore_hash"]: o for o in (cfg.get("overrides") or [])}
    return by_id, overrides


def load_corpus_rules() -> dict:
    """Editorial rules from config/corpus.yaml (split allowlist)."""
    path = ROOT / "config" / "corpus.yaml"
    if not path.exists():
        return {"split_books": set(), "split_hashes": set()}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sp = cfg.get("split_sections") or {}
    return {
        "split_books": set(sp.get("books") or []),
        "split_hashes": {int(h) for h in (sp.get("lore_hashes") or [])},
    }


def season_release_map(seasons: dict, releases: dict) -> dict[int, str]:
    """Map DestinySeasonDefinition hash -> our release id, by date proximity.

    Works on the seasons that carry a startDate, but almost no item
    references a season (52 of ~39k items in the current manifest), so it
    attributes almost nothing. Kept for the few items it does cover.
    """
    dated = []
    for rid, r in releases.items():
        d = r.get("date")
        if d:
            dated.append((rid, d if isinstance(d, date) else date.fromisoformat(str(d))))

    mapping: dict[int, str] = {}
    for h, s in seasons.items():
        start = (s.get("startDate") or "")[:10]
        if not start:
            continue
        sd = date.fromisoformat(start)
        best = min(dated, key=lambda x: abs((x[1] - sd).days), default=None)
        if best and abs((best[1] - sd).days) <= 7:
            mapping[int(h)] = best[0]
    return mapping


# --------------------------------------------------------------------------
# main extraction
# --------------------------------------------------------------------------

def main() -> None:
    global MANIFEST
    MANIFEST = _manifest_dir()
    OUT_CHUNKS.mkdir(parents=True, exist_ok=True)
    OUT_LOOKUP.mkdir(parents=True, exist_ok=True)

    releases, overrides = load_releases()
    rules = load_corpus_rules()

    lore = load_table("en", "DestinyLoreDefinition")
    items = load_table("en", "DestinyInventoryItemDefinition")
    records = load_table("en", "DestinyRecordDefinition")
    nodes = load_table("en", "DestinyPresentationNodeDefinition")
    seasons = load_table("en", "DestinySeasonDefinition")

    season_to_release = season_release_map(seasons, releases)

    # ---- join 1: lore hash -> unlocking record hash (Phase 2 gating key)
    lore_to_record: dict[int, int] = {}
    for rh, rec in records.items():
        lh = rec.get("loreHash")
        if lh:
            lore_to_record.setdefault(int(lh), int(rh))

    # ---- join 2: book/page structure from presentation nodes
    lore_book: dict[int, tuple[str, int]] = {}
    for nh, node in nodes.items():
        children = (node.get("children") or {}).get("records") or []
        if not children:
            continue
        title = clean((node.get("displayProperties") or {}).get("name"))
        if not title:
            continue
        order = 0
        for child in children:
            rec = records.get(str(child.get("recordHash", "")))
            if not rec or not rec.get("loreHash"):
                continue
            lh = int(rec["loreHash"])
            if lh not in lore_book:          # first placement wins
                lore_book[lh] = (title, order)
                order += 1

    # ---- release attribution (best-effort, see season_release_map)
    item_release: dict[int, str] = {}
    lore_release_hint: dict[int, str] = {}
    for ih, item in items.items():
        rid = season_to_release.get(int(item.get("seasonHash") or 0), "unknown")
        item_release[int(ih)] = rid
        lh = item.get("loreHash")
        if lh and rid != "unknown":
            lore_release_hint.setdefault(int(lh), rid)

    def release_fields(rid: str, lore_hash: int | None = None) -> dict:
        if lore_hash is not None and lore_hash in overrides:
            rid = overrides[lore_hash].get("release", rid)
        r = releases.get(rid)
        if not r:
            return {"release": "unknown", "release_date": None,
                    "vaulted": None, "spoils": []}
        vaulted = r["vaulted"]
        if lore_hash is not None and lore_hash in overrides:
            vaulted = overrides[lore_hash].get("vaulted", vaulted)
        return {"release": rid, "release_date": str(r.get("date") or "") or None,
                "vaulted": vaulted, "spoils": list(r.get("spoils") or [])}

    # ---- lore: collect candidates -------------------------------------
    candidates: list[dict] = []
    for lh_str, entry in lore.items():
        lh = int(lh_str)
        dp = entry.get("displayProperties") or {}
        body = clean(dp.get("description"))
        if not body:
            continue
        book, order = lore_book.get(lh, ("", 0))
        candidates.append({
            "lh": lh,
            "title": clean(dp.get("name")) or "Untitled",
            "subtitle": clean(entry.get("subtitle")),
            "body": body,
            "book": book,
            "order": order,
            "rid": lore_release_hint.get(lh, "unknown"),
            "record": lore_to_record.get(lh),
            "how": "",
        })

    # ---- lore: collapse duplicates -------------------------------------
    # Preference order decides which member of a group is kept.
    candidates.sort(key=lambda c: (not c["book"], -len(c["body"]),
                                   c["title"] == "Untitled", c["lh"]))
    groups: dict[int, list[dict]] = {}
    seen_exact: dict[str, int] = {}
    seen_tail: dict[str, int] = {}
    for c in candidates:
        ek, tk = norm_key(c["body"]), tail_key(c["body"])
        canon, how = seen_exact.get(ek), "EXACT"
        if canon is None and tk is not None:
            t = seen_tail.get(tk)
            # Two book pages sharing an ending are still two pages.
            if t is not None and not (c["book"] and groups[t][0]["book"]):
                canon, how = t, "TAIL"
        if canon is None:
            groups[c["lh"]] = [c]
            seen_exact[ek] = c["lh"]
            if tk is not None:
                seen_tail.setdefault(tk, c["lh"])
        else:
            c["how"] = how
            groups[canon].append(c)

    # ---- lore: build chunks in memory (flavour pass may add aliases) ----
    lore_chunks: dict[str, list[dict]] = {}       # base id -> its chunks
    lore_title: dict[str, str] = {}
    line_owner: dict[str, str] = {}               # line key -> base id
    books: dict[str, list[dict]] = defaultdict(list)
    split_log: list[str] = []
    dedup_log: list[str] = []
    n_sections = n_split_candidates = n_split_applied = 0

    for canon_lh, members in groups.items():
        head = members[0]
        rid = next((m["rid"] for m in members if m["rid"] != "unknown"),
                   "unknown")
        recs = [m["record"] for m in members if m["record"]]
        aliases = sorted({m["title"] for m in members[1:]
                          if m["title"] not in (head["title"], "Untitled")})

        detected = candidate_sections(head["body"])
        is_candidate = len(detected) > 1
        apply_split = is_candidate and (
            head["book"] in rules["split_books"]
            or canon_lh in rules["split_hashes"])
        sections = detected if apply_split else [head["body"]]
        if head["subtitle"]:
            sections[0] = f"{head['subtitle']}\n\n{sections[0]}"

        base_id = f"lore-{canon_lh}"
        chunks = []
        for i, sec in enumerate(sections):
            chunks.append({
                "id": base_id if i == 0 else f"{base_id}#s{i}",
                "text": sec,
                "title": head["title"],
                "book": head["book"],
                "page_order": head["order"],
                "source_type": "lore_entry",
                **release_fields(rid, canon_lh),
                "unlock_record_hash": recs[0] if recs else None,
                "alt_unlock_records": recs[1:],
                "aliases": aliases,
                "section": i,
                "section_count": len(sections),
                "lang": "en",
            })
        lore_chunks[base_id] = chunks
        lore_title[base_id] = head["title"]

        # Lines of the kept text, plus subtitles of collapsed members
        # (dedup ignores subtitles, and a subtitle is often a flavour line).
        owned = line_keys(head["subtitle"] + "\n" + head["body"])
        for m in members[1:]:
            owned += line_keys(m["subtitle"])
        for k in owned:
            line_owner.setdefault(k, base_id)

        n_sections += len(sections)
        if is_candidate:
            n_split_candidates += 1
            n_split_applied += int(apply_split)
            tag = "APPLIED  " if apply_split else "candidate"
            split_log.append(f"=== [{tag}] {head['title']} ({base_id}) "
                             f"book={head['book'] or '-'} — "
                             f"{len(detected)} sections")
            for i, sec in enumerate(detected):
                preview = sec.replace("\n", " ")[:110]
                split_log.append(f"  [{i}] {preview}…")
        if len(members) > 1:
            dedup_log.append(f"KEPT          {head['title']} ({base_id}) "
                             f"book={head['book'] or '-'}")
            for m in members[1:]:
                dedup_log.append(f"  DROPPED {m['how']:<5} {m['title']} "
                                 f"(lore-{m['lh']}) book={m['book'] or '-'}")

        # Every member keeps its wiki page, pointing at the kept text.
        for m in members:
            if m["book"]:
                books[m["book"]].append({"order": m["order"],
                                         "id": base_id,
                                         "title": m["title"]})

    # ---- flavour text ---------------------------------------------------
    flavor_chunks: list[dict] = []
    flavor_log_lore: list[str] = []
    flavor_log_game: list[str] = []
    n_flavor_reissue = 0
    seen_flavor: set[str] = set()

    for ih_str, item in items.items():
        flavor = clean(item.get("flavorText"))
        if not flavor or len(flavor) < 25:      # skip stubs
            continue
        key = norm_key(flavor)
        if key in seen_flavor:                  # reissues of the same item
            n_flavor_reissue += 1
            continue
        seen_flavor.add(key)
        ih = int(ih_str)
        name = clean((item.get("displayProperties") or {}).get("name")) \
            or "Unknown item"
        one_line = flavor.replace("\n", " ")

        if is_gameplay_text(flavor):
            flavor_log_game.append(f"  {name} (flavor-{ih}): {one_line[:140]}")
            continue

        owner = line_owner.get(key)
        if owner is not None:
            flavor_log_lore.append(f"  {name} (flavor-{ih}) -> "
                                   f"{lore_title[owner]} ({owner}): "
                                   f"{one_line[:100]}")
            if name != lore_title[owner]:
                for ch in lore_chunks[owner]:
                    if name not in ch["aliases"]:
                        ch["aliases"] = sorted(ch["aliases"] + [name])
            continue

        flavor_chunks.append({
            "id": f"flavor-{ih}",
            "text": flavor,
            "title": name,
            "book": "",
            "page_order": 0,
            "source_type": "flavor_text",
            **release_fields(item_release.get(ih, "unknown")),
            "unlock_record_hash": None,
            "alt_unlock_records": [],
            "aliases": [],
            "section": 0,
            "section_count": 1,
            "lang": "en",
        })

    # ---- write chunks -----------------------------------------------------
    chunks_path = OUT_CHUNKS / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as out:
        for chunks in lore_chunks.values():
            for ch in chunks:
                out.write(json.dumps(ch, ensure_ascii=False) + "\n")
        for ch in flavor_chunks:
            out.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # ---- book structure for the wiki ------------------------------------
    for pages in books.values():
        pages.sort(key=lambda p: p["order"])
    (OUT_CHUNKS / "books.json").write_text(
        json.dumps(books, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- review files: heuristics must be checked by a human ------------
    (OUT_CHUNKS / "review_splits.txt").write_text(
        "\n".join(split_log) or "(no split candidates)", encoding="utf-8")
    (OUT_CHUNKS / "review_dedup.txt").write_text(
        "\n".join(dedup_log) or "(no duplicates found)", encoding="utf-8")
    (OUT_CHUNKS / "review_flavor.txt").write_text(
        "\n".join(
            [f"=== GAMEPLAY TEXT — dropped ({len(flavor_log_game)})"]
            + (flavor_log_game or ["  (none)"])
            + ["", f"=== ALREADY IN A LORE ENTRY — dropped, item name kept "
                   f"as alias ({len(flavor_log_lore)})"]
            + (flavor_log_lore or ["  (none)"])),
        encoding="utf-8")

    # ---- localized lookup layer (README rule #1: presentation only) -----
    db = sqlite3.connect(OUT_LOOKUP / "strings.sqlite")
    db.execute("""CREATE TABLE IF NOT EXISTS strings
                  (hash INTEGER, lang TEXT, kind TEXT, name TEXT, text TEXT,
                   PRIMARY KEY (hash, lang, kind))""")
    for lang_dir in sorted(MANIFEST.iterdir()):
        lang = lang_dir.name
        if lang == "en" or not lang_dir.is_dir():
            continue
        rows = []
        for kind, table in (("lore", "DestinyLoreDefinition"),
                            ("item", "DestinyInventoryItemDefinition")):
            for h, d in load_table(lang, table).items():
                dp = d.get("displayProperties") or {}
                rows.append((int(h), lang, kind, clean(dp.get("name")),
                             clean(dp.get("description") or d.get("flavorText"))))
        db.executemany("INSERT OR REPLACE INTO strings VALUES (?,?,?,?,?)", rows)
        print(f"lookup: {lang} -> {len(rows)} localized strings")
    db.commit()
    db.close()

    # ---- summary: the baseline numbers ---------------------------------
    n_lore = len(groups)
    n_lore_dup = len(candidates) - n_lore
    n_tail = sum(1 for c in candidates if c["how"] == "TAIL")
    n_flavor = len(flavor_chunks)
    total = n_sections + n_flavor
    unknown = sum(1 for chunks in lore_chunks.values() for ch in chunks
                  if ch["release"] == "unknown") \
        + sum(1 for ch in flavor_chunks if ch["release"] == "unknown")
    print(f"lore entries:  {len(candidates)} found, {n_lore_dup} duplicates "
          f"collapsed ({n_tail} by TAIL match) -> {n_lore} kept")
    print(f"               {n_split_candidates} split candidates, "
          f"{n_split_applied} applied -> {n_sections} lore chunks")
    print(f"flavour texts: {n_flavor} kept — dropped: {n_flavor_reissue} "
          f"reissues, {len(flavor_log_lore)} already in a lore entry, "
          f"{len(flavor_log_game)} gameplay text")
    print(f"total chunks:  {total}")
    print(f"release attribution: {unknown}/{total} chunks 'unknown' "
          f"(manifest barely carries seasonHash — see tools/inspect_seasons.py)")
    print("REVIEW: data/chunks/review_splits.txt, review_dedup.txt, "
          "review_flavor.txt")


if __name__ == "__main__":
    main()