"""Manifest -> chunks (JSONL) + localized lookup + book structure.

Outputs
-------
data/chunks/chunks.jsonl      one chunk per line, schema below
data/chunks/books.json        book -> ordered pages (drives the wiki)
data/lookup/strings.sqlite    (hash, lang, name, text) official localizations

Chunk schema (THE load-bearing wall — see README rule #2):
{
  "id": "lore-<hash>" | "flavor-<hash>",
  "text": "...",
  "title": "...", "book": "...", "page_order": 3,
  "source_type": "lore_entry" | "flavor_text",
  "release": "witch_queen" | "unknown",
  "release_date": "2022-02-22" | null,
  "vaulted": true | false | null,       # null = unknown release
  "unlock_record_hash": 1234567890 | null,
  "spoils": ["lightfall"],
  "lang": "en"
}

Release attribution is best-effort in Phase 1: seasonHash where the manifest
provides one, else "unknown". Ishtar spoiler-dates improve this in Phase 3.
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_CHUNKS = ROOT / "data" / "chunks"
OUT_LOOKUP = ROOT / "data" / "lookup"


def _manifest_dir() -> Path:
    """Resolve the current manifest snapshot via the version pointer file
    (cross-platform replacement for a symlink)."""
    pointer = ROOT / "data" / "manifest" / "current.txt"
    if not pointer.exists():
        sys.exit("No manifest snapshot. Run pipeline/download_manifest.py first.")
    return ROOT / "data" / "manifest" / pointer.read_text().strip()


MANIFEST = None  # resolved in main()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_table(lang: str, table: str) -> dict:
    path = MANIFEST / lang / f"{table}.json"
    if not path.exists():
        sys.exit(f"Missing {path}. Run pipeline/download_manifest.py first.")
    return json.loads(path.read_text())


TAG_RE = re.compile(r"<[^>]+>")

def clean(text: str | None) -> str:
    """Strip residual HTML/entities, normalize unicode and whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = TAG_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def norm_key(text: str) -> str:
    """Normalization key for dedup (reissued items share flavour text)."""
    return re.sub(r"\W+", "", text.lower())


def load_releases() -> tuple[dict, dict]:
    cfg = yaml.safe_load((ROOT / "config" / "releases.yaml").read_text())
    by_id = {r["id"]: r for r in cfg["releases"]}
    overrides = {o["lore_hash"]: o for o in (cfg.get("overrides") or [])}
    return by_id, overrides


def season_release_map(seasons: dict, releases: dict) -> dict[int, str]:
    """Map DestinySeasonDefinition hash -> our release id, by date proximity.

    Best-effort: a season whose start date matches a configured release date
    (within a few days) gets that id. Everything else stays 'unknown'.
    """
    from datetime import date, timedelta

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

    # ---- join 2: book/page structure from presentation nodes.
    # Lore books are presentation nodes whose children are records that
    # carry loreHash. Walk every node; where its child records resolve to
    # lore, register (book title, page order).
    lore_book: dict[int, tuple[str, int]] = {}
    books: dict[str, list[dict]] = defaultdict(list)
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

    # ---- release attribution for items (lore defs carry no season signal;
    # they inherit best-effort from a referencing item where possible)
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
            ov = overrides[lore_hash]
            rid = ov.get("release", rid)
        r = releases.get(rid)
        if not r:
            return {"release": "unknown", "release_date": None,
                    "vaulted": None, "spoils": []}
        vaulted = r["vaulted"]
        if lore_hash is not None and lore_hash in overrides:
            vaulted = overrides[lore_hash].get("vaulted", vaulted)
        return {"release": rid, "release_date": str(r.get("date") or "") or None,
                "vaulted": vaulted, "spoils": list(r.get("spoils") or [])}

    # ---- emit chunks --------------------------------------------------
    chunks_path = OUT_CHUNKS / "chunks.jsonl"
    n_lore = n_flavor = n_dup = 0
    seen_flavor: set[str] = set()

    with chunks_path.open("w") as out:
        # lore entries: one entry = one chunk (self-contained narrative unit)
        for lh_str, entry in lore.items():
            lh = int(lh_str)
            dp = entry.get("displayProperties") or {}
            body = clean(dp.get("description"))
            if not body:
                continue
            title = clean(dp.get("name")) or "Untitled"
            subtitle = clean(entry.get("subtitle"))
            book, order = lore_book.get(lh, ("", 0))
            rid = lore_release_hint.get(lh, "unknown")
            chunk = {
                "id": f"lore-{lh}",
                "text": (f"{subtitle}\n\n{body}" if subtitle else body),
                "title": title,
                "book": book,
                "page_order": order,
                "source_type": "lore_entry",
                **release_fields(rid, lh),
                "unlock_record_hash": lore_to_record.get(lh),
                "lang": "en",
            }
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            if book:
                books[book].append({"order": order, "id": chunk["id"],
                                    "title": title})
            n_lore += 1

        # flavour text: dedup across reissues on normalized text
        for ih_str, item in items.items():
            flavor = clean(item.get("flavorText"))
            if not flavor or len(flavor) < 25:      # skip stubs
                continue
            key = norm_key(flavor)
            if key in seen_flavor:
                n_dup += 1
                continue
            seen_flavor.add(key)
            ih = int(ih_str)
            name = clean((item.get("displayProperties") or {}).get("name"))
            chunk = {
                "id": f"flavor-{ih}",
                "text": flavor,
                "title": name or "Unknown item",
                "book": "",
                "page_order": 0,
                "source_type": "flavor_text",
                **release_fields(item_release.get(ih, "unknown")),
                "unlock_record_hash": None,
                "lang": "en",
            }
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            n_flavor += 1

    # ---- book structure for the wiki ----------------------------------
    for pages in books.values():
        pages.sort(key=lambda p: p["order"])
    (OUT_CHUNKS / "books.json").write_text(
        json.dumps(books, ensure_ascii=False, indent=1))

    # ---- localized lookup layer (README rule #1: presentation only) ----
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

    print(f"chunks: {n_lore} lore entries, {n_flavor} flavour texts "
          f"({n_dup} reissue duplicates dropped)")
    unknown = sum(1 for line in chunks_path.open()
                  if '"release": "unknown"' in line)
    print(f"release attribution: {unknown} chunks 'unknown' "
          f"(expected in Phase 1 — improves with Ishtar dates in Phase 3)")


if __name__ == "__main__":
    main()
