"""Diagnose why release attribution comes out 'unknown'.

Read-only. Prints what the manifest actually offers for season/release
attribution, so the fix in extract_chunks.py is based on data, not on
guesses about the manifest's structure.

Usage:
    .venv\\Scripts\\python tools\\inspect_seasons.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def manifest_dir() -> Path:
    pointer = ROOT / "data" / "manifest" / "current.txt"
    if not pointer.exists():
        sys.exit("No manifest snapshot. Run pipeline/download_manifest.py first.")
    return ROOT / "data" / "manifest" / pointer.read_text(encoding="utf-8").strip()


def load(mdir: Path, table: str) -> dict:
    return json.loads((mdir / "en" / f"{table}.json").read_text(encoding="utf-8"))


def main() -> None:
    mdir = manifest_dir()
    seasons = load(mdir, "DestinySeasonDefinition")
    items = load(mdir, "DestinyInventoryItemDefinition")
    lore = load(mdir, "DestinyLoreDefinition")
    cfg = yaml.safe_load(
        (ROOT / "config" / "releases.yaml").read_text(encoding="utf-8"))
    release_by_name = {r["name"].lower(): r["id"] for r in cfg["releases"]}

    # ---- seasons -------------------------------------------------------
    print("=== DestinySeasonDefinition ===")
    print(f"definitions: {len(seasons)}")
    fields = Counter(k for s in seasons.values() for k in s)
    print("fields (how many definitions have each):")
    print("  " + ", ".join(f"{k}={n}" for k, n in fields.most_common()))
    print()
    rows = []
    for h, s in seasons.items():
        name = (s.get("displayProperties") or {}).get("name", "")
        rows.append((s.get("seasonNumber", -1), name,
                     (s.get("startDate") or "")[:10], h))
    for num, name, start, h in sorted(rows):
        match = release_by_name.get(name.lower(), "")
        print(f"  #{num:>3}  start={start or '-':<10}  {name[:42]:<42}"
              f"  -> {match or 'no name match in releases.yaml'}")

    # ---- items ---------------------------------------------------------
    print("\n=== DestinyInventoryItemDefinition ===")
    season_hashes = {int(h) for h in seasons}
    with_season = [i for i in items.values() if i.get("seasonHash")]
    dangling = sum(1 for i in with_season
                   if int(i["seasonHash"]) not in season_hashes)
    with_lore = [i for i in items.values() if i.get("loreHash")]
    watermarks = Counter(i.get("iconWatermark") for i in items.values()
                         if i.get("iconWatermark"))
    print(f"items:                         {len(items)}")
    print(f"  with seasonHash:             {len(with_season)}"
          f"  ({dangling} pointing to no known season)")
    print(f"  with loreHash:               {len(with_lore)}")
    print(f"  with loreHash AND seasonHash:"
          f" {sum(1 for i in with_lore if i.get('seasonHash'))}")
    print(f"  with iconWatermark:          {sum(watermarks.values())}"
          f"  ({len(watermarks)} distinct watermarks)")

    # ---- lore coverage -------------------------------------------------
    print("\n=== Lore coverage ===")
    lore_ids = {int(h) for h in lore}
    via_item = {int(i["loreHash"]) for i in with_lore} & lore_ids
    via_seasoned = {int(i["loreHash"]) for i in with_lore
                    if i.get("seasonHash")} & lore_ids
    print(f"lore entries:                          {len(lore_ids)}")
    print(f"  referenced by any item:              {len(via_item)}")
    print(f"  referenced by an item w/ seasonHash: {len(via_seasoned)}")
    print(f"  referenced by no item (book-only):   {len(lore_ids - via_item)}")


if __name__ == "__main__":
    main()