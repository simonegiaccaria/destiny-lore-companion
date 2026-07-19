"""Download and snapshot the Destiny 2 manifest tables we need.

Downloads the JSON world-component tables (per language) and stores them
under data/manifest/<version>/<lang>/<Table>.json, plus a 'current' symlink.
Everything is kept on disk permanently: the game is in maintenance mode and
this snapshot is the project's insurance against the API going dark.

English is mandatory (retrieval corpus). Extra languages in LANGUAGES are
lookup layers only (official localized names/text for the wiki and the
Ghost's terminology) — they are downloaded but never indexed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "manifest"

BASE = "https://www.bungie.net"
MANIFEST_URL = f"{BASE}/Platform/Destiny2/Manifest/"

# Tables needed for Phase 1. Keys are Bungie's component names.
TABLES = [
    "DestinyLoreDefinition",            # the lore corpus
    "DestinyInventoryItemDefinition",   # flavour text (+ localized names)
    "DestinyRecordDefinition",          # record -> loreHash join (Phase 2 gating)
    "DestinyPresentationNodeDefinition",# book/page hierarchy for the wiki
    "DestinySeasonDefinition",          # best-effort release attribution
]


def _load_config() -> tuple[str, list[str]]:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("BUNGIE_API_KEY", "").strip()
    if not api_key or api_key == "put-your-key-here":
        sys.exit("BUNGIE_API_KEY missing. Copy .env.example to .env and set it.")
    langs = [
        l.strip().lower()
        for l in os.environ.get("LANGUAGES", "en").split(",")
        if l.strip()
    ]
    # English is the retrieval corpus: mandatory, always first.
    langs = ["en"] + [l for l in langs if l != "en"]
    return api_key, langs


def _get(session: requests.Session, url: str) -> dict:
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    api_key, langs = _load_config()
    session = requests.Session()
    session.headers["X-API-Key"] = api_key

    manifest = _get(session, MANIFEST_URL)
    if manifest.get("ErrorCode") != 1:
        sys.exit(f"Bungie API error: {manifest.get('Message')}")

    resp = manifest["Response"]
    version = resp["version"]
    components = resp["jsonWorldComponentContentPaths"]

    out_root = DATA / version
    print(f"Manifest version: {version}")
    print(f"Languages: {', '.join(langs)}  (en = retrieval corpus)")

    for lang in langs:
        if lang not in components:
            print(f"  !! language '{lang}' not offered by the API, skipping "
                  f"(available: {', '.join(sorted(components))})")
            continue
        lang_dir = out_root / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        for table in TABLES:
            path = components[lang].get(table)
            if not path:
                print(f"  !! {lang}/{table}: not in manifest, skipping")
                continue
            dest = lang_dir / f"{table}.json"
            if dest.exists():
                print(f"  ok {lang}/{table} (already downloaded)")
                continue
            print(f"  -> {lang}/{table} ...")
            data = _get(session, BASE + path)
            dest.write_text(json.dumps(data, ensure_ascii=False))
            print(f"     {len(data):>7} definitions")

    # Version pointer file (NOT a symlink: symlinks need admin rights on
    # Windows). Downstream steps read this to find the current snapshot.
    (DATA / "current.txt").write_text(version)
    print(f"Snapshot complete: {out_root}")


if __name__ == "__main__":
    main()
