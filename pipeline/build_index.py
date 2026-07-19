"""Chunks -> local embeddings -> Chroma + SQLite FTS5 (hybrid retrieval).

Two indexes, by design (see docs/decisions.md):
 - Chroma (embedded, persistent): multilingual semantic search. BGE-M3 maps
   an Italian query and English lore into the same space — this is what
   makes "ask in Italian, corpus in English" work.
 - SQLite FTS5: exact keyword match. Destiny is full of invented proper
   nouns ("Ahamkara", "aiat", "Oryx") that pure semantic search dilutes.

Long lore entries are split with overlap; every split carries the parent's
full metadata so spoiler-gating and retcon handling never break.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"

MAX_WORDS = 700       # split threshold (safe for BGE-M3's window)
OVERLAP_WORDS = 80


def split_long(text: str) -> list[str]:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    parts, start = [], 0
    while start < len(words):
        parts.append(" ".join(words[start:start + MAX_WORDS]))
        start += MAX_WORDS - OVERLAP_WORDS
    return parts


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not CHUNKS.exists():
        raise SystemExit("No chunks. Run pipeline/extract_chunks.py first.")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    print(f"Loading embedding model {model_name} (GPU if available)...")
    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
    try:
        client.delete_collection("lore")
    except Exception:
        pass
    col = client.create_collection("lore", metadata={"hnsw:space": "cosine"})

    fts = sqlite3.connect(INDEX_DIR / "fts.sqlite")
    fts.execute("DROP TABLE IF EXISTS lore_fts")
    fts.execute("""CREATE VIRTUAL TABLE lore_fts USING fts5
                   (id UNINDEXED, title, book, text)""")

    ids, docs, metas, fts_rows = [], [], [], []
    total = 0
    with CHUNKS.open() as fh:
        for line in fh:
            c = json.loads(line)
            for i, part in enumerate(split_long(c["text"])):
                cid = c["id"] if i == 0 else f"{c['id']}#p{i}"
                ids.append(cid)
                # Prefix title+book: cheap context that measurably helps
                # retrieval of fragments quoted out of their book.
                docs.append(f"{c['book']} — {c['title']}\n{part}"
                            if c["book"] else f"{c['title']}\n{part}")
                metas.append({
                    "title": c["title"],
                    "book": c["book"],
                    "source_type": c["source_type"],
                    "release": c["release"],
                    "release_date": c["release_date"] or "",
                    # Chroma metadata: no None, no lists -> encode compactly
                    "vaulted": {True: "yes", False: "no"}.get(c["vaulted"], "unknown"),
                    "unlock_record_hash": c["unlock_record_hash"] or 0,
                    "spoils": ",".join(c["spoils"]),
                })
                fts_rows.append((cid, c["title"], c["book"], part))
                total += 1

    print(f"Embedding {total} chunks...")
    B = 128
    for s in range(0, total, B):
        emb = model.encode(docs[s:s + B], normalize_embeddings=True,
                           show_progress_bar=False)
        col.add(ids=ids[s:s + B], documents=docs[s:s + B],
                metadatas=metas[s:s + B], embeddings=emb.tolist())
        if (s // B) % 10 == 0:
            print(f"  {min(s + B, total)}/{total}")

    fts.executemany("INSERT INTO lore_fts VALUES (?,?,?,?)", fts_rows)
    fts.commit()
    fts.close()
    print(f"Done. Semantic + FTS indexes in {INDEX_DIR}")


if __name__ == "__main__":
    main()
