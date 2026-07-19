"""Hybrid retrieval: semantic (Chroma) + keyword (FTS5), fused with RRF.

Phase 2 note: spoiler-gating is a metadata filter here, not a new system.
`retrieve(..., unlocked_records=set_of_hashes)` already implements the
policy 'chunk is visible if vaulted OR its unlock record is in the set'.
In Phase 1 pass unlocked_records=None -> no gating.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"

RRF_K = 60  # standard reciprocal-rank-fusion constant


@dataclass
class Hit:
    id: str
    text: str
    meta: dict
    score: float = 0.0
    spoils: list[str] = field(default_factory=list)


class Retriever:
    def __init__(self) -> None:
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.model = SentenceTransformer(model_name)
        client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
        self.col = client.get_collection("lore")
        self.fts_path = INDEX_DIR / "fts.sqlite"

    # -- semantic ---------------------------------------------------------
    def _semantic(self, query: str, k: int) -> list[tuple[str, str, dict]]:
        emb = self.model.encode([query], normalize_embeddings=True)
        res = self.col.query(query_embeddings=emb.tolist(), n_results=k,
                             include=["documents", "metadatas"])
        return list(zip(res["ids"][0], res["documents"][0],
                        res["metadatas"][0]))

    # -- keyword ----------------------------------------------------------
    def _keyword(self, query: str, k: int) -> list[tuple[str, str, dict]]:
        # Quote each term: user queries are natural language, FTS5 syntax
        # characters in them must not explode the MATCH expression.
        terms = [t for t in query.split() if len(t) > 2]
        if not terms:
            return []
        match = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
        db = sqlite3.connect(self.fts_path)
        try:
            rows = db.execute(
                "SELECT id, title, book, text FROM lore_fts "
                "WHERE lore_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, k)).fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            db.close()
        out = []
        for cid, title, book, text in rows:
            meta = self._meta_of(cid)
            doc = f"{book} — {title}\n{text}" if book else f"{title}\n{text}"
            out.append((cid, doc, meta))
        return out

    def _meta_of(self, cid: str) -> dict:
        res = self.col.get(ids=[cid], include=["metadatas"])
        return res["metadatas"][0] if res["ids"] else {}

    # -- fusion + policy filter --------------------------------------------
    def retrieve(self, query: str, k: int = 8,
                 unlocked_records: set[int] | None = None) -> list[Hit]:
        pool = max(k * 4, 24)
        ranked: dict[str, Hit] = {}

        for rank, (cid, doc, meta) in enumerate(self._semantic(query, pool)):
            h = ranked.setdefault(cid, Hit(cid, doc, meta))
            h.score += 1.0 / (RRF_K + rank + 1)
        for rank, (cid, doc, meta) in enumerate(self._keyword(query, pool)):
            h = ranked.setdefault(cid, Hit(cid, doc, meta))
            h.score += 1.0 / (RRF_K + rank + 1)

        hits = sorted(ranked.values(), key=lambda h: h.score, reverse=True)

        # Spoiler policy (Phase 2): visible if vaulted OR record unlocked.
        if unlocked_records is not None:
            hits = [h for h in hits
                    if h.meta.get("vaulted") == "yes"
                    or int(h.meta.get("unlock_record_hash") or 0)
                    in unlocked_records
                    or int(h.meta.get("unlock_record_hash") or 0) == 0]

        for h in hits:
            h.spoils = [s for s in (h.meta.get("spoils") or "").split(",") if s]
        return hits[:k]
