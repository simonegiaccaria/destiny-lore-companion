"""Hybrid retrieval: semantic (Chroma) + keyword (FTS5), fused with RRF,
then balanced by source type.

Why the quota (see docs/decisions.md #9): flavour texts outnumber lore
entries by an order of magnitude and are ~10 words long, so both semantic
and keyword search favour them for short queries. Unconstrained, the top-k
is all one-line item quotes and the model has nothing real to ground on.
Pools are therefore filled per source_type, not by global rank.

Phase 2 note: spoiler-gating is a metadata filter here, not a new system.
`retrieve(..., unlocked_records=set_of_hashes)` implements the policy
'chunk is visible if vaulted OR its unlock record is in the set'.
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

# How many hits each source type may contribute to the final k.
# Tune here, not in the caller: this is a corpus property, not a per-query one.
DEFAULT_QUOTAS = {"lore_entry": 6, "flavor_text": 2}


@dataclass
class Hit:
    id: str
    text: str
    meta: dict
    score: float = 0.0          # RRF score: rank-based, NOT a quality measure
    sim: float | None = None    # raw cosine similarity, for diagnostics
    spoils: list[str] = field(default_factory=list)


class Retriever:
    def __init__(self) -> None:
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.model = SentenceTransformer(model_name)
        client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
        self.col = client.get_collection("lore")
        self.fts_path = INDEX_DIR / "fts.sqlite"

    # -- semantic ---------------------------------------------------------
    def _semantic(self, query: str, k: int) -> list[tuple[str, str, dict, float]]:
        emb = self.model.encode([query], normalize_embeddings=True)
        res = self.col.query(query_embeddings=emb.tolist(), n_results=k,
                             include=["documents", "metadatas", "distances"])
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            out.append((cid, doc, meta, 1.0 - float(dist)))  # cosine similarity
        return out

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

    # -- fusion + quota + policy filter ------------------------------------
    def retrieve(self, query: str, k: int = 8,
                 unlocked_records: set[int] | None = None,
                 quotas: dict[str, int] | None = None) -> list[Hit]:
        quotas = dict(quotas or DEFAULT_QUOTAS)
        # Pool must be deep enough that the minority source type is present
        # at all — the whole point is that it loses on global rank.
        pool = max(k * 12, 120)
        ranked: dict[str, Hit] = {}

        for rank, (cid, doc, meta, sim) in enumerate(self._semantic(query, pool)):
            h = ranked.setdefault(cid, Hit(cid, doc, meta))
            h.score += 1.0 / (RRF_K + rank + 1)
            h.sim = sim if h.sim is None else max(h.sim, sim)
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

        # Fill per source type, in rank order within each type.
        selected: list[Hit] = []
        remaining = dict(quotas)
        leftovers: list[Hit] = []
        for h in hits:
            st = h.meta.get("source_type", "")
            if remaining.get(st, 0) > 0:
                remaining[st] -= 1
                selected.append(h)
            else:
                leftovers.append(h)
            if len(selected) >= k:
                break

        # If a quota could not be filled (e.g. no lore entry matches at all),
        # top up with the best remaining hits rather than returning fewer.
        for h in leftovers:
            if len(selected) >= k:
                break
            selected.append(h)

        selected.sort(key=lambda h: h.score, reverse=True)
        for h in selected:
            h.spoils = [s for s in (h.meta.get("spoils") or "").split(",") if s]
        return selected[:k]