"""FastAPI server: Ghost chat (streaming) + lore-book wiki.

Run:  uvicorn app.server:app --host 0.0.0.0 --port 8000
Then open http://<lan-ip>:8000 from your phone.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from app.llm import answer_stream            # noqa: E402
from app.retrieval import Retriever          # noqa: E402

app = FastAPI(title="Destiny Lore Companion")
retriever: Retriever | None = None
BOOKS = json.loads((ROOT / "data" / "chunks" / "books.json").read_text()) \
    if (ROOT / "data" / "chunks" / "books.json").exists() else {}


@app.on_event("startup")
def _load() -> None:
    global retriever
    retriever = Retriever()


class ChatIn(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/chat")
def chat(payload: ChatIn):
    hits = retriever.retrieve(payload.question, k=8, unlocked_records=None)
    return StreamingResponse(
        answer_stream(payload.question, hits, payload.history),
        media_type="text/plain; charset=utf-8")


# ---------------------------------------------------------------- wiki ----

def _chunk_text(chunk_id: str) -> str:
    db = sqlite3.connect(ROOT / "data" / "index" / "fts.sqlite")
    try:
        rows = db.execute(
            "SELECT text FROM lore_fts WHERE id = ? OR id LIKE ? ORDER BY id",
            (chunk_id, chunk_id + "#p%")).fetchall()
    finally:
        db.close()
    return "\n".join(r[0] for r in rows)


@app.get("/api/books")
def api_books():
    return {"books": sorted(BOOKS.keys())}


@app.get("/api/book/{name}")
def api_book(name: str):
    pages = BOOKS.get(name, [])
    return {"name": name,
            "pages": [{"title": p["title"], "text": _chunk_text(p["id"])}
                      for p in pages]}


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "app" / "static" / "index.html").read_text()
