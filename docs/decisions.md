# Decision log (why the design is what it is)

1. **English-only retrieval corpus; extra languages are lookup layers.**
   Indexing translations duplicates chunks semantically, poisons retrieval,
   and reintroduces adaptation losses. Localized manifests feed the wiki
   renderer and the Ghost's official terminology only.
2. **Chunk metadata schema is frozen** (`release`, `release_date`,
   `vaulted`, `spoils`, `unlock_record_hash`). Phase 2 gating and retcon
   handling are metadata filters — dropping fields forces full reprocessing.
3. **Vaulted = known to everyone** (project policy). Vaulted lore that
   foreshadows still-playable stories carries `spoils=[...]` and triggers a
   pre-answer warning instead of being hidden.
4. **Spoiler gating (Phase 2) is account-wide**: union of record state
   across all characters via the profile endpoint.
5. **Hybrid retrieval from day one** (semantic + FTS5): invented proper
   nouns are keyword signals that pure embeddings dilute.
6. **LLM backend is a config value**, not an architecture. Local (Ollama)
   vs API is decided empirically on eval/reference_questions.md.
7. **Repo ships code, never data** — each user builds their corpus with
   their own Bungie API key (copyright line we do not cross).
8. **Every download is snapshotted** — the game is in maintenance mode
   (final update 2026-06-09); the API's longevity is not guaranteed.
