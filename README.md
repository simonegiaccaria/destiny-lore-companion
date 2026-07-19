# Destiny Lore Companion

A self-hosted, spoiler-aware lore companion for Destiny 2. It ingests the
official Bungie manifest, builds a local hybrid-search index (semantic +
keyword), and serves a Ghost-flavoured chat plus browsable lore-book pages
on your own machine — reachable from your phone on your LAN while you play.

**This repository ships code, not data.** No Bungie-owned text is committed
here. Every user builds their own corpus locally with their own API key.
Do not open PRs that add extracted game text, transcripts, or prebuilt
indexes to the repo — they will be rejected (copyright).

---

## Requirements

- **Python 3.11+** — check with `python --version`
  (Windows: install from https://python.org, tick *"Add python.exe to
  PATH"*. The Microsoft Store Python sometimes ships broken `venv`; if
  venv creation fails, use the python.org installer.)
- **A free Bungie API key** — https://www.bungie.net/en/Application
  (create an app; no OAuth needed in Phase 1, only the API key)
- **~15 GB disk** total: Python packages incl. PyTorch (~3 GB), embedding
  model (~2 GB, auto-downloaded), local LLM via Ollama (~6 GB, manual),
  manifest snapshots (~200 MB per language)
- **Optional but recommended:** an NVIDIA GPU (8 GB+ VRAM) for fast
  embeddings and the local LLM. Everything also runs CPU-only, slowly.
- **Optional:** [Ollama](https://ollama.com) for the free local chat
  model. Skip it if you use a commercial API instead (see LLM backend).

---

## Setup — Linux / macOS

```bash
cd destiny-lore-companion

# 1. Virtual environment
python3 -m venv .venv
source .venv/bin/activate            # do this in every new terminal

# 2. Dependencies
pip install -r requirements.txt

# 3. GPU check (should print True on an NVIDIA machine)
python -c "import torch; print(torch.cuda.is_available())"

# 4. Config
cp .env.example .env
# edit .env: set BUNGIE_API_KEY, optionally LANGUAGES=en,it

# 5. Local LLM (skip if using a commercial API)
#    install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b-instruct-q5_K_M

# 6. Pipeline (in order)
python pipeline/download_manifest.py
python pipeline/extract_chunks.py
python pipeline/build_index.py

# 7. Serve
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

## Setup — Windows (PowerShell)

```powershell
cd destiny-lore-companion

# 1. Virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1           # do this in every new terminal
# If PowerShell refuses ("execution policy"), run once:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# then retry. From classic cmd.exe use: .venv\Scripts\activate.bat

# 2. Dependencies
pip install -r requirements.txt

# 3. GPU check (should print True with an NVIDIA card)
python -c "import torch; print(torch.cuda.is_available())"
# If it prints False, install the CUDA build of PyTorch
# (pick the exact command for your setup at https://pytorch.org):
#   pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

# 4. Config
copy .env.example .env
# edit .env: set BUNGIE_API_KEY, optionally LANGUAGES=en,it

# 5. Local LLM (skip if using a commercial API)
#    install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b-instruct-q5_K_M

# 6. Pipeline (in order)
python pipeline\download_manifest.py
python pipeline\extract_chunks.py
python pipeline\build_index.py

# 7. Serve
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

**No activation needed (both OSes):** you can always call the venv's own
Python directly and skip step "activate" entirely:
`.venv\Scripts\python pipeline\download_manifest.py` (Windows) or
`.venv/bin/python pipeline/download_manifest.py` (Linux/macOS).

## Reaching it from your phone (same Wi-Fi)

1. Find your PC's LAN address:
   - Windows: `ipconfig` → "IPv4 Address" (e.g. `192.168.1.42`)
   - Linux: `ip addr` · macOS: `ipconfig getifaddr en0`
2. Open `http://<that-ip>:8000` in the phone browser.
3. **Windows firewall:** the first `uvicorn` launch pops a firewall
   dialog — allow access on *private* networks, or the phone will never
   connect. If you missed the dialog: Windows Security → Firewall →
   Allow an app → allow Python on private networks.

## LLM backend

Default is a local model via Ollama (free, runs fine on a 12 GB GPU). To
switch to a commercial API, edit `.env` (`LLM_BASE_URL`, `LLM_MODEL`,
`LLM_API_KEY`). The client code is identical for both — deliberate, so
quality can be compared empirically on `eval/reference_questions.md`.
The model in `.env.example` is a reasonable placeholder, not a strong
recommendation: check what currently runs best in the 8–14B class for
your GPU and change that one line.

---

## Architecture (Phase 1)

```
[ingest]  Bungie API -> manifest JSON -> pipeline/extract_chunks.py -> data/chunks/
[index]   data/chunks/ -> local embeddings -> Chroma (data/index/) + SQLite FTS5
[serve]   FastAPI: /chat (streaming) + lore-book wiki pages
[llm]     one OpenAI-compatible client -> Ollama (local) or a commercial API
```

Design rules NOT up for renegotiation without reading `docs/decisions.md`:

1. **The retrieval corpus is English only.** Extra languages are a
   presentation/lookup layer (official localized names for the wiki and
   the Ghost's terminology), never a second indexed corpus.
2. **Chunk metadata is the load-bearing wall.** `unlock_record_hash`,
   `release`, `vaulted`, `spoils` exist so Phase 2 spoiler-gating and
   retcon handling work without re-processing the corpus.
3. **Everything downloaded is preserved on disk.** The game is in
   maintenance mode; if the API ever goes dark, this tool keeps working
   from its last snapshot.

## Phase 1 exit criteria

- [ ] Pipeline runs end-to-end from a fresh clone with a virgin API key
- [ ] The 30 reference questions in `eval/` get acceptable answers
      (score them with `eval/SCORING.md`)
- [ ] Lore books are browsable from a phone on the LAN

## Known limitations (Phase 1)

- No spoiler-gating yet (Phase 2: OAuth + account-wide record union).
- `config/releases.yaml` release/vault flags are hand-curated — verify
  them; they encode editorial judgment, not API data.
- Release attribution of individual chunks is best-effort (`unknown`
  where the manifest gives no signal); improves in Phase 3 with Ishtar's
  per-document spoiler dates.
- Dialogue/cutscene transcripts are Phase 3 (Ishtar Collective) and exist
  in English only regardless of your LANGUAGES setting.
