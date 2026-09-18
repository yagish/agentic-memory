# CLAUDE.md — agentic-memory

## What this project is

A local-first persistent memory system for Claude Code (and pi). It captures every conversation, extracts structured memory via a local Ollama model in the background, and injects relevant context at the start of each new prompt.

## Running and testing

```bash
# Install dependencies
python3 -m pip install --user sentence-transformers fastapi uvicorn pydantic psutil setproctitle debugpy
ollama pull qwen2.5:7b

# Install hooks and launchd services
./install.sh

# Start services manually (launchd handles this after install)
python3 memory/servers/ingest_server.py    # port 7747
python3 memory/servers/dashboard_server.py # port 7748
python3 -m memory.daemon                   # background extractor
python3 -m memory.daemon --once            # one extraction pass

# Run tests
pytest -q
```

Tests suppress file log writes (`MEMORY_DISABLE_FILE_LOGS=1` is set by the test suite via conftest). Never create real `~/.memory/` files in tests.

## Project structure

```
integrations/
  claude/save_hook.py      Stop hook — parses JSONL transcript, saves to DB
  claude/wake_up.py        UserPromptSubmit hook — recalls memory, injects or answers
  pi/extension.ts          Pi agent extension (TypeScript, spawns adapter.py)
  pi/adapter.py            Python bridge called by the Pi extension
  common.py                Shared save/retrieve policy — THE seam all adapters call

memory/
  servers/
    ingest_server.py       FastAPI, port 7747: POST /ingest, POST /recall, GET /status
    dashboard_server.py    FastAPI, port 7748: /memory/* and /ops/* API routes + dashboard.html
    client.py              Stdlib HTTP client (no requests dependency)
    ingest_pipeline.py     Session upsert (used by ingest_server and save_hook)
  daemon/
    __init__.py            Main loop + five extractor functions + run() / process_one()
    _core.py               Constants (CPU_THRESHOLD=70, POLL_INTERVAL=300s, TTLs)
    compaction.py          Transcript summarisation helper (currently bypassed in pipeline)
    pruning.py             TTL pruning + embedding warm-up
  retrieval/
    _fetch.py              Per-type vector/DB queries, all wrapped in try/except
    _rank.py               _row_score: similarity + 0.10*overlap + 0.05*recency
    _intent.py             Embedding-based "resume" intent detection
    _format.py             Format ranked rows into injection text
    _models.py             WakeUpContext, RetrievalWarning, thresholds/limits
  facts/                   extract, normalize, render, repository, text (semantic content)
  episodic/                extract, repository
  procedural/              extract, backfill, repository
  working_memory/          extract, repository
  session/                 extract, repository
  llm/
    inference.py           generate_text() + embed_text() — Ollama seam
    ollama.py              start_ollama_if_needed() / stop_ollama()
  vectors/
    _model.py              all-MiniLM-L6-v2 via sentence-transformers, 384-dim, offline
    _ops.py                cosine_distance() and pack_vector() in pure Python
  db/
    schema.py              SQLite schema, bootstrap_db(), open_db(), migrations
    *.py                   Per-table helpers (facts, episodic, procedural, sessions, …)
  utils/
    logger.py              activity_log() and error_log() — structured JSON-lines to activity.log
    debug.py               debugpy attach helper

cli.py                     Nine-command CLI entry point
dashboard.html             Single-page dashboard (vanilla JS, no build step)
install.sh                 Idempotent installer (hooks + launchd + Pi extension)
scripts/                   One-off migration scripts
tests/                     pytest suite — fixture-based extraction, contract, integration tests
```

## Architecture — how data flows

**Write path (non-blocking):**
1. Claude session ends → `save_hook.py` reads the JSONL transcript from disk.
2. Visible turns (user text + final assistant replies) are extracted; tool calls and intermediate events are dropped.
3. Session is upserted into `sessions` table with `daemon_processed_at = NULL`.

**Extraction path (background):**
1. Daemon polls every 5 minutes (30 minutes when idle). Before each cycle it checks CPU; if above 70% it sleeps 60 s and retries.
2. For each unprocessed session, it builds a session text and runs five extractors **sequentially**:
   `facts → working_memory → session_memory → episodic → procedural`
3. Each extractor calls Ollama (`qwen2.5:7b` by default), parses the JSON response, and writes the result plus a 384-dim embedding to the relevant table.
4. Session is marked `daemon_processed_at = now`. If a session is re-ingested, `daemon_processed_at` is reset to NULL and extraction runs again.
5. Facts make **two Ollama calls**: one to extract the structured triple, one to generate `semantic_content` for embedding-quality text.
6. After processing, the daemon prunes facts older than 180 days and episodic memories older than 90 days.

**Retrieval path (per prompt):**
1. `wake_up.py` calls `POST /recall` on the ingest server (port 7747).
2. The ingest server embeds the prompt with `all-MiniLM-L6-v2` (loaded once at startup; offline mode enforced).
3. Vector search runs over all typed tables (linear cosine scan in Python — no vector index).
4. Results are ranked by `similarity + 0.10*lexical_overlap + 0.05*recency_decay` and capped per type.
5. Decision: if only facts match → render a direct answer and block the prompt. If any contextual memory exists → format as injection and prepend to the prompt via `additionalContext`. Otherwise pass through unchanged.

## Key invariants

- **`integrations/common.py` is the only seam** between adapters and the DB/retrieval stack. Claude hooks and Pi adapter must call it, not re-implement save/retrieve logic.
- **Retrieval never calls Ollama.** One embedding + cosine search + string formatting only. If you see an Ollama call in the retrieval path, that is a bug.
- **The daemon is the only writer to typed memory tables.** The ingest server and hooks write only to `sessions`. Typed tables (facts, episodic, …) are written exclusively by the daemon.
- **All retrieval errors are non-fatal.** Every fetch function in `retrieval/_fetch.py` catches exceptions and appends a `RetrievalWarning`. The wake-up hook must always return something (pass-through at minimum), never crash.
- **Embeddings are stored as little-endian float32 BLOB** (packed by `memory/vectors/_ops.py:pack_vector`). The cosine function expects this format; do not change the packing without updating both sides.

## Adding a new memory type

1. Create `memory/<type>/extractor.py` with an `extract_<type>_from_session_text()` function.
2. Create `memory/<type>/repository.py` with `save_extracted_<type>()` and `retrieve_<type>_memories()`.
3. Add a table in `memory/db/schema.py` (add migration to `_MIGRATIONS` list).
4. Wire a query helper in the appropriate `memory/db/*.py` module.
5. Add the extractor to the `extractors` list in `memory/daemon/__init__.py:_process_session`.
6. Add a `_retrieve_<type>` call in `memory/retrieval/_fetch.py` and add the result to `WakeUpContext` in `_models.py`.
7. Wire formatting in `memory/retrieval/_format.py` and `_build_injection`.
8. Add contract tests in `tests/test_contracts_<type>.py`.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `MEMORY_OLLAMA_MODEL` | `qwen2.5:7b` | LLM for all extraction |
| `MEMORY_INGEST_PORT` | `7747` | Ingest server port |
| `MEMORY_QUERY_PORT` | `7748` | Dashboard server port |
| `MEMORY_FACT_TTL_DAYS` | `180` | Fact pruning age |
| `MEMORY_EPISODIC_TTL_DAYS` | `90` | Episodic pruning age |
| `MEMORY_EXTRACTION_TIMEOUT_SECONDS` | `600` | Per-extractor Ollama timeout |
| `MEMORY_OLLAMA_RETRIES` | `3` | Ollama retry count (exponential backoff) |
| `MEMORY_COMPACT_INPUT_CHARS` | `40000` | Max transcript chars fed to LLM |
| `MEMORY_DISABLE_FILE_LOGS` | *(unset)* | Set `1` to suppress all file log writes |

## Dashboard API route prefixes

The dashboard server uses two FastAPI routers:
- `/memory/*` — data endpoints (sessions, facts, episodic, procedural, working memory, session memory, logs)
- `/ops/*` — operational endpoints (process-one, activity stats, performance stats, Ollama status/start)

The root `/` serves `dashboard.html`. Port 7748.
