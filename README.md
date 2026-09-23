# Agentic Memory

A local-first persistent memory system for AI agents. It records every conversation, extracts five complementary memory types via a local LLM, and injects relevant context at the start of each new prompt — without blocking any agent session.

## Documentation map

- [`docs/README.md`](docs/README.md) — repository documentation index
- [`docs/generated/README.md`](docs/generated/README.md) — full generated system docs
- [`CLAUDE.md`](CLAUDE.md) — contributor and coding-agent guide for this repo
- [`AGENTS.md`](AGENTS.md) — symlink to `CLAUDE.md` for cross-agent compatibility

## How it works

Every Claude Code session is captured by a **Stop hook** that writes the transcript to SQLite. A **background daemon** then picks up unprocessed sessions and runs five extractors sequentially against each one, calling a local Ollama model to produce structured memory. When a new prompt arrives, a **UserPromptSubmit hook** calls the recall server, embeds the prompt, does a cosine-similarity search over all stored memory, and either answers directly (for simple fact lookups) or prepends retrieved context to the prompt.

```
Claude session ends
       │
       ▼
save_hook.py  ──── writes turns ──────► sessions table (SQLite)
                                               │
                              daemon polls every 5 min
                                               │
                                               ▼
                              five sequential extractors
                            (facts → working memory → session memory
                              → episodic → procedural)
                            each calls Ollama qwen2.5:7b
                                               │
                              results written to typed tables
                                               │
Next prompt arrives                            │
       │                                       │
       ▼                                       │
wake_up.py  ──── POST /recall ──────► ingest_server
                                      embeds prompt (all-MiniLM-L6-v2)
                                      cosine search over all tables
                                      rank & budget-fit results
                                               │
                              ┌────────────────┴───────────────────┐
                         fact-only hit                      contextual memory
                              │                                     │
                         answer directly                   inject as context
                         (block prompt)                    (additionalContext)
```

## Memory types

| Type | What it stores | Table |
|---|---|---|
| **Facts** | Durable key-value triples: `entity.attribute = value` | `facts` |
| **Episodic** | Narrative summaries: title, abstract, decisions, outcomes, follow-ups | `episodic_memory` |
| **Procedural** | How-to patterns with trigger phrases and step-by-step instructions | `procedural_memory` |
| **Working memory** | Point-in-time snapshot: current goal, focus, next step, status | `working_memory` |
| **Session memory** | Compacted handoff: `left_off_at` + `next_steps` for session continuity | `session_memory` |

## Architecture

### Components

```
integrations/
  claude/save_hook.py      — Claude Code Stop hook (writes session to DB)
  claude/wake_up.py        — Claude Code UserPromptSubmit hook (retrieves memory)
  pi/extension.ts          — Pi agent extension (TypeScript)
  pi/adapter.py            — Python bridge for the Pi extension
  common.py                — Shared save/retrieve policy (seam for all adapters)

memory/
  servers/
    ingest_server.py       — FastAPI server, port 7747 (POST /ingest, POST /recall, GET /status)
    dashboard_server.py    — FastAPI server, port 7748 (dashboard API + static UI)
    client.py              — Stdlib HTTP client for the ingest server
    ingest_pipeline.py     — Session upsert logic
  daemon/
    __init__.py            — Main loop, five extractors, run() and process_one()
    _core.py               — Constants: CPU_THRESHOLD=70%, POLL_INTERVAL=5min, TTLs
    compaction.py          — Optional transcript summarisation before LLM extraction
    pruning.py             — TTL-based pruning for facts (180d) and episodic (90d)
  retrieval/
    _fetch.py              — Per-type DB queries, graceful degradation
    _rank.py               — Composite score: similarity + lexical overlap + recency decay
    _intent.py             — "Resume" intent detection for recent-episode fallback
    _format.py             — Format retrieved rows into injection text
    _models.py             — WakeUpContext, RetrievalWarning, similarity thresholds
  facts/ episodic/ procedural/ working_memory/ session/
                           — Per-type extractor + repository modules
  llm/
    inference.py           — Ollama text generation seam (retry with exponential backoff)
    ollama.py              — Ollama process lifecycle (start/stop)
  vectors/
    _model.py              — sentence-transformers all-MiniLM-L6-v2, 384-dim, offline
    _ops.py                — Cosine distance in pure Python (no sqlite-vec dependency)
  db/
    schema.py              — SQLite schema + bootstrap + forward-only migrations
    *.py                   — Per-table query helpers

cli.py                     — Nine-command CLI (bootstrap, status, search, semantic, …)
dashboard.html             — Single-page dashboard UI
install.sh                 — Wires hooks, launchd plists, Pi extension
```

### Design decisions

- **Async write path** — `save_hook.py` writes the transcript immediately (cheap SQLite write); the daemon does all LLM extraction after the session ends. Claude sessions are never blocked.
- **Sequential extraction** — the daemon runs all five extractors one after the other in a plain `for` loop. Each extractor is independently fault-tolerant; an error in one does not skip the rest.
- **CPU gate** — the daemon checks `psutil.cpu_percent` before each polling cycle and skips when CPU usage exceeds 70%, preventing background extraction from interfering with active work.
- **Model-free retrieval** — wake-up injection requires no LLM call at runtime: one embedding, a vector search, and string formatting. All "intelligence" is pre-baked into the stored extraction.
- **Composite ranking** — `_row_score` blends semantic similarity (dominant), lexical token overlap (small bonus), and exponential recency decay (tie-breaker). A memory from three days ago at 0.61 similarity beats a six-month-old one at 0.62.
- **Graceful degradation** — every retrieval step is wrapped in `try/except` producing a `RetrievalWarning` instead of a crash. The "resume" intent fallback retrieves recent episodes when semantic search returns nothing.
- **Linear vector scan** — embeddings are stored as BLOB in SQLite and compared in Python with a hand-rolled cosine function. There is no vector index; retrieval scans all rows at query time.
- **Two Ollama calls per fact** — each fact runs one call to extract the structured triple and a second call to generate the `semantic_content` text used for embedding-based search.

### Retrieval thresholds (hardcoded)

| Memory type | Min similarity | Fetch limit | Result cap |
|---|---|---|---|
| Facts | 0.38 | 5 | 3 |
| Episodic | 0.58 | 8 | 2 |
| Procedural | 0.58 | 4 | 2 |
| Session memory | 0.60 | 4 | 2 |

## Storage

Database at `~/.memory/memory.db` (SQLite, WAL mode).

**Active tables:**
- `sessions` — raw transcripts + `daemon_processed_at` flag
- `facts` — `entity` / `attribute` / `value` / `semantic_content` / `embedding`
- `episodic_memory` — `title` / `abstract` / `happened_at` / `details` / `embedding`
- `procedural_memory` — `title` / `summary` / `updated_at` / `details` / `embedding`
- `working_memory` — `current_goal` / `current_focus` / `next_step` / `status` / `embedding`
- `session_memory` — `title` / `summary` / `left_off_at` / `next_steps` / `embedding`

**TTL pruning** (daemon, on each cycle):
- Facts: 180 days (env: `MEMORY_FACT_TTL_DAYS`)
- Episodic: 90 days (env: `MEMORY_EPISODIC_TTL_DAYS`)
- Procedural, working memory, session memory: no TTL pruning

Log files at `~/.memory/`: `daemon.log`, `wake_up.log`, `save_hook.log`, `facts.log`, `episodic.log`, `procedural.log`, `working_memory.log`, `session_memory.log`, `ingest.log`, `activity.log`.

## Quick start

```bash
python3 -m pip install --user sentence-transformers fastapi uvicorn pydantic psutil setproctitle debugpy
ollama pull qwen2.5:7b
./install.sh
```

Start services manually if needed:

```bash
python3 memory/servers/ingest_server.py    # recall server (port 7747)
python3 memory/servers/dashboard_server.py # dashboard (port 7748)
python3 -m memory.daemon                   # extraction daemon
python3 -m memory.daemon --once            # one extraction pass and exit
```

Or use the helper scripts:

```bash
./scripts/start-memory.sh
./scripts/restart-memory.sh
./scripts/stop-memory.sh
```

## Integrations

### Claude Code

`install.sh` wires Claude's hooks to:
- `integrations/claude/save_hook.py` (Stop event)
- `integrations/claude/wake_up.py` (UserPromptSubmit event)

### Pi

`install.sh` installs a global Pi wrapper at `~/.pi/agent/extensions/agentic-memory.ts`. This repo also includes a project-local auto-discovered wrapper at `.pi/extensions/agentic-memory.ts`, so inside this project you can run `pi` directly.

To verify the extension loaded:
```text
/memory-status
```

The Pi extension calls `integrations/pi/adapter.py`, which uses the shared save/retrieval helpers in `integrations/common.py`.

## CLI

```bash
python3 cli.py bootstrap          # create DB schema
python3 cli.py status             # session/fact/episode/procedure counts
python3 cli.py search "query"     # full-text search over transcripts
python3 cli.py semantic "query"   # cosine-similarity search
python3 cli.py get-session <id>   # print one session as JSON
python3 cli.py tail               # 10 most recent sessions
python3 cli.py add-fact user name Yash --tag identity
python3 cli.py delete-fact <fact_id>
python3 cli.py dashboard          # open dashboard in browser
```

## Tests

```bash
pytest -q
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_OLLAMA_MODEL` | `qwen2.5:7b` | LLM used for all extraction |
| `MEMORY_INGEST_PORT` | `7747` | Ingest/recall server port |
| `MEMORY_QUERY_PORT` | `7748` | Dashboard server port |
| `MEMORY_FACT_TTL_DAYS` | `180` | Fact pruning age |
| `MEMORY_EPISODIC_TTL_DAYS` | `90` | Episodic pruning age |
| `MEMORY_EXTRACTION_TIMEOUT_SECONDS` | `600` | Per-extractor Ollama timeout |
| `MEMORY_OLLAMA_RETRIES` | `3` | Ollama retry count |
| `MEMORY_COMPACT_INPUT_CHARS` | `40000` | Max transcript chars sent to LLM |
| `MEMORY_COMPACT_OUTPUT_CHARS` | `5000` | Target compaction output length |
| `MEMORY_INGEST_CORS_ORIGINS` | `copilot.microsoft.com,github.com` | Allowed CORS origins |
| `MEMORY_DISABLE_FILE_LOGS` | *(unset)* | Set to `1` to suppress file log writes |
| `MEMORY_AGENT_NAME` | `claude` | Agent tag written to session records |

## Migration scripts

For existing databases, run the one-off migration before starting the updated app:

```bash
python3 scripts/migrate_facts_semantic_text.py --db ~/.memory/memory.db
```

To backfill procedural memory for sessions that predate procedural extraction:

```bash
python3 scripts/backfill_procedural_memory.py --db ~/.memory/memory.db --dry-run
python3 scripts/backfill_procedural_memory.py --db ~/.memory/memory.db
python3 scripts/backfill_procedural_memory.py --db ~/.memory/memory.db --force --limit 25
```

## Generated documentation

Full technical documentation (endpoints, flows, business rules, troubleshooting) lives in [`docs/generated/README.md`](docs/generated/README.md).

For a higher-level document index, including ADRs and planning notes, start at [`docs/README.md`](docs/README.md).
