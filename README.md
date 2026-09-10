# Agentic Memory

A simplified persistent memory system with separate integrations for Claude Code and pi.

Current runtime scope is intentionally small:
- store full **sessions**
- extract durable **facts**
- extract **episodes** (episodic memory)
- extract durable **procedures** (procedural memory)
- extract session-scoped **working memory** snapshots
- extract compacted **session memory** handoffs
- retrieve **working memory + session memory + facts + episodes + procedures** during wake-up

Everything else from the older design was removed.

## Design Strengths

- **Async extraction pipeline** — `save_hook.py` writes the transcript immediately (cheap SQLite write); the daemon does all LLM extraction after the session ends. Claude sessions are never blocked.
- **Adaptive token budget** — the intent classifier (`resume` / `quick` / `task`) adjusts context size: 200 tokens for trivial questions, 500 for normal work, 1 500 for "where did we leave off." Exemplar embeddings are cached after the first call so subsequent prompts pay zero extra embedding cost.
- **Composite retrieval ranking** — `_row_score` blends semantic similarity (dominant), lexical token overlap (small bonus), and exponential recency decay (tie-breaker). A memory from three days ago at similarity 0.61 beats a six-month-old one at 0.62.
- **Model-free retrieval path** — wake-up injection requires no LLM call at runtime: one embedding, a vector search, and string formatting. The only "intelligence" is pre-baked into the stored extraction.
- **Parallel session processing** — `ThreadPoolExecutor(max_workers=5)` runs all five extractors concurrently per session. Since each extractor blocks on an Ollama HTTP response, the GIL releases and they run in true parallel; wall-clock time drops from ~5× to ~1× the slowest extractor.
- **CPU gate** — the daemon checks `psutil.cpu_percent` before each polling cycle and skips processing when usage exceeds 70 %, preventing background extraction from interfering with active work.
- **Five complementary memory types** — facts (durable key-value), episodic (what happened), procedural (how-to patterns with trigger phrases), working memory (current goal/tasks), and session memory (handoff: `left_off_at` + `next_steps`). Each type covers a different recovery dimension.
- **Graceful degradation** — every retrieval step is wrapped in `try/except` producing a `RetrievalWarning` instead of a crash. The "resume" intent fallback retrieves recent episodes when semantic search returns nothing.
- **Good test coverage** — fixture-based extraction tests, contract tests for each memory type, semantic search tests, and a dedicated token-economics test.

## Components

### 1. Claude integration
`integrations/claude/save_hook.py`
- Claude Stop hook adapter
- reads Claude Code transcript JSONL
- stores the session in SQLite

`integrations/claude/wake_up.py`
- Claude UserPromptSubmit hook adapter
- searches memory on each prompt
- deterministically answers from fact-only hits
- returns prompt enrichment with `hookSpecificOutput.additionalContext` when contextual memory exists

### 2. Pi integration
`integrations/pi/extension.ts`
- pi extension adapter
- searches memory on each prompt
- either answers directly from memory or injects retrieved memory context into the turn
- saves the session transcript after each completed turn

`integrations/pi/adapter.py`
- Python bridge used by the pi extension to call the shared ingest/retrieval seams

### 3. Daemon
`memory/daemon.py`
- polls for unprocessed sessions
- extracts facts, then episodes, then procedures, then working memory, then session memory
- marks sessions processed
- `--once` forces one immediate extraction pass and skips the CPU gate
- writes only `~/.memory/daemon.log`

### 4. Recall server
`memory/ingest_server.py`
- long-lived singleton embedding model for recall
- `POST /ingest`
- `POST /recall`
- `GET /status`

### 5. Dashboard server
`memory/dashboard_server.py`
- sessions view
- facts view
- episodes view
- procedural-memory view
- working-memory view
- session-memory view
- logs for daemon, facts, episodes, procedures, working memory, session memory

## Storage

Retained database tables:
- `sessions`
- `facts`
- `episodic_memory`
- `procedural_memory`
- `working_memory`
- `session_memory`

Dropped on bootstrap/migration:
- `session_vecs`
- `retrievals`
- `chunks`
- `insights`
- `topic_clusters`
- `cluster_memberships`
- `summaries`
- `compressed_memory`
- `compacted_sessions`
- `response_cache`

Retained log files:
- `~/.memory/daemon.log`
- `~/.memory/wake_up.log`
- `~/.memory/save_hook.log`
- `~/.memory/facts.log`
- `~/.memory/episodic.log`
- `~/.memory/procedural.log`
- `~/.memory/working_memory.log`
- `~/.memory/session_memory.log`

Unit tests suppress file-log writes.

Fact storage keeps:
- structured canonical fields: `entity`, `attribute`, `value`
- computed canonical text for deterministic UI/tests, e.g. `user.name = Yash`
- model-generated semantic text for embeddings/retrieval, e.g. `My name is Yash. What's my name? Yash.`

For existing databases, run the one-off migration before starting the updated app:

```bash
python3 scripts/migrate_facts_semantic_text.py --db ~/.memory/memory.db
```

If you have older stored sessions that predate procedural extraction, backfill procedural memory explicitly:

```bash
python3 scripts/backfill_procedural_memory.py --db ~/.memory/memory.db --dry-run
python3 scripts/backfill_procedural_memory.py --db ~/.memory/memory.db
```

To re-extract procedures for sessions that already have procedural memory, use `--force`:

```bash
python3 scripts/backfill_procedural_memory.py --db ~/.memory/memory.db --force --limit 25
```

## Quick start

```bash
python3 -m pip install --user sentence-transformers fastapi uvicorn pydantic psutil setproctitle debugpy
ollama pull qwen2.5:7b
./install.sh
```

Start services manually if needed:

```bash
python3 memory/ingest_server.py   # recall server
python3 memory/dashboard_server.py
python3 memory/daemon.py
python3 memory/daemon.py --once
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

- `integrations/claude/save_hook.py`
- `integrations/claude/wake_up.py`

### Pi

`install.sh` installs a global Pi wrapper at:

- `~/.pi/agent/extensions/agentic-memory.ts`

This repo also includes a project-local auto-discovered wrapper at:

- `.pi/extensions/agentic-memory.ts`

So after install you can run `pi` anywhere, and inside this project you can also just run:

```bash
pi
```

To verify the extension loaded inside pi, run:

```text
/memory-status
```

For one-off/manual loading you can still use:

```bash
pi -e /absolute/path/to/agentic-memory/integrations/pi/extension.ts
```

The pi extension calls `integrations/pi/adapter.py`, which uses the shared save/retrieval helpers in `integrations/common.py`.

## CLI

```bash
python3 cli.py bootstrap
python3 cli.py status
python3 cli.py search "auth middleware"
python3 cli.py semantic "login loop bug"
python3 cli.py get-session <session_id>
python3 cli.py tail
python3 cli.py add-fact user name Yash --tag identity
python3 cli.py delete-fact <fact_id>
python3 cli.py dashboard
```

## Tests

```bash
pytest -q
```

## Notes

- `WakeUpContext` still keeps some old fields for compatibility, but runtime retrieval uses working memory, session memory, plus `episodic`, `facts`, and `procedural` memory.
- Model selection:
  - `MEMORY_OLLAMA_MODEL` sets the model used for fact extraction, episodic extraction, procedural extraction, working-memory extraction, and session-memory extraction.
  - Recall is model-free apart from the singleton embedding model hosted by `memory/ingest_server.py`.
- `memory.db.log_retrieval()` is a compatibility no-op because retrieval-event storage was removed.
- `integrations/common.py` holds the shared save/retrieve policy used by Claude and pi.
- `dashboard.html` supports click-row details for Sessions, Facts, Episodes, Procedural memory, Working memory, and Session memory.
