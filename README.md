# Agentic Memory

A simplified persistent memory system with separate integrations for Claude Code and pi.

Current runtime scope is intentionally small:
- store full **sessions**
- extract durable **facts**
- extract **episodes** (episodic memory)
- extract durable **procedures** (procedural memory)
- retrieve **facts + episodes + procedures** during wake-up

Everything else from the older design was removed.

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
- cannot inject prompt text because Claude's hook contract only supports allow/block here

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
- extracts facts, then episodes, then procedures
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
- logs for daemon, facts, episodes, procedures

## Storage

Retained database tables:
- `sessions`
- `facts`
- `episodic_memory`
- `procedural_memory`

Dropped on bootstrap/migration:
- `session_vecs`
- `retrievals`
- `chunks`
- `insights`
- `topic_clusters`
- `cluster_memberships`
- `summaries`
- `compressed_memory`
- `working_memory`
- `compacted_sessions`
- `response_cache`

Retained log files:
- `~/.memory/daemon.log`
- `~/.memory/wake_up.log`
- `~/.memory/save_hook.log`
- `~/.memory/facts.log`
- `~/.memory/episodic.log`
- `~/.memory/procedural.log`

Unit tests suppress file-log writes.

Fact storage keeps:
- structured canonical fields: `entity`, `attribute`, `value`
- computed canonical text for deterministic UI/tests, e.g. `user.name = Yash`
- model-generated semantic text for embeddings/retrieval, e.g. `My name is Yash. What's my name? Yash.`

For existing databases, run the one-off migration before starting the updated app:

```bash
python3 scripts/migrate_facts_semantic_text.py --db ~/.memory/memory.db
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

This repo includes a project-local auto-discovered wrapper at:

- `.pi/extensions/agentic-memory.ts`

So inside this project you can just run:

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

- `WakeUpContext` still keeps some old fields for compatibility, but runtime retrieval uses `episodic`, `facts`, and `procedural` memory.
- Model selection:
  - `MEMORY_OLLAMA_MODEL` sets the model used for fact extraction, episodic extraction, and procedural extraction.
  - Recall is model-free apart from the singleton embedding model hosted by `memory/ingest_server.py`.
- `memory.db.log_retrieval()` is a compatibility no-op because retrieval-event storage was removed.
- `integrations/common.py` holds the shared save/retrieve policy used by Claude and pi.
- `dashboard.html` supports click-row details for Sessions, Facts, and Episodes. Procedural storage exists in the runtime, but the dashboard UI has not been expanded for it yet.
