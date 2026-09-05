# Agentic Memory

A simplified persistent memory system for Claude Code.

Current runtime scope is intentionally small:
- store full **sessions**
- extract durable **facts**
- extract **episodes** (episodic memory)
- retrieve only **facts + episodes** during wake-up

Everything else from the older design was removed.

## Components

### 1. Save hook
`hooks/save_hook.py`
- reads Claude Code transcript JSONL
- stores the session in SQLite
- does not write its own log file

### 2. Daemon
`memory/daemon.py`
- polls for unprocessed sessions
- extracts facts, then episodes
- marks sessions processed
- `--once` forces one immediate extraction pass and skips the CPU gate
- writes only `~/.memory/daemon.log`

### 3. Wake-up hook
`hooks/wake_up.py`
- retrieves semantic matches from:
  - `facts`
  - `episodic_memory`
- fact-only hits can block with a deterministic fact answer
- no working memory, compaction, procedural memory, or cache layers remain

### 4. Ingest server
`memory/ingest_server.py`
- `POST /ingest`
- `GET /status`

### 5. Dashboard server
`memory/dashboard_server.py`
- sessions view
- facts view
- episodes view
- logs for daemon, facts, episodes

## Storage

Retained database tables:
- `sessions`
- `facts`
- `episodic_memory`

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
- `procedural_memory`

Retained log files:
- `~/.memory/daemon.log`
- `~/.memory/facts.log`
- `~/.memory/episodic.log`

Unit tests suppress file-log writes.

## Quick start

```bash
python3 -m pip install --user sentence-transformers fastapi uvicorn pydantic psutil setproctitle debugpy
ollama pull llama3.2:3b
./install.sh
```

Start services manually if needed:

```bash
python3 memory/ingest_server.py
python3 memory/dashboard_server.py
python3 memory/daemon.py
python3 memory/daemon.py --once
```

## CLI

```bash
python3 cli.py bootstrap
python3 cli.py status
python3 cli.py search "auth middleware"
python3 cli.py semantic "login loop bug"
python3 cli.py get-session <session_id>
python3 cli.py tail
python3 cli.py add-fact "user.name = Yash" --tag identity
python3 cli.py delete-fact <fact_id>
python3 cli.py dashboard
```

## Tests

```bash
pytest -q
```

## Notes

- `WakeUpContext` still keeps some old fields for compatibility, but runtime retrieval only uses `episodic` and `facts`.
- `memory.db.log_retrieval()` is a compatibility no-op because retrieval-event storage was removed.
- `dashboard.html` supports click-row details for Sessions, Facts, and Episodes.
