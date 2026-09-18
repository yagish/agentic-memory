# Repository Recon — agentic-memory

## Version stamp
- Commit: `7df34ac`
- Branch: `refactor-integrations-pi-claude-memory`
- Date: 2026-09-18

---

## Language and runtime

- **Language:** Python 3 (no version pin found in repo; uses f-strings, `from __future__ import annotations`, `match`-free; compatible with 3.10+)
- **TypeScript:** one file — `integrations/pi/extension.ts` (pi agent extension; not built locally, symlinked/bundled by pi runtime)
- **Frameworks:** FastAPI + uvicorn (two HTTP servers); no Django/Flask
- **Build / task tooling:** none (no Makefile, no pyproject.toml, no requirements.txt); dependencies installed manually via `pip3 install`
- **Key external libraries:** `sentence-transformers`, `fastapi`, `uvicorn`, `pydantic`, `psutil`, `setproctitle`, `debugpy`
- **LLM runtime:** Ollama (local, `http://localhost:11434/api/generate`); default model `qwen2.5:7b` (env-overridable via `MEMORY_OLLAMA_MODEL`)
- **Embedding model:** `all-MiniLM-L6-v2` from sentence-transformers (384 dimensions, loaded in-process at ingest server startup)
- **Database:** SQLite at `~/.memory/memory.db`; WAL mode; busy_timeout = 5 000 ms; no ORM

---

## Repository type

Primary types (all apply):
- **Async worker / background daemon** — `memory/daemon/` polls SQLite for unprocessed sessions, runs five LLM extractors per session, writes structured memory types back to DB
- **Synchronous HTTP service (dual)** — `memory/servers/ingest_server.py` (port 7747) handles ingest and recall; `memory/servers/dashboard_server.py` (port 7748) serves the dashboard UI and admin API
- **CLI tool** — `cli.py` (nine sub-commands)
- **Library / integration adapter** — `integrations/claude/` and `integrations/pi/` wire external agent runtimes to the shared save/retrieve seam in `integrations/common.py`

---

## Repository shape

Single Python package tree. No monorepo structure, no published artifact. Installed locally and run as macOS launchd services.

**Directory map:**

| Directory | Purpose |
|---|---|
| `memory/daemon/` | Background extraction daemon (main loop, compaction, pruning, constants) |
| `memory/db/` | SQLite schema, bootstrap, migrations, per-table query helpers |
| `memory/servers/` | FastAPI ingest server, FastAPI dashboard server, stdlib HTTP client |
| `memory/retrieval/` | Wake-up context assembly: fetch, rank, format, intent detection |
| `memory/facts/` | Fact extraction, normalization, rendering, repository, text helpers |
| `memory/episodic/` | Episodic memory extraction and repository |
| `memory/procedural/` | Procedural memory extraction, backfill, repository |
| `memory/working_memory/` | Working-memory extraction and repository |
| `memory/session/` | Session-memory extraction and repository |
| `memory/llm/` | Ollama text generation seam; embedding seam |
| `memory/vectors/` | Embedding model wrapper (sentence-transformers); cosine distance; pack/unpack |
| `memory/utils/` | Logger (activity + error logs), debug helpers |
| `memory/contracts/` | Shared type/data contracts (empty `__init__`) |
| `integrations/claude/` | Claude Code Stop hook (save_hook.py) and UserPromptSubmit hook (wake_up.py) |
| `integrations/pi/` | Pi extension (extension.ts) and Python bridge (adapter.py) |
| `integrations/common.py` | Shared save/retrieve policy — the seam all integration adapters call |
| `scripts/` | One-off migration scripts (facts semantic text, procedural backfill) |
| `tests/` | pytest suite with fixture-based extraction tests, contract tests, integration tests |
| `hooks/` | Deprecated top-level symlinks (now superseded by `integrations/claude/`) |
| `docs/` | ADRs and generated documentation (this tree) |
| `.github/` | CI configuration (workflows) |

---

## Public surface inventory

### HTTP endpoints — ingest server (port 7747)

| Method | Path | Purpose |
|---|---|---|
| GET | `/status` | Health check; returns DB path, embed model ready flag, embed model name |
| POST | `/ingest` | Store a session transcript (session_id, agent, turns, started_at, metadata) |
| POST | `/recall` | Retrieve ranked memory context for a prompt; returns action + injection/answer |

### HTTP endpoints — dashboard server (port 7748)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serve `dashboard.html` (single-page UI) |
| GET | `/api/sessions` | List sessions, paginated |
| GET | `/api/sessions/{id}` | Get one session with full transcript |
| GET | `/api/facts` | List facts |
| GET | `/api/episodic` | List episodic memories |
| GET | `/api/procedural` | List procedural memories |
| GET | `/api/working_memory` | List working-memory snapshots |
| GET | `/api/session_memory` | List session-memory entries |
| GET | `/api/logs/{source}` | Tail log files (daemon, wake_up, save_hook, facts, episodic, procedural, working_memory, session_memory, memory_search_engine) |
| POST | `/api/process-one` | Trigger one daemon extraction pass (for manual use) |
| GET | `/api/activity-stats` | Cached parse of activity.log for LLM-calls-avoided and estimated tokens-saved |
| GET | `/api/performance-stats` | Cached parse of activity.log for embedding/search/daemon latency percentiles |
| GET | `/api/ollama-status` | Check if Ollama is running |
| POST | `/api/start-ollama` | Start Ollama if not running |

### CLI commands — `cli.py`

| Command | Purpose |
|---|---|
| `bootstrap` | Create DB schema at `~/.memory/memory.db` |
| `status` | Print session, turn, fact, episode, procedure, working-memory, session-memory counts |
| `search <query>` | Full-text search over session transcripts |
| `semantic <query>` | Cosine-similarity search over sessions |
| `get-session <id>` | Print one session as JSON |
| `tail [n]` | Print N most recent sessions (default 10) |
| `add-fact <entity> <attribute> <value>` | Insert a manual fact |
| `delete-fact <fact_id>` | Delete a fact by ID |
| `dashboard` | Open the dashboard in a browser |

### Claude Code hooks (invoked by Claude Code runtime)

| Event | Handler | Purpose |
|---|---|---|
| `Stop` | `integrations/claude/save_hook.py` | Parse JSONL transcript, save session to DB |
| `UserPromptSubmit` | `integrations/claude/wake_up.py` | Retrieve memory context, inject or answer |

### Pi agent extension

| Interface | File | Purpose |
|---|---|---|
| Pi extension (TS) | `integrations/pi/extension.ts` | Intercept prompts, call Python adapter for recall/save |
| Python bridge | `integrations/pi/adapter.py` | Expose recall and save functions to the TS extension |

### Scheduled / background service (macOS launchd)

| Service label | Script | Purpose |
|---|---|---|
| `com.memory.daemon` | `memory/daemon/__main__.py` | Background extraction daemon |
| `com.memory.ingest` | `memory/servers/ingest_server.py` | Ingest + recall HTTP server |
| `com.memory.query` | `memory/servers/dashboard_server.py` | Dashboard HTTP server |

---

## Domain terminology

| Term | Meaning |
|---|---|
| **Session** | One Claude Code or pi agent conversation, stored as JSON turns in the `sessions` table |
| **Fact** | A durable structured triple: entity + attribute + value (e.g., `user.name = Yash`) |
| **Episode / Episodic memory** | A narrative summary of what happened in a session: title, abstract, decisions, outcomes, follow-ups |
| **Procedure / Procedural memory** | A how-to pattern extracted from a session: title, summary, steps, trigger phrases |
| **Working memory** | A point-in-time snapshot of the active goal, focus, next step, and status in a session |
| **Session memory** | A compacted handoff record: `left_off_at` + `next_steps`, used to resume at the start of the next session |
| **Daemon** | The background process that picks up unprocessed sessions and runs all five extractors |
| **Wake-up** | The retrieval event at the start of each prompt: fetches ranked memory context and injects it |
| **Recall server** | The ingest server's `/recall` endpoint (or the client's `recall()` method) |
| **Ingest** | The act of saving a session transcript to SQLite |
| **Compaction** | Summarizing a long raw transcript to a shorter text for LLM extraction |
| **Embedding** | A 384-dimensional float vector produced by `all-MiniLM-L6-v2` |
| **Semantic search** | Cosine-similarity search over stored embeddings |
| **Token budget** | The character/token limit applied during retrieval to keep injected context small |
| **Activity log** | `~/.memory/activity.log` — structured JSON-lines log used for performance/stats |
| **WakeUpContext** | The dataclass that carries all retrieved memory for one prompt |
| **RecallOutcome** | The decision (answer / inject / noop) produced after retrieval |

---

## Existing documentation found

- `README.md` — describes design strengths, components, storage tables, quick-start, CLI, and integrations. Largely current.
- `ENHANCEMENTS.md` — feature ideas list; not implementation.
- `TESTING_PI_MEMORY.md` — manual test instructions for pi integration.
- `docs/adr/` — ADR directory exists but no ADR files are present at inspection time.
- `docs/memory-type-checklist.md` — internal checklist.

---

## Exclusions in effect

`__pycache__/`, `.venv/`, `.git/`, `.pytest_cache/`, `tests/fixtures/` (test data only), lockfiles (none present).
