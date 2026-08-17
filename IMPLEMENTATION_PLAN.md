# Agentic Memory System — Implementation Plan

> Grilled and agreed: 2026-08-16.
> Scope: Claude Code only. Save transcripts first. Retrieval deferred to Phase 4+.

---

## Decisions Made

| Decision | Choice | Reason |
|---|---|---|
| Agent target | Claude Code only | Simplest adapter; expand later |
| Storage | SQLite + FTS5 | Zero deps, built-in full-text search, right foundation |
| Save strategy | Full transcript upsert per session | No statefulness, always a complete snapshot |
| Trigger | Claude Code Stop hook | Native, blocks AI briefly, then lets it through |
| Retrieval | Out of scope (Phase 4+) | Validate save first; retrieval depends on clean data |
| Vector embeddings | Deferred to Phase 6 | Small local model when we get there: `all-MiniLM-L6-v2` or `snowflake-arctic-embed:xs` |

---

## Phase 1 — Storage Layer

**Goal**: a Python module that initialises the SQLite database and provides `upsert_session()` and `search()`.

### Tasks

1. **Create `memory/db.py`**
   - `init_db(path)` — create the database file and run schema migrations
   - `upsert_session(session_id, agent, transcript, timestamp)` — insert or replace the full session row
   - `search(query, limit)` — FTS5 full-text search across all sessions, returns ranked rows

2. **Define the schema** (runs inside `init_db`)
   ```sql
   CREATE TABLE IF NOT EXISTS sessions (
     session_id   TEXT PRIMARY KEY,
     agent        TEXT NOT NULL DEFAULT 'claude',
     started_at   TEXT,
     updated_at   TEXT,
     turn_count   INTEGER,
     transcript   TEXT  -- verbatim JSON array of {role, content} turns
   );

   CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts
     USING fts5(session_id UNINDEXED, transcript, content='sessions');

   CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
     INSERT INTO sessions_fts(session_id, transcript) VALUES (new.session_id, new.transcript);
   END;
   CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
     INSERT INTO sessions_fts(sessions_fts, session_id, transcript) VALUES ('delete', old.session_id, old.transcript);
     INSERT INTO sessions_fts(session_id, transcript) VALUES (new.session_id, new.transcript);
   END;
   ```

3. **Write unit tests** (`tests/test_db.py`)
   - Insert a session, read it back, assert verbatim match
   - Upsert the same session with more turns, assert updated
   - FTS5 search returns the right session for a known keyword

### Deliverable
`memory/db.py` + passing tests. No hook, no server yet.

---

## Phase 2 — Claude Code Stop Hook

**Goal**: a hook script that fires on every Claude Code `stop` event, reads the transcript from stdin, and calls `upsert_session()`.

### Tasks

4. **Create `hooks/save_hook.py`**
   - Reads JSON from stdin (Claude Code Stop hook payload)
   - Extracts `session_id`, `transcript` (list of turns), `timestamp`
   - Calls `memory.db.upsert_session()`
   - Exits 0 on success (non-zero blocks the AI)
   - Writes errors to a log file, never raises to stdout

5. **Understand the hook payload shape**
   - Review Claude Code's Stop hook JSON format
   - Confirm which fields carry `session_id` and transcript turns
   - Add a `--dry-run` flag that prints what would be saved without writing

6. **Register the hook in `.claude/settings.json`**
   ```json
   {
     "hooks": {
       "Stop": [
         {
           "matcher": "",
           "hooks": [
             {
               "type": "command",
               "command": "python /path/to/hooks/save_hook.py"
             }
           ]
         }
       ]
     }
   }
   ```

7. **Manual smoke test**
   - Have a short conversation with Claude
   - Verify the session row appears in SQLite with the correct transcript
   - Check `updated_at` updates on subsequent turns in the same session

### Deliverable
Hook wired and saving real transcripts from a live Claude session.

---

## Phase 3 — MCP Server (Search Interface)

**Goal**: expose the saved transcripts to Claude via MCP tools so it can answer "what did we talk about last week?"

### Tasks

8. **Create `memory/mcp_server.py`**
   - Uses the Python `mcp` library
   - Wraps `db.search()` and basic status queries

9. **Implement these MCP tools**
   - `memory_status` — total session count, date range, total turns stored
   - `memory_search(query, limit?)` — FTS5 search, returns matching session excerpts with timestamps
   - `memory_get_session(session_id)` — retrieve full verbatim transcript for one session

10. **Register the MCP server in `.claude/settings.json`**
    ```json
    {
      "mcpServers": {
        "memory": {
          "command": "python",
          "args": ["/path/to/memory/mcp_server.py"]
        }
      }
    }
    ```

11. **Manual test**
    - Ask Claude "what did we talk about in my last session?"
    - Verify it calls `memory_search` and returns a coherent answer

### Deliverable
Claude can query its own saved transcripts on demand.

---

## Phase 4 — Wake-up Injection (Retrieval, L0+L1)

**Goal**: inject a digest of recent sessions into the system prompt at conversation start.

> Deferred until Phase 1–3 are stable and you have enough real data to validate the digest quality.

### Sketch (to be detailed when we get here)
- `PreToolUse` or system prompt injection hook at session start
- L0: static `identity.md` — who the user is (~100 tokens)
- L1: last N sessions, summarised to key topics by turn weight (~600 tokens)
- Injected as a fenced block in the system prompt

---

## Phase 5 — Semantic Search (Option C)

**Goal**: add vector embeddings alongside FTS5 for "find conversations *like* this" queries.

> Deferred. When we get here, decide between:
> - `all-MiniLM-L6-v2` via `sentence-transformers` (~90MB, no Ollama needed)
> - `snowflake-arctic-embed:xs` via Ollama (~30MB, requires Ollama running)
>
> Add `sqlite-vec` extension to the existing SQLite file — additive, no schema rewrite.

---

## Phase 6 — CLI + Dashboard

**Goal**: a command-line tool for inspecting the memory database and a dashboard showing stored memory and retrieval value.

### CLI (`cli.py`)

```
python3 cli.py status                   # session count, turn count, date range
python3 cli.py search "query"           # FTS5 search, prints ranked results
python3 cli.py get-session <id>         # print full transcript for one session
python3 cli.py tail [N]                 # show last N sessions (default 10)
```

### Dashboard metrics

What we can measure accurately:
- **Sessions stored** — total count, growth over time
- **Turns stored** — total across all sessions
- **Tokens stored (estimated)** — character count ÷ 4 (rough approximation)
- **Retrievals** — how many times `memory_search` / `memory_get_session` was called
- **Tokens retrieved (estimated)** — size of content Claude pulled from memory

What we intentionally do NOT claim:
- "Tokens saved" vs baseline — requires knowing what you would have re-typed, which is not instrumentable

### Dashboard output

Rendered as a local HTML file (`dashboard.html`) — opened in the browser via `python3 cli.py dashboard`.
Shows: session timeline, cumulative turns, estimated tokens stored, retrieval count.

### Schema addition needed

Add a `retrievals` table to log each MCP tool call (timestamp, tool name, query, result size) so retrieval counts are real, not estimated:

```sql
CREATE TABLE IF NOT EXISTS retrievals (
  id          TEXT PRIMARY KEY,
  tool        TEXT,   -- 'memory_search' | 'memory_get_session' | 'memory_status'
  query       TEXT,
  result_size INTEGER,  -- bytes returned
  called_at   TEXT
);
```

---

## File Layout

```
agentic-memory/
  memory/
    __init__.py
    db.py          ← Phase 1
    mcp_server.py  ← Phase 3
  hooks/
    save_hook.py   ← Phase 2
  tests/
    test_db.py     ← Phase 1
  cli.py           ← Phase 6
  .claude/
    settings.json  ← Phase 2
  .mcp.json        ← Phase 3
  MEMORY_SYSTEM_DESIGN_BRIEF.md
  IMPLEMENTATION_PLAN.md
```

---

## Build Order

```
Phase 1 (schema + db.py) → Phase 2 (hook) → Phase 3 (MCP) → Phase 4 (wake-up) → Phase 5 (vectors)
         ↑                        ↑
     testable alone         testable with live Claude

Completed: 1 → 2 → 3 → 4 → 5 → 6 (CLI) → 7 (chunking) → 8 (facts + MCP write tools)

Next (remaining):
  10 (relevance wake-up) → 11-01 (RRF hybrid search) → 11-02 (hybrid MCP tool)
  → 12-01 (consolidation) → 12-02 (decay/pruning)
  → 13-01 (daemon tables) → 13-03 (topic clusters) → 13-02 (daemon loop)
  → 13-04 (MCP/wake-up exposure) → 13-05 (launchd)

Note: Phase 9 removed — fact extraction moved into Phase 13 daemon (async, no hook cost).
```

Each phase produces something independently verifiable before the next starts.

---

## Phase 7 — Sub-session Chunking

**Goal**: replace one-vector-per-session with finer-grained chunk embeddings so semantic search can pinpoint topics within a session, not just the session as a whole.

**Problem**: a 50-turn conversation touching authentication, database design, and UI layout collapses into one blob. A search for "authentication bug" may miss it entirely because the session was mostly about something else.

### Ticket 7-01 — Add `chunks` table and storage functions to `db.py`

```sql
CREATE TABLE IF NOT EXISTS chunks (
  id          TEXT PRIMARY KEY,   -- "{session_id}:{chunk_index}"
  session_id  TEXT NOT NULL REFERENCES sessions(session_id),
  chunk_index INTEGER NOT NULL,
  text        TEXT NOT NULL,      -- concatenated turn content for this chunk
  embedding   BLOB,               -- 384-float vector (same model as session_vecs)
  created_at  TEXT
);
```

- Add `store_chunk(conn, session_id, chunk_index, text, embedding)` to `db.py`
- Add `get_chunks_for_session(conn, session_id)` to `db.py`
- Write unit tests for both functions

### Ticket 7-02 — Implement transcript chunker

- Add `chunk_transcript(turns, window=6, overlap=1)` to `db.py`
  - Sliding window of 6 turns, 1-turn overlap between chunks
  - Each chunk is the concatenated `content` of those turns
  - Returns `list[str]` — one string per chunk
- Write unit tests: correct chunk count, overlap present, no turns dropped

### Ticket 7-03 — Wire chunking into `save_hook.py` and update semantic search

- After `upsert_session()`, call `chunk_transcript()`, embed each chunk, store via `store_chunk()`
- Add `semantic_search_chunks(conn, query, limit)` to `db.py` — searches `chunks.embedding` instead of `session_vecs`, returns `(session_id, chunk_index, distance, text)` tuples
- Update `memory_semantic_search` MCP tool to use chunk-level search; group results by session before returning
- Write tests: a keyword appearing only in the 3rd chunk of a session is returned by semantic search

---

## Phase 8 — Memory Write MCP Tools

**Goal**: give Claude the ability to proactively save structured facts mid-conversation, not just passively have transcripts archived at session end.

### Ticket 8-01 — Add `facts` table and CRUD functions to `db.py`

```sql
CREATE TABLE IF NOT EXISTS facts (
  id         TEXT PRIMARY KEY,   -- UUID
  content    TEXT NOT NULL,      -- the fact as plain text
  tags       TEXT,               -- JSON array of strings e.g. ["preference", "auth"]
  source     TEXT,               -- "extracted" | "manual" | "agent"
  session_id TEXT,               -- which session it came from (nullable)
  created_at TEXT,
  updated_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
  USING fts5(id UNINDEXED, content, tags, content='facts');
```

- Add `insert_fact(conn, content, tags, source, session_id)` → returns new `id`
- Add `update_fact(conn, id, content, tags)` 
- Add `delete_fact(conn, id)`
- Add `search_facts(conn, query, limit)` — FTS5 search over `facts_fts`
- Add `list_facts(conn, tag_filter, limit)` — filter by tag, ordered by `updated_at` DESC
- Write unit tests for all five functions

### Ticket 8-02 — Add fact write/read MCP tools to `mcp_server.py`

Add four new tools:

| Tool | Args | What it does |
|---|---|---|
| `memory_save_fact` | `content, tags?` | Insert a new fact; returns its `id` |
| `memory_update_fact` | `id, content, tags?` | Replace content/tags of an existing fact |
| `memory_delete_fact` | `id` | Remove a fact by id |
| `memory_list_facts` | `tag?, limit?` | Return facts filtered by tag, newest first |

- Log each call to the `retrievals` table (tool name, query/content, result size)
- Write MCP-layer tests for all four tools

---

## ~~Phase 9 — Structured Fact Extraction~~ (removed)

**Decision**: automated fact extraction was originally planned in the Stop hook (calling Claude API on every session save). This adds API cost, network dependency, and latency to a path that must complete quickly and offline.

**Resolution**: fact extraction is handled in two better ways:
1. **Proactively** — Claude calls `memory_save_fact` (Phase 8 MCP tool) mid-conversation when it observes something worth remembering.
2. **Asynchronously** — the Phase 13 background daemon runs `re_extract_facts()` on unprocessed sessions at its own pace, with no time pressure.

The Stop hook remains fast and offline-safe.

---

## Phase 10 — Relevance-based Wake-up

**Goal**: replace the recency-only L1 digest with a semantically relevant one — inject sessions and facts that are topically related to what the user is actually asking about right now.

**Problem**: current `wake_up.py` always shows the last 5 sessions regardless of the current prompt. Opening a session to debug an auth bug gets yesterday's unrelated UI conversation instead.

### Ticket 10-01 — Rewrite `wake_up.py` to use prompt-driven retrieval

- Parse the incoming user prompt from the hook payload (`payload["prompt"]` or equivalent field)
- Run `semantic_search_chunks(conn, prompt, limit=5)` to find topically relevant chunks
- Also run `search_facts(conn, prompt, limit=5)` to surface relevant extracted facts
- Build the digest from these results (grouped by session) rather than from `ORDER BY updated_at DESC`
- Keep L0 identity injection unchanged
- Fall back to recency-based L1 if semantic search returns nothing (no sessions yet, model not loaded)
- Write tests: a prompt about "authentication" retrieves the auth-related session, not an unrelated recent one

---

## Phase 11 — Hybrid Search

**Goal**: a single search tool that combines FTS5 keyword results and semantic vector results into one ranked list, which is almost always better than either alone.

### Ticket 11-01 — Implement Reciprocal Rank Fusion in `db.py`

- Add `hybrid_search(conn, query, limit) -> list[dict]` to `db.py`
- Run `search()` (FTS5) and `semantic_search_chunks()` in sequence
- Apply RRF: `score = 1/(k + rank_fts) + 1/(k + rank_semantic)` where `k=60`
- Deduplicate by `session_id`, take top `limit` by fused score
- Return same shape as existing search results (session_id, agent, updated_at, snippet)
- Write unit tests: a result that appears in both lists ranks above one that appears in only one

### Ticket 11-02 — Add `memory_hybrid_search` MCP tool

- Wrap `hybrid_search()` as a new MCP tool in `mcp_server.py`
- Log to `retrievals` table
- Update the system-level guidance (CLAUDE.md or README) to recommend `memory_hybrid_search` as the default retrieval tool

---

## Phase 12 — Memory Consolidation & Decay

**Goal**: prevent unbounded database growth and ensure old valuable conversations remain accessible as compressed summaries rather than being lost in noise.

**LLM backend**: all generation uses a local **ollama** model — no cloud API calls. Default model: `llama3.2:3b` (~2GB, ~2-3GB RAM while active, unloads after 5 min idle). Configurable via `MEMORY_OLLAMA_MODEL` env var so users can swap to a smaller (`qwen2.5:1.5b`) or larger model based on their hardware.

### Ticket 12-01 — Build `memory/consolidation.py` — L2 topic summaries

- `consolidate_old_sessions(conn, days_threshold=30)` — finds sessions older than N days that haven't been summarised
- For each, calls **ollama** (`ollama.chat(model=..., messages=[...])`) to produce a 3–5 sentence summary of key topics, decisions, and outcomes
- Stores the summary in a new `summaries` table:
  ```sql
  CREATE TABLE IF NOT EXISTS summaries (
    id          TEXT PRIMARY KEY,
    session_id  TEXT UNIQUE REFERENCES sessions(session_id),
    summary     TEXT,
    created_at  TEXT
  );
  ```
- Marks the session as `summarised = 1` in the `sessions` table (add column via migration)
- Add `--consolidate` flag to `cli.py` to trigger manually: `python3 cli.py consolidate`
- Write tests: sessions older than threshold get summaries; recent sessions are skipped

### Ticket 12-02 — Add memory decay / raw transcript pruning

- Add `prune_old_transcripts(conn, days_threshold=90)` to `consolidation.py`
  - Only prunes sessions that are already summarised (`summarised = 1`)
  - Replaces `transcript` JSON with `null` (or empty) to free disk space
  - Preserves the session row, metadata, facts, chunks, and summary
- Add `--prune` flag to `cli.py`: `python3 cli.py prune --days 90`
- Write tests: pruned session loses transcript but retains all other data; un-summarised sessions are never pruned
- Update dashboard to show: sessions with full transcript vs summarised-only vs both

---

## Phase 13 — Background Relearning Daemon

**Goal**: a long-running background process that continuously improves the memory system as the user works — re-processing sessions, extracting richer facts, discovering cross-session patterns, and updating the knowledge base without any user intervention.

**Why this matters**: the quality of a memory system should compound over time. More sessions = more signal to learn from. Without a relearning loop, the system is static — a fact extracted from session 1 is never revised even if 50 later sessions add nuance or contradict it. This phase makes the system genuinely self-improving.

**LLM backend**: same ollama setup as Phase 12 (local, offline, no API key). The daemon checks `psutil.cpu_percent()` before running inference and skips a cycle if the machine is under load (threshold: 70%). Poll interval backs off from 5 min → 30 min when nothing new is found, so it's quiet on an idle machine.

### Ticket 13-01 — Add `processed_at` tracking and an `insights` table to `db.py`

Schema additions:

```sql
-- Track which sessions have been fully processed by the daemon
ALTER TABLE sessions ADD COLUMN daemon_processed_at TEXT;

-- Cross-session patterns and meta-insights
CREATE TABLE IF NOT EXISTS insights (
  id           TEXT PRIMARY KEY,
  insight_type TEXT NOT NULL,  -- "pattern" | "preference" | "topic_cluster" | "skill"
  content      TEXT NOT NULL,  -- plain-text description of the insight
  evidence     TEXT,           -- JSON array of session_ids that support this insight
  confidence   REAL,           -- 0.0–1.0, updated as more evidence accumulates
  created_at   TEXT,
  updated_at   TEXT
);
```

- Add `get_unprocessed_sessions(conn, limit)` — returns sessions where `daemon_processed_at IS NULL`
- Add `mark_session_processed(conn, session_id)` — stamps `daemon_processed_at`
- Add `upsert_insight(conn, insight_type, content, evidence, confidence)` — insert or merge insight
- Add `list_insights(conn, insight_type?, limit?)` — for dashboard and wake-up injection
- Write unit tests for all four functions

### Ticket 13-02 — Build `memory/daemon.py` — the relearning loop

A polling loop that runs as a background process:

```
while True:
    new_sessions = get_unprocessed_sessions(conn, limit=10)
    for session in new_sessions:
        re_extract_facts(session)          # may surface facts missed on first pass
        update_topic_clusters(session)     # assign session to topic buckets
        mark_session_processed(session)
    if enough_sessions_since_last_insight_run:
        generate_cross_session_insights()  # expensive — runs less frequently
    sleep(POLL_INTERVAL)  # default 5 minutes
```

- `re_extract_facts(conn, session)` — calls **ollama** with full transcript; upserts new facts, skips duplicates by content similarity
- `update_topic_clusters(conn, session)` — embeds the session summary (via `sentence-transformers`, same model as Phase 5), assigns it to the nearest existing topic cluster or creates a new one; stores cluster assignment in a `topic_clusters` table
- `generate_cross_session_insights(conn)` — called every N sessions (default: every 10 new sessions); sends a sample of recent facts + session summaries to **ollama** asking for patterns, recurring preferences, and skill observations; upserts results into `insights`
- Poll interval: 5 minutes while sessions are accumulating; backs off to 30 minutes when nothing new is found
- Writes all activity to `~/.memory/daemon.log`
- Exits cleanly on SIGTERM; resumes from where it left off (stateless — `daemon_processed_at` is the checkpoint)

### Ticket 13-03 — Add `topic_clusters` table and clustering functions to `db.py`

```sql
CREATE TABLE IF NOT EXISTS topic_clusters (
  id           TEXT PRIMARY KEY,
  label        TEXT NOT NULL,     -- human-readable topic name e.g. "authentication & security"
  centroid     BLOB,              -- average embedding of all member sessions
  member_count INTEGER DEFAULT 0,
  updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS cluster_memberships (
  session_id  TEXT NOT NULL REFERENCES sessions(session_id),
  cluster_id  TEXT NOT NULL REFERENCES topic_clusters(id),
  distance    REAL,               -- cosine distance from centroid
  PRIMARY KEY (session_id, cluster_id)
);
```

- `assign_to_cluster(conn, session_id, embedding)` — finds nearest cluster within threshold; creates new cluster if none is close enough; updates centroid as rolling average
- `get_cluster_sessions(conn, cluster_id)` — returns all session_ids in a cluster
- `get_clusters(conn)` — returns all clusters with label and member count
- Write unit tests: two similar sessions land in the same cluster; a dissimilar session creates a new cluster

### Ticket 13-04 — Expose insights and clusters in wake-up and MCP

- Update `wake_up.py` (Phase 10): after injecting relevant sessions, also append top 3 `insights` relevant to the current prompt — these are the system's learned meta-knowledge about the user
- Add `memory_list_insights(insight_type?, limit?)` MCP tool so Claude can query what the system has learned
- Add `memory_list_clusters()` MCP tool so Claude can see what topic areas exist
- Update the dashboard (`cli.py dashboard`) to show: topic cluster map, insight count by type, daemon last-run time, sessions processed vs unprocessed

### Ticket 13-05 — Wire daemon into system startup

- Add a `launchd` plist (`com.memory.daemon.plist`) for macOS so the daemon starts automatically on login and restarts if it crashes
- Add `python3 cli.py daemon start|stop|status` commands to manage it manually
- Add `--once` flag: `python3 cli.py daemon --once` runs one full pass and exits (useful for testing and CI)
- Write an integration test using `--once`: save 3 sessions, run daemon once, assert facts and cluster assignments were created

---

## Updated Build Order

```
Phase 7 (chunking)  →  Phase 8 (fact write tools)
       ↓
Phase 10 (relevance wake-up)  →  Phase 11 (hybrid search)
       ↓
Phase 12 (consolidation + decay)   [local ollama — no API]
       ↓
Phase 13 (background daemon)       [local ollama — no API]
       ↓
Phase 16 (agent-agnostic ingest)   [parallel track — can start after Phase 8]
       ↓
Phase 14 (installer)   →   Phase 15 (token economics)
```

**Recommended order**: 10-01 → 11-01 → 11-02 → 12-01 → 12-02 → 13-01 → 13-03 → 13-02 → 13-04 → 13-05 → 16-01 → 16-02 → 16-03 → 16-04 → 14-01 → 14-02 → 15-01 → 15-02

**Key constraints**:
- Phase 12 (ollama summarisation) requires `ollama` installed and a model pulled (`ollama pull llama3.2:3b`)
- Phase 13 daemon depends on Phase 12 (reuses consolidation) and Phase 13-01/03 (tables must exist first)
- Phase 16 is independent of 12/13 — can be built in parallel
- Phase 9 removed: fact extraction moved into the Phase 13 daemon (async, no hook overhead, local model)

---

## Phase 14 — Plugin Installer

**Goal**: make the system installable by anyone with a single command. Currently hooks and MCP config contain hardcoded absolute paths to this repo, so the system only works for the original developer.

### Ticket 14-01 — Write `install.sh`

The installer runs from wherever the user cloned the repo. It:

1. Detects `INSTALL_DIR` = the directory containing `install.sh` (via `$(cd "$(dirname "$0")"; pwd)`)
2. Checks Python 3.8+ is available, exits with a clear message if not
3. Installs Python dependencies: `pip3 install mcp fastmcp sentence-transformers`
4. Creates `~/.memory/` directory and writes a starter `~/.memory/identity.md` template if it doesn't already exist
5. Patches `~/.claude/settings.json` (global Claude Code settings) to add the two hooks, using Python's `json` module as the parser (jq not universally available). Must merge into any existing hooks, not overwrite them.
6. Writes (or merges) the MCP server entry into `~/.claude/mcp.json` (Claude Code's global MCP registry) pointing `python3 <INSTALL_DIR>/memory/mcp_server.py`
7. Prints a clear success summary and tells the user to restart Claude Code

Design constraints:
- Idempotent: running twice must not duplicate hook entries or break anything
- Never overwrites existing identity.md (the user may have personalised it)
- Uses Python for all JSON manipulation — no `jq` dependency
- Works on macOS and Linux; Windows is out of scope (Claude Code CLI runs on those two)

### Ticket 14-02 — Uninstall option

Add `install.sh --uninstall` that:
- Removes the two hook entries from `~/.claude/settings.json`
- Removes the `memory` server entry from `~/.claude/mcp.json`
- Leaves `~/.memory/` untouched (data preservation — user keeps their history)
- Prints confirmation of what was removed

---

## Phase 16 — Agent-agnostic Ingest Interface

**Goal**: decouple the memory store from Claude Code so any agent (Cursor, custom Python scripts, LangChain agents, future CLI tools) can write sessions to and read facts from the same database.

**Problem**: `save_hook.py` reads Claude Code's JSONL transcript format. An agent running in a different framework has no way to write to the memory store without replicating that format.

**Design**: expose a lightweight HTTP ingest endpoint (FastAPI, already available via `fastmcp` deps) alongside the existing MCP server. Any agent posts a session in the canonical shape and gets it stored, chunked, and embedded exactly as if it came from the Stop hook.

### Ticket 16-01 — Define the canonical session shape

The ingest API accepts a single JSON body:

```json
{
  "session_id": "string (required)",
  "agent":      "string (e.g. 'cursor', 'langchain', 'custom') — default 'unknown'",
  "turns": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "started_at": "ISO timestamp (optional)",
  "metadata":   {}  // arbitrary key-value, stored as JSON in a new column
}
```

- Add `metadata TEXT` column to `sessions` via migration in `db.py`
- Document the shape in `README.md`

### Ticket 16-02 — Build `memory/ingest_server.py` — HTTP ingest endpoint

- Single POST endpoint: `POST /ingest` — accepts the canonical shape, runs the full save pipeline (upsert → embed → chunk), returns `{"ok": true, "session_id": "..."}`
- `GET /status` — returns session count, newest session, db size (for health checks)
- Runs on `localhost:7747` by default (configurable via `MEMORY_INGEST_PORT` env var)
- Reuses all existing `db.py` functions — no duplication of save logic
- Write tests: POST a session, assert it appears in `sessions` table and `chunks` table

### Ticket 16-03 — Add `python3 cli.py ingest-server start|stop|status`

- `start` — launches `ingest_server.py` as a background process, writes PID to `~/.memory/ingest.pid`
- `stop` — sends SIGTERM to the PID
- `status` — checks if the process is running and prints the port
- Add ingest server startup to `install.sh` (optional, off by default — user opts in)

### Ticket 16-04 — Add a thin client helper `memory/client.py`

So other agents can ingest without knowing the HTTP API:

```python
from memory.client import MemoryClient
client = MemoryClient()  # connects to localhost:7747
client.save_session(session_id="...", agent="cursor", turns=[...])
```

- `save_session(session_id, agent, turns, started_at=None, metadata=None)`
- Raises `ConnectionError` if the ingest server is not running (clear message: "start it with `python3 cli.py ingest-server start`")
- Write tests with a mock HTTP server

---

## Phase 15 — Token Economics Measurement

**Goal**: measure whether the memory system is actually saving tokens or costing more. This is an open question — the honest answer is "it depends", and the system should surface enough data to let users judge for themselves.

### The Economics (design brief)

**What the system costs (tokens added):**

| Source | Est. tokens per session | Notes |
|---|---|---|
| Wake-up injection | ~220–350 | Identity (~100) + 5 session previews (~30 each) |
| MCP tool schemas | ~400 | Added to system prompt when MCP server is active |
| MCP tool calls | ~50–200 per call | Tool invocation + result overhead |
| Daemon (Phase 13) | 0 cloud tokens | Local ollama inference only — no API cost |

**What the system could save (unmeasurable without control group):**

| Scenario | Potential saving | Why unmeasurable |
|---|---|---|
| User skips pasting old context | 500–5,000 tokens | We don't know what they would have pasted |
| Claude skips clarification turns | 400–800 per averted turn | We don't know which turns were averted |
| Relevant snippet vs re-derivation | 200–2,000 per retrieval | We don't know if Claude would have needed it |

**The honest conclusion**: we cannot claim a net saving without a control group. What we CAN show:
- Injection overhead per session (what memory adds, guaranteed)
- Retrieval volume (what was pulled back via MCP)
- Sessions with zero retrievals (pure overhead, no benefit)
- Coverage ratio = retrieval bytes / injection bytes (≥1.0 means retrievals pulled back more than was injected; <1.0 means the system is a net cost)

A coverage ratio ≥ 1.0 does not prove savings — the user might not have needed that context — but a ratio consistently < 1.0 is strong evidence the system is costing more than it returns.

### Ticket 15-01 — Log wake-up injection size

Edit `hooks/wake_up.py`: after writing the flag file and before printing the JSON response, compute `injection_tokens = len(digest) // 4` and call `log_retrieval(conn, tool="wake_up_injection", query=None, result_size=injection_tokens)`. This reuses the existing `retrievals` table — no schema change needed. Guard with try/except so a DB failure never blocks the injection.

### Ticket 15-02 — Add token economics to `cli.py status`

Query the `retrievals` table to compute:
- `injection_total` = SUM(result_size) WHERE tool = 'wake_up_injection'
- `retrieval_total` = SUM(result_size) WHERE tool != 'wake_up_injection'
- `coverage_ratio` = retrieval_total / injection_total (or "N/A" if injection_total = 0)
- `zero_retrieval_sessions` = count of sessions where no MCP tool was called (pure overhead)

Print these in `cmd_status` under a new `=== Token Economics ===` block, with a one-line plain-English interpretation ("retrievals cover X% of injection overhead").

### Ticket 15-03 — Token economics panel in the dashboard

Add a new card to `cli.py dashboard` with:
- Stat tiles: Injected tokens, Retrieved tokens, Coverage ratio
- A bar chart (two bars per day: injected vs. retrieved) so the trend is visible over time — is retrieval volume growing faster than injection overhead as the system learns more?
- A table of sessions with zero retrievals (date, turn count) — these are the cases where the system was pure overhead
- Honest footnote: "Coverage ratio ≥ 1.0 means retrievals returned more context than was injected. It does not measure tokens saved vs. a baseline — that would require a control group."

### Open question: session-level opt-out

If a user notices coverage < 1.0 (they rarely use MCP retrieval), we could add a `--no-wakeup` flag or a per-project disable to avoid paying injection overhead in projects where memory isn't useful. Record this as a design decision once we see real data.

---

## Phase 16 — Token Efficiency

**Goal**: make the system net token-positive by default. The current push model (inject unconditionally) pays overhead on every session regardless of whether memory is useful. These four tickets convert it to a pull-oriented model that earns its keep.

**Dependency**: Phase 15 token tracking must be in place first so we can measure whether each change actually improves the coverage ratio.

### Break-even target

| Scenario | Current overhead | After Phase 16 |
|---|---|---|
| One-off session | ~700 tokens (always paid) | ~0 (gated out) |
| Ongoing project | ~700 tokens | ~100 tokens (facts + relevance filter) |
| Averted clarification turn | saves ~400 tokens | saves ~400 tokens |

A single averted clarification on an ongoing project covers the reduced overhead 4×. One-off sessions pay nothing.

---

### Ticket 16-01 — Relevance gate in `wake_up.py`

**Problem**: wake-up injects ~300 tokens into every session, including completely unrelated one-off questions.

**Fix**: embed the user's first prompt using the local `all-MiniLM-L6-v2` model (already loaded for Phase 5), compute cosine similarity against the 10 most recent session embeddings, and only proceed with injection if `max_similarity >= GATE_THRESHOLD` (default: 0.35). If no past session is similar enough, output `{}` (allow, no injection) and log the skip to `wake_up.log`.

Implementation notes:
- Read the user's first prompt from `payload["userPrompt"]` (available in the hook payload)
- Reuse `embed()` from `memory/db.py` — it's already loaded in the process
- The gate adds ~5ms of local CPU; no API call, no network
- Make `GATE_THRESHOLD` a constant at the top of the file so it's easy to tune
- Log: `"skipped injection (max_similarity=0.22, threshold=0.35)"` on skips; `"injecting (max_similarity=0.61)"` on inject
- Write a unit test: a prompt with no related sessions skips; a prompt matching a past session injects

---

### Ticket 16-02 — Project-directory scoping

**Problem**: wake-up and MCP search query all sessions globally. A Python project gets injected with context from a Go project — irrelevant tokens paid, relevant context diluted.

**Fix**:
- Add a `project_dir` column to the `sessions` table (nullable `TEXT`). Backfill existing rows with `NULL`.
- In `save_hook.py`: read `project_dir` from the hook payload (Claude Code sends `cwd` in the Stop hook payload) and pass it to `upsert_session()`.
- In `wake_up.py` `fetch_recent_sessions()`: add `WHERE project_dir = ? OR project_dir IS NULL` so results are scoped to the current working directory.
- In MCP `memory_search` and `memory_semantic_search`: accept an optional `project_dir` filter parameter.
- Update the dashboard to show sessions grouped by project.

---

### Ticket 16-03 — Switch L1 digest to extracted facts

**Dependency**: Phase 9 (fact extraction) must be complete.

**Problem**: the L1 wake-up digest injects 5 raw session previews (~150 tokens). A one-line preview of "first user message" carries little signal — it's the cheapest possible summary of a session.

**Fix**: replace session previews with extracted facts filtered to the current project:
- Query `SELECT content FROM facts WHERE source='extracted' AND session_id IN (recent project sessions) ORDER BY created_at DESC LIMIT 8`
- Format as a compact bullet list: `• <fact content>` — typically 30–60 chars each
- Cap the total fact digest at 300 chars to bound the token cost regardless of how many facts accumulate
- Keep the identity (L0) block unchanged
- Estimated token cost: ~80 tokens for facts vs. ~150 tokens for raw session previews

If no facts exist yet (Phase 9 not run or new installation), fall back to the current session-preview format.

---

### Ticket 16-04 — Per-project MCP opt-in

**Problem**: the MCP tool schemas (~400 tokens) are added to every Claude session where the server is registered — even sessions that will never use memory retrieval. On a fresh installation with no sessions stored, this is pure overhead.

**Fix**: add a lightweight opt-in check to `mcp_server.py`. On startup, count sessions in the database. If `total_sessions < MIN_SESSIONS_FOR_MCP` (default: 3), the server starts but all tools return a short message: `"Memory index not ready — fewer than 3 sessions stored. Run a few sessions first."` This doesn't remove the schema overhead (that's set by Claude Code, not us) but it prevents wasted API calls before the index has useful data.

Longer term (out of scope for this ticket): document how to scope the MCP server to specific projects using a per-project `.mcp.json` instead of the global `~/.claude/mcp.json`. This lets users opt in project-by-project and avoids paying schema overhead in unrelated workspaces.
