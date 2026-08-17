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

## Phase 9 — Structured Fact Extraction

**Goal**: after every session save, automatically extract durable facts (decisions, preferences, entities) from the transcript using an LLM call, and store them in the `facts` table.

**Dependency**: Phase 8 (facts table + `insert_fact()` must exist first).

### Ticket 9-01 — Implement `memory/fact_extractor.py`

- `extract_facts(transcript_text) -> list[dict]` — calls Claude API (`claude-haiku-4-5-20251001`, low cost) with a structured prompt
- Prompt instructs the model to return a JSON array of `{content, tags}` objects
- Extract only: decisions made, user preferences, reusable patterns, named entities with context
- Skip: transient task details, code that already lives in files, ephemeral questions
- Parse and validate the JSON response; return an empty list on failure (never raise)
- Write unit tests with mocked API responses: valid JSON parsed correctly, malformed JSON returns `[]`

### Ticket 9-02 — Wire fact extraction into `save_hook.py`

- After `upsert_session()` succeeds, call `extract_facts(full_text)`
- For each returned fact, call `insert_fact(conn, content, tags, source="extracted", session_id=session_id)`
- Guard with a try/except — extraction failure must never prevent the session save
- Log extraction count to `save_hook.log` (e.g. "extracted 4 facts from session abc123")
- Write integration test: a transcript containing a clear decision results in at least one fact row

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

### Ticket 12-01 — Build `memory/consolidation.py` — L2 topic summaries

- `consolidate_old_sessions(conn, days_threshold=30)` — finds sessions older than N days that haven't been summarised
- For each, calls Claude API to produce a 3–5 sentence summary of key topics, decisions, and outcomes
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

- `re_extract_facts(conn, session)` — calls Claude API (Haiku) with full transcript; upserts new facts, skips duplicates by content similarity
- `update_topic_clusters(conn, session)` — embeds the session summary, assigns it to the nearest existing topic cluster or creates a new one; stores cluster assignment in a `topic_clusters` table
- `generate_cross_session_insights(conn)` — called every N sessions (default: every 10 new sessions); sends a sample of recent facts + session summaries to Claude API asking for patterns, recurring preferences, and skill observations; upserts results into `insights`
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

## Updated Build Order (Phases 7–12)

```
Phase 7 (chunking)  →  Phase 8 (fact write tools)  →  Phase 9 (fact extraction)
       ↓                                                        ↓
Phase 11 (hybrid search)  ←  Phase 10 (relevance wake-up)  ←  (uses chunks + facts)
       ↓
Phase 12 (consolidation)   ←  can start any time after Phase 7
```

**Recommended order**: 7-01 → 7-02 → 7-03 → 8-01 → 8-02 → 9-01 → 9-02 → 10-01 → 11-01 → 11-02 → 12-01 → 12-02 → 13-01 → 13-03 → 13-02 → 13-04 → 13-05

Phase 13 depends on Phase 12 (consolidation must exist before the daemon re-uses it), and on Phase 9 (fact extraction must exist before re-extraction). Phase 13-02 (the daemon loop) can start once 13-01 and 13-03 are done.
