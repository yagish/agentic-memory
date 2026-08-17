# Agentic Memory System

A local-first memory system for Claude Code. Every conversation is automatically saved to a SQLite database and made searchable — by keyword or meaning — in future sessions.

---

## How It Works

```
┌─────────────────────────────────────────────┐
│              Claude Code Session             │
│                                             │
│  You type → Claude responds → You type...  │
└──────┬───────────────────────────┬──────────┘
       │ UserPromptSubmit hook     │ Stop hook
       │ (fires on first message)  │ (fires after every response)
       ▼                           ▼
┌─────────────┐           ┌──────────────────┐
│  wake_up.py │           │  save_hook.py    │
│             │           │                  │
│ Reads last  │           │ Reads transcript │
│ 5 sessions  │           │ from disk, saves │
│ + identity  │           │ to SQLite +      │
│             │           │ stores a vector  │
│ Appends     │           └────────┬─────────┘
│ digest to   │                    │
│ your prompt │                    ▼
└─────────────┘           ┌─────────────────┐
                          │  ~/.memory/     │
                          │  memory.db      │
                          │                 │
                          │  sessions       │  full transcripts
                          │  sessions_fts   │  keyword search index
                          │  session_vecs   │  embedding vectors
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  mcp_server.py  │
                          │                 │
                          │  memory_status          │
                          │  memory_search          │  keyword search
                          │  memory_get_session     │  fetch transcript
                          │  memory_semantic_search │  meaning search
                          └─────────────────┘
```

---

## Components

### `hooks/save_hook.py` — Stop Hook

Registered in `.claude/settings.json`. Fires after every Claude response.

- Receives the JSONL transcript path from Claude Code via stdin
- Parses turns, skipping internal `isMeta` and `thinking` blocks
- Upserts the full transcript into `sessions` (FTS5 triggers keep the index in sync)
- Runs the transcript through `all-MiniLM-L6-v2` to produce a 384-float vector, stores it in `session_vecs`
- Always exits 0 — errors are logged to `~/.memory/save_hook.log`, never surfaced to Claude

### `hooks/wake_up.py` — UserPromptSubmit Hook

Registered in `.claude/settings.json`. Fires before your first message each session.

- Runs once per session; a flag file at `/tmp/memory_injected_{session_id}` prevents re-firing
- Reads `~/.memory/identity.md` (L0 — who you are, ~100 tokens)
- Queries the 5 most recent sessions, extracts a one-line preview from each (L1 — recent context)
- Appends the digest to your first message via `hookSpecificOutput.userPromptSuffix` — Claude sees it, you don't
- Falls back to `{"decision": "allow"}` on any error

### `memory/db.py` — Storage Layer

All SQLite logic lives here. Other modules call these functions; none write SQL directly.

| Function | What it does |
|---|---|
| `init_db(path)` | Opens/creates the database, runs schema migrations |
| `upsert_session(...)` | Inserts or replaces a full session transcript |
| `search(conn, query, limit)` | FTS5 full-text search, returns ranked excerpts |
| `embed(text)` | Converts text to a 384-float vector using `all-MiniLM-L6-v2` |
| `store_embedding(conn, session_id, vector)` | Saves a vector to `session_vecs` |
| `semantic_search(conn, query, limit)` | Cosine similarity search across all stored vectors |

**Schema:**

```sql
sessions        -- one row per session; transcript stored as JSON text
sessions_fts    -- FTS5 virtual table; kept in sync by INSERT/UPDATE triggers
session_vecs    -- one BLOB per session; 384 × float32 = 1,536 bytes
```

**Embedding:** `all-MiniLM-L6-v2` via `sentence-transformers` (~90 MB, runs fully locally, no Ollama needed). Cosine similarity is computed in Python — macOS system Python doesn't support SQLite extension loading, so `sqlite-vec`'s SQL functions aren't used.

### `memory/mcp_server.py` — MCP Server

Registered in `.mcp.json`. Claude Code starts it automatically as a subprocess.

| Tool | What Claude can do with it |
|---|---|
| `memory_status` | Total sessions, turns, date range |
| `memory_search(query, limit?)` | Keyword search — exact/partial word matches, ranked by relevance |
| `memory_get_session(session_id)` | Full verbatim transcript for one session |
| `memory_semantic_search(query, limit?)` | Meaning-based search — finds sessions even when exact words differ |

---

## File Layout

```
agentic-memory/
  memory/
    __init__.py
    db.py              storage layer (SQLite, FTS5, embeddings)
    mcp_server.py      MCP server (4 tools Claude can call)
  hooks/
    save_hook.py       Stop hook — saves every transcript
    wake_up.py         UserPromptSubmit hook — injects memory digest
  tests/
    test_db.py         storage layer tests
    test_mcp_server.py MCP tool tests
    test_save_hook.py  hook parsing and save tests
    test_wake_up.py    wake-up digest tests
    test_semantic_search.py  embedding and vector search tests
  .claude/
    settings.json      hook registrations
  .mcp.json            MCP server registration
  ~/.memory/
    memory.db          the database (created on first save)
    identity.md        your L0 identity profile (edit this yourself)
    save_hook.log      save hook activity log
    wake_up.log        wake-up hook error log
  IMPLEMENTATION_PLAN.md   phase-by-phase build plan
  MEMORY_SYSTEM_DESIGN_BRIEF.md   original design notes
  README.md            this file
```

---

## Setup

**Install dependencies:**

```bash
pip3 install mcp fastmcp sentence-transformers
```

**Create your identity file** (optional but recommended):

```bash
mkdir -p ~/.memory
cat > ~/.memory/identity.md << 'EOF'
Name: <your name>
Role: <your role>
Current focus: <what you're working on>
EOF
```

**Run the tests:**

```bash
python3 -m unittest discover -s tests -v
```

---

## Phases

| Phase | Status | What it delivers |
|---|---|---|
| 1 — Storage Layer | ✅ Done | `db.py`, SQLite schema, FTS5, unit tests |
| 2 — Save Hook | ✅ Done | `save_hook.py`, transcript parsing, Stop hook wired |
| 3 — MCP Server | ✅ Done | `mcp_server.py`, 3 tools, `.mcp.json` registration |
| 4 — Wake-up Injection | ✅ Done | `wake_up.py`, L0 + L1 digest on session start |
| 5 — Semantic Search | ✅ Done | `embed()`, `session_vecs`, `memory_semantic_search` tool |
| 6 — CLI + Dashboard | ✅ Done | `cli.py` commands, retrieval logging, local HTML dashboard |

### Phase 6 — CLI + Dashboard

```bash
python3 cli.py status              # session count, turn count, date range
python3 cli.py search "query"      # FTS5 keyword search, ranked results
python3 cli.py semantic "query"    # semantic vector search
python3 cli.py get-session <id>    # print full transcript
python3 cli.py tail 10             # last 10 sessions
python3 cli.py dashboard           # generate dashboard.html and open in browser
```

A `retrievals` table logs every MCP tool call (timestamp, tool name, query, result size). The dashboard shows session count over time, total turns, estimated tokens stored, and retrieval activity broken down by tool.

---

## Data Flow: Save

```
Claude stops responding
  → Claude Code reads transcript JSONL from disk
  → Passes path to save_hook.py via stdin JSON
  → Hook parses turns (skips meta/thinking blocks)
  → upsert_session() → sessions table + FTS5 index (via trigger)
  → embed(full_text) → all-MiniLM-L6-v2 → 384 floats
  → store_embedding() → session_vecs table
  → Hook exits 0
```

## Data Flow: Wake-up

```
You send first message of a session
  → UserPromptSubmit hook fires
  → Checks /tmp/memory_injected_{session_id} — if exists, skip
  → Reads ~/.memory/identity.md (L0)
  → Queries last 5 sessions from SQLite (L1)
  → Builds digest text
  → Writes flag file
  → Outputs {"hookSpecificOutput": {"userPromptSuffix": "\n\n=== MEMORY WAKE-UP ===\n..."}}
  → Claude sees your message + digest; you only see your message
```

## Data Flow: Search

```
You ask Claude "what did we discuss about authentication?"
  → Claude calls memory_semantic_search("authentication")
  → embed("authentication") → query vector
  → Fetch all session_vecs rows
  → Compute cosine distance in Python for each
  → Return top N sorted by distance
  → Claude calls memory_get_session(session_id) for the best match
  → Reads full verbatim transcript
```
