# Agentic Memory System

A local-first, agent-agnostic memory system that gives AI assistants persistent memory across sessions — without sending data to any cloud service.

Every conversation is automatically saved, embedded, and made searchable by keyword or meaning. On your next session, the most relevant past context is silently injected before your first message. Over time, a background daemon extracts facts, discovers patterns, and builds a richer picture of your preferences and working style.

All inference runs locally via [ollama](https://ollama.com) and [sentence-transformers](https://www.sbert.net). No data leaves your machine.

---

## Why this exists

AI assistants have no memory between sessions. Every conversation starts from scratch. You paste the same context repeatedly, re-explain decisions already made, and watch Claude forget what you told it yesterday.

This system fixes that by:
- **Automatically saving** every conversation to a local SQLite database
- **Injecting relevant past context** at the start of each new session (before your first message, invisibly)
- **Letting Claude search its own memory** mid-conversation via MCP tools
- **Learning over time** — a background daemon extracts facts, clusters topics, and generates cross-session insights

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code Session                       │
│                                                             │
│   You type → Claude responds → You type...                  │
└──────┬──────────────────────────────────┬────────────────────┘
       │ UserPromptSubmit hook             │ Stop hook
       │ (fires on first message)          │ (fires after each response)
       ▼                                   ▼
┌─────────────────┐               ┌──────────────────────┐
│   wake_up.py    │               │    save_hook.py       │
│                 │               │                      │
│ 1. Read prompt  │               │ 1. Parse JSONL turns │
│ 2. Semantic     │               │ 2. upsert_session()  │
│    search past  │               │ 3. embed() → vector  │
│    sessions     │               │ 4. chunk_transcript()│
│ 3. FTS5 facts   │               │    → chunk embeddings│
│ 4. List insights│               └──────────┬───────────┘
│ 5. Inject digest│                          │
└────────┬────────┘                          ▼
         │                        ┌──────────────────────┐
         │                        │    ~/.memory/        │
         ▼                        │    memory.db         │
 Appended to your                 │                      │
 first message                    │  sessions            │ transcripts
 (you don't see it)               │  sessions_fts        │ keyword index
                                  │  session_vecs        │ full vectors
                                  │  chunks              │ sub-session vectors
                                  │  facts               │ extracted facts
                                  │  summaries           │ ollama summaries
                                  │  insights            │ cross-session patterns
                                  │  topic_clusters      │ topic groupings
                                  │  retrievals          │ usage tracking
                                  └──────────┬───────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
         ┌──────────────────┐    ┌─────────────────────┐   ┌─────────────────┐
         │  mcp_server.py   │    │     daemon.py        │   │ ingest_server.py│
         │                  │    │                      │   │                 │
         │ MCP tools Claude │    │ Background process:  │   │ HTTP API for    │
         │ can call mid-    │    │ - extract facts      │   │ any agent to    │
         │ conversation:    │    │ - cluster topics     │   │ write sessions  │
         │                  │    │ - generate insights  │   │ POST /ingest    │
         │ memory_hybrid_   │    │ - consolidate old    │   │ GET /status     │
         │  search          │    │   sessions           │   │                 │
         │ memory_search    │    │ Uses ollama locally  │   │ port 7747       │
         │ memory_get_      │    │ CPU-aware scheduling │   └─────────────────┘
         │  session         │    └─────────────────────┘
         │ memory_save_fact │
         │ memory_list_     │
         │  insights        │
         │ ...              │
         └──────────────────┘
```

---

## What happens at each stage

### On every Claude response — `save_hook.py`

Claude Code fires this Stop hook after each assistant response. It:
1. Reads the session transcript from Claude Code's JSONL file
2. Saves all turns to the `sessions` table (FTS5 index updated automatically via trigger)
3. Embeds the full transcript with `all-MiniLM-L6-v2` → stores in `session_vecs`
4. Splits the transcript into overlapping 6-turn windows, embeds each chunk → stores in `chunks`
5. Logs everything to `~/.memory/activity.log`

The hook always exits 0 — errors go to `~/.memory/error.log`, never to Claude.

### On your first message — `wake_up.py`

Claude Code fires this UserPromptSubmit hook before your first message. It:
1. Checks a flag file (`/tmp/memory_injected_{session_id}`) — skips if already run this session
2. Reads your identity profile from `~/.memory/identity.md` (L0)
3. Runs **semantic search** on your prompt against stored chunks → finds the most relevant past sessions (L1)
4. Runs **FTS5 fact search** on your prompt → finds relevant stored facts (L2)
5. Reads top 3 cross-session insights from the daemon (L3)
6. Falls back to recency (last 5 sessions) if no prompt or if sentence-transformers isn't installed
7. Appends the digest to your message via `userPromptSuffix` — Claude sees it, you don't
8. Logs injection size to the `retrievals` table for token economics tracking

### Mid-conversation — `mcp_server.py`

Claude can call these tools whenever relevant context is needed:

| Tool | What it does |
|---|---|
| `memory_hybrid_search(query)` | **Default retrieval** — fuses FTS5 + semantic results via Reciprocal Rank Fusion |
| `memory_search(query)` | Keyword-only FTS5 search |
| `memory_semantic_search(query)` | Meaning-based vector search |
| `memory_get_session(id)` | Full verbatim transcript for one session |
| `memory_status()` | Session count, turn count, date range |
| `memory_save_fact(content, tags?)` | Save a fact worth remembering across sessions |
| `memory_update_fact(id, ...)` | Update a stored fact |
| `memory_delete_fact(id)` | Delete a stored fact |
| `memory_list_facts(tag?)` | Browse stored facts |
| `memory_list_insights(type?)` | List cross-session patterns discovered by the daemon |
| `memory_list_clusters()` | List topic clusters |

### In the background — `daemon.py`

A long-running process (managed by launchd on macOS) that continuously improves the memory store:

1. **Fact extraction** — for each unprocessed session, calls ollama (`llama3.2:3b`) to extract factual statements, preferences, and decisions; deduplicates before storing
2. **Topic clustering** — embeds each session and assigns it to a topic cluster (or creates a new one); centroid updated as a rolling average
3. **Cross-session insights** — every 10 new sessions, sends a sample of recent facts + summaries to ollama asking for patterns, recurring preferences, and skill observations
4. **CPU-aware** — checks `psutil.cpu_percent()` before each inference cycle; skips if above 70%
5. **Backs off** — polls every 5 minutes while sessions are accumulating; backs off to 30 minutes when idle
6. **Stateless checkpoint** — `daemon_processed_at` column tracks progress; safe to restart at any time

### Periodically — `memory/consolidation.py`

Run manually or via cron to compact old sessions:
- `consolidate_old_sessions()` — sessions older than 30 days get a 2–3 sentence ollama summary stored in `summaries`
- `prune_old_transcripts()` — sessions older than 90 days that already have a summary get their raw transcript nulled (saves disk space while preserving the summary)

---

## File layout

```
agentic-memory/
├── install.sh                   one-command installer
├── com.memory.daemon.plist      launchd plist for daemon auto-start (macOS)
├── cli.py                       command-line interface
├── hooks/
│   ├── save_hook.py             Stop hook — saves every transcript
│   └── wake_up.py              UserPromptSubmit hook — injects memory digest
├── memory/
│   ├── db.py                   storage layer (all SQLite/FTS5/embedding logic)
│   ├── mcp_server.py           MCP server (11 tools)
│   ├── daemon.py               background relearning daemon
│   ├── consolidation.py        session summarisation and transcript pruning
│   ├── ingest_server.py        HTTP ingest API for non-Claude agents
│   ├── client.py               thin Python client for ingest API
│   └── logger.py               structured activity and error logging
└── tests/                       151 tests (pytest)

~/.memory/                       created on first use
├── memory.db                   SQLite database
├── identity.md                 your L0 identity profile (edit this)
├── activity.log                one line per memory operation
├── error.log                   errors with tracebacks
├── save_hook.log               save hook activity
├── wake_up.log                 wake-up hook errors
└── daemon.log                  daemon activity
```

---

## Installation

**Prerequisites:**
- Python 3.8+
- [ollama](https://ollama.com) installed and running, with `llama3.2:3b` pulled:
  ```bash
  ollama pull llama3.2:3b
  ```
  The daemon and consolidation use this model. The core save/search/wake-up path works without it.

**Install:**

```bash
git clone <repo-url> ~/agentic-memory
cd ~/agentic-memory
bash install.sh
```

The installer:
1. Checks Python 3.8+
2. Installs Python deps: `mcp fastmcp sentence-transformers fastapi uvicorn pydantic psutil`
3. Creates `~/.memory/` and writes a starter `~/.memory/identity.md` (skipped if already exists)
4. Idempotently patches `~/.claude/settings.json` to register both hooks
5. Writes the MCP server entry to `~/.claude/mcp.json`
6. Substitutes real paths into `com.memory.daemon.plist`

**Then restart Claude Code** for hooks and MCP server to take effect.

**Uninstall** (preserves all data in `~/.memory/`):

```bash
bash install.sh --uninstall
```

---

## Personalise your identity profile

Edit `~/.memory/identity.md`. This is injected at the start of every session as L0 context:

```markdown
# Identity

Name: Alex Chen
Role: Senior backend engineer at Acme Corp

## About me
10 years of Go and Python. Currently leading the authentication refactor.
Prefer concise explanations over long prose.

## Preferences
- Always show file paths when referencing code
- I use pytest, not unittest
- Dark mode everywhere
```

---

## CLI reference

```bash
# Database overview
python3 cli.py status                    # sessions, turns, date range, token economics

# Search
python3 cli.py search "auth bug"         # FTS5 keyword search
python3 cli.py semantic "login timeout"  # semantic vector search
python3 cli.py get-session <id>          # print full transcript
python3 cli.py tail 10                   # last 10 sessions

# Dashboard
python3 cli.py dashboard                 # generate dashboard.html and open in browser

# Logs
python3 cli.py logs                      # tail activity.log (last 50 lines)
python3 cli.py logs --errors             # tail error.log
python3 cli.py logs --tail 100           # last 100 lines

# Consolidation (manual — or set up a cron job)
python3 cli.py consolidate               # summarise sessions older than 30 days
python3 cli.py consolidate --days 14     # lower threshold
python3 cli.py consolidate --dry-run     # preview only, no writes
python3 cli.py prune                     # null raw transcripts older than 90 days
python3 cli.py prune --days 60 --dry-run

# Daemon management
python3 cli.py daemon start              # launch background daemon
python3 cli.py daemon stop               # stop it
python3 cli.py daemon status             # running/stopped, PID
python3 cli.py daemon --once             # run one pass and exit (useful for testing)

# Ingest server (for non-Claude agents)
python3 cli.py ingest-server start       # start HTTP server on port 7747
python3 cli.py ingest-server stop
python3 cli.py ingest-server status
```

---

## Pi integration

Your Pi conversations can be saved to the same memory store so Claude can reference what you and Pi already discussed.

**Prerequisites:** the ingest server must be running (it receives sessions from the browser):

```bash
python3 cli.py ingest-server start
```

If you ran `install.sh`, the launchd agent keeps it running automatically after login.

### Option A — bookmarklet (click to save after each conversation)

1. Generate the bookmark URL:
   ```bash
   python3 integrations/pi/make_bookmarklet.py
   ```
2. Open the generated file `integrations/pi/bookmarklet_url.txt`, select all, and copy.
3. Create a new bookmark in your browser (name it "Save to Memory").
4. Paste the copied text as the bookmark **URL** (not the name).
5. Go to [pi.ai](https://pi.ai), have a conversation, then click the bookmark.
6. A confirmation alert will appear: "Saved N turns to memory."

### Option B — Tampermonkey userscript (fully automatic, no clicking)

1. Install [Tampermonkey](https://www.tampermonkey.net) for your browser.
2. Open Tampermonkey > Dashboard > **+** (new script tab).
3. Delete the placeholder code and paste the entire contents of `integrations/pi/pi_memory.user.js`.
4. Save (Ctrl+S).

The script watches the Pi page for new messages. 15 seconds after each response, it automatically POSTs the conversation to the ingest server. You'll see `[pi-memory] saved N turns` in the browser console (F12 > Console).

### How Pi sessions appear in Claude

Once saved, Pi sessions are stored alongside Claude sessions in the same database. The wake-up hook searches across all sessions — so when you start a Claude session on a topic you discussed with Pi, those Pi turns surface in the memory digest automatically. You do not need to do anything extra; the search is agent-agnostic.

---

## Using from another agent

Any agent (Cursor, LangChain, custom scripts) can write sessions to the same store.

**Option 1 — Python client:**

```python
from memory.client import MemoryClient

client = MemoryClient()  # connects to localhost:7747
client.save_session(
    session_id="my-session-001",
    agent="cursor",
    turns=[
        {"role": "user",      "content": "How does the auth flow work?"},
        {"role": "assistant", "content": "The auth flow starts with..."},
    ],
)
```

**Option 2 — HTTP directly:**

```bash
# Start the server first
python3 cli.py ingest-server start

# POST a session
curl -X POST http://localhost:7747/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-001",
    "agent": "cursor",
    "turns": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }'

# Check status
curl http://localhost:7747/status
```

The ingest server runs the same full pipeline as the Claude Code Stop hook (upsert → embed → chunk), so sessions from any agent are fully searchable and factored into wake-up injection.

---

## Auto-start daemon on macOS login

```bash
# Substitute your real paths into the plist (install.sh already does this)
cp com.memory.daemon.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.memory.daemon.plist
```

To stop it: `launchctl unload ~/Library/LaunchAgents/com.memory.daemon.plist`

---

## Token economics

The system tracks the overhead it adds vs the context it returns:

```
python3 cli.py status
```

```
=== Token Economics ===
  Wake-up injections:    47 sessions
  Est. tokens injected:  16,450
  Est. tokens retrieved: 23,810
  Coverage ratio:        1.45x (145%)
  Interpretation:        retrievals returned more context than was injected
```

A **coverage ratio ≥ 1.0** means the MCP retrievals returned more context than was injected at session start — the system is net-positive on information density. This does not measure tokens saved vs. a no-memory baseline (that would require a control group), but a ratio consistently below 1.0 is a signal the system costs more than it returns for your usage pattern.

---

## Local model choices

| Component | Model | Size | When it runs |
|---|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | ~90 MB | Save hook, wake-up, daemon |
| Summarisation | `llama3.2:3b` (ollama) | ~2 GB | Consolidation, daemon |
| Fact extraction | `llama3.2:3b` (ollama) | ~2 GB | Daemon only |
| Cross-session insights | `llama3.2:3b` (ollama) | ~2 GB | Daemon only |

Override the ollama model: `export MEMORY_OLLAMA_MODEL=llama3.1:8b`

The ollama model is loaded into RAM only while actively running (~2–3 GB) and unloaded automatically after 5 minutes of idle. The daemon checks CPU load before each inference cycle and skips if the machine is above 70% load.

---

## Running tests

```bash
source venv/bin/activate
python3 -m pytest -q       # 151 tests
```

---

## Data privacy

- All data stays on your machine in `~/.memory/memory.db`
- No data is sent to any external service
- The MCP server, hooks, daemon, and ingest server all run as local processes
- `~/.memory/` is never touched by `install.sh --uninstall` — your history is preserved
