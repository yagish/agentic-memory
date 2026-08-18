# Agentic Memory

A local-first memory system that gives AI assistants persistent memory across sessions — without sending data to any cloud service.

Every conversation is automatically saved and indexed. Before each prompt, the system checks whether a cached answer already exists and injects it directly. Relevant facts and cross-session patterns are surfaced silently. A background daemon continuously learns from past sessions, extracts facts, and prunes stale data.

All inference runs locally via [Ollama](https://ollama.com) and [sentence-transformers](https://www.sbert.net). No data leaves your machine.

---

## How it works — end to end

```
You type a prompt
       │
       ▼
┌─────────────────────────────────────────────┐
│  wake_up.py  (UserPromptSubmit hook)         │
│                                             │
│  1. Semantic search: does memory have an    │
│     answer ≥ 93% similar to this prompt?   │
│     → YES: inject [Cached Answer] block     │
│             Claude responds with [From      │
│             Memory] prefix                  │
│     → NO:  continue to LLM                 │
│  2. FTS5 search: relevant facts for prompt  │
│  3. First message of session: inject        │
│     cross-session insights (once per        │
│     session only)                           │
│  4. Always inject identity profile (L0)     │
└──────────────────┬──────────────────────────┘
                   │ suffix appended to prompt
                   ▼
          Claude LLM call (or skipped if cached)
                   │
                   ▼
         Claude responds
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  save_hook.py  (Stop hook)                  │
│                                             │
│  Runs after EVERY assistant response:       │
│  1. Reads JSONL transcript                  │
│  2. Upserts session to SQLite (FTS5 index)  │
│  3. Embeds transcript → session_vecs        │
│  4. Chunks transcript into 6-turn windows   │
│     → embeds each chunk for retrieval       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
          ~/.memory/memory.db

                   │ (every 5 min, background)
                   ▼
┌─────────────────────────────────────────────┐
│  daemon.py                                  │
│                                             │
│  For each unprocessed session:              │
│  1. Extract facts (LLM)                     │
│  2. Assign topic cluster                    │
│  Every 10 sessions:                         │
│  3. Generate cross-session insights (LLM)   │
│  Every cycle:                               │
│  4. Prune processed sessions > 30 days old  │
└─────────────────────────────────────────────┘
```

---

## Services

Three background services run at all times (managed by launchd on macOS):

| Service | Port | Purpose | Log |
|---|---|---|---|
| **daemon** | — | Extracts facts, clusters topics, generates insights, prunes old sessions | `~/.memory/daemon.log` |
| **ingest server** | 7747 | HTTP write API — any agent can POST sessions here | `~/.memory/ingest.log` |
| **dashboard server** | 7748 | Dashboard UI + operational API (restart, compress, logs) | `~/.memory/query.log` |

**Dashboard:** `http://localhost:7748`

---

## Memory layers

Each prompt injects context in priority order:

| Layer | What | When |
|---|---|---|
| **Cached Answer** | A past Q&A pair ≥ 93% semantically similar to this prompt | Every prompt (if found) |
| **Identity (L0)** | `~/.memory/identity.md` — your name, role, preferences | Every prompt |
| **Relevant Facts (L2)** | Facts from the DB matching this prompt (FTS5) | Every prompt |
| **Insights (L3)** | Cross-session patterns learned by the daemon | Once per session (first message only) |

When a cached answer is used, Claude prefixes its response with `[From Memory]` so you know the answer came from stored memory, not a fresh LLM call.

---

## Local models

| Component | Model | Size | When it runs |
|---|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | ~90 MB | Save hook, wake-up hook, daemon |
| Fact extraction | `llama3.2:3b` (Ollama) | ~2 GB | Daemon, every new session |
| Insights | `llama3.2:3b` (Ollama) | ~2 GB | Daemon, every 10 sessions |
| Compression | `llama3.2:3b` (Ollama) | ~2 GB | Manual or dashboard button |

Override the Ollama model: `export MEMORY_OLLAMA_MODEL=llama3.1:8b`

Ollama loads the model into RAM only while actively running and unloads it after idle. The daemon checks CPU load before each inference cycle and skips if load is above 70%.

---

## Installation

**Prerequisites:**
- macOS (launchd required for background services)
- Python 3.8+
- [Ollama](https://ollama.com) — install via `brew install ollama`
- Claude Code

```bash
git clone <repo-url> ~/agentic-memory
cd ~/agentic-memory
bash install.sh
```

The installer runs these steps:

1. Checks Python 3.8+
2. Installs Python deps: `mcp fastmcp sentence-transformers fastapi uvicorn pydantic psutil setproctitle`
3. Downloads and imports the Ollama model (see [Model download](#model-download) below)
4. Creates `~/.memory/` directory
5. Writes `~/.memory/identity.md` (interactive prompts for name, role, tech stack — skipped if file exists)
6. Registers `save_hook.py` (Stop hook) in `~/.claude/settings.json`
7. Registers `wake_up.py` (UserPromptSubmit hook) in `~/.claude/settings.json`
8. Registers the MCP server in `~/.claude/mcp.json`
9. Installs and starts `com.memory.daemon` via launchd
10. Installs and starts `com.memory.ingest` (port 7747) via launchd
11. Installs and starts `com.memory.query` (port 7748) via launchd

**Then restart Claude Code** for hooks and MCP server to take effect.

### Model download

The installer uses `qwen2.5:3b` by default and handles corporate proxy environments that block the Ollama registry:

1. **Try `ollama pull qwen2.5:3b`** directly from the Ollama registry
2. **If that fails** (e.g. proxy blocks `registry.ollama.ai`), download the GGUF file from HuggingFace (`~2 GB`) to `~/.memory/models/` and import it with `ollama create`
3. **Skip** the whole step if the model is already present in Ollama

HuggingFace is used as the fallback because most corporate proxies allow it while blocking the Ollama registry.

The GGUF is stored at `~/.memory/models/` — it survives reinstalls and is never in the repo.

**Override env vars:**

```bash
# Use a different Ollama model (must be available via pull or manual import)
export MEMORY_OLLAMA_MODEL=phi3.5:mini

# Use a different HuggingFace GGUF URL for the fallback download
export MEMORY_HF_GGUF_URL=https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf
```

**Uninstall** (all data in `~/.memory/` is preserved):

```bash
bash install.sh --uninstall
```

---

## File layout

```
agentic-memory/
├── install.sh                   one-command installer + uninstaller
├── hooks/
│   ├── wake_up.py               UserPromptSubmit hook — cached answer + memory injection
│   └── save_hook.py             Stop hook — upserts session after every response
├── memory/
│   ├── db.py                    storage layer (SQLite, FTS5, embeddings, facts)
│   ├── mcp_server.py            MCP server — 11 tools Claude can call mid-conversation
│   ├── daemon.py                background learner (facts, insights, topic clusters, pruning)
│   ├── ingest_server.py         HTTP write API for non-Claude agents (port 7747)
│   ├── dashboard_server.py      dashboard UI + ops API (port 7748)
│   ├── compress.py              LLM-based map-reduce compression of all sessions
│   ├── consolidation.py         session summarisation and transcript pruning
│   ├── client.py                thin Python client for the ingest API
│   └── logger.py                structured activity and error logging
├── dashboard.html               dashboard frontend (served by dashboard_server.py)
└── tests/

~/.memory/                       created on first use, never deleted by uninstall
├── memory.db                    SQLite — sessions, facts, chunks, insights, retrievals
├── identity.md                  your L0 identity profile (edit freely)
├── compressed_memory.md         latest compressed memory document (if compression was run)
├── activity.log                 one line per memory operation
├── daemon.log                   daemon activity (fact extraction, insights, pruning)
├── ingest.log                   ingest server
├── query.log                    dashboard server
└── wake_up.log                  wake-up hook errors
```

---

## Database schema

```
sessions         — raw transcripts, turn count, dates, daemon_processed_at
sessions_fts     — FTS5 full-text index (auto-synced via trigger)
session_vecs     — whole-session embeddings (semantic search)
chunks           — 6-turn overlapping windows with embeddings (sub-session retrieval)
facts            — extracted facts with tags and source session
insights         — cross-session patterns (type, confidence)
topic_clusters   — session groupings with centroid embeddings
retrievals       — usage log (tool, query, result size, timestamp)
compressed_memory — output of map-reduce compression runs
```

---

## MCP tools

Claude can call these tools mid-conversation when it needs memory:

| Tool | What it does |
|---|---|
| `memory_hybrid_search(query)` | Fuses FTS5 + semantic results via Reciprocal Rank Fusion |
| `memory_search(query)` | Keyword-only FTS5 search |
| `memory_semantic_search(query)` | Meaning-based vector search |
| `memory_get_session(id)` | Full verbatim transcript for one session |
| `memory_status()` | Session count, turn count, date range |
| `memory_save_fact(content, tags?)` | Save a fact worth remembering |
| `memory_update_fact(id, ...)` | Update a stored fact |
| `memory_delete_fact(id)` | Delete a stored fact |
| `memory_list_facts(tag?)` | Browse stored facts |
| `memory_list_insights(type?)` | Cross-session patterns from the daemon |
| `memory_list_clusters()` | Topic clusters |

---

## Personalise your identity profile

Edit `~/.memory/identity.md`. Injected at the start of every session:

```markdown
# Identity

Name: Alex Chen
Role: Senior backend engineer

## About me
Primary tech: Python, Go, AWS
Experience: 8 years

## Preferences
- Concise responses — skip the preamble
- Always show file paths when referencing code
- I use pytest, not unittest
```

The assistant will silently add more facts here as it learns them.

---

## Send sessions from another agent

Any agent (Cursor, LangChain, custom scripts) can write to the same store.

**Python client:**

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

**HTTP directly:**

```bash
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
```

---

## Compress memory (manual)

The dashboard has a **Compress Memory** button that runs a map-reduce LLM pass over all sessions, produces a single structured markdown document, and deletes the raw sessions. Use it when the DB grows large or before handing off to a new agent.

CLI:

```bash
python3 memory/compress.py             # compress + delete sessions
python3 memory/compress.py --dry-run   # preview only
python3 memory/compress.py --model llama3.1:8b
```

Output saved to `~/.memory/compressed_memory.md` and the `compressed_memory` table.

---

## Data privacy

- All data stays on your machine in `~/.memory/memory.db`
- No data is sent to any external service
- Ollama runs 100% locally — the LLM never sees your prompts over the network
- `bash install.sh --uninstall` removes hooks and services but never touches `~/.memory/`
