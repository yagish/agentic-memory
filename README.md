# Agentic Memory

A persistent memory system for Claude Code that makes every new session feel like a continuation — not a cold start. It stores, compacts, and retrieves conversation history so Claude always has the right context without you having to re-explain anything.

**Two goals:**
1. Remove the "start from scratch" annoyance across sessions.
2. Reduce wasted tokens re-establishing context at the beginning of every conversation.

---

## How it works — end to end

```
User sends a prompt
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  hooks/wake_up.py  (Claude Code UserPromptSubmit)   │
│                                                     │
│  1. Embed prompt                                    │
│  2. Search compacted_sessions by similarity         │
│     ≥ 96%  → inject full cached session summary     │
│     70–95% → inject as enrichment context           │
│  3. Pull working_memory for best-matching cluster   │
│     (first message of session only)                 │
│  4. Search facts via FTS                            │
│  5. Search procedural_memory for how-to prompts     │
│  6. Build one suffix under a 500-token budget       │
└─────────────────────────────────────────────────────┘
        │
        ▼
   Claude processes prompt + injected context
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  hooks/save_hook.py  (Claude Code Stop hook)        │
│                                                     │
│  • Saves full transcript → sessions                 │
│  • Generates whole-session embedding → session_vecs │
│  • Chunks transcript → chunks                       │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  memory/daemon.py  (background process)             │
│                                                     │
│  Pass 1 — per new session:                          │
│  • LLM → episodic entry (title + abstract)          │
│  • Assign to topic cluster                          │
│  • Update working_memory for that cluster           │
│  • If cluster ≥ 3 sessions → compact                │
│  • LLM → extract procedural patterns                │
│                                                     │
│  Pass 2 — periodic sweep:                           │
│  • Close stale working_memory (>14 days)            │
│  • Merge near-duplicate compacted_sessions          │
│  • Prune old processed sessions                     │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  memory/ingest_server.py  (agent-facing HTTP API)   │
│                                                     │
│  • POST /ingest  → remote session writes            │
│  • POST /recall  → wake-up digest for other agents  │
│  • POST /answer  → repeated-question lookup         │
└─────────────────────────────────────────────────────┘
```

---

## Memory layers

| Layer | Table | Lifetime | What it contains |
|---|---|---|---|
| **Working** | `working_memory` | 14 days of inactivity | Rolling task context — what you've been working on across recent sessions |
| **Episodic** | `episodic_memory` | Forever | Thin log: one title + 2-sentence abstract per session |
| **Semantic cache** | `compacted_sessions` | Forever | LLM-compacted session summaries with vectors — the primary retrieval target |
| **Procedural** | `procedural_memory` | Forever | Reusable how-to patterns and workflows extracted from sessions |
| **Facts** | `facts` | Until updated | Searchable facts, including identity/profile facts synced from `~/.memory/identity.md` |
| **Insights** | `insights` | Forever | Cross-session patterns and meta-observations surfaced by APIs and dashboard |

---

## Token efficiency design

`hooks/wake_up.py` always starts with retrieval, then only injects what actually matches.

**Semantic threshold** — compacted session context is only injected if cosine similarity ≥ 70%. Below that the match is treated as noise.

**Working memory — first message only** — working memory is only considered on the first prompt of a Claude Code session.

**Procedural memory — conditional** — procedural patterns are only searched for prompts containing how-to markers (`how`, `steps`, `best way`, `should i`, `approach`, `workflow`, etc.).

**Facts on demand** — fact injection is driven by an FTS query built from the current prompt. This is how identity/profile facts surface when relevant.

**500-token budget** — total injection is capped at 500 tokens. Priority if budget is exceeded: cache hit > working memory > enrichment context > facts > procedural.

---

## Compacted session format

The daemon compacts sessions into a structured summary:

```
Task: [one line — what the session was about]
Context: [repo, language, key components involved]
What was tried: [bullet points]
Outcome: [what worked / current state]
Left off at: [where to pick up next time]
```

Summary length scales with session size:

| Session size | Summary target |
|---|---|
| < 5 turns | Skipped |
| 5–15 turns | ~150 tokens |
| 16–40 turns | ~400 tokens |
| 40+ turns | ~800 tokens |

At the moment, `hooks/wake_up.py` injects the full compacted summary block for high-similarity matches and does not slice out individual fields.

---

## Retrieval modes in the current codebase

### Claude Code hook (`hooks/wake_up.py`)

The hook builds a prompt suffix from these sources, in order:

1. `compacted_sessions` cache hit (`>= 0.96` similarity)
2. `working_memory` for the best matching cluster on the first message only
3. enrichment summaries from other similar compacted sessions (`0.70–0.95`)
4. fact matches from `facts`
5. procedural matches from `procedural_memory`

A high-similarity cache hit injects the stored compacted session summary plus an instruction to answer from that memory and prefix the reply with `[From Memory]`.

### Agent HTTP API (`memory/ingest_server.py`)

Other agents can use two related retrieval endpoints:

- `POST /recall` — builds a digest from `identity.md`, recent/relevant sessions, facts, and insights
- `POST /answer` — returns a previously seen assistant answer for a near-identical user prompt using `find_direct_answer()`

---

## Services

| Process | How to run | What it does |
|---|---|---|
| Daemon | `python3 memory/daemon.py` | Compacts sessions, updates episodic/working/procedural memory |
| Ingest server | `python3 memory/ingest_server.py` | HTTP API for `/ingest`, `/recall`, and `/answer` |
| Dashboard | `python3 memory/dashboard_server.py` | Web UI and operational API at `http://localhost:7748` |
| MCP server | `python3 memory/mcp_server.py` | Exposes memory tools to Claude via MCP |

The daemon is the only required background process for Claude Code hook-based memory. Ingest server, dashboard, and MCP server are optional depending on how you use the system.

---

## Local models

All LLM calls in the daemon go through a local [ollama](https://ollama.com) instance — no Anthropic API tokens are consumed for compaction.

```bash
# Install ollama, then pull the default model
ollama pull llama3.2:3b
```

Override the model with the environment variable:
```bash
MEMORY_OLLAMA_MODEL=mistral python3 memory/daemon.py
```

Embeddings use `sentence-transformers/all-MiniLM-L6-v2` (~90 MB, runs fully locally, no GPU required).

---

## Installation

```bash
# 1. Clone the repo
git clone <repo>
cd agentic-memory

# 2. Install everything with the bootstrap script
./install.sh

# 3. If you want to do it manually instead of using install.sh:
python3 -m pip install --user \
  mcp fastmcp sentence-transformers fastapi uvicorn \
  pydantic psutil setproctitle debugpy

# 4. Start the daemon if install.sh did not already register/run it for you
python3 memory/daemon.py &

# 5. Sync your identity profile into the facts table
python3 cli.py sync-identity
```

## Remote debugging

All entry scripts support env-gated `debugpy` attach via `memory/debug.py`.

Per-script environment variables:

- `MEMORY_DEBUG_DAEMON_PORT`
- `MEMORY_DEBUG_INGEST_PORT`
- `MEMORY_DEBUG_QUERY_PORT`
- `MEMORY_DEBUG_MCP_PORT`
- `MEMORY_DEBUG_SAVE_HOOK_PORT`
- `MEMORY_DEBUG_WAKE_UP_PORT`
- `MEMORY_DEBUG_CLI_PORT`

Optional globals:

- `MEMORY_DEBUG_HOST` — defaults to `127.0.0.1`
- `MEMORY_DEBUG_WAIT=1` — wait for debugger attach before continuing

Examples:

```bash
# Long-running service: stop launchd copy, then run under the normal script entrypoint
./scripts/stop-memory.sh daemon
MEMORY_DEBUG_DAEMON_PORT=5678 MEMORY_DEBUG_WAIT=1 python3 memory/daemon.py

# Or use the helper start script. If a debug port is set, it bypasses launchctl
# and launches the service directly so the env vars are preserved.
./scripts/stop-memory.sh ingest
MEMORY_DEBUG_INGEST_PORT=5679 MEMORY_DEBUG_WAIT=1 ./scripts/start-memory.sh ingest

# Hooks are short-lived, so replay them manually instead of blocking Claude Code.
MEMORY_DEBUG_SAVE_HOOK_PORT=5682 python3 hooks/save_hook.py --dry-run < sample-save-payload.json
MEMORY_DEBUG_WAKE_UP_PORT=5683 python3 hooks/wake_up.py < sample-wake-payload.json
```

For remote machines, keep the debug listener bound to `127.0.0.1` and tunnel it:

```bash
ssh -L 5678:127.0.0.1:5678 your-host
```

The debugger writes attach events to `~/.memory/debug.log`.

### Model download

On first run, `save_hook.py` downloads the embedding model (~90 MB):

```bash
# Pre-download to avoid a delay on the first hook invocation
python3 -c "from memory.db import embed; embed('warmup')"
```

---

## File layout

```
agentic-memory/
├── hooks/
│   ├── wake_up.py          # Claude Code UserPromptSubmit hook
│   └── save_hook.py        # Claude Code Stop hook
├── memory/
│   ├── db.py               # SQLite schema + CRUD + retrieval helpers
│   ├── daemon.py           # Background compaction / clustering / episodic updates
│   ├── ingest_server.py    # HTTP API for other agents
│   ├── mcp_server.py       # MCP tools
│   ├── dashboard_server.py # Dashboard + ops API
│   └── client.py           # Thin Python client for ingest_server
├── scripts/
│   ├── start-memory.sh
│   ├── stop-memory.sh
│   └── restart-memory.sh
├── cli.py                  # Management commands
└── ~/.memory/
    ├── memory.db         # SQLite database (all memory tables)
    ├── identity.md       # User profile — source of truth for identity facts
    ├── wake_up.log       # Hook injection log
    └── save_hook.log     # Session save log
```

---

## Database schema

| Table | Purpose |
|---|---|
| `sessions` | Raw transcripts plus processing metadata |
| `session_vecs` | Whole-session embeddings |
| `chunks` | Sub-session chunk embeddings for fine-grained search |
| `working_memory` | Active task context per topic cluster (14-day TTL) |
| `episodic_memory` | Thin event log — title + abstract per session |
| `compacted_sessions` | LLM-compacted summaries + vectors (primary hook retrieval target) |
| `procedural_memory` | Reusable how-to patterns with confidence scores |
| `facts` | Searchable facts, including identity facts synced from `identity.md` |
| `insights` | Cross-session patterns and learned observations |
| `topic_clusters` | Cluster centroids for topic grouping |
| `cluster_memberships` | Session → cluster assignments |
| `retrievals` | Audit log of injections and retrieval tool usage |

---

## CLI commands

```bash
python3 cli.py install           # Wire hooks into ~/.claude/settings.json (run once after cloning)
python3 cli.py sync-identity     # Re-sync identity.md → facts table
python3 cli.py compact           # Manually trigger daemon compaction pass
python3 cli.py status            # Show DB stats (sessions, facts, compacted entries)
```

---

## Personalise your identity profile

Edit `~/.memory/identity.md` using this format:

```markdown
- **Name**: Alex Johnson
- **Role**: Senior backend engineer
- **Tech stack**: Python, Go, PostgreSQL, Kubernetes
- **Working style**: Prefers concise responses, no trailing summaries
```

Run `python3 cli.py sync-identity` after changes. The daemon picks up changes automatically on the next session's first message.

---

## Send sessions from another agent

Set the `MEMORY_AGENT_NAME` environment variable before running save_hook:

```bash
MEMORY_AGENT_NAME=cursor python3 hooks/save_hook.py
```

Sessions are tagged with the agent name in the database.

---

## Data privacy

All data stays local:
- SQLite database at `~/.memory/memory.db`
- Embeddings computed locally via `sentence-transformers`
- Compaction LLM calls go to local ollama — nothing leaves your machine
- No cloud sync, no telemetry
