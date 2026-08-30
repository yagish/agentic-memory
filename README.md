# Agentic Memory

A persistent memory system for Claude Code that makes every new session feel like a continuation — not a cold start. It stores, compacts, and retrieves conversation history so Claude always has the right context without you having to re-explain anything.

**Two goals:**
1. Remove the "start from scratch" annoyance across sessions.
2. Reduce wasted tokens re-establishing context at the beginning of every conversation.

---

## Quick Start

### Prerequisites
- Python 3.9+
- Ollama (for local LLM inference) — [install here](https://ollama.com)
- A Claude Code workspace

### Installation

```bash
# 1. Clone the repo
git clone <repo>
cd agentic-memory

# 2. Run the installer (handles dependencies, hooks, and launchd setup)
./install.sh

# 3. Alternatively, install manually
python3 -m pip install --user \
  mcp fastmcp sentence-transformers fastapi uvicorn \
  pydantic psutil setproctitle debugpy

# 4. Pull the default Ollama model
ollama pull llama3.2:3b

# 5. Sync your identity/profile into the facts table (optional)
python3 cli.py sync-identity
```

### First Run

```bash
# Start the daemon (processes sessions, compacts memory, maintains retention)
python3 memory/daemon.py &

# Check status
python3 cli.py status

# Open the dashboard (optional)
python3 cli.py dashboard
```

---

## System Architecture

Agentic Memory is a **multi-layer system** with three concurrent services and a deep module architecture:

### Three Services

| Service | Port | Purpose |
|---------|------|---------|
| **MCP Server** | N/A | Exposes memory tools to Claude via Model Context Protocol. Started automatically by Claude Code. |
| **Ingest Server** | 7747 | HTTP API for ingesting sessions from other agents and querying memory via `/ingest`, `/recall`, `/answer`. |
| **Dashboard Server** | 7748 | Web UI and operational API for monitoring, searching, and managing memory. |
| **Daemon** | N/A | Long-lived background process that processes raw sessions into compacted, searchable memory. |

### Four Deep Modules (ADR-0001)

The system is built on four deep, domain-focused modules:

| Module | Responsibility |
|--------|-----------------|
| **`memory/inference.py`** | All model calls (Ollama): text generation, embeddings, JSON parsing, model config |
| **`memory/ingest_pipeline.py`** | End-to-end session ingestion: upsert, chunk, embed, return structured outcomes |
| **`memory/retrieval.py`** | Retrieval policy: wake-up assembly, recall digests, ranking, budget decisions |
| **`memory/vectors.py`** | Vector operations: embeddings, packing, cosine distance, model loading |

These modules are stateless and testable, with adapters (`save_hook.py`, `ingest_server.py`, `mcp_server.py`, `daemon.py`) orchestrating them.

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

## How the Daemon Works

The daemon is a **long-lived background process** that transforms raw session transcripts into organized, compacted, and searchable memory. It runs two types of passes:

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Raw Sessions (unprocessed transcripts)           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   PASS 1: Per-Session Work    │
        │   (runs on each new session)   │
        └──────────────────────────────┘
           │    │    │    │    │
           ▼    ▼    ▼    ▼    ▼
        1️⃣  2️⃣  3️⃣  4️⃣  5️⃣
        Episodic  Clustering  Working  Compaction  Procedural
        Memory    Assignment   Memory   (when ≥3)   Patterns
           │                              │
           └──────────────┬───────────────┘
                          ▼
        ┌──────────────────────────────┐
        │ PASS 2: Maintenance (every   │
        │ 20 sessions processed)        │
        └──────────────────────────────┘
           │    │    │
           ▼    ▼    ▼
        Close   Merge  Prune
        Stale   Near-  Old
        WM      Dupes  Sessions
```

### Concrete Example: Processing 3 Related Sessions

Imagine you work on authentication in a TypeScript project across 3 sessions:
- **Session 1:** "Added JWT token validation"
- **Session 2:** "Fixed refresh token bugs"
- **Session 3:** "Implemented role-based access control (RBAC)"

#### Pass 1 — Session 1

```
STEP 1: Create Episodic Entry
  • Ollama prompt: "Summarize this conversation in title + abstract"
  • Output: title = "Added JWT token validation"
           abstract = "Implemented JWT validation middleware..."
  • Stored in episodic_memory table

STEP 2: Assign to Topic Cluster
  • Embed session content
  • Assign to cluster_id = "cluster_auth_001" (or create new)

STEP 3: Update Working Memory
  • Append to cluster's working memory:
    "**Added JWT token validation** - Implemented JWT..."
  • Stored in working_memory for future reference

STEP 4: Try to Compact (needs ≥ 3 sessions)
  • Only 1 session in cluster → SKIP (wait for sessions 2 & 3)

STEP 5: Extract Procedural Patterns
  • Ollama prompt: "Identify reusable how-to patterns"
  • Output: [
      {title: "How to validate JWTs", steps: "1. Import lib\n2. ..."},
      {title: "Verify token expiration", steps: "1. Check exp claim\n..."}
    ]
  • Stored in procedural_memory table
```

#### Pass 1 — Sessions 2 & 3 follow the same steps

After session 3, `cluster_auth_001` now contains 3 sessions with full transcripts.

#### Pass 1 — Compaction Trigger (when ≥ 3 uncompacted sessions exist)

```
STEP 4 (on session 3): Compact cluster_auth_001
  • Gather all turns from sessions 1, 2, 3 (~24 turns total)
  • Ollama prompt:
    "Summarize these 3 related work sessions into a structured note.
     Task: [what the work was about]
     Context: [repo, language, key components]
     What was tried: [bullet points]
     Outcome: [what worked]
     Left off at: [where to pick up next time]"
  
  • Output stored in compacted_sessions:
    {
      cluster_id: "cluster_auth_001",
      content: "Task: TypeScript authentication system
               Context: Express + TypeScript, PostgreSQL
               What was tried:
                 • JWT token validation middleware
                 • Refresh token rotation logic
                 • Role-based access control
               Outcome: Authentication flow complete. All secured.
               Left off at: Need integration tests & rate limiting.",
      vector: [0.23, 0.45, ...],  // embedded summary
      source_sessions: [sess_1, sess_2, sess_3]
    }
  
  • Transcript columns pruned for all 3 sessions (saved to NULL)
    → Saves disk space; episodic/compacted records preserve memory
```

#### Pass 2 — Maintenance (every 20 sessions processed)

After processing 20 new sessions, the daemon runs:

```
1. CLOSE STALE WORKING MEMORY
   • Find cluster WM entries inactive > 14 days
   • Ollama: "Was anything significant accomplished?"
   • If YES → create final episodic entry, then close WM

2. MERGE NEAR-DUPLICATE COMPACTED SESSIONS
   • Find pairs with vector similarity ≥ 92%
   • Ollama: combine into single coherent summary
   • Keep merged version, discard duplicate

3. PRUNE OLD SESSIONS
   • Delete raw processed sessions > 30 days old
   • Keep: episodic entries, compacted summaries, procedural patterns
   • Effect: Save disk space; memory persists in structured form
```

### Key Design Patterns

| Pattern | Benefit |
|---------|---------|
| **One bounded text_sample per session** | Reused across episodic + procedural LLM calls → consistent reasoning, less overhead |
| **CPU threshold check (70%)** | Skip expensive LLM work when machine is busy |
| **Poll intervals (5 min busy, 30 min idle)** | Responsive yet efficient |
| **Signal handling (SIGTERM)** | Graceful shutdown; finishes current session before exiting |
| **Ollama process lifecycle** | Starts before batch, stops after → local LLM only when needed |
| **Transcript pruning after compaction** | Reclaim space; compacted vector summary is the source of truth |

### Running the Daemon

```bash
# Long-lived background process (production)
python3 memory/daemon.py

# One pass for testing
python3 memory/daemon.py --once

# Graceful stop from another terminal
kill -TERM $(pgrep -f AgenticMemoryDaemon)
```

Logs to `~/.memory/daemon.log` for each step.

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

## Services & Running

### Startup sequence

```bash
# 1. Start the daemon (processes sessions in the background)
python3 memory/daemon.py &

# 2. Start the ingest server (optional, for remote agents)
python3 memory/ingest_server.py &

# 3. Start the dashboard (optional, for web UI)
python3 memory/dashboard_server.py &

# MCP server starts automatically when Claude Code launches
```

### Service reference

| Process | Port | Required? | What it does |
|---------|------|-----------|-------------|
| **Daemon** | N/A | ✅ Yes | Compacts sessions, updates episodic/working/procedural memory, prunes old data. Runs continuously. |
| **MCP Server** | N/A | ✅ Yes (Claude Code) | Exposes memory tools to Claude via Model Context Protocol. Auto-started by Claude Code. |
| **Ingest Server** | 7747 | ❌ Optional | HTTP API for remote agents: `/ingest` (write sessions), `/recall` (query memory), `/answer` (repeated questions). |
| **Dashboard Server** | 7748 | ❌ Optional | Web UI for searching, monitoring, and managing memory. Also serves operational APIs. |

### Daemon operation

```bash
# Normal mode: continuous background processing
python3 memory/daemon.py

# One-pass mode: process pending sessions and exit
python3 memory/daemon.py --once

# Graceful shutdown (from another terminal)
kill -TERM $(pgrep -f AgenticMemoryDaemon)
```

Logs to `~/.memory/daemon.log` for monitoring.

---

## CLI Reference

### Database & memory management

```bash
python3 cli.py status              # Show memory DB stats, token economics
python3 cli.py search "<query>"    # Full-text keyword search across transcripts
python3 cli.py semantic "<query>"  # Semantic (vector) search by meaning
python3 cli.py get-session <id>    # Print full verbatim transcript for one session
python3 cli.py tail [N]            # Last N sessions (default 10)
python3 cli.py dashboard           # Open dashboard at http://localhost:7748
```

### Administration

```bash
python3 cli.py bootstrap          # Initialize the database (auto-done by install.sh)
python3 cli.py install            # Wire hooks into ~/.claude/settings.json
python3 cli.py sync-identity       # Re-sync identity.md → facts table
python3 cli.py compact             # Manually trigger compaction pass
```

### Debugging

```bash
python3 cli.py logs                # Show activity and error logs
python3 cli.py delete-old          # Prune sessions > 30 days old
python3 cli.py export              # Export sessions to JSON
```

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_OLLAMA_MODEL` | `llama3.2:3b` | Local LLM model for compaction. Must be installed via `ollama pull` |
| `MEMORY_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `MEMORY_INGEST_PORT` | `7747` | Ingest server listen port |
| `MEMORY_DASHBOARD_PORT` | `7748` | Dashboard server listen port |
| `MEMORY_DAEMON_POLL_BUSY_SECS` | `300` | How often daemon checks for new sessions when CPU > 70% |
| `MEMORY_DAEMON_POLL_IDLE_SECS` | `1800` | How often daemon checks for new sessions when CPU ≤ 70% |
| `MEMORY_DAEMON_CPU_THRESHOLD` | `70` | Skip compaction if CPU usage exceeds this % |
| `MEMORY_COMPACTION_TRIGGER` | `3` | Compact cluster after this many uncompacted sessions |
| `MEMORY_RETENTION_DAYS` | `30` | Keep raw session transcripts for this many days before pruning |
| `MEMORY_WORKING_MEMORY_TTL_DAYS` | `14` | Close working memory entries inactive for this many days |

### Database location

```bash
~/.memory/memory.db           # SQLite database (all tables)
~/.memory/identity.md          # User profile (source of truth for identity facts)
~/.memory/daemon.log           # Daemon activity log
~/.memory/activity.log         # System activity log
~/.memory/error.log            # Error and warning log
~/.memory/debug.log            # Debugger attachment log
```

### Hooks

Claude Code hooks are registered in `~/.claude/settings.json` by `./install.sh`:

- **UserPromptSubmit hook** → `hooks/wake_up.py` (injects context before Claude sees the prompt)
- **Stop hook** → `hooks/save_hook.py` (saves transcript after Claude responds)

---

## Local Models

All LLM calls in the daemon go through a local [ollama](https://ollama.com) instance — no Anthropic API tokens are consumed for compaction.

```bash
# Install ollama, then pull the default model
ollama pull llama3.2:3b
```

Override the model:
```bash
MEMORY_OLLAMA_MODEL=mistral python3 memory/daemon.py
```

**Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` (~90 MB, CPU-only, no GPU required).

---

## Remote Debugging

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

## Personalize your identity profile

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
