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
│  wake_up.py  (UserPromptSubmit hook)                │
│                                                     │
│  1. Trivial prompt? → skip all injection            │
│  2. Embed prompt → semantic search compacted_sessions│
│     ≥ 96%  → inject as [From Memory] cache hit     │
│     70–95% → inject as enrichment context           │
│  3. Working memory (first message of session only)  │
│  4. Relevant facts (identity surfaces when needed)  │
│  5. Procedural memory (how-to prompts only)         │
│  6. Enforce 500-token injection budget              │
└─────────────────────────────────────────────────────┘
        │
        ▼
   Claude processes prompt + injected context
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  save_hook.py  (Stop hook)                          │
│                                                     │
│  • Saves full transcript → sessions table           │
│  • Generates session embedding → session_vecs       │
│  • Chunks transcript → session_chunks (fine-grained)│
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  daemon.py  (background process)                    │
│                                                     │
│  Pass 1 — per new session:                          │
│  • LLM → episodic entry (title + abstract)          │
│  • Assign to topic cluster                          │
│  • Update working_memory for that cluster           │
│  • If cluster ≥ 3 sessions → compact:               │
│    - LLM generates structured summary               │
│    - Tiered length by session size                  │
│    - Merge into compacted_sessions + store vector   │
│    - Null out raw transcripts of source sessions    │
│  • LLM → extract procedural patterns               │
│                                                     │
│  Pass 2 — periodic sweep:                           │
│  • Close working_memory inactive > 14 days          │
│  • Merge near-duplicate compacted_sessions (≥ 92%)  │
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
| **Identity facts** | `facts` | Until updated | User name, role, preferences — synced from `~/.memory/identity.md` |

### Why no Insights table?

The old `insights` table stored cross-session patterns. In the new architecture those naturally fall into either procedural memory (workflow patterns) or the compacted session store (stable knowledge). The separate table added complexity without adding value.

---

## Token efficiency design

Wake_up.py applies several gates before injecting anything:

**Trivial prompt gate** — if the prompt is under 25 characters or is a continuation word (`yes`, `ok`, `continue`, `done`, `sure`, etc.), injection is skipped entirely. These prompts appear constantly in multi-step tasks and need no memory context.

**Semantic threshold** — compacted session context is only injected if cosine similarity ≥ 70%. Below that the match is noise, not signal.

**Working memory — first message only** — working memory is task context that's useful once at session start, not re-read on every turn.

**Procedural memory — conditional** — only injected when the prompt contains how-to markers (`how`, `steps`, `best way`, `should I`, `approach`, `workflow`).

**Identity — demand only** — identity facts live in the `facts` table and surface through the same FTS5 search as any other fact. They appear when the prompt is about the user; they're skipped for unrelated prompts. `identity.md` is never injected verbatim.

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

The structured format lets wake_up.py inject just the `Task` + `Left off at` fields (≈ 40 tokens) for quick enrichment, or the full summary for a 96%+ cache hit.

---

## Retrieval — how a cache hit works

Because Claude Code's UserPromptSubmit hook can only inject a prompt suffix (it cannot intercept the response), a 96%+ semantic match works by injecting a strong instruction:

```
[Cached Answer — 98% match]
Q: <stored question>
A: <stored answer>
Return this answer verbatim, prefixed with [From Memory].
```

Claude reads the injected instruction and returns the cached answer. The effect is identical to a cache hit — Claude doesn't need to reason through the problem again.

---

## Services

| Process | How to run | What it does |
|---|---|---|
| Daemon | `python3 memory/daemon.py` | Compacts sessions, builds memory layers in background |
| Dashboard | `python3 memory/dashboard_server.py` | Web UI at `http://localhost:8765` |
| MCP server | `python3 memory/mcp_server.py` | Exposes memory tools to Claude via MCP |

The daemon is the only required background process. Dashboard and MCP server are optional.

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
# 1. Clone and install Python dependencies
git clone <repo>
cd agentic-memory
pip install -r requirements.txt

# 2. Create your identity profile
mkdir -p ~/.memory
cat > ~/.memory/identity.md << 'EOF'
- **Name**: Your Name
- **Role**: What you do
- **Tech stack**: Languages/frameworks you use
- **Working style**: Any preferences Claude should know
EOF

# 3. Wire the hooks into the global Claude Code settings
python3 cli.py install

# 4. Start the daemon
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
│   ├── wake_up.py        # UserPromptSubmit hook — injects memory context
│   └── save_hook.py      # Stop hook — saves transcripts + embeddings
├── memory/
│   ├── db.py             # All SQLite operations — schema, CRUD, search
│   ├── daemon.py         # Background compaction and memory extraction
│   ├── mcp_server.py     # MCP tools (memory_search, memory_save_fact, etc.)
│   └── dashboard_server.py # Web UI
├── cli.py                # Management commands (sync-identity, compact, status)
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
| `sessions` | Raw transcripts (transcript nulled after compaction) |
| `session_vecs` | Whole-session embeddings |
| `session_chunks` | Sub-session chunk embeddings for fine-grained search |
| `working_memory` | Active task context per topic cluster (14-day TTL) |
| `episodic_memory` | Thin event log — title + abstract per session |
| `compacted_sessions` | LLM-compacted summaries + vectors (primary retrieval target) |
| `procedural_memory` | Reusable how-to patterns with confidence scores |
| `facts` | Identity facts synced from identity.md |
| `topic_clusters` | Cluster centroids for topic grouping |
| `cluster_memberships` | Session → cluster assignments |
| `retrievals` | Audit log of every memory injection |

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
