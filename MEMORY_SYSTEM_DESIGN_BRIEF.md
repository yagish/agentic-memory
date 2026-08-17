# Agent-Agnostic Memory System — Design Brief

> Session capture: 2026-08-14. Use this as a starting context doc in a new session.
> Working directory this came from: `/Users/ys68052/tutorials/mempalace` (the MemPalace reference implementation).

---

## Background: What MemPalace Is

MemPalace is a local-first, verbatim memory system for AI agents. It stores every word you share with an agent and makes it retrievable — no summarization, no paraphrasing, no cloud. Built on ChromaDB (vector store) + SQLite (knowledge graph).

The name comes from the ancient "method of loci" memory technique + Zettelkasten (Niklas Luhmann's cross-referenced index card system).

---

## How MemPalace Works

### Storage Hierarchy

```
WING (broad namespace: person, project, topic)
  └── ROOM (time/topic slice within a wing)
        └── DRAWER (verbatim text chunk — never summarized)
              └── CLOSET (AAAK-compressed index pointer TO drawers)
```

- **Drawers** = actual content, stored in ChromaDB with vector embeddings
- **Closets** = lossy symbolic summaries that point to drawers; scanned cheaply by an LLM to decide which drawers to retrieve
- **Knowledge Graph** = SQLite temporal entity-relationship store (Entity → Predicate → Entity with `valid_from`/`valid_to`)

### The 4-Layer Memory Stack (L0–L3)

| Layer | Always loaded? | Token budget | What it is |
|---|---|---|---|
| **L0 — Identity** | Yes | ~50–100 | Static `identity.txt` — who the user is |
| **L1 — Essential Story** | Yes | ~500–800 | Auto-generated from highest-weight recent drawers, grouped by room |
| **L2 — On-Demand** | No (triggered) | ~200–500 | Wing/room filtered ChromaDB retrieval when a topic surfaces mid-session |
| **L3 — Deep Search** | No (triggered) | Unlimited | Full hybrid vector + BM25 semantic search across all drawers |

L0+L1 are injected into the system prompt at session start ("wake-up"). L2/L3 are triggered mid-conversation by the AI calling MCP tools.

### The AAAK Index Format

AAAK (the closet layer) is a lossy symbolic format that compresses drawers into scannable pointers:

```
FILE_NUM|PRIMARY_ENTITY|DATE|TITLE
ZID:ENTITIES|topic_keywords|"key_quote"|WEIGHT|EMOTIONS|FLAGS
T:ZID<->ZID|label        ← tunnel (cross-reference)
ARC:emotion->emotion     ← emotional arc
```

Flags: `ORIGIN`, `CORE`, `SENSITIVE`, `PIVOT`, `GENESIS`, `DECISION`, `TECHNICAL`

This is NOT lossless. The verbatim content lives in drawers. AAAK just tells the LLM "drawer X is relevant to your query — go get it."

### How Saves Work (The Hook Loop)

Claude Code's **Stop hook** fires after every assistant response. Every N human messages:
1. Hook **blocks** the AI from stopping
2. Returns a reason telling it to save a diary entry
3. AI classifies and files content via `mempalace_add_drawer` into the right wing/room
4. Next Stop: hook sees `stop_hook_active=true` → lets AI through

**Key insight**: the AI does the classification because it has conversation context. No regex, no ML classifier needed.

### MCP Interface (Read/Write Tools)

```
mempalace_status          — total drawers, wing/room breakdown
mempalace_list_wings      — all wings with drawer counts
mempalace_get_taxonomy    — full wing → room → count tree
mempalace_search          — semantic search with optional wing/room filter
mempalace_add_drawer      — file verbatim content into wing/room
mempalace_delete_drawer   — remove by ID
mempalace_check_duplicate — dedup check before filing
mempalace_reconnect       — cache invalidation after external writes
```

### Key Files in MemPalace

```
mempalace/mcp_server.py      — MCP server, all tools
mempalace/layers.py          — L0–L3 memory stack
mempalace/palace.py          — shared palace operations
mempalace/knowledge_graph.py — temporal SQLite KG
mempalace/searcher.py        — hybrid BM25 + vector search
mempalace/dialect.py         — AAAK compression
mempalace/entity_detector.py — auto-detect people/projects from content
mempalace/backends/base.py   — abstract storage backend interface
hooks/mempal_save_hook.sh    — Claude Code Stop hook
```

---

## Memory Types Analysis

Five standard cognitive memory types mapped to MemPalace:

| Memory Type | What it means | MemPalace support |
|---|---|---|
| **Episodic** | Events, conversations, "what happened when" | ✅ Full — drawers in time-keyed rooms, `authored_at` metadata |
| **Semantic** | Facts, world knowledge, entity relationships | ✅ Partial — SQLite knowledge graph for entities; wings for topics |
| **Procedural** | How to do things, skills, reusable patterns | ❌ Gap — can store in drawers but no dedicated structure or retrieval path |
| **Working** | Active in-context memory for current session | ✅ Via L0+L1 injection; precompact hook saves before context compression |
| **Prospective** | Future reminders, scheduled tasks, to-dos | ❌ Not supported |

**Gaps to fill in your own system**: procedural and prospective memory.

---

## Design: Agent-Agnostic Memory System

### Core Principles (Non-Negotiable)

1. **Verbatim storage** — never summarize user content
2. **Separation of index from content** — index is lossy; content is lossless; retrieval bridges both
3. **Local-first, external API optional** — no cloud dependency for core operations
4. **Agent-neutral trigger surface** — save mechanism cannot depend on any one agent's hook format
5. **AI does the classification** — block the agent, give it the transcript, let it decide where to file

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Adapters                        │
│  Claude Hook │ Copilot Extension │ REST API │ Webhook   │
└──────────────────────┬──────────────────────────────────┘
                       │ normalized MemoryEvent
┌──────────────────────▼──────────────────────────────────┐
│                   Memory Core                            │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ Episodic │  │ Semantic │  │Procedural│  │Prospect│  │
│  │  Store   │  │  Graph   │  │  Store   │  │  -ural │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│        └──────────────┴─────────────┴────────────┘      │
│               SQLite + FTS5 + vector index               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  Retrieval Layer                          │
│  wake_up() → inject into system prompt / first message   │
│  recall(topic) → on-demand filtered pull                 │
│  search(query) → hybrid BM25 + vector                    │
└─────────────────────────────────────────────────────────┘
```

### The 5 Memory Stores

**1. Episodic** — what happened
```sql
CREATE TABLE episodes (
  id TEXT PRIMARY KEY,
  session_id TEXT,
  agent_id TEXT,
  timestamp TEXT,
  content TEXT,       -- verbatim, never summarized
  entities JSON,      -- ["Alice", "project-x"]
  tags JSON,
  importance REAL     -- scored at save time
);
-- FTS5 index on content
-- Optional: vector column via sqlite-vec
```

**2. Semantic** — facts and relationships
```sql
CREATE TABLE triples (
  id TEXT PRIMARY KEY,
  subject TEXT,
  predicate TEXT,
  object TEXT,
  valid_from TEXT,
  valid_to TEXT,      -- NULL = still true
  confidence REAL,
  source_episode_id TEXT  -- back-link to verbatim
);
```
Temporal: `valid_from`/`valid_to` lets you ask "what was true about X on date Y?"

**3. Procedural** — how-to patterns (MemPalace's gap)
```sql
CREATE TABLE procedures (
  id TEXT PRIMARY KEY,
  skill_name TEXT,
  trigger_pattern TEXT,   -- "how do we usually...", "when X happens..."
  steps TEXT,             -- verbatim
  success_count INTEGER DEFAULT 0,
  last_used TEXT
);
```
Retrieved when agent detects a task that matches a stored pattern.

**4. Working** — active session buffer
```sql
CREATE TABLE working_memory (
  session_id TEXT,
  slot_name TEXT,
  content TEXT,
  priority REAL,
  expires_at TEXT   -- short TTL, session-scoped
);
```
Small structured buffer (10–20 slots). Gets injected into system prompt. Evicted by recency + relevance.

**5. Prospective** — future intent (MemPalace's gap)
```sql
CREATE TABLE prospective (
  id TEXT PRIMARY KEY,
  trigger_type TEXT,    -- 'time' | 'event' | 'keyword'
  trigger_value TEXT,   -- ISO datetime, event name, or keyword
  intent TEXT,          -- verbatim
  created_at TEXT,
  fired_at TEXT         -- NULL = not yet fired
);
```
Checked at session start: inject any unfired prospective memories whose trigger has been met.

### The Normalized MemoryEvent (Agent-Neutral Interface)

All agent adapters emit this same shape:

```json
{
  "agent": "claude | copilot | codex | pi | unknown",
  "session_id": "uuid",
  "trigger": "session_end | interval | manual",
  "transcript": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "timestamp": "2026-08-14T10:00:00Z"
}
```

The Memory Core processes this without caring about the source agent.

### Agent Adapter Map

| Agent | Trigger mechanism |
|---|---|
| **Claude Code** | Stop hook (JSON on stdin) — blocks AI, returns save instruction |
| **Codex CLI** | `.codex/hooks.json` Stop hook — same JSON format |
| **Copilot / VS Code** | Extension event listener → REST POST to Memory Core |
| **Pi / web agents** | Session-end webhook or periodic polling → REST POST |
| **Any agent** | `POST /memory/save` with MemoryEvent body |
| **MCP-native agents** | MCP server wrapping the same REST API |

### The Retrieval Stack

```
L0: identity.md         — who the user is, always injected (~100 tokens)
L1: wake-up digest      — top N recent+important episodes, compact (~600 tokens)
L2: topic recall        — filtered by entity or topic on demand (~200-500 tokens)
L3: semantic search     — hybrid vector + BM25, triggered by query
```

The index layer (your AAAK equivalent) — compact pointer format:
```
ENTITY|topic_keywords|importance_score|→verbatim_id
```
LLM scans this at wake-up to decide which verbatim content to pull at L2/L3.

### Tech Stack Recommendation

| Component | Choice | Why |
|---|---|---|
| Episode + KG + procedure store | SQLite + FTS5 | Local, zero deps, full-text search built in |
| Vector index | `sqlite-vec` (preferred) or ChromaDB | `sqlite-vec` keeps everything in one file |
| API layer | FastAPI | Simple, async, OpenAPI docs free |
| Embeddings | `nomic-embed-text` via Ollama | Local, good quality, free |
| Hook format | JSON over stdin (same as MemPalace) | Already works for Claude + Codex |
| MCP server | Python `mcp` library | Wraps REST API for MCP-native agents |

### What MemPalace Gets Right (Replicate These)

- **AI does classification** — block the agent mid-session with the transcript; it files content correctly because it has context. No regex, no classifier.
- **Verbatim-only storage** — the system's core trust promise. Never break it.
- **Pluggable storage backend** — abstract base class, swap ChromaDB for anything else without touching the API.
- **4-layer stack** — cheap always-on (L0+L1) + expensive on-demand (L2+L3) keeps token costs low.
- **Temporal knowledge graph** — `valid_from`/`valid_to` is the right model for facts that change over time. SQLite, not Neo4j.

### What to Add That MemPalace Is Missing

- **Procedural store** with trigger-pattern matching
- **Prospective memory** checked at session start
- **REST API** as the universal interface (not just MCP)
- **Normalized MemoryEvent** so all adapters speak the same language
- **Working memory TTL** — auto-evict stale session slots

---

## Open Questions to Resolve Before Building

1. **Vector store**: `sqlite-vec` (single file, simpler) vs ChromaDB (more mature, more deps)?
2. **Embedding model**: local-only (Ollama/nomic) or allow BYOK (OpenAI/Anthropic) as opt-in?
3. **Trigger interval**: how many exchanges between auto-saves? (MemPalace uses configurable N)
4. **Classification strategy**: always block the agent and ask it to classify, or add a lightweight local classifier for bulk ingest?
5. **Multi-user**: single palace per machine, or namespaced by user identity?

---

## Next Steps

1. Define the SQLite schema for all 5 stores
2. Build the REST API (FastAPI) with `/memory/save`, `/memory/search`, `/memory/wake-up`
3. Wrap it in an MCP server for Claude/Codex
4. Write the Claude Code Stop hook adapter
5. Add the VS Code extension adapter for Copilot
6. Build the L0–L3 retrieval stack on top of the stores
7. Add procedural trigger matching
8. Add prospective memory injection at session start
