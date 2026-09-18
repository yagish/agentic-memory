# agentic-memory — Generated Documentation

Agentic Memory is a local-first persistent memory system for AI agents (Claude Code and pi). It records every conversation, extracts five complementary memory types via a local LLM, and injects relevant context at the start of each new prompt — without blocking any agent session.

## Documents

| Document | Lines | Surface covered |
|---|---|---|
| [Overview](overview/overview.md) | 721 | System context, architecture, all cross-cutting concerns: storage, security, observability, error handling, resiliency, deployment, configuration, testing, glossary |
| [Ingest & Recall Server](ingest-recall/ingest-recall.md) | 671 | `POST /ingest`, `POST /recall`, `GET /status` (port 7747); retrieval pipeline; composite ranking; MemoryClient; budget-fitting; fact answer renderer |
| [Extraction Daemon](daemon/daemon.md) | 660 | Background extraction loop; CPU gate; five sequential extractors (facts, episodic, procedural, working memory, session memory); Ollama lifecycle; compaction; TTL pruning; launchd service |
| [Dashboard](dashboard/dashboard.md) | 662 | All `/memory/*` and `/ops/*` API routes (port 7748); single-page UI; log viewer; performance and activity stats |
| [Integrations & CLI](integrations-cli/integrations-cli.md) | 714 | Claude Code Stop hook; Claude Code UserPromptSubmit hook; Pi TypeScript extension; Pi Python bridge; shared seam (`integrations/common.py`); all 9 CLI commands; installer |

## Provenance

Generated from `agentic-memory` @ `7df34ac` (branch: `refactor-integrations-pi-claude-memory`) on 2026-09-18. Regenerate rather than hand-edit.

## Known gaps

Items marked `[confirm]` or `[unknown]` across the five documents that require a human to resolve:

**Cross-cutting**
- CI/CD pipeline details: `.github/workflows/` directory exists but was not examined (`[unknown]`)
- No-TTL policy for `procedural_memory`, `working_memory`, and `session_memory` tables — only `facts` and `episodic_memory` have TTL-based pruning; whether this is intentional is not stated in the code (`[confirm]`)

**Ingest & Recall**
- `include_working_memory` flag in `retrieve_wake_up_context()` has no discriminating effect in the current implementation — working memory is retrieved iff `session_id` is provided regardless of the flag (`[confirm]`)
- `embed_fn` parameter in `ingest_session()` is declared but immediately deleted (`del embed_fn`); no embeddings are computed at ingest time — future intent unknown (`[confirm]`)
- `cache_hit` and `enrichment` fields in `WakeUpContext` are always `None`/`[]` — present for anticipated use but not yet wired (`[confirm]`)
- CORS is restricted to `copilot.microsoft.com` and `github.com` by default; rationale not documented (`[confirm]`)
- Ingest server binds to `127.0.0.1` only; whether this is intentional for all deployment scenarios (`[confirm]`)

**Daemon**
- No aggregate session extraction timeout: `MEMORY_EXTRACTION_TIMEOUT_SECONDS` (default 600 s) applies per extractor, so a worst-case session could run 5 × 600 s = 3 000 s (`[confirm]`)
- Fact persistence makes two Ollama calls per fact (one to extract the triple, one to generate `semantic_content` text via `generate_semantic_fact_text` in `memory/facts/text.py`) — the second call is a non-obvious cost (`[inferred]` — confirmed by reading source, rationale `[confirm]`)

**Dashboard**
- `GET /memory/sessions/{session_id}/transcript` returns HTTP 200 with `{"transcript": null}` when a session is not found, rather than HTTP 404 — whether this is intentional (`[confirm]`)
- Pagination is not implemented on list endpoints; whether it is planned (`[confirm]`)
- Business rationale for the `/memory/sessions/compacted` alias path alongside `/memory/sessions/{id}/transcript` (`[unknown]`)

**Integrations & CLI**
- `_build_injection` is imported in `wake_up.py` but is not called in `main()` — the server path via `MemoryClient` constructs the injection instead (`[confirm]`)
- Pi extension hard-codes `include_working_memory: false`; whether this is intentional or incomplete (`[confirm]`)
- `MEMORY_AGENT_NAME` env variable purpose: whether it is intended for multi-tenant tagging or a placeholder (`[unknown]`)

**Documentation drift (confirmed)**

The following discrepancies between existing documentation and the implementation were found:

1. **README parallel vs. sequential extraction:** The README claims `ThreadPoolExecutor(max_workers=5)` runs all five extractors in parallel. The current implementation at `memory/daemon/__init__.py:342–376` runs them sequentially in a plain `for` loop. The daemon itself logs `"sequential extractors"`. The README has not been updated since the refactor.
2. **README module paths stale:** README references `memory/daemon.py`, `memory/ingest_server.py`, and `memory/dashboard_server.py` at the top level. Actual post-refactor locations are `memory/daemon/__init__.py`, `memory/servers/ingest_server.py`, and `memory/servers/dashboard_server.py`.
3. **Plist template files have stale paths:** The committed `com.memory.daemon.plist`, `com.memory.ingest.plist`, and `com.memory.query.plist` files in the repository root contain `__INSTALL_DIR__` placeholders and pre-refactor script paths. `install.sh` generates the deployed plists correctly; the template files themselves are not used at runtime but are misleading.
4. **Dashboard route prefixes:** `_recon.md` listed routes without prefixes. The actual dashboard API routes are under `/memory/` and `/ops/` prefixes (e.g., `GET /memory/sessions`, `POST /ops/process-one`).
