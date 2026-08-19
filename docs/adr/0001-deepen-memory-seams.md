# ADR-0001: Deepen memory seams around inference, ingest, retrieval, and lifecycle

- **Status**: Accepted
- **Date**: 2026-08-19

## Context

The current memory system works, but the major modules are shallow:

- `memory/db.py` mixes schema, storage helpers, retrieval logic, clustering, retention helpers, and vector math.
- `hooks/save_hook.py` and `memory/ingest_server.py` duplicate the ingest implementation.
- `hooks/wake_up.py` and `memory/ingest_server.py` each assemble retrieval policy directly.
- `memory/daemon.py`, `memory/consolidation.py`, and `memory/compress.py` each own part of the session lifecycle.
- Ollama and embedding orchestration are repeated across multiple modules.

This reduces locality. Callers must know too much about implementation details, table layout, ranking thresholds, and failure handling.

## Decision

Introduce four deep modules and migrate adapters onto them incrementally.

### Module map

#### `memory/inference.py`
Owns model-facing work:
- text generation requests
- embedding requests
- parsing model JSON responses
- model/runtime configuration

Adapters and domain modules should depend on this seam instead of open-coding Ollama or embedder calls.

#### `memory/ingest_pipeline.py`
Owns end-to-end session ingestion:
- upsert session
- store full-session embedding
- replace chunk embeddings
- return structured outcomes and warnings

Adapters (`save_hook.py`, `ingest_server.py`) should parse inputs and log outcomes, but not orchestrate ingest steps directly.

#### `memory/retrieval.py`
Owns retrieval policy:
- wake-up context assembly
- recall/digest selection
- direct-answer policy
- ranking and budget decisions

Adapters should format and transport retrieval results, not implement policy.

#### `memory/lifecycle.py`
Owns session state transitions:
- processed
- compacted
- transcript pruned
- archived/deleted
- retention and destructive transition rules

Daemon, consolidation, and compression flows should call this seam instead of each owning separate lifecycle rules.

## Migration plan

1. Add the new module seams with stable interfaces.
2. Migrate `save_hook.py` to `memory/ingest_pipeline.py`.
3. Migrate `memory/ingest_server.py` ingest path to the same ingest seam.
4. Migrate `hooks/wake_up.py` retrieval policy into `memory/retrieval.py`.
5. Migrate `/recall` and `/answer` onto retrieval seams.
6. Move lifecycle transitions from daemon/consolidation/compress into `memory/lifecycle.py`.
7. Move remaining raw SQL callers behind deeper modules.
8. Split `memory/db.py` into storage adapters once callers no longer depend on its broad interface.

## Consequences

### Positive

- Better locality for ingest, retrieval, lifecycle, and inference changes.
- Smaller adapter interfaces with more leverage.
- Cleaner test surface centered on behavior instead of table details.
- Easier future migrations away from SQLite-specific or Ollama-specific implementations.

### Negative

- Temporary duplication while adapters are migrated.
- Some wrappers will initially be thin until later migration steps land.

## Notes

This ADR intentionally prefers incremental migration over a large rewrite. Each task should be independently revertible via a small commit.
