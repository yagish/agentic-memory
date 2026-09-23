# agentic-memory documentation

This directory holds both generated system documentation and hand-written project notes.

## Start here

- [`../README.md`](../README.md) — top-level overview, quick start, architecture summary
- [`generated/README.md`](generated/README.md) — full generated documentation set
- [`../CLAUDE.md`](../CLAUDE.md) — contributor guide and repository invariants
- [`../AGENTS.md`](../AGENTS.md) — symlink to `CLAUDE.md` for cross-agent runtimes

## Generated documentation

- [`generated/overview/overview.md`](generated/overview/overview.md) — system overview, architecture, storage, deployment, observability
- [`generated/ingest-recall/ingest-recall.md`](generated/ingest-recall/ingest-recall.md) — ingest server, recall pipeline, ranking, response behavior
- [`generated/daemon/daemon.md`](generated/daemon/daemon.md) — background extraction loop, extractors, pruning, Ollama lifecycle
- [`generated/dashboard/dashboard.md`](generated/dashboard/dashboard.md) — dashboard UI and `/memory/*` / `/ops/*` APIs
- [`generated/integrations-cli/integrations-cli.md`](generated/integrations-cli/integrations-cli.md) — Claude hooks, pi integration, shared seam, CLI commands

## Architecture notes

- [`adr/0001-deepen-memory-seams.md`](adr/0001-deepen-memory-seams.md) — ADR for deeper memory seams

## Planning notes

- [`memory-enhancements-implementation-plan.md`](memory-enhancements-implementation-plan.md)
- [`memory-enhancements-session-prompts.md`](memory-enhancements-session-prompts.md)
- [`memory-type-checklist.md`](memory-type-checklist.md)
