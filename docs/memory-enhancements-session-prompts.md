# Memory Enhancements Session Prompts

Use one fresh session per task to keep context small.

---

## General session prompt template

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

[TASK NAME]

Scope:
- [specific files/modules to touch]
- [specific behavior to add]
- [specific behavior not to change]

Requirements:
- Keep changes minimal and cohesive.
- Follow existing project architecture and invariants in `CLAUDE.md`.
- Use `integrations/common.py` as the seam for adapters.
- Retrieval must not call Ollama.
- The daemon is the only writer to typed memory tables unless the existing architecture already does otherwise.
- Add/update tests for the changed behavior.
- Do not implement future phases unless required for this task.
- At the end, summarize:
  1. files changed
  2. behavior added
  3. tests added/updated
  4. follow-up work left for the next session

Before coding:
1. Read the plan file.
2. Read only the files relevant to this task.
3. Briefly restate the task and implementation approach.
4. Then implement it.
```

---

## Session 1 — project context schema + plumbing

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Add project-context plumbing to ingest and recall.

Scope:
- Add normalized project context support for sessions/retrieval.
- Capture/pass through project metadata needed for ambient project-aware retrieval.
- Touch only the plumbing needed now; do not implement project-aware ranking yet.

Files likely relevant:
- `memory/db/schema.py`
- `memory/db/sessions.py`
- `memory/servers/ingest_pipeline.py`
- `memory/servers/ingest_server.py`
- `memory/servers/client.py`
- `integrations/pi/extension.ts`
- `integrations/pi/adapter.py`
- `integrations/claude/save_hook.py`
- `integrations/claude/wake_up.py`
- `integrations/common.py`

Requirements:
- Prefer project identity fields matching the plan:
  - `project_id`
  - `repo_root`
  - `cwd`
  - `git_remote`
  - `git_branch`
- Ambient context only; do not implement prompt-based override.
- Keep backward compatibility with existing sessions where possible.
- Add tests for ingest/save/recall plumbing.

Do not implement:
- project-aware gating/ranking
- FTS
- telemetry tables
- retention logic
- eval harness

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Session 2 — telemetry schema + event foundation

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Add telemetry storage foundations.

Scope:
- Add SQLite telemetry tables and minimal logging helpers for:
  - recall request/outcome events
  - retrieval lane metrics
  - latency breakdowns
  - system/process stats hooks
- Keep existing log behavior working.

Files likely relevant:
- `memory/db/schema.py`
- new DB telemetry helpers under `memory/db/`
- `memory/utils/logger.py`
- `memory/servers/ingest_server.py`
- `memory/servers/dashboard_server.py`

Requirements:
- Add schema and helper APIs only as needed for later retrieval instrumentation.
- Keep it local-first and simple.
- Add tests for schema/helpers.

Do not implement:
- full dashboard UI
- project-aware ranking
- FTS
- retention logic
- full eval harness

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Session 3 — offline eval scaffold

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Add an offline eval scaffold for retrieval quality.

Scope:
- Create the initial eval harness structure.
- Support evaluating prompts against expected memory/project outcomes.
- Add enough baseline fixtures/sample format so future sessions can populate real labeled cases.

Files likely relevant:
- `tests/`
- possibly `cli.py` if a runner entry point is appropriate
- retrieval modules only if needed for test harness integration

Requirements:
- Optimize for future regression gating on wrong-project contamination.
- Keep the harness lightweight and local.
- Add clear fixture/data format for labeled eval cases.

Do not implement:
- tuning changes to retrieval
- telemetry UI
- FTS
- retention logic

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Session 4 — project-aware gating/ranking

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Implement project-aware retrieval gating and ranking.

Scope:
- Add same-project hard scoping for:
  - `working_memory`
  - `session_memory`
  - `episodic`
  - repo/project facts
- Keep global retrieval for:
  - `procedural`
  - user/global facts
- Use fetch-time narrowing and rank-time project-aware logic.
- Use ambient project context only.

Files likely relevant:
- `memory/retrieval/_fetch.py`
- `memory/retrieval/_rank.py`
- `memory/retrieval/_models.py`
- new helper e.g. `memory/retrieval/_project.py`
- fact classification logic in `memory/facts/`
- any DB/session metadata access helpers needed

Requirements:
- Safe default for uncertain facts = project-specific.
- Better to return nothing than inject wrong-project context.
- Add unit + integration tests covering cross-project contamination reduction.

Do not implement:
- FTS
- RRF
- retention logic
- telemetry polish beyond what this task needs

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Session 5 — fact scope classification

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Add fact scope classification for global vs project-specific facts.

Scope:
- Implement rules + LLM-assisted classification for facts.
- Safe default = project-specific.
- Make classification available to retrieval policy.

Files likely relevant:
- `memory/facts/` modules
- fact repository/save path
- retrieval logic that consumes fact scope
- schema changes only if truly needed

Requirements:
- Deterministic rules first.
- LLM use only where aligned with existing extraction architecture.
- Retrieval path must not call Ollama.
- Add tests for user/global facts vs repo/project facts vs uncertain facts.

Do not implement:
- FTS
- retention logic
- telemetry UI
- unrelated retrieval rewrites

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Session 6 — typed-memory FTS

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Add SQLite FTS5 indexing/search for typed memories.

Scope:
- Add typed-memory FTS index/tables.
- Index enriched search text, not just semantic summaries.
- Include lexical anchors where feasible.

Files likely relevant:
- `memory/db/schema.py`
- new DB/index helpers
- typed memory repository modules
- retrieval helpers

Requirements:
- Retrieval object remains typed memories.
- Keep semantic retrieval untouched unless needed for integration.
- Add tests for exact keyword hits.

Do not implement:
- session FTS
- RRF fusion
- retention logic

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Session 7 — session FTS as expansion only

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Add SQLite FTS5 indexing/search for sessions, used only to expand typed-memory candidates.

Scope:
- Add session FTS index/tables.
- Session hits must map to typed memories from the same session.
- No raw session snippet injection.

Files likely relevant:
- `memory/db/schema.py`
- session DB helpers
- retrieval helpers
- session repository/lookup helpers

Requirements:
- Session FTS is a search surface only.
- If a session hit does not map to a good typed memory candidate, return nothing.
- Add tests covering session-hit-to-memory expansion.

Do not implement:
- raw transcript injection
- retention logic
- dashboard polish

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Session 8 — RRF fusion

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Add hybrid retrieval fusion using Reciprocal Rank Fusion (RRF).

Scope:
- Combine:
  - semantic typed-memory results
  - typed-memory FTS results
  - session-FTS-derived typed-memory results
- Deduplicate and produce final ranked candidates.

Files likely relevant:
- `memory/retrieval/_fetch.py`
- `memory/retrieval/_rank.py`
- new helper e.g. `memory/retrieval/_hybrid.py`
- tests

Requirements:
- Preserve project-aware gating/policy.
- Better to return nothing than inject risky context.
- Add unit tests for RRF and integration tests for multi-lane retrieval.

Do not implement:
- retention logic
- telemetry UI
- prompt-based project override

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Session 9 — short-term TTL cleanup

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Add cleanup for short-term memories.

Scope:
- Add TTL/cleanup behavior for:
  - `working_memory`
  - `session_memory`
- Preserve overwrite/current-state behavior where applicable.

Files likely relevant:
- `memory/daemon/pruning.py`
- `memory/db/schema.py`
- `memory/session/repository.py`
- `memory/working_memory/repository.py`
- related DB helpers/tests

Requirements:
- Keep behavior deterministic and testable.
- Do not change long-term memory behavior yet.
- Add tests for pruning behavior.

Do not implement:
- episodic reinforcement
- fact/procedural conflict replacement
- telemetry UI

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Session 10 — episodic reinforcement

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Add episodic reinforcement signals.

Scope:
- Support reinforcement from:
  - repeated extraction in new sessions
  - repeated retrieval/use
- Make episodic pruning reinforcement-aware.

Files likely relevant:
- `memory/episodic/repository.py`
- `memory/daemon/pruning.py`
- retrieval instrumentation helpers
- schema/DB helpers

Requirements:
- Keep pruning safe and simple.
- Add tests for reinforcement extending episodic lifetime.

Do not implement:
- fact/procedural hard replacement unless needed
- dashboard polish

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Session 11 — long-term conflict replacement

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Add hard replacement/de-dup behavior for conflicting long-term memories.

Scope:
- For `facts` and `procedural`, prefer one canonical current memory.
- Replace/delete older conflicting duplicates as decided in the plan.

Files likely relevant:
- `memory/facts/repository.py`
- `memory/procedural/repository.py`
- `memory/db/schema.py`
- related DB helpers/tests

Requirements:
- Keep canonical memory selection deterministic.
- Add tests for conflict replacement and duplicate cleanup.

Do not implement:
- unrelated retrieval changes
- dashboard polish

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Session 12 — dashboard/reporting polish

```text
We are implementing the plan in `docs/memory-enhancements-implementation-plan.md`.

Read that file first, then implement ONLY this task:

Task: Expose telemetry/eval summaries in dashboard/operational endpoints.

Scope:
- Add endpoints and/or dashboard views for:
  - retrieval lane hit rates
  - wrong-project filtering counts
  - latency breakdowns
  - prune/reinforcement activity
  - eval summaries if available

Files likely relevant:
- `memory/servers/dashboard_server.py`
- `dashboard.html`
- telemetry DB helpers

Requirements:
- Keep the UI simple and useful for debugging/improvement.
- Do not redesign the dashboard.
- Add tests for new API endpoints.

Do not implement:
- new retrieval behavior
- new pruning behavior

Before coding:
1. Read the plan file.
2. Read only the relevant files.
3. Briefly restate the task and approach.
4. Then implement it.
```

---

## Short reusable prompt

```text
Read `docs/memory-enhancements-implementation-plan.md` and `CLAUDE.md` first.

Implement ONLY this task: [TASK NAME]

Task scope:
[describe exact scope]

Relevant files/modules:
[list files]

Constraints:
- Keep changes minimal and cohesive.
- Do not implement future phases.
- Retrieval must not call Ollama.
- Better to return nothing than inject harmful/wrong-project context.
- Add/update tests for this task.
- At the end, summarize files changed, behavior added, tests added, and follow-up work for the next session.

Before coding:
1. Read the plan file and relevant files.
2. Briefly restate the task and your approach.
3. Then implement it.
```
