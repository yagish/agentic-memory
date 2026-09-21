# Memory Enhancements Implementation Plan

## Goals

Implement five major enhancements to the memory system:

1. Project-aware memory retrieval
2. Hybrid retrieval with semantic + keyword search
3. Short-term memory cleanup with long-term relearning
4. Telemetry
5. Eval loop for continuous improvement

Primary optimization goal: **minimize harmful and wrong-project injections, even if recall drops**.

---

## Decisions Reached

### 1) Project-aware memory
- Use **mixed scoping by memory type**.
- Project identity:
  - **primary:** git remote origin
  - **secondary:** repo root
  - **tertiary:** cwd
- Use **ambient workspace only** in phase 1.
- **Explicit prompt-based project override** is deferred.
- Enforce project awareness in **both** places:
  - fetch-time narrowing
  - rank-time reranking
- Hard scoping policy:
  - `working_memory` → same-project only
  - `session_memory` → same-project only
  - `episodic` → same-project only
  - repo/project facts → same-project only
  - `procedural` → global
  - user preferences / portable facts → global
- For facts, scope classification will use:
  - **rules + LLM**
  - safe default: **project-specific**

### 2) Hybrid retrieval
- Use **SQLite FTS5** for keyword retrieval.
- Implement **Option C from the start**:
  - semantic retrieval over typed memories
  - FTS over typed memories
  - FTS over sessions
- Session-level FTS is **retrieval expansion only**:
  - session hits map to typed memories from that session
  - **no raw session snippet injection**
  - better to return nothing than inject weak context
- Combine lanes with **RRF**.

### 3) Memory lifecycle
- Short-term:
  - `working_memory`
  - `session_memory`
- Decaying but reinforceable:
  - `episodic`
- Long-term:
  - `facts`
  - `procedural`
- Episodic reinforcement comes from **both**:
  - repeated extraction in later sessions
  - repeated retrieval/use
- `working_memory` and `session_memory` cleanup:
  - overwrite-like behavior where appropriate
  - plus TTL pruning
- Long-term conflicts:
  - **hard replace/delete old conflicting memories**
  - prefer one canonical current memory

### 4) Telemetry
- Add **both** product telemetry and system telemetry.
- Store telemetry in **both**:
  - SQLite tables
  - structured local logs
- Phase-1 product telemetry = implicit only:
  - request
  - candidate counts
  - filtering counts
  - selected memories
  - outcome
  - token/latency summaries
- Phase-1 system telemetry:
  - retrieval latencies by lane
  - daemon/extractor latencies
  - prune counts/latencies
  - retries/failures/timeouts
  - CPU/memory/process stats

### 5) Eval loop
- Do **both** offline and online evals.
- Offline is the initial source of truth.
- Offline dataset source:
  - sampled real historical prompts/sessions
  - manually label a subset
- Test bar before merge:
  - **strong**
  - benchmark must show no regression on wrong-project contamination

### 6) Success criteria / delivery
- Optimize for **precision/safety first**.
- Primary goal:
  - reduce wrong-project / harmful injections
- Accept lower recall if needed.
- Delivery mode:
  - **big bang**
  - no feature-flag rollout required

---

## Assumptions

- `sessions.metadata` is not enough by itself for efficient project-aware retrieval; normalized project context will likely need to be queryable for ranking/filtering.
- Pi already provides `cwd`; Claude likely needs extra project-context capture.
- Typed memories remain the final retrieval object; raw sessions are only a search surface.
- Existing semantic retrieval paths can initially be wrapped inside a new hybrid candidate/fusion layer.
- The dashboard can be extended for telemetry/eval reporting without a new service.

---

## Remaining Decisions Needed

1. **Exact TTLs**
   - `working_memory`: hours or days?
   - `session_memory`: days or weeks?
   - `episodic`: base TTL plus reinforcement extension formula?

2. **Canonical project metadata capture**
   - exactly what to persist per session:
     - `project_id`
     - `repo_root`
     - `cwd`
     - `git_remote`
     - `branch`
     - aliases/repo name?

3. **Fact-scope rules**
   - exact heuristics for:
     - user/global facts
     - repo/project facts
     - uncertain facts

4. **Telemetry retention**
   - how long to keep telemetry tables/logs

5. **Offline eval set size**
   - how many labeled prompt cases are required for merge gating

---

## Priority Ranking

### Product-level ranking
1. **Project-aware memory**
2. **Telemetry foundation**
3. **Hybrid retrieval**
4. **Short-term cleanup + long-term relearning**
5. **Eval loop as an ongoing system**

### Important nuance
Even though eval ranks later as a product capability, the **offline eval harness must start early** because it is required as a strong pre-merge gate.

---

## Recommended Implementation Order

## Phase 0 — Safety net first
Build guardrails before changing retrieval behavior.

### Scope
- add offline eval harness
- define labeled benchmark slice
- add baseline telemetry event schema
- snapshot current retrieval metrics

### Why first
- precision/safety is the top priority
- big bang delivery still needs internal safety checks
- benchmark is needed to prevent regressions

### Likely areas
- `tests/` new eval/benchmark suite
- `memory/servers/dashboard_server.py`
- new telemetry DB helpers under `memory/db/`
- `memory/utils/logger.py`

---

## Phase 1 — Project context plumbing
Make project identity first-class in ingest and recall.

### Scope
- capture normalized project metadata on ingest
- pass current project context on recall
- backfill/derive project context from existing sessions where possible

### Likely implementation
#### Pi
- extend `integrations/pi/extension.ts`
- capture:
  - cwd
  - repo root
  - git remote
  - branch

#### Claude
- extend `integrations/claude/save_hook.py`
- derive git metadata where possible
- extend recall path so current ambient project context is available during retrieval

#### Shared/request path
- `integrations/claude/wake_up.py`
- `integrations/common.py`
- `memory/servers/ingest_server.py`
- `memory/servers/client.py`

### Schema direction
Add normalized session project context rather than relying only on opaque JSON metadata.

Recommended persisted context:
- `project_id`
- `repo_root`
- `cwd`
- `git_remote`
- `git_branch`

Likely in:
- `memory/db/schema.py`
- and associated DB/session helpers

---

## Phase 2 — Project-aware retrieval policy
Fix contamination before increasing retrieval power.

### Scope
- hard same-project gating for:
  - `working_memory`
  - `session_memory`
  - `episodic`
  - repo/project facts
- global treatment for:
  - `procedural`
  - user/global facts
- add project-aware ranking helpers
- add fact scope classification

### Likely files
- `memory/retrieval/_fetch.py`
- `memory/retrieval/_rank.py`
- `memory/retrieval/_models.py`
- new helper e.g. `memory/retrieval/_project.py`
- fact classification path in `memory/facts/`

### Deliverable
Wrong-project injection rate should drop sharply.

---

## Phase 3 — Hybrid retrieval
Add exact-match retrieval after scoping is safe.

### Scope
- add FTS5 index for typed memories
- add FTS5 index for sessions
- session FTS maps back to typed memories by `session_id`
- merge:
  - semantic lane
  - typed-memory FTS lane
  - session-derived lane
- fuse with **RRF**

### Recommended shape
Use unified search/index tables so hybrid retrieval can operate over shared candidate docs, then resolve back to typed records.

Potential structure:
- typed memory index
- session index

### Likely files
- `memory/db/schema.py`
- new DB/index helpers under `memory/db/`
- `memory/retrieval/_fetch.py`
- new helper e.g. `memory/retrieval/_hybrid.py`
- `memory/retrieval/_rank.py`

### Important policy
- session FTS **never injects session text**
- session FTS only boosts/promotes typed memories from matching sessions

---

## Phase 4 — Memory lifecycle and relearning
Once retrieval is safer and stronger, clean and reinforce the store.

### Scope
- TTL cleanup for:
  - `working_memory`
  - `session_memory`
- reinforcement stats for:
  - `episodic`
- hard replacement for conflicting long-term memories:
  - `facts`
  - `procedural`
- add canonicalization/dedup strategy where needed

### Likely files
- `memory/daemon/pruning.py`
- `memory/db/schema.py`
- typed repositories:
  - `memory/facts/repository.py`
  - `memory/procedural/repository.py`
  - `memory/episodic/repository.py`
  - `memory/session/repository.py`
  - `memory/working_memory/repository.py`

### Likely data additions
- `retrieval_count`
- `last_retrieved_at`
- `reinforcement_count`
- `expires_at`
- canonical/dedup identity for facts/procedures

---

## Phase 5 — Telemetry and dashboard expansion
Telemetry starts early, but reporting matures here.

### Scope
- telemetry tables
- dashboard views for:
  - wrong-project filtering counts
  - lane hit rates
  - selected-memory provenance
  - latency breakdowns
  - prune/reinforcement activity
  - DB growth

### Likely files
- `memory/servers/dashboard_server.py`
- `dashboard.html`
- `memory/utils/logger.py`
- new DB telemetry modules

---

## Phase 6 — Eval loop as a living improvement system
Turn benchmark and telemetry into an ongoing tuning loop.

### Scope
- offline eval runner
- benchmark summaries in dashboard/CLI
- compare retrieval variants over time
- identify top failure modes:
  - wrong-project injection
  - missed keyword anchor
  - stale memory leak
  - overaggressive filtering

### Likely files
- `tests/` benchmark harness
- maybe `cli.py` command for eval/report
- dashboard endpoints/views

---

## Concrete Start Order

If starting implementation immediately, do this order:

1. **Schema + project metadata plumbing**
2. **Offline eval scaffold + telemetry event tables**
3. **Project-aware gating/ranking**
4. **FTS indexes + RRF hybrid retrieval**
5. **Retention/reinforcement/conflict cleanup**
6. **Dashboard/eval reporting polish**

---

## Biggest Risks

1. **Claude project metadata capture may be weaker than Pi**
   - may require best-effort git discovery

2. **Fact scope classification could be noisy**
   - mitigate with safe default = project-specific

3. **FTS over AI summaries alone would be weak**
   - mitigate by indexing lexical anchors, not just semantic paraphrases

4. **Big bang means benchmark discipline matters**
   - required because retrieval behavior is changing deeply

---

## Recommended Immediate Next Step

Start with:
- schema changes for project context and telemetry
- project metadata plumbing through ingest/recall
- offline eval scaffold

Then implement:
- project-aware retrieval gating/ranking
- hybrid retrieval with FTS + RRF
- retention/relearning improvements

This gives the best chance of landing the big bang safely while protecting against harmful retrieval regressions.
