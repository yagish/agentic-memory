# Agentic Memory — Finalized Implementation Plan

Last updated: 2026-09-04
Status: active plan of record

## 1. Goal

Build the memory system in small, testable slices.

We already know when we are saving and when we are retrieving, so we are **not** building a general LLM controller to decide `store` vs `retrieve`.

Instead, we will:
- keep session save as the authoritative base
- treat **facts** as the reference implementation pattern
- implement each additional memory type using the same sequence:
  1. contract
  2. extraction / generation
  3. persistence
  4. retrieval
  5. eval harness
  6. tests
- only after all individual memory types work independently, add an LLM-based router that identifies which memory type(s) apply
- only after the pieces and eval harnesses are in place, automate population through the daemon

## 2. Core Planning Decisions

1. **No operation controller**
   - save vs retrieve is already known by the call site
   - save path = session save / ingest
   - retrieve path = recall / answer path

2. **Per-memory-type linear delivery**
   - each memory type is built fully before moving to the next
   - no daemon-first expansion for unfinished memory types

3. **Eval harness required for every memory type**
   - every type must have a dedicated evaluation harness before automation

4. **Router comes after the memory types**
   - the LLM-based router/classifier is a later phase
   - it decides which memory type(s) should be extracted or retrieved
   - it does not decide `store` vs `retrieve`

5. **Composition comes after routing**
   - if multiple memory types match, retrieve all relevant memory and compose a response/context from them
   - this requires its own tests and eval harness

6. **Daemon automation comes last**
   - the daemon should follow already-proven contracts and eval harnesses
   - it should not be the place where memory-type behavior is first discovered

7. **MCP / other-agent integration is deferred**
   - not part of the active roadmap
   - freeze and ignore for now unless explicit cleanup is later requested

## 3. Scope and Guardrails

### In scope now
- local session save foundation
- facts
- episodic memory
- procedural memory
- working memory
- session / compacted-session memory
- per-type eval harnesses
- later: extraction router
- later: retrieval router
- later: multi-memory composition
- later: daemon automation

### Explicitly out of scope for now
- MCP work
- multi-agent integration strategy
- agent-facing integration polish
- generalized `store/retrieve/update/delete/none` controller

### Engineering guardrails
- prefer TDD and vertical slices
- keep changes small and linear
- preserve dashboard/log route compatibility where feasible
- do not add heuristic hacks where contract-driven behavior should exist
- keep session storage authoritative
- use real local LLM extraction tests where those harnesses exist

## 4. Cross-Cutting Logging Requirements

Logging is part of the implementation plan, not an afterthought.

### Principles
- each memory type gets **one dedicated log stream/file**
- do **not** split per-memory-type logs into separate extract/retrieve files
- daemon has its **own dedicated runtime log**
- wake-up has its **own dedicated runtime log**
- memory-type logs are for deep domain debugging
- daemon and wake-up logs are for orchestration/runtime debugging

### Required log separation
At minimum, maintain separate logs for:
- session save
- daemon
- wake_up
- facts
- episodic memory
- procedural memory
- working memory
- session / compacted-session memory
- later: extraction router
- later: retrieval router
- later: composition

### Recommended log file names
Use stable, predictable filenames under `~/.memory/`:
- `save_hook.log`
- `daemon.log`
- `wake_up.log`
- `facts.log`
- `episodic.log`
- `procedural.log`
- `working_memory.log`
- `session_memory.log`
- later: `extraction_router.log`
- later: `retrieval_router.log`
- later: `composition.log`

### Dashboard requirement for logs
The dashboard must be updated as new log files are introduced.

Requirements:
- every active log file should be visible from the dashboard log viewer
- dashboard log views should auto-refresh by polling every few seconds
- dashboard should show the latest log lines without requiring manual reload
- when a new memory-type log is added, dashboard log source configuration and UI visibility must be updated in the same slice
- daemon and wake-up logs remain available as runtime/orchestration logs
- per-memory-type logs remain available as deep debugging logs

### Per-memory-type logging requirements
Each memory-type log should contain both extraction and retrieval events, tagged by event/source.

For each memory type, log:
- timestamp
- event name (for example: `extract_start`, `extract_result`, `retrieve_start`, `retrieve_result`, `validation_error`, `persist_result`)
- source (`daemon`, `wake_up`, test harness, manual debug, etc.)
- session ID or query context where applicable
- input prompt / extraction prompt / retrieval prompt as applicable
- raw model output where applicable
- validated object(s)
- retrieved record(s)
- similarity / ranking information where applicable
- rejected/invalid outputs with reason
- persisted record IDs / summary of what was saved

### Daemon logging requirements
Daemon log stays separate and should log runtime orchestration:
- polling / cycle boundaries
- what work was attempted
- which memory type pipelines were invoked
- success / failure per stage
- timings / counts where useful
- no mixing detailed per-memory-type extraction payloads into daemon orchestration logs

### Wake-up logging requirements
Wake-up log stays separate and should log retrieval/runtime orchestration:
- incoming prompt
- retrieval flow used
- which memory types were queried
- top-level decision made
- whether memory was found / blocked / allowed
- no mixing detailed per-memory-type retrieval payloads into wake-up orchestration logs

## 5. Phase 1 — Session Save Foundation

### Goal
Keep session capture stable and authoritative.

### Scope
- save session transcript
- save timestamps / metadata required by downstream extractors
- ensure stored sessions remain usable by later memory pipelines
- keep session save tests green

### Deliverables
- stable session save path
- session save tests
- session storage remains the authoritative source for downstream memory extraction
- dedicated session-save logging

## 6. Phase 2 — Facts Slice

### Goal
Use facts as the reference implementation pattern for all later memory types.

### Scope
1. fact contract
2. fact extractor
3. fact persistence
4. fact retrieval
5. fact eval harness
6. fact renderer / answer path as needed
7. dedicated fact logs

### Deliverables
- validated fact contract
- extraction fixtures and tests
- retrieval tests
- eval harness
- fact log showing prompt and extracted/retrieved facts

### Status note
Facts are the current baseline and pattern to copy.

## 7. Phase 3 — Episodic Memory Slice

### Goal
Implement episodic memory end-to-end using the same pattern as facts.

### Scope
1. episodic contract
2. episodic extractor
3. episodic persistence
4. episodic retrieval
5. episodic eval harness
6. episodic logging

### Intended shape
Event / discussion / decision memory, such as:
- what happened
- what was decided
- participants if needed
- optional todos / outcomes depending on final contract

### Deliverables
- `EpisodicMemory` contract
- extractor tests
- repository/storage tests
- retrieval tests
- episodic eval harness
- dedicated episodic log showing extraction and retrieval activity

## 8. Phase 4 — Procedural Memory Slice

### Goal
Implement procedural memory end-to-end.

### Scope
1. procedural contract
2. procedural extractor
3. procedural persistence
4. procedural retrieval
5. procedural eval harness
6. procedural logging

### Intended shape
Reusable how-to knowledge:
- repeatable workflow
- steps
- trigger / confidence later if needed

### Deliverables
- `ProceduralMemory` contract
- extraction tests for reusable procedures
- retrieval tests for how-to prompts
- procedural eval harness
- dedicated procedural log showing extraction and retrieval activity

## 9. Phase 5 — Working Memory Slice

### Goal
Implement working memory end-to-end.

### Scope
1. working-memory contract
2. working-memory builder / extractor
3. persistence
4. retrieval
5. eval harness
6. working-memory logging

### Intended shape
Current temporary active context:
- current goal
- current focus
- constraints
- active tasks
- expiration / closure behavior

### Deliverables
- `WorkingMemory` contract
- create/update/close tests
- retrieval tests
- working-memory eval harness
- dedicated working-memory log showing extraction and retrieval activity

## 10. Phase 6 — Session / Compacted-Session Slice

### Goal
Implement session memory as its own first-class slice.

### Scope
1. session / compacted-session contract
2. compaction generator
3. persistence
4. retrieval
5. eval harness
6. session-memory logging

### Intended shape
Compressed conversation memory, likely including:
- task
- context
- what was tried
- outcome
- left off at

### Deliverables
- `SessionMemory` or `CompactedSessionMemory` contract
- compaction validation tests
- retrieval tests
- session-memory eval harness
- dedicated session-memory log showing extraction and retrieval activity

## 11. Phase 7 — Extraction Type Router

### Goal
Given a saved session, identify which memory type(s) should be extracted.

### Important
This phase happens **after** the individual memory types above are proven independently.

### Scope
- LLM-based classification / router for extraction
- router output indicates one or more memory types to extract
- eval harness for classification accuracy
- router logging

### Example
A session may contain:
- facts
- an episodic event
- a procedural pattern

The router may decide:
- extract fact + episodic + procedural

### Deliverables
- extraction-router contract
- extraction-router eval harness
- unit tests with fixtures / goldens
- dedicated extraction-router log showing router prompt and chosen memory types

## 12. Phase 8 — Retrieval Type Router

### Goal
Given a user prompt, identify which memory type(s) should be queried.

### Scope
- prompt -> memory type classification
- support mixed retrieval:
  - fact
  - episodic
  - procedural
  - working
  - session
- router output contract
- eval harness
- router logging

### Deliverables
- retrieval-router contract
- prompt classification eval harness
- unit tests and fixtures
- dedicated retrieval-router log showing prompt and selected memory types

## 13. Phase 9 — Multi-Memory Response Composition

### Goal
If multiple memory types match, retrieve all relevant memory and compose a response/context using all of them.

### Scope
- deterministic assembly rules
- combine results from multiple memory types
- avoid obvious duplication/conflicts where possible
- tests and eval harness
- composition logging

### Deliverables
- composition layer
- unit tests for multi-memory prompts
- eval harness for combined-memory behavior
- dedicated composition log showing prompt, retrieved memory bundle, and final assembled output

## 14. Phase 10 — Daemon Automation (One Type at a Time)

### Goal
Automate population only after contracts, retrieval, and eval harnesses are proven.

### Principle
The daemon follows proven behavior. It does not define it first.

### Rollout order
Suggested order:
1. facts remain active
2. episodic automation
3. procedural automation
4. working-memory automation
5. session / compacted-session automation
6. later: router-driven multi-type daemon flow

### Deliverables
- each memory type automated behind its already-proven path
- daemon invokes only tested pipelines
- daemon logs remain separate from per-memory-type logs

### Current planning truth
Treat facts as the only daemon-populated memory type we actively trust right now. Other daemon-driven memory types are deferred until their slices are individually complete.

## 15. Phase 11 — Integration Later

### Goal
Revisit MCP and other-agent integration only after the local system is stable.

### Status
Deferred.

### Deferred areas
- `memory/mcp_server.py`
- cross-agent save/recall flows
- external integration strategy

## 16. Standard Template for Every Memory Type

Every memory type must follow this checklist before automation:

1. **Contract**
   - Pydantic model
   - validation rules
   - invalid-shape rejection

2. **Extractor / generator**
   - LLM prompt or deterministic builder
   - parse + validate against contract
   - extraction logging

3. **Persistence**
   - repository / storage layer
   - insert/update/read tests

4. **Retrieval**
   - exact / semantic / mixed as appropriate
   - retrieval logging
   - recall tests

5. **Eval harness**
   - labeled fixtures / expected outputs
   - acceptance tests
   - failure cases

6. **Only then automation**
   - hook into daemon later

## 17. Immediate Execution Order

The active intended order is:

1. keep session save foundation stable
2. keep facts as the reference slice
3. build episodic slice end-to-end
4. build procedural slice end-to-end
5. build working-memory slice end-to-end
6. build session / compacted-session slice end-to-end
7. build extraction type router + eval harness
8. build retrieval type router + eval harness
9. build multi-memory composition + eval harness
10. automate daemon population one memory type at a time
11. revisit integration later

## 18. Definition of Done for the Pre-Daemon Milestone

Before expanding daemon automation beyond facts, the following should be true:
- each memory type has an explicit validated contract
- each memory type has extraction/generation logic
- each memory type has persistence
- each memory type has retrieval
- each memory type has a dedicated eval harness
- each memory type has one dedicated log showing extraction and retrieval prompts/results
- extraction router exists with eval harness
- retrieval router exists with eval harness
- multi-memory composition exists with tests and eval harness

Only after that should broader daemon automation proceed.
