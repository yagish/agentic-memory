# Memory Type Implementation Checklist

Use this checklist when adding the **next memory type**.

This is based on what we did for **episodic memory** and, where useful, the stronger fixture/prompt workflow we used for **facts**.

---

## Goal

For each new memory type, we want:

- a clear validated contract
- extraction logic with a pinned prompt version
- durable storage
- retrieval behavior integrated into recall
- good observability/logging
- broad fixture coverage
- real-model acceptance tests against Ollama
- a repeatable tuning loop driven by fixtures, not guesswork

---

## Phase 1: Define the memory type

- [ ] Write down what this memory type is for.
- [ ] Write down what it is **not** for.
- [ ] Decide whether it is:
  - [ ] durable
  - [ ] session-scoped
  - [ ] directly answerable
  - [ ] retrieval-only context
- [ ] Identify the minimum fields required to make it useful.
- [ ] Identify optional fields that improve rendering/retrieval.
- [ ] Decide what counts as a valid extraction.
- [ ] Decide what must be rejected.

### Output of this phase

- clear semantic definition
- field list
- boundaries vs other memory types

---

## Phase 2: Create fixtures before writing the prompt

This should happen **earlier than almost everything else**.

The fixtures are what tell us what the prompt should do.
Without them, prompt work turns into guesswork.

- [ ] Create fixture directories under `tests/fixtures/<type>/`.
- [ ] Add:
  - [ ] `sessions/`
  - [ ] `expected/`
- [ ] Add a **broad initial corpus immediately**, before prompt tuning.
- [ ] Write expected outputs for **desired behavior**, not current model behavior.
- [ ] Include positive, hard, and boundary cases from the start.
- [ ] Add `discover_<type>_fixture_cases(...)` helper if useful.
- [ ] Add a fixture inventory test that asserts:
  - [ ] session/expected parity
  - [ ] minimum fixture count guard
- [ ] Add a real-model fixture test file similar to `tests/test_episodic_extraction_fixtures.py`.
- [ ] For the **next new memory type**, require **at least 100 fixtures**.
- [ ] Facts can use a lower target like **50 fixtures** when the coverage spread is already broad.

### Important lesson

Do **not** stop at “fixtures exist.”
The useful workflow is:

1. create many fixtures
2. define expected outputs for **desired behavior**
3. build/tune the prompt against those tests

Not:

1. run model
2. copy current outputs into expected files
3. declare success

Also do **not** hardcode code-path hacks just to make one scenario pass.

When a scenario fails:

1. add or improve the fixture
2. add or improve the expected output
3. adjust the prompt and minimal generic validation
4. re-run the real-model fixture suite

Do **not** solve fixture failures by adding brittle special cases like:

- regex matching for one phrase pattern
- attribute-specific `if/else` trees just for one example
- hardcoded scenario gates
- ad hoc string checks that only make a single fixture pass

### Include early in the corpus

- [ ] straightforward extraction
- [ ] mixed conversation with assistant turns
- [ ] decisions
- [ ] outcomes
- [ ] follow-ups / next steps
- [ ] named people
- [ ] exact literals that matter
  - [ ] env vars
  - [ ] branch names
  - [ ] flags
  - [ ] file names
  - [ ] model names
  - [ ] numeric thresholds
  - [ ] dates / rollout percentages
- [ ] multiple events in one transcript
- [ ] one event with several outcomes
- [ ] handoff / owner assignment
- [ ] rollback / incident / remediation flow
- [ ] architecture decision vs implementation result
- [ ] transcript with tool-like noise if relevant
- [ ] generic wording that tempts the model to over-summarize
- [ ] no real memory of this type present
- [ ] another memory type should handle it instead
- [ ] transcript mostly contains requests/instructions/tool traces
- [ ] enforce `100+` fixtures as the default minimum for a new memory type
- [ ] allow `50+` as the lower bound for facts when that memory type already has broad category coverage

---

## Phase 3: Add the contract

- [ ] Add a new validated contract to `memory/contracts.py`.
- [ ] Normalize fields that should have stable comparisons.
- [ ] Deduplicate list-like fields if needed.
- [ ] Reject unknown fields with strict validation.
- [ ] Add contract tests.

### Pattern from episodic memory

Episodic memory ended up with:

- `title`
- `abstract`
- `participants`
- `decisions`
- `outcomes`
- `follow_ups`
- `confidence`
- `source_quote`
- persistence fields like `source_session_id`, `happened_at`

### Tests

- [ ] valid payload accepted
- [ ] normalization works
- [ ] missing required field rejected
- [ ] invalid ranges rejected
- [ ] unknown fields rejected

---

## Phase 4: Build extraction helpers

- [ ] Create or extend `memory/<type>.py`.
- [ ] Add a pinned prompt version constant.
- [ ] Add `build_<type>_extraction_prompt(...)`.
- [ ] Add `parse_extracted_<type>(...)`.
- [ ] Add `extract_<type>_from_session_text(...)`.
- [ ] Add one strict retry path for malformed model output.
- [ ] Add logging for:
  - [ ] extraction start
  - [ ] raw model result
  - [ ] validation errors

### Pattern from episodic memory

We used:

- prompt builder
- strict JSON parser/validator
- retry reminder
- dedicated log file
- normalized helper for fixture assertions

### Tests

- [ ] wrapped JSON still parses if parser allows it
- [ ] lists normalize/dedupe correctly
- [ ] prompt includes the right transcript context
- [ ] prompt includes critical instructions/anti-examples

---

## Phase 5: Tune the prompt using the fixture corpus

This is the key loop.

- [ ] Run the fixture suite against Ollama.
- [ ] Inspect failures by category.
- [ ] Tighten prompt instructions.
- [ ] Re-run.
- [ ] Repeat until behavior is good enough.
- [ ] Keep expected outputs aimed at desired behavior while iterating.
- [ ] Prefer prompt changes and minimal generic validation over hardcoded scenario logic.

### Common prompt improvements we learned

- [ ] ask for one main event only
- [ ] force concrete titles
- [ ] require current end state in abstract
- [ ] preserve literal details when central
- [ ] explicitly include named people when they materially matter
- [ ] tell the model not to genericize important specifics
- [ ] tell the model not to invent missing details
- [ ] tell the model not to copy from examples

### For episodic memory specifically

Prompt tuning improved when we explicitly told the model to preserve:

- names
- branch names
- env vars
- flags
- file names
- model names
- percentages
- dates
- numeric thresholds

---

## Phase 6: Design storage and repository APIs

- [ ] Decide DB table shape.
- [ ] Decide what gets embedded for retrieval.
- [ ] Add DB insert/read/search helpers in `memory/db.py`.
- [ ] Add repository wrapper in `memory/<type>_repository.py` if needed.
- [ ] Keep runtime APIs aligned with the canonical shape.
- [ ] Avoid long-term runtime support for legacy formats unless migration truly needs it.

### Questions to answer

- [ ] Is there one row per extracted item?
- [ ] Is there an embedding column?
- [ ] Do we need semantic text distinct from canonical stored fields?
- [ ] Do we need explicit migration support?
- [ ] Can retrieval stay semantic/vector-based end to end, without keyword/`LIKE` fallback?

### Tests

- [ ] save persists expected structured fields
- [ ] list/read returns normalized shape
- [ ] semantic retrieval returns correct rows
- [ ] filters/thresholds behave as expected

---

## Phase 7: Integrate retrieval into recall

- [ ] Decide how this memory type participates in recall:
  - [ ] direct answer
  - [ ] context injection
  - [ ] ranking signal only
- [ ] Add retrieval call(s) in `memory/retrieval.py`.
- [ ] Use semantic/vector retrieval as the default and preferred path.
- [ ] Do **not** add regex matching, SQL `LIKE`, or keyword fallback just to rescue one query shape.
- [ ] Add thresholds/limits only if needed.
- [ ] Add fallbacks only when clearly justified.
- [ ] Make sure weak hits do not hijack unrelated prompts.
- [ ] Update integration behavior in `integrations/common.py` if needed.

### Pattern from episodic memory

Episodic memory is primarily used for **context injection**, not direct answering.

### Tests

- [ ] good match returns this memory type
- [ ] weak/unrelated match is filtered out
- [ ] mixed memory-type queries behave correctly
- [ ] no-memory path returns noop cleanly

---

## Phase 8: Keep expectations aimed at desired behavior

When using fixtures for prompt improvement:

- [ ] expected outputs should reflect what we want
- [ ] do not weaken expectations just to match a bad prompt
- [ ] only loosen assertions where genuine model variance is acceptable
- [ ] prefer snippet-based assertions for robustness

### Good assertion style

Use partial semantic checks like:

- `title_contains`
- `abstract_contains`
- `participants_contains`
- `decisions_contains`
- `outcomes_contains`
- `follow_ups_contains`

This worked well for episodic extraction because exact phrasing varies, but important content should still appear.

---

## Phase 9: Add manual/live verification

- [ ] run extraction manually on representative transcripts
- [ ] inspect stored rows in the real DB
- [ ] inspect retrieval on real prompts
- [ ] verify actual recall behavior through the integration path
- [ ] confirm the memory type helps real prompts without causing hijacks

### Verify both

- [ ] the extraction is good
- [ ] the retrieval/rendering behavior is good

These are different problems.

---

## Phase 10: Add observability

- [ ] add dedicated logs if the memory type has complex extraction/retrieval
- [ ] expose relevant data in dashboard endpoints if helpful
- [ ] surface stored rows in the dashboard if useful for debugging
- [ ] make it easy to inspect why a memory was or was not returned

### Good debugging signals

- extraction prompt version
- raw model output
- validated object
- retrieval similarity
- threshold decisions
- fallback path usage

---

## Phase 11: Handle migration/backfill explicitly

If the new memory type changes schema or embedding text:

- [ ] prefer an explicit external migration script over hidden runtime migration
- [ ] make backfill/re-embedding one-off and inspectable
- [ ] document the command in `README.md` if user-facing

---

## Phase 12: Support long-running real-model test runs

Real-model fixture suites can take a long time.

### Background run pattern

```bash
mkdir -p .tmp
nohup env MEMORY_RUN_OLLAMA_TESTS=1 python3 -m pytest tests/test_<type>_extraction_fixtures.py -q > .tmp/<type>-fixtures.log 2>&1 &
echo $! > .tmp/<type>-fixtures.pid
```

### Check status

```bash
ps -p "$(cat .tmp/<type>-fixtures.pid)"
tail -f .tmp/<type>-fixtures.log
```

### Stop it

```bash
kill "$(cat .tmp/<type>-fixtures.pid)"
```

---

## Definition of done for a new memory type

A memory type is not “done” just because extraction exists.

### It is done when:

- [ ] contract exists and is tested
- [ ] extraction helper exists and is tested
- [ ] storage exists and is tested
- [ ] retrieval integration exists and is tested
- [ ] broad fixture corpus exists
- [ ] fixture inventory/count guards exist
- [ ] real-model fixture suite passes or is explicitly documented as still in tuning
- [ ] manual/live verification has been done
- [ ] logging/debugging is good enough to diagnose failures
- [ ] migrations/backfills are explicit if needed

---

## Anti-patterns to avoid

- [ ] tuning retrieval before the extraction shape is stable
- [ ] calibrating expected fixtures to current bad model behavior
- [ ] overfitting to one transcript instead of using a large corpus
- [ ] adding brittle hardcoded heuristics when prompt/validation should do the work
- [ ] hardcoding special-case code just to make one fixture pass
- [ ] solving semantic retrieval problems with regex matching, SQL `LIKE`, or keyword fallbacks
- [ ] silently changing runtime compatibility without a migration path
- [ ] letting weak matches directly answer unrelated prompts
- [ ] relying only on unit tests without real-model acceptance tests

---

## Suggested implementation order for the next memory type

1. [ ] define purpose and boundaries
2. [ ] create fixture harness + inventory guards immediately
3. [ ] add a large fixture corpus immediately
4. [ ] write expected outputs for desired behavior
5. [ ] add contract + contract tests
6. [ ] add extraction helper + helper tests
7. [ ] expand the corpus until the new memory type has **at least 100 fixtures**
8. [ ] tune prompt with real-model fixture runs
9. [ ] add DB/repository storage + tests
10. [ ] add retrieval integration + tests
11. [ ] do live manual verification
12. [ ] add migration/backfill if needed
13. [ ] add dashboard/log visibility if useful

---

## Files to touch for most new memory types

Usually some subset of:

- `memory/contracts.py`
- `memory/<type>.py`
- `memory/<type>_repository.py`
- `memory/db.py`
- `memory/retrieval.py`
- `integrations/common.py`
- `tests/test_<type>.py`
- `tests/test_<type>_repository.py`
- `tests/test_contracts_<type>.py`
- `tests/test_<type>_extraction_fixtures.py`
- `tests/fixtures/<type>/sessions/*`
- `tests/fixtures/<type>/expected/*`
- `README.md`
- `scripts/*` if migration/backfill is needed

---

## Short version

If we want to do the next memory type well, repeat this pattern:

- define strict contract
- build extraction helper
- store canonical data cleanly
- integrate retrieval carefully
- create lots of fixtures
- make expected fixtures represent desired behavior
- tune prompt against real-model tests
- verify live behavior

That is the reusable playbook.
