# Working memory prompt review

- **Builder:** `build_working_memory_extraction_prompt` — [memory/working_memory/extractor.py:64](../../../memory/working_memory/extractor.py#L64)
- **Version:** `working-memory-v1`
- **Contract:** `ExtractedWorkingMemory` — current_goal, current_focus, active_tasks (min 1), constraints, next_step, status, confidence?, source_quote?
- **Corpus:** 100 fixtures, 10 `expect_none`

## Verdict

The clearest *definition* of the five — the is-for / is-not-for block is
genuinely good and should be copied to the other prompts. Its problems are
mechanical: it contradicts its own contract twice, one of its four status
values is logically unreachable, and it has the fewest examples of any prompt
carrying a six-field output shape.

---

## 1. "Return empty arrays when a list has no items" — but `active_tasks` cannot be empty

**What.** `active_tasks: tuple[str, ...] = Field(default=(), min_length=1)`
([contracts/__init__.py:222](../../../memory/contracts/__init__.py#L222)).
Verified:

```python
>>> ExtractedWorkingMemory.model_validate({'current_goal':'g','current_focus':'f',
...     'active_tasks':[],'constraints':[],'next_step':'n','status':'in_progress'})
ValidationError: active_tasks — Tuple should have at least 1 item after validation, not 0
```

The prompt says the opposite, without exception. Identical to
[procedural.md §1](procedural.md).

**Why it matters.** The natural case that triggers it: work is genuinely
blocked on an external thing, so there is a goal, a focus, a constraint and a
next step, but no task the user can act on. The model follows the prompt,
returns `active_tasks: []`, and the extraction is lost entirely — including the
blocker, which was the most valuable part.

**Suggested change.** Shape block:

```
  "active_tasks": ["unfinished task 1", "unfinished task 2"],   // at least one required
```

Rules:

```
- active_tasks must contain at least one item. If there is genuinely no
  unfinished task, there is no active working context — return {} instead.
- constraints may be an empty array.
```

Add to `_with_strict_json_retry`: *"active_tasks must not be empty."*

---

## 2. `status: "done"` is allowed but should never be produced

**What.** Two rules in tension:

> Allowed status values: in_progress, blocked, ready_to_resume, **done**

> - Use done only when the transcript clearly says the work is wrapped up and no active continuation is needed; otherwise prefer ready_to_resume or in_progress.

But the type definition says working memory is *"not for: fully completed
conversations with no next step"*, and Example 3 shows exactly that case
returning `{}`. And `next_step` is a required non-empty field — so a `done`
record must still name a next step, which contradicts "no active continuation
is needed."

**Why it matters.** `done` is a trap: every transcript that qualifies for it
should have returned `{}` one rule earlier. A model that reaches for it
produces a record that is simultaneously valid and meaningless, and
`_format_working_memory` will then inject "Status: done." into a future
wake-up context ([_format.py:53](../../../memory/retrieval/_format.py#L53)) —
telling the assistant about finished work as if it were live state.

**Suggested change.** Pick one:

- **Remove `done`** from the allowed values in the prompt (leave it in
  `_WORKING_MEMORY_STATUSES` for stored-row compatibility), and replace the
  hedging rule with: *"If the work is finished with no continuation, return {}
  rather than a done status."*
- **Or** keep it and give it a real job: *"Use done only when the work is
  finished but a verification step remains — for example a deploy that
  succeeded and needs monitoring. Put that verification in next_step."*

The first is cleaner and matches the existing Example 3.

---

## 3. Three examples for a six-field output is thin

**What.** Examples are: ready_to_resume, blocked, `{}`. Missing: `in_progress`
— the most common status by far, and the default a model will reach for.
Compare: procedural has 10 examples for a five-field output.

**Why it matters.** `in_progress` vs `ready_to_resume` is the distinction the
model makes most often and has the least guidance on. Neither is defined. My
reading of the intent: `in_progress` = mid-task, the next action is obvious and
immediate; `ready_to_resume` = at a clean stopping point, someone could pick it
up cold. Nothing in the prompt says that.

**Suggested change.** Add definitions to the status list and one `in_progress`
example:

```
Allowed status values:
- in_progress — actively mid-task; the work was interrupted rather than parked
- blocked — cannot proceed until something external arrives or is decided
- ready_to_resume — at a clean stopping point; someone could pick this up cold
```

```
Example 3:
Transcript:
User: I'm partway through splitting the retrieval module — _fetch and _rank are
moved, _format still has the old imports.
Assistant: The test suite is red on three format tests until those imports are updated.
User: Keep going, I want the whole split done before I look at ranking.

Output:
{
  "current_goal": "Split the retrieval module into separate submodules",
  "current_focus": "Updating the old imports in _format after moving _fetch and _rank",
  "active_tasks": ["Update the old imports in _format", "Get the three failing format tests passing"],
  "constraints": ["Finish the whole split before starting on ranking"],
  "next_step": "Update the old imports in _format so the three failing format tests pass",
  "status": "in_progress",
  "confidence": 0.9
}
```

(Renumber the existing `{}` example to 4.)

---

## 4. "the same session or the next immediate handoff" is undefined at extraction time

**What.** The opening definition:

> A valid working memory is the current temporary active context that would help resume the work **in the same session or the next immediate handoff**.

**Why it matters.** The extractor runs in the daemon *after* the session has
ended ([daemon/__init__.py:340](../../../memory/daemon/__init__.py#L340)), so
"the same session" is always false by the time this prompt executes. More
importantly, "next immediate" has no shelf life attached, and nothing in
retrieval enforces one — `_format_working_memory` will happily inject a
three-week-old `in_progress` goal into today's context. Stale working memory is
worse than absent working memory: it asserts that something is currently in
motion when it is not.

**Suggested change.** Prompt side, make the horizon concrete:

```
A valid working memory is the active context someone would need to pick this
work back up in their next sitting. Write it for a reader who returns tomorrow
having forgotten the details but not the goal.
```

Code side, flag for the next agent: consider a recency cutoff or a staleness
decay on working-memory retrieval. Working memory is the one type where age
should probably disqualify rather than merely de-rank.

---

## 5. Boundary against session memory is drawn on the wrong axis

**What.** The is-not-for list excludes durable facts, procedures, and past
events — but never mentions session memory, which is its nearest neighbor.
Compare the two shapes:

| working memory | session memory |
|---|---|
| `current_goal` | `title` / `summary` |
| `active_tasks` | `next_steps` |
| `next_step` | `next_steps[0]` |
| `constraints` | — |
| `status` | `left_off_at` |

`next_step` / `next_steps` and `status` / `left_off_at` are close to the same
information. On the fixture corpus they are visibly close:
`working_backfill_atlas` and `session_backfill_atlas` describe the same
situation.

**Why it matters.** Both are retrieved and both are formatted into the same
envelope ([_format.py:101-103](../../../memory/retrieval/_format.py#L101-L103)),
working memory first. Duplication costs budget in the type where budget is
tightest.

**Suggested change.** State the axis explicitly in both prompts. The one that
holds up: **working memory is forward-looking state, session memory is
backward-looking narrative.** Working memory should carry no history at all.

```
- Do not summarize what already happened. Another memory type records the
  session's history. Working memory contains only what is still true and still
  pending right now.
```

---

## 6. Minor

- **`current_goal` vs `current_focus`** are distinguished only by "overall
  task" vs "the specific slice currently in motion." In Example 1 the focus is
  a 20-word sentence restating the goal plus the tasks. Suggest tightening:
  *"current_focus must be narrower than current_goal. If you cannot state a
  narrower slice, repeat nothing — use the single task currently in hand."*
- **`constraints` conflates four things** — "active blockers, dependencies,
  deadlines, or guardrails." A blocker (cannot proceed) and a guardrail (may
  proceed but not that way) are different enough that merging them loses the
  `blocked` signal. Low priority given the field is free text.
- **No length caps stated** — `current_goal`, `current_focus`, `next_step` are
  all `max_length=300`. See [README.md §C4](README.md).
- **The is-for / is-not-for block is the best framing in the repo.** Worth
  lifting into the episodic prompt, which has no equivalent.
