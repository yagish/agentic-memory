# Session memory prompt review

- **Builder:** `build_session_memory_extraction_prompt` — [memory/session/extractor.py:65](../../../memory/session/extractor.py#L65)
- **Version:** `session-memory-v1`
- **Contract:** `ExtractedSessionMemory` — title, summary, what_was_tried, outcomes, left_off_at, next_steps, confidence?, source_quote?
- **Corpus:** 100 fixtures, 10 `expect_none`

## Verdict

Internally the cleanest prompt here: the is-for / is-not-for block is sharp,
the rules are stated generically with no repo-specific leakage, and unlike
procedural it has no fixture-driven keyword hacks. Its problem is entirely
external — it substantially duplicates the episodic prompt, and neither prompt
acknowledges the other. That is the finding worth acting on; the rest is minor.

---

## 1. This prompt and the episodic prompt extract nearly the same thing

**What.** Both run on the identical `text_sample`
([daemon/__init__.py:340-347](../../../memory/daemon/__init__.py#L340-L347)),
both produce one object, and their fields line up almost one-to-one:

| episodic | session memory | overlap |
|---|---|---|
| `title` | `title` | same |
| `abstract` — "what happened and where it ended" | `summary` — "the session and current state" | same |
| `outcomes` | `outcomes` | same field name, same rule |
| `follow_ups` — "next steps or unresolved work" | `next_steps` — "useful concrete continuation steps" | same |
| `decisions` | — | episodic only |
| `participants` | — | episodic only |
| — | `what_was_tried` | session only |
| — | `left_off_at` | session only |

Compare the two instructions directly:

- episodic: *"Focus on the single main event, discussion, or decision from this session."*
- session: *"Focus on the main session handoff, not every side remark."*

On a single-topic session these select the same content. The fixture corpora
confirm it — `episodic/auth_refactor_*` and `session_auth_*` describe the same
situation with the same expected literals ("token validation", "redirect
loop", "refresh-token").

**Why it matters.** Three concrete costs:

1. **Two LLM calls per session** where one would do. These are sequential in
   `_process_session`, each with a 180s timeout.
2. **Two embeddings and two stored rows** for the same content.
3. **Double injection into the wake-up budget.** `_context_sections`
   ([_format.py:101-107](../../../memory/retrieval/_format.py#L101-L107))
   emits session memory then episodic, so a `resume`-intent prompt gets
   "Relevant prior session: Auth middleware refactor. The session moved token
   validation…" immediately followed by "Recent related episode: Auth
   middleware fix. The session moved token validation…". At the 500-token
   `task` budget that duplication can crowd out facts entirely — facts are
   formatted *last* ([_format.py:106](../../../memory/retrieval/_format.py#L106))
   and truncated first.

**Suggested change.** Decide the boundary, then write it into both prompts.
The option that fits the existing storage and field names best:

> **Episodic = one event, plural per session, permanent record of what happened.
> Session memory = one handoff per session, describing state at the end.**

Evidence this is the intended split: episodic stores `happened_at` (a point in
time) and has `list_recent_episodic` (plural retrieval); session memory stores
`updated_at` and has `left_off_at` (a state). The prompts just never say it.

Concretely, add to **this** prompt:

```
- This is the whole-session handoff, not a single event. If the session covered
  several topics, cover them all at the level a returning reader needs.
- Do not describe events for their own sake. Every line should help someone
  decide what to do next. Another memory type records the events themselves.
```

and to the **episodic** prompt:

```
- Record what happened, not what to do next. A separate memory type owns the
  session handoff. Keep follow_ups to concrete loose ends this event left open,
  not a plan for the next session.
```

If instead the decision is that one of them is redundant, session memory is the
more useful of the two to keep: `left_off_at` + `next_steps` is exactly what
the `resume` intent needs, and the 1,500-token resume budget
([_intent.py:24](../../../memory/retrieval/_intent.py#L24)) exists to serve it.
That is a larger call than a prompt edit.

---

## 2. `what_was_tried` vs `outcomes` blurs attempts and results

**What.**

> - what_was_tried should capture the most important implementation or investigation steps.
> - outcomes should capture concrete validated results, fixes, failures, or conclusions.

**Why it matters.** "Failures" appear under `outcomes`, but a failed attempt is
also a "step that was tried." Example 1 puts "Retested the login redirect flow
locally" in `what_was_tried` and "Redirect loop stopped reproducing locally" in
`outcomes` — the same action, split across two fields. That is defensible, but
the prompt never states the principle behind it, so the split is arbitrary in
any case the examples do not cover.

The distinction that matters for a returning reader is **what not to retry**:
an approach that was tried and failed is the single most valuable thing a
handoff can record, and right now it has no clear home.

**Suggested change.**

```
- what_was_tried: the approaches taken, whether or not they worked. This is the
  field that saves a returning reader from repeating a dead end — record
  abandoned approaches here even when nothing came of them.
- outcomes: what is now known to be true as a result — fixes that hold, tests
  that pass, conclusions that were validated, and approaches that were ruled out.
```

---

## 3. `left_off_at` is the highest-value field and gets the least guidance

**What.** One rule: *"left_off_at should clearly say what remains unresolved or
what state the work is in now."* It is capped at 300 chars, required non-empty,
and surfaced verbatim in recall as `"Left off at: …"`
([_format.py:96](../../../memory/retrieval/_format.py#L96)).

**Why it matters.** For a "where did we leave off" prompt — the exact phrase in
`_RESUME_INTENT_EXEMPLARS` ([_intent.py:11](../../../memory/retrieval/_intent.py#L11))
— this field *is* the answer. It deserves more than the `summary` gets, and
currently gets less. In both examples it restates the summary's second half.

**Suggested change.**

```
- left_off_at is the single most useful line for someone resuming. Write it as
  the state of the work right now, not as a recap: what is done and holding,
  what is half-finished, and what is untouched. It must be readable on its own
  without the summary. Maximum 300 characters.
```

Add a rule against restating the summary, since both examples currently do:

```
- Do not repeat the summary in left_off_at. If they would say the same thing,
  make left_off_at more specific.
```

---

## 4. No guidance for multi-topic sessions

**What.** *"Focus on the main session handoff, not every side remark"* and
*"not for: one tiny sub-event when the broader session has a clearer handoff."*
Both assume a dominant topic exists.

**Why it matters.** Long sessions routinely cover three unrelated things. The
prompt gives no way to represent that: `title` and `summary` are singular, and
"focus on the main one" silently discards the other two. Since session memory
is one row per session, there is no second row to catch them. Both example
transcripts are three lines on one topic, so the corpus cannot surface this —
and at 100 fixtures across 10 themes, that is a real gap.

**Suggested change.**

```
- When the session covered several unrelated threads, do not pick one and drop
  the rest. Title it for the session as a whole, name each thread in the
  summary, and let next_steps carry each thread's continuation.
```

Add 3-5 multi-topic fixtures.

---

## 5. Both examples are the same shape

**What.** Example 1: change made → it worked → tests still needed. Example 2:
command run → counts matched → watch dashboard. Both are *successful work with
a clean next step.*

**Why it matters.** Missing: a session that ended in failure, and a session
that ended mid-investigation with no conclusion. Those are the sessions where a
handoff matters most, and the model has no pattern for them. A failed session
under the current examples is likely to be written optimistically, because both
examples model optimism.

**Suggested change.** Add one:

```
Example 3:
Transcript:
User: The memory daemon keeps timing out on long sessions.
Assistant: I raised the Ollama timeout to 300s, but it still times out — the
model appears to stall rather than run slowly.
User: I also tried the smaller model and it did not stall, so it may be a
context-length problem. Out of time for today.

Output:
{
  "title": "Daemon extraction timeout investigation",
  "summary": "The session investigated daemon timeouts on long sessions. Raising the Ollama timeout to 300s did not help, but the smaller model did not stall, pointing at context length rather than speed.",
  "what_was_tried": ["Raised the Ollama timeout to 300s", "Ran the same session against the smaller model"],
  "outcomes": ["The 300s timeout did not fix the stall", "The smaller model did not stall, suggesting a context-length limit rather than slow inference"],
  "left_off_at": "Root cause is unconfirmed; the leading hypothesis is context length, and no fix has been attempted",
  "next_steps": ["Measure the token count of the sessions that stall", "Test whether truncating the transcript avoids the stall"],
  "confidence": 0.85
}
```

(Renumber the `{}` example to 4.)

This example also demonstrates §2 — a ruled-out approach living in `outcomes`.

---

## 6. Minor

- **No length caps stated.** `title` 200, `summary` 600, `left_off_at` 300. See
  [README.md §C4](README.md). `left_off_at` is the riskiest: it is required, and
  the guidance in §3 above pushes toward longer text.
- **The negative example is weak.** *"Thanks for the explanation of embeddings"*
  is obviously empty. A harder negative earns more: a session with real
  discussion but no work — "we talked through three approaches to caching and
  did not pick one." Is that `{}`, or is the unresolved decision itself the
  handoff? The prompt should say. My read: it is a valid session memory, with
  the decision as `left_off_at` — which is worth demonstrating rather than
  leaving to chance.
- **`_with_strict_json_retry` ends with a statement, not an instruction** —
  *"A valid compacted session memory should make it easy to resume the session
  later."* True, but it does not tell the model what to change on a retry.
  Replace with the concrete constraint that most likely failed: *"Output only a
  JSON object starting with { and ending with }. If there is no resumable work,
  output {}."*
