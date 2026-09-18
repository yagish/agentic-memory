# Episodic prompt review

- **Builder:** `build_episodic_extraction_prompt` — [memory/episodic/extractor.py:66](../../../memory/episodic/extractor.py#L66)
- **Version:** `episodic-v2`
- **Contract:** `ExtractedEpisode` — title, abstract, participants, decisions, outcomes, follow_ups, confidence?, source_quote?
- **Corpus:** 100 fixtures, **0 negative**

## Verdict

The best-balanced prompt of the five: the rules are stated generically, the
four examples each teach a distinct shape (plain fix, named owner, incident
rollback, staged rollout), and none of them are repo-specific. Two structural
gaps: it is the only extractor with no way to decline, and its boundary against
session memory is undefined in both prompts.

---

## 1. No escape hatch — the model must always invent an episode

**What.** Procedural, working memory, and session memory all end with an
explicit "return `{}`" path. Episodic does not:

| Prompt | Can decline? |
|---|---|
| procedural | yes — `return an empty JSON object: {}` |
| working_memory | yes — `return exactly: {}` |
| session_memory | yes — `return exactly: {}` |
| **episodic** | **no** |

`parse_extracted_episode` ([extractor.py:206](../../../memory/episodic/extractor.py#L206))
has no `None` branch either, and `ExtractedEpisode` requires non-empty `title`
and `abstract`. So on a transcript with no event — "thanks, that explains it",
a failed one-turn question, an aborted session — the model's only valid move is
to manufacture one.

**Why it matters.** Fabricated episodes are stored, embedded, and retrieved.
`_format_episode` injects up to two of them into every wake-up context
([_format.py:29-31](../../../memory/retrieval/_format.py#L29-L31)). An invented
"Discussion about embeddings" episode competes for the same budget as real
recall and can beat it on semantic similarity for a vague prompt. This is the
one failure mode that actively degrades the product rather than merely missing
value.

Fixture evidence: **0 of 100** episodic fixtures are negative, while every
other type has 10. The corpus cannot see this gap because it was built assuming
an episode always exists.

**Suggested change.** Three coordinated edits.

*Prompt* — after the shape block:

```
If the transcript contains no real event, discussion, or decision — for example
a one-off question and answer, a greeting, an aborted session, or only tool
noise — return exactly:
{}
```

and in Rules:

```
- Do not manufacture an episode from a transcript where nothing happened. An
  empty result is correct and useful; an invented episode is not.
```

*Example 5* (replacing nothing, purely additive):

```
Example 5:
Transcript:
User: What does cosine similarity mean?
Assistant: It measures the angle between two vectors, ignoring magnitude.
User: Got it, thanks.

Output:
{}
```

*Code* — `parse_extracted_episode` returns `ExtractedEpisode | None`, mirroring
`parse_extracted_working_memory` ([working_memory/extractor.py:171](../../../memory/working_memory/extractor.py#L171)).
`_create_episodic_entry` ([daemon/__init__.py:118](../../../memory/daemon/__init__.py#L118))
already handles a falsy return via `emitted=bool(result)`.

*Fixtures* — add ~10 `expect_none` episodic cases to match the other types.

---

## 2. The boundary against session memory is undefined

**What.** Both prompts run on the same full transcript and ask for
overlapping things:

| | episodic | session_memory |
|---|---|---|
| name | `title` | `title` |
| narrative | `abstract` (1-2 sentences, "what happened and where it ended") | `summary` (1-2 sentences, "the session and current state") |
| results | `outcomes` | `outcomes` |
| forward | `follow_ups` | `next_steps` |
| extra | `participants`, `decisions` | `what_was_tried`, `left_off_at` |

Episodic says "Focus on the single main event"; session memory says "Focus on
the main session handoff, not every side remark." On a single-topic session —
the common case — those are the same instruction.

**Why it matters.** See [README.md §C7](README.md). Both get embedded, both get
formatted into the same envelope
([_format.py:101-107](../../../memory/retrieval/_format.py#L101-L107)) as
"Recent related episode: …" and "Relevant prior session: …". The user pays two
LLM calls and two embeddings for one paragraph delivered twice.

**Suggested change.** Pick a boundary and state it in *both* prompts. Two
coherent options:

- **By scope (recommended).** Episodic = one event, and a session with three
  distinct events yields three episodes. Session memory = exactly one per
  session, the handoff. This makes episodic plural and session memory singular,
  which matches the storage (episodic is `list_recent_episodic`; session memory
  is one row per session) and matches the field names (`happened_at` vs
  `updated_at`).

  Episodic gains: *"A session may contain several distinct episodes. Extract
  only the single most significant one."* — or change the contract to return a
  list, which is the more honest fix.

  Session memory gains: *"Do not describe one sub-event. Summarize the whole
  session as a handoff, even when it covered several topics."*

- **By durability.** Episodic = what happened, permanent record. Session memory
  = resumable state, superseded by the next session. This is closer to the
  cognitive-science framing but harder for a model to apply to a transcript.

This needs a decision before either prompt is edited; it is the one finding
here that is architectural.

---

## 3. "max 10 words" is the only stated length constraint, and it is the wrong one

**What.** `title` is capped at "max 10 words" in the prompt but 200 characters
in the contract. `abstract` is capped at "1-2 sentences" in the prompt but 600
characters in the contract — and 1-2 sentences that "preserve important literal
details when central: names, branch names, env vars, flags, file names, model
names, percentages, dates, and numeric thresholds" can easily run past 600.

**Why it matters.** Blowing 600 chars is a hard `ValidationError` → retry →
possible total loss of the episode. The retry reminder
([extractor.py:57-65](../../../memory/episodic/extractor.py#L57-L65)) does not
mention length, so the retry is likely to fail the same way. Meanwhile the word
limit on titles is stricter than the contract needs.

**Suggested change.** State contract limits in the shape block:

```
  "title": "short title (max 10 words, 200 characters)",
  "abstract": "1-2 sentence summary of what happened and where it ended (max 600 characters)",
```

and add to `_with_strict_json_retry`: *"Keep title under 200 characters and
abstract under 600 characters."*

---

## 4. The literal-preservation list is a closed enumeration

**What.** Repeated three times across the prompt and the retry:

> names, branch names, env vars, flags, file names, model names, percentages, dates, and numeric thresholds

**Why it matters.** It generalizes by luck. Port numbers, error codes, ticket
IDs, table names, queue names, region names, customer names, commit SHAs, and
time windows are all equally central when they appear, and none are listed. A
model that has been given nine categories tends to treat them as the set.

**Suggested change.** Lead with the principle, keep the list as illustration:

```
- Preserve any literal string that a future reader would need to act on this
  episode — anything you could copy-paste or search for. That includes names,
  branch names, env vars, flags, file names, model names, percentages, dates,
  numeric thresholds, error codes, ticket IDs, and identifiers of any other
  kind. When unsure whether a literal matters, keep it.
```

This replaces three separate enumerations with one rule.

---

## 5. `participants` excludes the two people who are always there

**What.**

> - do not include generic roles like "user" or "assistant"

**Why it matters.** Correct as a rule, but it makes `participants` empty on
most real transcripts — 2 of the 4 examples have `[]`. A field that is empty
most of the time still costs tokens in the shape block and still invites the
model to fill it. The genuinely useful signal is *third parties*: who was
assigned, who requested, who must be told.

**Suggested change.** Reframe positively so the field earns its place:

```
- Participants:
  - list third parties named in the transcript who have a stake in this episode:
    people assigned ownership, who requested the work, who must be notified, or
    who are blocking it
  - never include the user or the assistant themselves
  - an empty list is normal and correct when no third party was named
```

---

## 6. Minor

- **`decisions` vs `outcomes` overlap.** Example 3 lists "Add a startup guard"
  in both. The prompt does not say whether a decision that was also executed
  belongs in one, the other, or both. Suggest: *"When a decision was carried
  out in the same session, record the choice in decisions and the completed
  result in outcomes. They may describe the same change from different sides."*
  (Which legitimizes what Example 3 already does.)
- **Completion-verb list is closed too:** "passed, green, fixed, works now,
  enabled, acknowledged, shipped, or added" appears in both the prompt and the
  retry. Same fix as §4 — state the principle ("a concrete completed result
  stated in the transcript") and let the verbs illustrate.
- **Examples 1 and 3 are structurally identical** (fix a thing, state the
  result, note a follow-up). Example 1 could be dropped or repurposed as the
  negative case from §1, keeping the example count at four.
