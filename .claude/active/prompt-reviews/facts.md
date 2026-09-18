# Facts prompt review

- **Builder:** `build_fact_extraction_prompt` — [memory/facts/extractor.py:126](../../../memory/facts/extractor.py#L126)
- **Version:** `facts-v7`
- **Contract:** `ExtractedFact` — entity, attribute, value, confidence?, evidence?, source_quote?
- **Corpus:** 56 fixtures, 6 negative

## Verdict

The *definition* is accurate — it correctly describes durable user-profile and
preference facts, and correctly excludes requests. But the prompt is the most
overfit of the five: roughly 55% of its body is examples, and a large fraction
of those examples were clearly added to defeat individual fixtures from this
repo's own dashboard work. There is also a real correctness bug in the input
pre-filter that happens before the prompt ever runs.

---

## 1. `_extract_user_lines` silently drops most of a real transcript

**What.** [extractor.py:88-102](../../../memory/facts/extractor.py#L88-L102)
keeps only lines starting with `user:`, then falls back to *all* lines if none
matched. Turns are built by `_build_session_text` as `f"{role}: {content}"`
([compaction.py:29](../../../memory/daemon/compaction.py#L29)) — one line per
turn — but `content` is real multi-line chat text.

Verified:

```python
>>> _extract_user_lines("user: My name is Yash.\nI work at Acme as a Tech lead.\nMy shell is zsh.\nassistant: ok")
['My name is Yash.']
```

Two durable facts were discarded before the model saw anything.

**Why it matters.** Every user turn longer than one line loses everything after
its first line. This is the normal case for real sessions, and it is invisible:
no warning, no log, and the fixtures never catch it because every fixture is
one line per turn. The 56-case corpus cannot detect the bug it would most
benefit from catching.

**Suggested change.** Track the current speaker and accumulate until the next
`assistant:` prefix:

```python
def _extract_user_lines(session_text: str) -> list[str]:
    """Return user-authored lines, including continuation lines of a user turn."""
    user_lines: list[str] = []
    in_user_turn = False
    saw_prefix = False
    for raw_line in session_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("user:"):
            saw_prefix = True
            in_user_turn = True
            content = line.split(":", 1)[1].strip()
            if content:
                user_lines.append(content)
            continue
        if lowered.startswith("assistant:"):
            saw_prefix = True
            in_user_turn = False
            continue
        if in_user_turn:
            # Continuation line of the user's turn — keep it.
            user_lines.append(line)
    if saw_prefix:
        return user_lines
    return [line.strip() for line in session_text.splitlines() if line.strip()]
```

Note the `saw_prefix` change too: today a transcript of *only* assistant turns
falls through to "return every line", feeding assistant text to a prompt whose
first rule is "Extract facts from USER messages only."

**Add fixtures:** a multi-line user turn with a fact on line 2; an
assistant-only transcript expecting `[]`.

---

## 2. Heavily overfit to this repo's own dashboard sessions

**What.** These negative examples are in the prompt verbatim
([extractor.py:243-268](../../../memory/facts/extractor.py#L243-L268)):

- `"Can we time how much time each prompt embedding takes..."`
- `"Show a graph for each performance metric in the dashboard so we can adjust the timeout."`
- `"Move the overview text at the bottom of the page into a tooltip for the Process One Session button..."`
- `"Remove ObsoleteTables from db.py."`
- `"Make daemon run facts first, then episodic."`
- `"Use --once as a force flag."`

Each maps 1:1 to a fixture added in the current working tree
(`feature_request_can_we`, `feature_request_show_graph`,
`feature_request_with_quoted_context`, …). This is precisely the pattern
[docs/memory-type-checklist.md](../../../docs/memory-type-checklist.md) warns
about under "Anti-patterns": *"overfitting to one transcript instead of using a
large corpus."* The rule-based section above already covers all six cases
generically:

> - "Can we / Could we / Should we / Shall we [do X]" — this is a request, not a fact.
> - "Show / Add / Build / Create / Make / Move / Fix / Implement [X]" — this is a request, not a fact.

**Why it matters.** Three costs. Tokens: this prompt is ~4,500 characters of
instruction prepended to every fact extraction, on every session, forever.
Generalization: a model shown six project-specific negatives learns "dashboard
and daemon talk is not a fact" more strongly than "imperatives are not facts",
so a request phrased in an unseen domain is more likely to slip through.
Maintenance: every new false positive invites a seventh example, and the prompt
grows monotonically. It is already at v7.

**Suggested change.** Cut the six repo-specific negatives down to two that
teach *distinct* shapes the rules do not already cover, and keep them generic:

```
Input:
User: <bash-input>pwd</bash-input>
Output:
[]

Input:
User: My current task is fixing auth middleware.
Output:
[]
```

The first teaches tool-noise rejection (a shape, not a topic). The second
teaches the genuinely subtle case: a first-person declarative sentence that
*looks* like a profile fact but is session-scoped. Everything else is already
covered by the rules.

Then verify against the full 56-fixture corpus. If removal regresses cases, the
right response is a better *rule*, not the example back — e.g. add:

> - A request stays a request even when it names a file, a metric, a UI element, or a system component. The subject of a request is never an entity in a fact.

---

## 3. The durable/non-durable boundary is asserted, never defined

**What.** The prompt's core test is:

> - A fact is stable, reusable knowledge that will likely still be useful in future sessions.

then enumerates instances: identity, role, company, timezone, location, editor,
shell, package manager, terminal. "Durable" is never given an operational test
the model can apply to an attribute *not* on the list.

**Why it matters.** The enumeration is the real specification, so anything off
it is a coin flip. Genuinely durable facts outside the list — "I'm colorblind,
use shapes not colors in charts", "our team's on-call rotation is weekly",
"I only have admin on staging" — have no clear home. `project_goal` and
`project_status` fixtures already sit awkwardly here: "the project status is
prototype" is expected to extract, but status is among the *least* durable
things about a project.

**Suggested change.** Add one operational test above the enumeration:

```
Test for durability — ask: "If the user opened a brand-new conversation in six
months and this fact were missing, would the assistant get something wrong?"
- Yes -> it is a durable fact.
- No, it only matters for the work happening right now -> it is not.
```

Then reframe the enumeration as illustration rather than specification:

```
- Facts that pass this test are usually about the user: identity, role,
  company, timezone, location, editor, shell, package manager, terminal, and
  stable preferences. That list is illustrative, not exhaustive — apply the
  test, not the list.
```

This is the single highest-leverage genericity change in this file: it gives
the model a rule it can extend, instead of a set it can only match.

---

## 4. `project.status = prototype` contradicts the durability rule

**What.** [tests/fixtures/facts/expected/project_goal.json](../../../tests/fixtures/facts/expected/project_goal.json)
expects `{"entity": "project", "attribute": "status", "value": "prototype"}`.
The prompt says non-user facts are kept only for "durable repo or project
metadata". A project's status changes by definition — it is the field most
likely to be stale on recall, and a stale fact is worse than a missing one
because facts can answer deterministically
([common.py](../../../integrations/common.py) fact-only answer path).

**Why it matters.** The fixture is teaching the boundary incorrectly, so prompt
tuning against it pulls in the wrong direction. Per the checklist's Phase 8 —
"expected outputs should reflect what we want" — this expectation looks like it
was calibrated to model behavior rather than desired behavior.

**Suggested change.** Decide explicitly, then make prompt and fixture agree:

- **Option A (recommended):** status is not durable. Change the fixture to
  expect only `project.goal`, and add to the prompt: *"Do not extract volatile
  project state such as status, current phase, progress, or version-in-flight.
  Those belong in session memory."*
- **Option B:** status is durable metadata. Then say so explicitly in the
  prompt so the model is not guessing, and accept staleness.

Flagging rather than choosing — this is a product call about what a fact means
in this system.

---

## 5. "Exact value rule" is scoped to four attributes but the reasoning is universal

**What.** [extractor.py:169-172](../../../memory/facts/extractor.py#L169-L172)
forbids synonym substitution only for role, title, name, and company.

**Why it matters.** The same failure mode applies everywhere: `zsh` → `Zsh`,
`Neovim` → `nvim`, `uv` → `UV`, `iTerm2` → `iTerm`. Facts are matched
deterministically on `entity.attribute = value`
([text.py:14](../../../memory/facts/text.py#L14)), and
`build_semantic_fact_text` embeds the value verbatim, so casing and spelling
drift degrades both the canonical string and the embedding.

**Suggested change.** Generalize the rule and keep the role example as the
illustration of the sharpest case:

```
Exact value rule — for every fact:
- Copy the user's exact words for the value. Never substitute synonyms,
  abbreviations, expansions, or different casing.
- "Tech lead" stays "Tech lead" — never "CTO", "Staff engineer", or
  "Engineering manager". Never infer seniority or elevate a stated role.
- "Neovim" stays "Neovim", never "nvim". "zsh" stays "zsh", never "Zsh".
```

---

## 6. `evidence` vs `source_quote` is never explained

**What.** The prompt says "Every fact must include: ... source_quote" and
"Optional fields allowed: confidence, evidence", with no definition of
`evidence` or how it differs from `source_quote`.

**Why it matters.** An undefined optional field is an invitation to
hallucinate structure. Since `extra="forbid"` means a near-miss key name is a
hard failure that loses the whole batch (see README §C6), an ambiguous field is
disproportionately expensive. Neither field is read by retrieval today.

**Suggested change.** Simplest: drop `evidence` from the prompt entirely
(leaving it in the contract for compatibility). If it should stay, define it:
*"evidence: optional additional context from elsewhere in the transcript that
supports the fact, when the source_quote alone is ambiguous."*

---

## 7. Minor

- **`_with_strict_json_retry` duplicates the base prompt's rules**
  ([extractor.py:73-81](../../../memory/facts/extractor.py#L73-L81)). If the
  first attempt failed *because* those rules did not land, restating them is
  unlikely to help. A retry is more useful when it changes the frame: *"Your
  previous response was not a valid JSON array. Output only a JSON array
  starting with [ and ending with ]. If in doubt, output []."*
- **`_facts_debug_log("session_text", ...)` writes the full transcript** to
  `~/.memory/facts.log` on every extraction
  ([extractor.py:451](../../../memory/facts/extractor.py#L451)), and the retry
  loop logs the full prompt twice more. That is roughly 3× transcript size per
  session on disk, unrotated. Not a prompt issue — noting it while in the file.
- **Ordering rule may conflict with the conflict rule.** "Preserve the order of
  first appearance" plus "Keep conflicting facts when the value differs" is
  correct as written, but the `conflicting_timezone` example is the only place
  it is demonstrated. Worth keeping that example — it earns its place, unlike
  the six in §2.
