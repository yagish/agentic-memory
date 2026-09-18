# Procedural prompt review

- **Builder:** `build_procedural_extraction_prompt` — [memory/procedural/extractor.py:68](../../../memory/procedural/extractor.py#L68)
- **Version:** `procedural-v2`
- **Contract:** `ExtractedProcedure` — title, summary, steps (min 1), trigger_phrases, tools, confidence?, source_quote?
- **Corpus:** 100 fixtures, 10 `expect_none`

## Verdict

The most thorough prompt and also the most overfit. Its definition of a
procedure is genuinely good — durable, repeatable, not a one-off — and the 10
negative examples are the best negative coverage in the repo. But it
contradicts its own contract in a way that causes hard failures, and the eight
positive examples are eight variations of a single shape: a service name plus a
named workflow label plus a linear command list. There is also a code-level
enrichment step that pollutes `tools` on ordinary English.

---

## 1. "Return empty arrays when a list has no items" — but `steps` cannot be empty

**What.** The contract declares `steps: tuple[str, ...] = Field(default=(), min_length=1)`
([contracts/__init__.py:174](../../../memory/contracts/__init__.py#L174)).
Verified:

```python
>>> ExtractedProcedure.model_validate({'title':'t','summary':'s','steps':[],'trigger_phrases':[],'tools':[]})
ValidationError: steps — Tuple should have at least 1 item after validation, not 0
```

The prompt never states that `steps` is different from the other lists. The
parallel working-memory prompt *does* say "Return empty arrays when a list has
no items" while `active_tasks` has the same `min_length=1` — see
[working-memory.md §1](working-memory.md), same bug.

**Why it matters.** A transcript with a clear procedure whose steps the model
cannot confidently order produces `steps: []` → `ValidationError` → retry → the
retry reminder also says nothing about steps → total loss. The correct output
in that situation was `{}`, and nothing tells the model that.

**Suggested change.** In the shape block:

```
  "steps": ["ordered repeatable steps — at least one is required"],
```

and in Rules:

```
- steps must contain at least one step. A procedure with no steps is not a
  procedure — return {} instead.
```

Add the same line to `_with_strict_json_retry`.

---

## 2. `_enrich_procedure_literals` injects ordinary English words into `tools`

**What.** [extractor.py:354-372](../../../memory/procedural/extractor.py#L354-L372)
post-processes the validated procedure, scanning title + summary + steps +
trigger_phrases with `re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", combined)` and
appending every match to `tools`.

Verified:

```python
>>> combined = 'Roll back the API deploy in staging. Page the SRE on-call. Check AWS and the CDN.'
>>> re.findall(r'\b[A-Z][A-Z0-9_]{2,}\b', combined)
['API', 'SRE', 'AWS', 'CDN']
```

None of those are tools. Any all-caps acronym qualifies — API, SRE, AWS, CDN,
CPU, SQL, JSON, HTTP, TODO, and the first word of a sentence written in caps.

**Why it matters.** `tools` is concatenated into the embedded semantic text
(`"Tools: " + "; ".join(...)` —
[procedural/repository.py:22](../../../memory/procedural/repository.py#L22)),
so noise directly degrades the retrieval vector for every procedure. This is
also exactly what the checklist's anti-patterns list calls out: *"adding
brittle hardcoded heuristics when prompt/validation should do the work."*

**Why it exists.** Almost certainly to satisfy the prompt rule *"If an env var
or literal is called out in a follow-up note like 'keep X', 'mention X', or
'compare X', include it in tools"* — that is, a code workaround for a prompt
instruction that did not land reliably.

**Suggested change.** Two parts.

- *Code:* narrow the regex to things that actually look like env vars —
  require an underscore or a digit, and a longer minimum:
  `r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"` matches `STAGING_DB_URL`,
  `PYTEST_ADDOPTS`, `CHECKOUT_DB_URL` and rejects `API`, `SRE`, `AWS`, `CDN`.
  Better still, delete the function and let the prompt own it (see next).
- *Prompt:* make the rule explicit enough that enrichment is unnecessary:

```
- tools must include every env var, command, script, flag, branch, job, dashboard,
  URL, stage, and channel that appears anywhere in the steps or in a follow-up
  note about the procedure. If a step mentions STAGING_DB_URL, tools must
  contain STAGING_DB_URL.
- Do not add generic acronyms (API, SRE, AWS, CDN, SQL) to tools. Only add a
  literal you could copy-paste and run, set, or open.
```

Then re-run the 100-fixture suite with enrichment disabled and see whether the
prompt alone holds.

---

## 3. Eight examples, one shape

**What.** Examples 1-8 are: web deploy, checkout flag rollback, checkout-api
local setup, checkout-api release cutoff, checkout-api backfill, checkout-api
docs publish, admin-web flaky test triage, pricing-api incident debug. Five of
eight use `checkout-api`. All eight are: *named service + named workflow label
+ linear list of shell commands + optional conditional escalation.*

The fixture corpus matches exactly — 8 positive themes (`deploy_workflow`,
`flag_rollback`, `local_setup`, `release_cutoff`, `backfill_rebuild`,
`docs_publish`, `flaky_test_triage`, `incident_debug`, `cache_bust`) × ~10
name variants (`atlas`, `birch`, `glint`, `jade`, …). So 100 fixtures test
roughly 9 distinct situations.

**Why it matters.** Procedural memory should capture *any* repeatable how-to.
Real ones that do not fit the shape: a review checklist with no commands; a
decision procedure ("when X, prefer Y, unless Z"); a debugging heuristic
sequence; a communication protocol ("escalate to #eng-oncall, then page after
15 min"); a convention ("name migrations `YYYYMMDD_description`"). None
resemble the eight examples, and the prompt rule *"The title should usually be:
<service or domain> + <exact workflow label>"* actively pushes against them —
a checklist with no service has no natural title under that rule.

Also: 4,900 characters of examples on every extraction, and the marginal
example teaches almost nothing the previous one did not.

**Suggested change.**

- Cut to four positive examples spanning *different shapes*, not different
  services: (a) a command-driven deploy — keep Example 1; (b) a conditional
  rollback — keep Example 2; (c) a **non-command checklist** — new; (d) a
  **decision/heuristic procedure** — new. Drop Examples 3-8 or keep at most one.
- Soften the title rule:

```
- When the transcript names a service and a workflow label, title it
  "<service> <workflow label>". When it does not, use a short specific name for
  what the procedure accomplishes.
```

- Add the two missing shapes as fixture themes so the corpus can see them.

Sketch for the checklist example:

```
Example 3:
Transcript:
User: Before approving any PR that touches the pricing module, we always check
that the rounding tests were updated, that there is a rollback note in the
description, and that someone from finance reviewed it.
Assistant: And if it changes a public API, it also needs a deprecation window.

Output:
{
  "title": "Pricing module PR review checklist",
  "summary": "Use this checklist when reviewing a PR that touches the pricing module. It verifies rounding tests, a rollback note, finance review, and a deprecation window for public API changes.",
  "steps": [
    "Check that the rounding tests were updated",
    "Check that there is a rollback note in the description",
    "Confirm someone from finance reviewed it",
    "Require a deprecation window if it changes a public API"
  ],
  "trigger_phrases": ["review a pricing PR", "pricing module review checklist"],
  "tools": ["pricing module"],
  "confidence": 0.9
}
```

---

## 4. The "rollback" keyword rule is a fixture artifact

**What.**

> - If the procedure is about rollback, disable, or recovery, include the word "rollback" in the summary.

**Why it matters.** This instructs the model to write a specific word into
prose to satisfy a `summary_contains` assertion. It is the checklist's
anti-pattern *"calibrating expected fixtures to current bad model behavior"*
inverted — calibrating the prompt to the assertion. And it is factually wrong
some of the time: a *recovery* procedure that restores from backup is not a
rollback, but this rule makes the model call it one.

**Suggested change.** Delete the rule. If retrieval genuinely needs rollback
procedures to match the query "how do I roll this back", that belongs in
`trigger_phrases` — which is the field that exists for exactly this — not
smuggled into the summary:

```
- When the procedure undoes, disables, or recovers from something, include a
  trigger phrase using the user's likely wording, such as "roll back <service>"
  or "disable <feature>".
```

Then relax the fixture assertion from `summary_contains: ["rollback"]` to
checking `trigger_phrases`.

---

## 5. The summary is over-specified into a template

**What.** Three rules constrain the summary's wording:

> - The summary should explicitly repeat the service name and the workflow label from the transcript when present.
> - If the procedure is about rollback… include the word "rollback"…

and every example's summary begins with the literal words "Use this …".

**Why it matters.** Eight examples all opening "Use this X for Y. It does A, B,
and C." teaches a template more strongly than any rule. Templates are fine
until the procedure does not fit one — and combined with §3's single shape, the
model has essentially been given a fill-in-the-blank form.

**Suggested change.** Collapse the three rules into one intent-level rule and
vary the example phrasing:

```
- The summary should say when to reach for this procedure and what it
  accomplishes, using the transcript's own names for the service and the
  workflow so a future search matches.
```

---

## 6. `trigger_phrases` carries retrieval weight that no rule acknowledges

**What.** The rule is ~90 words of prescription about article-free phrases,
label-preserving phrases, and "how do I …" shapes. What it never says is *why*:
`trigger_phrases` is embedded into the retrieval text as `"Useful for: …"`
([repository.py:20](../../../memory/procedural/repository.py#L20)), so these
phrases are the primary thing a user's future prompt is matched against.

**Why it matters.** The model is following formatting rules without knowing the
objective, so it optimizes shape over coverage — three phrases of the same
shape rather than three genuinely different ways someone might ask.

**Suggested change.** Lead with the purpose, then keep one or two shape hints:

```
- trigger_phrases are matched against a future user prompt to decide whether to
  surface this procedure. Write 2-4 phrases covering genuinely different ways
  someone might ask for it: the short imperative ("deploy checkout-api"), the
  question form ("how do I deploy checkout-api"), and the phrasing of the
  problem that leads here ("checkout deploy is stuck"). Prefer variety of
  wording over variety of formatting.
```

---

## 7. Minor

- **"one durable repeatable procedure only"** — same plurality question as
  episodic ([episodic.md §2](episodic.md)). A long session can contain a deploy
  flow *and* a rollback flow; the second is silently dropped. Worth an explicit
  rule: *"If the transcript contains more than one distinct procedure, extract
  the one most likely to be needed again and ignore the rest."*
- **`_repair_common_json_string_escapes`**
  ([extractor.py:302](../../../memory/procedural/extractor.py#L302)) exists
  because Example 7 puts `PYTEST_ADDOPTS="-x"` — a value containing quotes —
  into a JSON string. The prompt says "Do not put unescaped double quotes
  inside JSON string values" while demonstrating the hard case. Consider using
  a single-quoted or quote-free literal in the example so the prompt is not
  fighting itself.
- **Negative examples 9 and 10 are excellent** and should be kept as-is. They
  teach the past-tense-incident and one-off-fix boundaries, which are the two
  genuinely hard negatives. This is the model for what the positive examples
  should look like after §3.
