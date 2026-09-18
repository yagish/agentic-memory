# Session compaction prompt review

- **Builder:** inline in `_compact_session_text` — [memory/daemon/compaction.py:57](../../../memory/daemon/compaction.py#L57)
- **Version:** *none*
- **Status:** **dead code** — not called by any runtime path

## Verdict

This prompt is not part of the pipeline and has not been for some time. It is
reviewed here for completeness and because the decision about it is a one-liner
either way: delete it, or give it the one job it would actually be good at.

## Evidence it is dead

`_get_or_compact_session_text` ([compaction.py:75](../../../memory/daemon/compaction.py#L75))
takes `conn`, immediately does `del conn`, and returns the raw session text.
It never calls `_compact_session_text`. Full call graph:

```
memory/daemon/compaction.py:44   def _compact_session_text(...)      <- definition
memory/daemon/__init__.py:44         _compact_session_text,          <- import only
```

Nothing else. The module docstring is explicit and states the reasoning:

> Memory extractors now run against the full raw session transcript rather than
> an LLM-compacted rewrite. This avoids hallucinated details in downstream facts,
> episodes, procedures, working memory, and session memory.
>
> The compaction helper is retained for optional/debug use, but the daemon no
> longer uses compacted text as the source for memory extraction.

That was the right call. Compacting before extraction means five extractors
inherit one summarizer's omissions and hallucinations, and the literal-preservation
rules the other four prompts work hard at would be defeated upstream.

## Findings, if it is ever revived

### 1. It asks for everything, which means it prioritizes nothing

```
Include:
- All key facts (names, settings, preferences, technical details)
- Decisions made and their rationale
- Steps taken or discussed
- Outcomes reached and open questions
```

That is the union of all five memory types' inputs, compressed into
`_COMPACT_OUTPUT_CHARS`. Under a hard character budget with no priority order,
the model decides what to drop — and it will drop exactly the rare literals
(env var names, branch names, flags) that are individually low-frequency but
high-value, because that is what compression does.

### 2. No length enforcement on the output

`f"Your output must be under {_COMPACT_OUTPUT_CHARS} characters"` is a
request. Nothing truncates or retries; the result goes straight back to the
caller. Every other prompt in the repo at least has a validating contract
behind it.

### 3. Silent input truncation

`input_text = full_text[:_COMPACT_INPUT_CHARS]` ([compaction.py:54](../../../memory/daemon/compaction.py#L54))
cuts mid-word with no marker. The model is not told the transcript was cut, so
it summarizes a fragment as though it were whole — and the *end* of a session,
where the handoff lives, is what gets dropped.

### 4. No transcript fencing

`"TRANSCRIPT:\n" + input_text` — same issue as [README.md §C3](README.md), and
worse here because there is no contract to reject a derailed response.

### 5. Unversioned and untested

No version constant, no fixtures, and `tests/test_daemon.py:198` patches
`_get_or_compact_session_text` wholesale, so neither this function nor its
prompt is exercised anywhere.

## Recommendation

**Delete it.** Remove `_compact_session_text` and the unused import at
[daemon/__init__.py:44](../../../memory/daemon/__init__.py#L44). Keep
`_get_or_compact_session_text` under its current name if other callers depend
on it, or rename to `_session_text_for_extraction` and drop the vestigial
`conn` parameter.

Rationale: it is ~30 lines carrying a prompt that contradicts the documented
architecture. "Retained for optional/debug use" has not been exercised, and
git history preserves it if it is ever wanted back.

**If instead it should live**, give it the narrow job it is suited to and the
pipeline arguably needs: a *pre-extraction size guard* for transcripts too
large for the extractors' context window. That is a different prompt — one that
reduces volume while preserving every literal, rather than one that
"summarizes." Roughly:

```
Reduce this transcript to fit a smaller context window while losing nothing an
extractor would need.

Keep verbatim, without exception:
- every user message, condensed only by removing filler
- every literal: names, commands, flags, env vars, file paths, branch names,
  URLs, error text, numbers, and dates
- every statement of a decision, a result, or a next step

Compress only:
- assistant explanation and reasoning prose
- repeated or superseded content
- tool output, to its first and last few lines

Do not summarize. Do not paraphrase a literal. Do not add anything not present
in the transcript. Output the reduced transcript in the same
"role: content" line format, nothing else.
```

Note the inversion: the current prompt says *"condense into a summary"*; this
one says *"delete the compressible parts and keep the rest verbatim."* That is
the only form of compaction safe to put in front of five literal-sensitive
extractors — and it should be gated on transcript size, not applied by default.
