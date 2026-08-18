# Enhancements

Suggested improvements, roughly ordered by value vs effort.

---

## 1. Fact deduplication in the daemon

**Problem:** The daemon extracts facts from every new session and stores them all. If the same preference ("user prefers pytest") appears in 20 sessions, it gets stored 20 times. Search results contain duplicates, and the facts table grows without bound.

**Fix:** Before inserting a new fact, embed it and compute cosine similarity against existing facts. Skip insertion if similarity ≥ 0.90 to an existing fact. If it's between 0.85 and 0.90, update the existing fact's content to the newer phrasing (reinforcement).

**Where:** `memory/daemon.py` — in the fact insertion loop.

---

## 2. Fact confidence decay

**Problem:** Facts learned six months ago may no longer be true (tech stack changed, project finished, preference updated). The system has no way to distinguish fresh facts from stale ones.

**Fix:** Add a `reinforced_at` timestamp column to the `facts` table, updated each time a session reinforces an existing fact. Add a `confidence` float (0–1). The daemon runs a nightly pass decaying confidence by a small factor for facts not seen in the last N sessions. Facts below a threshold (e.g. 0.1) are flagged for review or auto-deleted.

**Where:** `memory/db.py` (schema), `memory/daemon.py` (decay pass).

---

## 3. DB size guard for compression trigger

**Problem:** Compression is only triggered manually (dashboard button or CLI). A long-running deployment will accumulate sessions indefinitely until the user notices.

**Fix:** The daemon checks `os.path.getsize(DB_PATH)` each cycle. If the DB exceeds a configurable threshold (default: 500 MB), it triggers compression automatically and logs the event. Add a `MEMORY_DB_MAX_MB` env var.

**Where:** `memory/daemon.py` — end of each processing cycle.

---

## 4. Soft-delete transcripts before full deletion

**Problem:** When sessions are deleted (by compression or the 30-day pruner), the raw transcript is gone permanently. If compression produced a poor summary, that context is lost forever.

**Fix:** Instead of `DELETE FROM sessions`, first `UPDATE sessions SET transcript = NULL` (soft-delete). Keep the row with metadata (session_id, dates, turn_count, facts) for 7 more days, then hard-delete. Gives a recovery window if compression output is wrong.

**Where:** `memory/db.py` → `delete_sessions()`.

---

## 5. Memory relevance scoring

**Problem:** Facts and insights are injected based on FTS5 keyword overlap with the current prompt. There's no signal for which facts are actually useful vs. which are always retrieved but never influence Claude's response.

**Fix:** Add a `times_surfaced` counter to the `facts` table, incremented each time a fact is injected by `wake_up.py`. Track `times_used` separately by adding a tool call in `save_hook.py` that detects if Claude referenced injected content in its response (heuristic: check for overlap between injected fact text and assistant turns). Surface low-`times_used` / high-`times_surfaced` facts in the dashboard as "possibly stale or irrelevant."

**Where:** `hooks/wake_up.py` (increment `times_surfaced`), `hooks/save_hook.py` (increment `times_used`), `memory/db.py` (schema), `dashboard.html` (new table view).

---

## 6. Cached answer invalidation

**Problem:** The 93% similarity threshold for cached answers is fixed. A cached answer from 3 months ago might be returned for a prompt that's superficially similar but asking about a different project or context.

**Fix:** Add a `cached_until` date to Q&A pairs (defaulting to 30 days from extraction). Expire old cached answers automatically. Add a session recency filter — only use cached answers from sessions in the same project/topic cluster as the current session.

**Where:** `memory/db.py` → `find_direct_answer()`.

---

## 7. Cross-agent identity linking

**Problem:** Sessions from different agents (Claude Code, Cursor, a custom script) are all stored in the same DB but there's no way to trace which agent contributed which facts or insights.

**Fix:** The `facts` table already has a `source` column. Extend the dashboard to show a per-agent breakdown of facts and sessions. Add an `agent` filter to `memory_hybrid_search` so Claude can ask "what did my Cursor sessions say about this?"

**Where:** `memory/dashboard_server.py` (API), `dashboard.html` (UI), `memory/mcp_server.py` (filter param).

---

## 8. Wake-up hook token budget

**Problem:** As facts and insights accumulate, the injected `=== MEMORY ===` block grows. Very long injections waste Claude's context window on low-relevance content.

**Fix:** Cap the total injection at a configurable token budget (default: 2,000 tokens). Prioritise: cached answer > identity > top-N facts by relevance score > insights. Truncate the least-relevant facts first. Log how often the cap is hit.

**Where:** `hooks/wake_up.py` → `_build_injection()`.
