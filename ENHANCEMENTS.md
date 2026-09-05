# Enhancements

Ideas still relevant to the simplified system.

## 1. Fact deduplication / reinforcement
Avoid inserting near-duplicate facts every time the daemon sees the same preference or project fact.

## 2. Fact freshness
Add `reinforced_at` and/or confidence decay so stale facts can weaken over time.

## 3. Episode deduplication
Avoid storing highly similar episodes when multiple sessions repeat the same work summary.

## 4. Retrieval budgeting
Add a configurable token budget for wake-up rendering so large fact/episode sets stay bounded.

## 5. Better dashboard filters
Add filtering by session, source, tag, and time range for facts and episodes.

## 6. Episode storage naming cleanup
Optionally rename internal `episodic_memory` table/API language to `episodes` for consistency with the UI.
