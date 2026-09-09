# Testing pi memory integration

## 1. Start memory services

In one terminal:

```bash
python3 memory/ingest_server.py  # recall server
```

In another terminal:

```bash
python3 memory/daemon.py
```

If pi is already running, use `/reload` after the extension change.

---

## 2. Start pi in this repo

```bash
pi
```

This repo auto-loads the extension from:

```text
.pi/extensions/agentic-memory.ts
```

Inside pi, first verify the extension loaded:

```text
/memory-status
```

Expected result:
- pi shows a notification like `agentic-memory loaded (noop)`

---

## 3. Test direct answer from memory

In another shell, add a known fact:

```bash
python3 cli.py add-fact user favorite_language Python --tag identity
```

Then in pi ask:

```text
what is my favorite language?
```

Expected result:
- pi answers directly from memory
- the normal agent turn is skipped
- you see a memory answer in chat

---

## 4. Test session saving

In pi, send a normal message like:

```text
I am debugging a login redirect loop in auth middleware
```

After the turn finishes, in another shell run:

```bash
sqlite3 ~/.memory/memory.db "select session_id, agent, turn_count, updated_at from sessions order by updated_at desc limit 10;"
```

Expected result:
- a recent row exists with `agent = pi`

---

## 5. Force memory extraction

Run one daemon pass:

```bash
python3 memory/daemon.py --once
```

Optional verification:

```bash
sqlite3 ~/.memory/memory.db "select id, content, source, session_id from facts order by updated_at desc limit 10;"
sqlite3 ~/.memory/memory.db "select id, title, abstract, session_id from episodic_memory order by happened_at desc limit 10;"
sqlite3 ~/.memory/memory.db "select id, title, summary, session_id from procedural_memory order by updated_at desc limit 10;"
```

Expected result:
- facts and/or episodes from the recent pi session appear
- if the transcript contained a reusable workflow, a procedural-memory row appears too

---

## 6. Test prompt enhancement

Back in pi, ask:

```text
continue debugging the auth middleware
```

Expected result:
- pi does not do a direct fact-only answer
- pi injects memory context into the turn
- you see a visible memory context message in chat

---

## 7. Optional dashboard check

Start dashboard:

```bash
python3 memory/dashboard_server.py
```

Then inspect:
- sessions from `pi`
- extracted facts
- extracted episodes
- extracted procedural memories
- recall log in the sidebar
- procedural log in the sidebar
- procedural rows in the Procedural tab

---

## Quick pass/fail checklist

### Save works if:
- a `sessions` row appears with `agent = pi`

### Direct-answer works if:
- a fact question like `what is my favorite language?` is answered immediately from memory

### Prompt-enhancement works if:
- a related follow-up prompt shows memory context and uses it in the response

### Procedural memory works if:
- a reusable workflow from a recent session appears in `procedural_memory`
- a prompt like `how do i deploy the auth service?` injects the stored procedure into recall context

### Dashboard works if:
- the Procedural overview tile is clickable
- the Procedural tab lists stored rows and opens a detail drawer
- Recall and Procedural logs both appear in the sidebar/log viewer
