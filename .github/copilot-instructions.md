# Persistent memory (agentic-memory MCP server)

This workspace runs a local memory MCP server (see `.vscode/mcp.json`) that gives
you long-term memory across sessions. Unlike Claude Code, you have no automatic
before/after hooks, so you must call these tools yourself:

- At the **start** of a new conversation/task, call `memory_recall` with the
  user's first message (reuse the same `session_id` for every turn in this
  conversation). If it returns a non-empty `injection`, treat it as background
  context about the user and prior related work. If it returns a `cache_hit`,
  you already answered this exact question before — reuse that answer.
- At the **end** of a conversation/task (or when the user says goodbye/moves
  on), call `memory_save_session` with the full turn history
  (`role`/`content` pairs) and `agent: "copilot"` so future sessions can
  recall it.
- Use `memory_search` / `memory_hybrid_search` / `memory_semantic_search`
  whenever you need to look something up mid-conversation instead of asking
  the user to repeat themselves.
- Use `memory_save_fact` for durable facts (preferences, decisions) worth
  remembering even outside this specific conversation.
