# Pi integration

This folder contains the pi-specific memory integration.

## Files

- `extension.ts` — pi extension that:
  - searches memory on each prompt
  - answers directly from memory when facts alone are sufficient
  - otherwise injects retrieved memory context into the next agent turn
  - saves the session transcript after each completed turn
- `adapter.py` — Python JSON adapter used by the extension to call the shared memory seams

## Usage

The repo installer writes a global wrapper to:

```text
~/.pi/agent/extensions/agentic-memory.ts
```

So after `./install.sh`, Pi auto-loads agentic-memory in any project.

For manual testing, run pi with the extension:

```bash
pi -e /absolute/path/to/agentic-memory/integrations/pi/extension.ts
```

Or copy/symlink the extension into one of pi's extension directories.

## Environment

- `MEMORY_PYTHON` — optional Python interpreter override for running `adapter.py`

The adapter writes to the same default DB as the Claude integration: `~/.memory/memory.db`.
