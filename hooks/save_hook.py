"""Compatibility wrapper for the Claude save hook.

The real Claude integration now lives in integrations/claude/save_hook.py.
"""

from integrations.claude.save_hook import main


if __name__ == "__main__":
    main()
