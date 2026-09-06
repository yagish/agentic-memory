"""Compatibility wrapper for the Claude wake-up hook.

The real Claude integration now lives in integrations/claude/wake_up.py.
"""

from integrations.claude.wake_up import main


if __name__ == "__main__":
    main()
