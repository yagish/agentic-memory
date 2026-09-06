import os

# Tests should never write retained runtime logs under ~/.memory.
os.environ.setdefault("MEMORY_DISABLE_FILE_LOGS", "1")
