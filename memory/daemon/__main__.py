"""CLI entry point for the memory daemon.

Run as:
  python3 -m memory.daemon           # runs forever
  python3 -m memory.daemon --once    # one extraction pass, then exit
"""

import argparse

import setproctitle

from memory.daemon import run
from memory.debug import enable_debug

setproctitle.setproctitle("AgenticMemoryDaemon")
enable_debug("daemon")

parser = argparse.ArgumentParser(
    description="Background memory daemon for fact and episodic extraction."
)
parser.add_argument(
    "--once",
    action="store_true",
    help="Force one extraction pass immediately, then exit.",
)
args = parser.parse_args()

run(once=args.once)
