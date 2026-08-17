#!/usr/bin/env bash
set -euo pipefail

# Detect the real absolute path to the directory containing this script.
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --uninstall: remove hooks and MCP entry that point to this INSTALL_DIR.
# Data in ~/.memory/ is preserved.
if [ "${1:-}" = "--uninstall" ]; then
    echo "Uninstalling memory hooks and MCP server..."
    export INSTALL_DIR
    python3 << PYEOF
import json, os

INSTALL_DIR = os.environ["INSTALL_DIR"]
SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")

if os.path.exists(SETTINGS_PATH):
    with open(SETTINGS_PATH) as f:
        settings = json.load(f)

    hooks = settings.get("hooks", {})

    def remove_hook(event_name, cmd_prefix):
        entries = hooks.get(event_name, [])
        new_entries = []
        removed = 0
        for entry in entries:
            new_hooks = [h for h in entry.get("hooks", []) if not h.get("command", "").startswith(cmd_prefix)]
            if new_hooks:
                entry["hooks"] = new_hooks
                new_entries.append(entry)
            elif entry.get("hooks") and not new_hooks:
                removed += 1  # whole entry removed
        hooks[event_name] = new_entries
        return removed

    r1 = remove_hook("Stop", f"python3 {INSTALL_DIR}/hooks/save_hook.py")
    r2 = remove_hook("UserPromptSubmit", f"python3 {INSTALL_DIR}/hooks/wake_up.py")

    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)

    print(f"  - Removed Stop hook: {r1} entr{'y' if r1==1 else 'ies'}")
    print(f"  - Removed UserPromptSubmit hook: {r2} entr{'y' if r2==1 else 'ies'}")
PYEOF

    python3 << PYEOF
import json, os

INSTALL_DIR = os.environ["INSTALL_DIR"]
MCP_PATH = os.path.expanduser("~/.claude/mcp.json")

if os.path.exists(MCP_PATH):
    with open(MCP_PATH) as f:
        mcp = json.load(f)

    servers = mcp.get("mcpServers", {})
    if "memory" in servers and INSTALL_DIR in str(servers["memory"].get("args", [])):
        del servers["memory"]
        with open(MCP_PATH, "w") as f:
            json.dump(mcp, f, indent=2)
        print("  - Removed MCP server 'memory'")
    else:
        print("  = MCP server 'memory' not found or points elsewhere (no change)")
PYEOF

    echo ""
    echo "Uninstall complete. ~/.memory/ data preserved."
    echo "Restart Claude Code to apply changes."
    exit 0
fi

echo "Installing agentic-memory from: $INSTALL_DIR"
echo ""

# Step 1: Check Python 3
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is required but not found in PATH." >&2
    exit 1
fi

# Check Python version is >= 3.8
if ! python3 -c "import sys; assert sys.version_info >= (3,8), f'Python 3.8+ required, got {sys.version}'" 2>/dev/null; then
    echo "ERROR: Python 3.8 or higher is required." >&2
    python3 --version >&2
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Step 2: Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --quiet --user mcp fastmcp sentence-transformers fastapi uvicorn pydantic psutil
echo "  + Dependencies installed"

# Step 3: Create ~/.memory/ directory
mkdir -p "$HOME/.memory"
echo ""
echo "Memory directory: $HOME/.memory/"

# Step 4: Write starter ~/.memory/identity.md (only if it doesn't already exist)
if [ ! -f "$HOME/.memory/identity.md" ]; then
    cat > "$HOME/.memory/identity.md" << 'EOF'
# Identity

Name: (your name)
Role: (your role, e.g. "Senior software engineer at Acme Corp")

## About me
(A few sentences about your background, expertise, and working style)

## Preferences
- (e.g. "I prefer concise explanations over long prose")
- (e.g. "Always show the file path when referencing code")
EOF
    echo "Created ~/.memory/identity.md — edit it to personalise your memory profile."
fi

# Step 5: Patch ~/.claude/settings.json — add hooks (idempotent)
echo ""
echo "Configuring ~/.claude/settings.json hooks..."
export INSTALL_DIR
python3 << PYEOF
import json, os, sys

SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
INSTALL_DIR = os.environ["INSTALL_DIR"]

# Load existing settings or start fresh
if os.path.exists(SETTINGS_PATH):
    with open(SETTINGS_PATH) as f:
        settings = json.load(f)
else:
    settings = {}

hooks = settings.setdefault("hooks", {})

# Helper: add a hook entry if not already present
def add_hook(event_name, command):
    entries = hooks.setdefault(event_name, [])
    # Check if this exact command is already registered
    for entry in entries:
        for h in entry.get("hooks", []):
            if h.get("command") == command:
                return False  # already present
    # Not found — add a new entry
    entries.append({"matcher": "", "hooks": [{"type": "command", "command": command}]})
    return True

save_cmd = f"python3 {INSTALL_DIR}/hooks/save_hook.py"
wake_cmd = f"python3 {INSTALL_DIR}/hooks/wake_up.py"

added_stop = add_hook("Stop", save_cmd)
added_wake = add_hook("UserPromptSubmit", wake_cmd)

with open(SETTINGS_PATH, "w") as f:
    json.dump(settings, f, indent=2)

if added_stop:
    print(f"  + Added Stop hook: {save_cmd}")
else:
    print(f"  = Stop hook already present (no change)")
if added_wake:
    print(f"  + Added UserPromptSubmit hook: {wake_cmd}")
else:
    print(f"  = UserPromptSubmit hook already present (no change)")
PYEOF

# Step 6: Write (or merge) ~/.claude/mcp.json — add the memory MCP server entry
echo ""
echo "Configuring ~/.claude/mcp.json..."
python3 << PYEOF
import json, os

MCP_PATH = os.path.expanduser("~/.claude/mcp.json")
INSTALL_DIR = os.environ["INSTALL_DIR"]

if os.path.exists(MCP_PATH):
    with open(MCP_PATH) as f:
        mcp = json.load(f)
else:
    mcp = {}

servers = mcp.setdefault("mcpServers", {})
server_path = f"{INSTALL_DIR}/memory/mcp_server.py"

if "memory" in servers and servers["memory"].get("args") == [server_path]:
    print("  = MCP server 'memory' already registered (no change)")
else:
    servers["memory"] = {"command": "python3", "args": [server_path]}
    with open(MCP_PATH, "w") as f:
        json.dump(mcp, f, indent=2)
    print(f"  + Registered MCP server 'memory': {server_path}")
PYEOF

# Step 7: Update com.memory.daemon.plist with real paths
echo ""
echo "Updating launchd plist..."
sed -i.bak \
    -e "s|PROJECT_PATH|$INSTALL_DIR|g" \
    -e "s|HOME_PATH|$HOME|g" \
    "$INSTALL_DIR/com.memory.daemon.plist"
rm -f "$INSTALL_DIR/com.memory.daemon.plist.bak"
echo "  + Updated com.memory.daemon.plist with real paths"
echo "  (To auto-start daemon at login: cp $INSTALL_DIR/com.memory.daemon.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.memory.daemon.plist)"

# Step 8: Print success summary
echo ""
echo "Installation complete!"
echo "  Memory DB will be stored at: $HOME/.memory/memory.db"
echo "  Identity profile:            $HOME/.memory/identity.md"
echo "  Activity log:                $HOME/.memory/activity.log"
echo ""
echo "Next steps:"
echo "  1. Edit ~/.memory/identity.md with your profile"
echo "  2. Restart Claude Code for hooks and MCP server to take effect"
echo "  3. Optional: python3 $INSTALL_DIR/cli.py daemon start"
echo "  4. Optional: python3 $INSTALL_DIR/cli.py ingest-server start"
