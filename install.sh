#!/usr/bin/env bash
set -euo pipefail

# Detect the real absolute path to the directory containing this script.
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --uninstall: remove hooks, MCP entry, and launchd plists that point to this
# INSTALL_DIR.  Data in ~/.memory/ is always preserved.
if [ "${1:-}" = "--uninstall" ]; then
    echo "Uninstalling agentic-memory..."

    # Remove Claude Code hooks
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
                removed += 1
        hooks[event_name] = new_entries
        return removed
    r1 = remove_hook("Stop", f"python3 {INSTALL_DIR}/hooks/save_hook.py")
    r2 = remove_hook("UserPromptSubmit", f"python3 {INSTALL_DIR}/hooks/wake_up.py")
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"  - Removed Stop hook: {r1} entr{'y' if r1==1 else 'ies'}")
    print(f"  - Removed UserPromptSubmit hook: {r2} entr{'y' if r2==1 else 'ies'}")
PYEOF

    # Remove MCP entry
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

    # Unload and remove launchd plists
    for LABEL in com.memory.daemon com.memory.ingest; do
        PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
        if [ -f "$PLIST" ]; then
            launchctl unload "$PLIST" 2>/dev/null || true
            rm -f "$PLIST"
            echo "  - Unloaded and removed $PLIST"
        fi
    done

    echo ""
    echo "Uninstall complete. ~/.memory/ data preserved."
    echo "Restart Claude Code to apply changes."
    exit 0
fi

echo "Installing agentic-memory from: $INSTALL_DIR"
echo ""

# ── Step 1: Python version check ────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is required but not found in PATH." >&2
    exit 1
fi
if ! python3 -c "import sys; assert sys.version_info >= (3,8), f'Python 3.8+ required, got {sys.version}'" 2>/dev/null; then
    echo "ERROR: Python 3.8 or higher is required." >&2
    python3 --version >&2
    exit 1
fi
echo "Python 3 found: $(python3 --version)"

# ── Step 2: Python dependencies ─────────────────────────────────────────────
echo ""
echo "Installing Python dependencies..."
pip3 install --quiet --user mcp fastmcp sentence-transformers fastapi uvicorn pydantic psutil
echo "  + Dependencies installed"

# ── Step 3: Memory directory ─────────────────────────────────────────────────
mkdir -p "$HOME/.memory"
echo ""
echo "Memory directory: $HOME/.memory/"

# ── Step 4: Identity profile ─────────────────────────────────────────────────
if [ ! -f "$HOME/.memory/identity.md" ]; then
    echo ""
    echo "Setting up your identity profile..."
    echo "(Your AI assistant will use this to remember who you are across sessions)"
    echo "(Press Enter to skip any question)"
    echo ""

    read -r -p "  Your name: " _NAME
    read -r -p "  Your role  (e.g. 'Senior engineer at Acme'): " _ROLE
    read -r -p "  Primary languages / tech  (e.g. 'Python, TypeScript, AWS'): " _TECH
    read -r -p "  Experience level  (e.g. 'Beginner', '5 years Python', 'Senior'): " _EXP
    read -r -p "  Communication style  (e.g. 'concise', 'detailed with examples'): " _STYLE

    {
        echo "# Identity"
        echo ""
        [ -n "$_NAME" ]  && echo "Name: $_NAME"
        [ -n "$_ROLE" ]  && echo "Role: $_ROLE"
        echo ""
        echo "## About me"
        [ -n "$_TECH" ]  && echo "Primary tech stack: $_TECH"
        [ -n "$_EXP" ]   && echo "Experience: $_EXP"
        echo ""
        echo "## Preferences"
        [ -n "$_STYLE" ] && echo "- Communication style: $_STYLE"
        echo ""
        echo "<!-- Your AI assistant will add more facts here as it learns them during conversations -->"
    } > "$HOME/.memory/identity.md"

    echo ""
    echo "  + Created ~/.memory/identity.md"
fi

# ── Step 5: Claude Code hooks ─────────────────────────────────────────────────
echo ""
echo "Configuring ~/.claude/settings.json hooks..."
export INSTALL_DIR
python3 << PYEOF
import json, os
SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
INSTALL_DIR = os.environ["INSTALL_DIR"]
if os.path.exists(SETTINGS_PATH):
    with open(SETTINGS_PATH) as f:
        settings = json.load(f)
else:
    settings = {}
hooks = settings.setdefault("hooks", {})
def add_hook(event_name, command):
    entries = hooks.setdefault(event_name, [])
    for entry in entries:
        for h in entry.get("hooks", []):
            if h.get("command") == command:
                return False
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

# ── Step 6: MCP server ───────────────────────────────────────────────────────
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

# ── Step 7: launchd — background daemon ─────────────────────────────────────
echo ""
echo "Installing background daemon (auto-extracts facts, builds insights)..."

# Substitute real paths into the plist (idempotent — sed replaces placeholders
# only when they still contain the placeholder text).
sed -i.bak \
    -e "s|PROJECT_PATH|$INSTALL_DIR|g" \
    -e "s|HOME_PATH|$HOME|g" \
    "$INSTALL_DIR/com.memory.daemon.plist"
rm -f "$INSTALL_DIR/com.memory.daemon.plist.bak"

cp "$INSTALL_DIR/com.memory.daemon.plist" "$HOME/Library/LaunchAgents/"
launchctl unload "$HOME/Library/LaunchAgents/com.memory.daemon.plist" 2>/dev/null || true
launchctl load   "$HOME/Library/LaunchAgents/com.memory.daemon.plist"
echo "  + Daemon loaded (starts on login, logs: ~/.memory/daemon.log)"

# ── Step 8: launchd — ingest server ─────────────────────────────────────────
echo ""
echo "Installing ingest server (receives sessions from other agents)..."

sed -i.bak \
    -e "s|PROJECT_PATH|$INSTALL_DIR|g" \
    -e "s|HOME_PATH|$HOME|g" \
    "$INSTALL_DIR/com.memory.ingest.plist"
rm -f "$INSTALL_DIR/com.memory.ingest.plist.bak"

cp "$INSTALL_DIR/com.memory.ingest.plist" "$HOME/Library/LaunchAgents/"
launchctl unload "$HOME/Library/LaunchAgents/com.memory.ingest.plist" 2>/dev/null || true
launchctl load   "$HOME/Library/LaunchAgents/com.memory.ingest.plist"
echo "  + Ingest server loaded on port 7747 (logs: ~/.memory/ingest.log)"

# ── Step 9: Final summary ────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  agentic-memory installed successfully!"
echo ""
echo "  Memory DB:        $HOME/.memory/memory.db"
echo "  Identity profile: $HOME/.memory/identity.md"
echo "  Activity log:     $HOME/.memory/activity.log"
echo "  Daemon log:       $HOME/.memory/daemon.log"
echo "  Ingest log:       $HOME/.memory/ingest.log"
echo ""
echo "  Daemon:           running (auto-restarts on login)"
echo "  Ingest server:    running on port 7747 (auto-restarts on login)"
echo "================================================================"
echo ""
echo "One step remaining:"
echo "  Restart your AI coding assistant for hooks and MCP server to take effect."
echo ""
echo "  After that — every session is saved automatically."
echo "  Other agents can POST sessions to the ingest server at localhost:7747."
echo ""
