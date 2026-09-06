#!/usr/bin/env bash
set -euo pipefail

# Detect the real absolute path to the directory containing this script.
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON3_EXEC="$(python3 -c 'import sys; print(sys.executable)' 2>/dev/null || command -v python3)"

# --uninstall: remove hooks and launchd plists that point to this
# INSTALL_DIR. Data in ~/.memory/ is always preserved.
if [ "${1:-}" = "--uninstall" ]; then
    echo "Uninstalling agentic-memory..."

    # Remove Claude Code hooks
    export INSTALL_DIR PYTHON3_EXEC
    python3 << PYEOF
import json, os
INSTALL_DIR = os.environ["INSTALL_DIR"]
PYTHON3_EXEC = os.environ["PYTHON3_EXEC"]
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
    r1 += remove_hook("Stop", f"{PYTHON3_EXEC} {INSTALL_DIR}/hooks/save_hook.py")
    r1 += remove_hook("Stop", f"python3 {INSTALL_DIR}/integrations/claude/save_hook.py")
    r1 += remove_hook("Stop", f"{PYTHON3_EXEC} {INSTALL_DIR}/integrations/claude/save_hook.py")
    r2 = remove_hook("UserPromptSubmit", f"python3 {INSTALL_DIR}/hooks/wake_up.py")
    r2 += remove_hook("UserPromptSubmit", f"{PYTHON3_EXEC} {INSTALL_DIR}/hooks/wake_up.py")
    r2 += remove_hook("UserPromptSubmit", f"python3 {INSTALL_DIR}/integrations/claude/wake_up.py")
    r2 += remove_hook("UserPromptSubmit", f"{PYTHON3_EXEC} {INSTALL_DIR}/integrations/claude/wake_up.py")
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"  - Removed Stop hook: {r1} entr{'y' if r1==1 else 'ies'}")
    print(f"  - Removed UserPromptSubmit hook: {r2} entr{'y' if r2==1 else 'ies'}")
PYEOF

    # Unload and remove launchd plists
    for LABEL in com.memory.daemon com.memory.ingest com.memory.query; do
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
echo "Python 3 found: $($PYTHON3_EXEC --version)"
echo "Python 3 executable: $PYTHON3_EXEC"

# ── Step 2: Python dependencies ─────────────────────────────────────────────
echo ""
echo "Installing Python dependencies into the same interpreter used by hooks/services..."
"$PYTHON3_EXEC" -m pip install --quiet --user sentence-transformers fastapi uvicorn pydantic psutil setproctitle debugpy
"$PYTHON3_EXEC" - << 'PYEOF'
import importlib
mods = [
    "sentence_transformers",
    "fastapi",
    "uvicorn",
    "pydantic",
    "psutil",
    "setproctitle",
    "debugpy",
]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
if missing:
    raise SystemExit("Missing Python modules after install: " + ", ".join(missing))
print("  + Dependencies installed and import-verified")
PYEOF

# ── Step 3: Ollama model ─────────────────────────────────────────────────────
# Default: qwen2.5:7b. Try the Ollama registry first, then fall back to a
# single-file GGUF from HuggingFace if the registry is flaky or blocked.
# Override via MEMORY_OLLAMA_MODEL / MEMORY_HF_GGUF_URL.
OLLAMA_MODEL="${MEMORY_OLLAMA_MODEL:-qwen2.5:7b}"
HF_GGUF_URL="${MEMORY_HF_GGUF_URL:-https://huggingface.co/lmstudio-community/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf}"
MODELS_DIR="$HOME/.memory/models"
GGUF_FILE="$MODELS_DIR/$(basename "$HF_GGUF_URL")"

echo ""
echo "Checking Ollama model ($OLLAMA_MODEL)..."

OLLAMA_BIN=""
for _candidate in /opt/homebrew/bin/ollama /usr/local/bin/ollama; do
    if [ -f "$_candidate" ]; then
        OLLAMA_BIN="$_candidate"
        break
    fi
done

if [ -z "$OLLAMA_BIN" ]; then
    echo "  ! ollama not found — skipping (install via: brew install ollama)"
elif "$OLLAMA_BIN" list 2>/dev/null | grep -q "^${OLLAMA_MODEL%%:*}"; then
    echo "  = Model $OLLAMA_MODEL already present in Ollama"
else
    # Check if it can be pulled directly from the Ollama registry first.
    # If that fails (e.g. corporate proxy), fall back to HuggingFace download.
    echo "  Trying ollama pull $OLLAMA_MODEL ..."
    if "$OLLAMA_BIN" pull "$OLLAMA_MODEL" 2>/dev/null; then
        echo "  + Model $OLLAMA_MODEL ready (via Ollama registry)"
    else
        echo "  ! Ollama registry pull failed — downloading GGUF from HuggingFace..."
        mkdir -p "$MODELS_DIR"
        if [ -f "$GGUF_FILE" ]; then
            echo "  = GGUF already downloaded: $GGUF_FILE"
        else
            echo "  Downloading $(basename "$HF_GGUF_URL") (~5 GB) ..."
            if curl -L --progress-bar -o "$GGUF_FILE" "$HF_GGUF_URL"; then
                echo "  + Downloaded to $GGUF_FILE"
            else
                rm -f "$GGUF_FILE"
                echo "  ! Download failed — retry manually:"
                echo "    curl -L -o $GGUF_FILE $HF_GGUF_URL"
                echo "    ollama create $OLLAMA_MODEL -f <(echo 'FROM $GGUF_FILE')"
                GGUF_FILE=""
            fi
        fi
        if [ -n "$GGUF_FILE" ] && [ -f "$GGUF_FILE" ]; then
            echo "  Importing into Ollama as $OLLAMA_MODEL ..."
            MODELFILE_TMP="$(mktemp)"
            echo "FROM $GGUF_FILE" > "$MODELFILE_TMP"
            if "$OLLAMA_BIN" create "$OLLAMA_MODEL" -f "$MODELFILE_TMP"; then
                echo "  + Model $OLLAMA_MODEL ready (imported from HuggingFace GGUF)"
            else
                echo "  ! Import failed — retry manually:"
                echo "    ollama create $OLLAMA_MODEL -f $MODELFILE_TMP"
            fi
            rm -f "$MODELFILE_TMP"
        fi
    fi
fi

# ── Step 4: Memory directory ─────────────────────────────────────────────────
mkdir -p "$HOME/.memory"
echo ""
echo "Memory directory: $HOME/.memory/"

# ── Step 5: Identity profile ─────────────────────────────────────────────────
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

# ── Step 6: Claude Code hooks ─────────────────────────────────────────────────
echo ""
echo "Configuring ~/.claude/settings.json hooks..."
export INSTALL_DIR PYTHON3_EXEC
python3 << PYEOF
import json, os
SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
INSTALL_DIR = os.environ["INSTALL_DIR"]
PYTHON3_EXEC = os.environ["PYTHON3_EXEC"]
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
save_cmd = f"{PYTHON3_EXEC} {INSTALL_DIR}/integrations/claude/save_hook.py"
wake_cmd = f"{PYTHON3_EXEC} {INSTALL_DIR}/integrations/claude/wake_up.py"
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

# ── Step 7: launchd — background daemon ─────────────────────────────────────
echo ""
echo "Installing background daemon (auto-extracts facts and episodes)..."

# Capture the user site-packages path so we can inject it into launchd's env.
# launchd runs with a minimal environment and doesn't add --user site-packages
# to sys.path automatically, which breaks imports of fastapi, uvicorn, etc.
PYTHON_USER_SITE="$($PYTHON3_EXEC -c 'import site; print(site.getusersitepackages())')"

# Write the daemon plist directly so paths are always correct, regardless of
# whether the source template has been modified by a previous install run.
cat > "$HOME/Library/LaunchAgents/com.memory.daemon.plist" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.memory.daemon</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON3_EXEC</string>
    <string>$INSTALL_DIR/memory/daemon.py</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key>
    <string>$PYTHON_USER_SITE</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>MEMORY_OLLAMA_MODEL</key>
    <string>$OLLAMA_MODEL</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$HOME/.memory/daemon.log</string>
  <key>StandardErrorPath</key>
  <string>$HOME/.memory/daemon.log</string>
</dict>
</plist>
PLIST_EOF

launchctl unload "$HOME/Library/LaunchAgents/com.memory.daemon.plist" 2>/dev/null || true
launchctl load   "$HOME/Library/LaunchAgents/com.memory.daemon.plist"
echo "  + Daemon loaded (starts on login, logs: ~/.memory/daemon.log)"

# ── Step 9: launchd — recall server ─────────────────────────────────────────
echo ""
echo "Installing recall server (singleton embeddings + ingest endpoint on port 7747)..."

# Same approach: write plist directly with the correct absolute python3 path.
cat > "$HOME/Library/LaunchAgents/com.memory.ingest.plist" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.memory.ingest</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON3_EXEC</string>
    <string>$INSTALL_DIR/memory/ingest_server.py</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key>
    <string>$PYTHON_USER_SITE</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>MEMORY_OLLAMA_MODEL</key>
    <string>$OLLAMA_MODEL</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$HOME/.memory/ingest.log</string>
  <key>StandardErrorPath</key>
  <string>$HOME/.memory/ingest.log</string>
</dict>
</plist>
PLIST_EOF

launchctl unload "$HOME/Library/LaunchAgents/com.memory.ingest.plist" 2>/dev/null || true
launchctl load   "$HOME/Library/LaunchAgents/com.memory.ingest.plist"
echo "  + Recall server loaded on port 7747 (logs: ~/.memory/ingest.log)"

# ── Step 10: launchd — query server ──────────────────────────────────────────
echo ""
echo "Installing query server (serves dashboard + read API on port 7748)..."

cat > "$HOME/Library/LaunchAgents/com.memory.query.plist" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.memory.query</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON3_EXEC</string>
    <string>$INSTALL_DIR/memory/dashboard_server.py</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key>
    <string>$PYTHON_USER_SITE</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>MEMORY_OLLAMA_MODEL</key>
    <string>$OLLAMA_MODEL</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$HOME/.memory/query.log</string>
  <key>StandardErrorPath</key>
  <string>$HOME/.memory/query.log</string>
</dict>
</plist>
PLIST_EOF

launchctl unload "$HOME/Library/LaunchAgents/com.memory.query.plist" 2>/dev/null || true
launchctl load   "$HOME/Library/LaunchAgents/com.memory.query.plist"
echo "  + Query server loaded on port 7748 (logs: ~/.memory/query.log)"

# ── Step 10b: launchd — log rotation ─────────────────────────────────────────
echo ""
echo "Installing log rotation (daily copy-truncate, keep 7 days)..."
chmod +x "$INSTALL_DIR/scripts/rotate-memory-logs.sh"
cat > "$HOME/Library/LaunchAgents/com.memory.logrotate.plist" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.memory.logrotate</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$INSTALL_DIR/scripts/rotate-memory-logs.sh</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>MEMORY_LOG_RETENTION_DAYS</key>
    <string>7</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>3</integer>
    <key>Minute</key>
    <integer>17</integer>
  </dict>
</dict>
</plist>
PLIST_EOF

launchctl unload "$HOME/Library/LaunchAgents/com.memory.logrotate.plist" 2>/dev/null || true
launchctl load   "$HOME/Library/LaunchAgents/com.memory.logrotate.plist"
echo "  + Log rotation loaded (archives in ~/.memory/log-archive, keeps 7 days)"

# ── Step 11: Final summary ────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  agentic-memory installed successfully!"
echo ""
echo "  Memory DB:        $HOME/.memory/memory.db"
echo "  Identity profile: $HOME/.memory/identity.md"
echo "  Activity log:     $HOME/.memory/activity.log"
echo "  Daemon log:       $HOME/.memory/daemon.log"
echo "  Recall log:       $HOME/.memory/ingest.log"
echo "  Query log:        $HOME/.memory/query.log"
echo "  Log archive:      $HOME/.memory/log-archive"
echo ""
echo "  Daemon:           running (auto-restarts on login)"
echo "  Recall server:    port 7747 — singleton embeddings + /ingest + /recall"
echo "  Query server:     port 7748 — dashboard at http://localhost:7748"
echo "  Log rotation:     daily at 03:17, keep 7 days"
echo "  Default model:    $OLLAMA_MODEL"
echo "================================================================"
echo ""
echo "One step remaining:"
echo "  Restart your AI coding assistant for hooks to take effect."
echo ""
echo "  After that — every session is saved automatically."
echo "  Dashboard: http://localhost:7748"
echo "  Other agents POST sessions to: http://localhost:7747/ingest"
echo "  Hooks/adapters recall via:    http://localhost:7747/recall"
echo ""
