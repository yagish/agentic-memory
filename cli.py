# cli.py — command-line tool for inspecting the memory database.
#
# Usage:
#   python3 cli.py status              # session count, turn count, date range
#   python3 cli.py search "query"      # FTS5 keyword search
#   python3 cli.py semantic "query"    # semantic (vector) search
#   python3 cli.py get-session <id>    # print full transcript
#   python3 cli.py tail [N]            # last N sessions (default 10)
#   python3 cli.py dashboard           # generate + open dashboard.html

import argparse      # parses command-line arguments (the words after "python3 cli.py")
import json          # for pretty-printing dicts
import os
import subprocess    # for launching the daemon as a background process
import sys
import webbrowser    # opens the dashboard HTML in the default browser

# Add the project root to the module search path so we can import memory.db.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re

from memory.db import init_db, search, semantic_search, insert_fact, delete_fact
from memory.consolidation import consolidate_old_sessions, prune_old_transcripts


# Where the database lives — must match the hook and MCP server.
DB_PATH = os.path.expanduser("~/.memory/memory.db")

# Where the generated dashboard HTML is written.
DASHBOARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


def get_conn():
    """Open the memory database. Exits with a clear message if it doesn't exist yet."""
    if not os.path.exists(DB_PATH):
        print("No memory database found at", DB_PATH)
        print("Start a Claude Code session to create it.")
        sys.exit(1)
    return init_db(DB_PATH)


# ---------------------------------------------------------------------------
# Command: status
# ---------------------------------------------------------------------------

def _get_token_economics(conn):
    """
    Query the retrievals table for injection and retrieval token totals.

    Returns a dict with:
      injection_total   — sum of est. tokens injected by wake_up_injection rows
      retrieval_total   — sum of est. tokens returned by MCP tool calls
      coverage_ratio    — retrieval_total / injection_total, or None if no injections
      injection_count   — number of wake-up injection events recorded
      retrieval_count   — number of distinct MCP tool queries recorded
    """
    # Total estimated tokens injected via wake-up (all 'wake_up_injection' rows).
    row = conn.execute(
        "SELECT COALESCE(SUM(result_size), 0) FROM retrievals WHERE tool = 'wake_up_injection'"
    ).fetchone()
    injection_total = row[0]

    # Total estimated tokens returned by MCP tool calls (everything that is NOT
    # a wake-up injection).
    row = conn.execute(
        "SELECT COALESCE(SUM(result_size), 0) FROM retrievals WHERE tool != 'wake_up_injection'"
    ).fetchone()
    retrieval_total = row[0]

    # Coverage ratio: how much context was retrieved per token injected.
    # A ratio >= 1.0 means retrievals returned at least as much context as was injected.
    if injection_total > 0:
        coverage_ratio = retrieval_total / injection_total
    else:
        # No injections recorded yet — ratio is undefined.
        coverage_ratio = None

    # Count how many wake-up injection events are recorded (proxy for sessions seen).
    row2 = conn.execute(
        "SELECT COUNT(*) FROM retrievals WHERE tool = 'wake_up_injection'"
    ).fetchone()
    injection_count = row2[0]

    # Count how many distinct MCP tool queries have been logged.
    row3 = conn.execute(
        "SELECT COUNT(DISTINCT query) FROM retrievals WHERE tool != 'wake_up_injection'"
    ).fetchone()
    retrieval_count = row3[0]

    return {
        "injection_total":  injection_total,
        "retrieval_total":  retrieval_total,
        "coverage_ratio":   coverage_ratio,
        "injection_count":  injection_count,
        "retrieval_count":  retrieval_count,
    }


def cmd_status(_args):
    """Print a summary of everything stored in the database."""
    conn = get_conn()

    total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    total_turns    = conn.execute("SELECT COALESCE(SUM(turn_count), 0) FROM sessions").fetchone()[0]
    date_row       = conn.execute("SELECT MIN(updated_at), MAX(updated_at) FROM sessions").fetchone()
    total_vecs     = conn.execute("SELECT COUNT(*) FROM session_vecs").fetchone()[0]
    total_retrievals = conn.execute("SELECT COUNT(*) FROM retrievals").fetchone()[0]

    # Estimate token count: average English word is ~4-5 characters, roughly 4 chars per token.
    # We sum the length of all stored transcripts for a rough estimate.
    char_count = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(transcript)), 0) FROM sessions"
    ).fetchone()[0]
    estimated_tokens = char_count // 4

    # Gather token economics data before closing the connection.
    # We wrap in try/except so a missing or empty table never crashes status.
    try:
        econ = _get_token_economics(conn)
    except Exception:
        econ = None

    conn.close()

    print("=== Memory Status ===")
    print(f"  Sessions stored   : {total_sessions}")
    print(f"  Turns stored      : {total_turns}")
    print(f"  Embeddings stored : {total_vecs}")
    print(f"  Tokens (estimated): {estimated_tokens:,}")
    print(f"  Retrievals logged : {total_retrievals}")
    print(f"  Oldest session    : {date_row[0] or 'none'}")
    print(f"  Newest session    : {date_row[1] or 'none'}")

    # Token economics section — shows injection overhead vs. retrieval yield.
    if econ is not None:
        print("\n=== Token Economics ===")
        print(f"  Wake-up injections:    {econ['injection_count']} sessions")
        print(f"  Est. tokens injected:  {econ['injection_total']:,}")
        print(f"  Est. tokens retrieved: {econ['retrieval_total']:,}")
        if econ['coverage_ratio'] is not None:
            # Format as a decimal ratio and a percentage (e.g. 1.25x  125%).
            ratio_pct = econ['coverage_ratio'] * 100
            print(f"  Coverage ratio:        {econ['coverage_ratio']:.2f}x ({ratio_pct:.0f}%)")
            # Plain-English sentence so readers do not have to interpret the number.
            if econ['coverage_ratio'] >= 1.0:
                interp = "retrievals returned more context than was injected"
            elif econ['coverage_ratio'] >= 0.5:
                interp = "retrievals cover about half of injection overhead"
            else:
                interp = "retrievals cover less than half of injection overhead"
            print(f"  Interpretation:        {interp}")
        else:
            print(f"  Coverage ratio:        N/A (no injections recorded yet)")


# ---------------------------------------------------------------------------
# Command: search
# ---------------------------------------------------------------------------

def cmd_search(args):
    """Full-text keyword search across all transcripts."""
    conn = get_conn()
    results = search(conn, args.query, limit=args.limit)
    conn.close()

    if not results:
        print(f"No results for: {args.query!r}")
        return

    print(f"Found {len(results)} result(s) for: {args.query!r}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['session_id']}")
        print(f"     Updated : {r['updated_at']}")
        # The snippet has matching words wrapped in [square brackets].
        print(f"     Excerpt : {r['snippet']}")
        print()


# ---------------------------------------------------------------------------
# Command: semantic
# ---------------------------------------------------------------------------

def cmd_semantic(args):
    """Semantic (vector) search — finds sessions by meaning, not exact words."""
    conn = get_conn()
    results = semantic_search(conn, args.query, limit=args.limit)
    conn.close()

    if not results:
        print(f"No results for: {args.query!r}")
        return

    print(f"Found {len(results)} result(s) for: {args.query!r}\n")
    for i, r in enumerate(results, 1):
        # Distance is cosine distance: 0.0 = identical, 2.0 = opposite.
        # We convert to a similarity percentage for readability.
        similarity = round((1 - r['distance'] / 2) * 100, 1)
        print(f"[{i}] {r['session_id']}")
        print(f"     Updated    : {r['updated_at']}")
        print(f"     Similarity : {similarity}%  (distance={r['distance']:.4f})")
        print()


# ---------------------------------------------------------------------------
# Command: get-session
# ---------------------------------------------------------------------------

def cmd_get_session(args):
    """Print the full verbatim transcript for one session."""
    conn = get_conn()

    row = conn.execute(
        "SELECT session_id, agent, started_at, updated_at, turn_count, transcript "
        "FROM sessions WHERE session_id = ?",
        (args.session_id,),
    ).fetchone()
    conn.close()

    if row is None:
        print(f"Session not found: {args.session_id}")
        sys.exit(1)

    print(f"=== Session: {row['session_id']} ===")
    print(f"Agent      : {row['agent']}")
    print(f"Started    : {row['started_at']}")
    print(f"Updated    : {row['updated_at']}")
    print(f"Turns      : {row['turn_count']}")
    print()

    # json.loads converts the stored JSON string back into a Python list of dicts.
    turns = json.loads(row["transcript"])
    for turn in turns:
        role = turn.get("role", "?").upper()
        content = turn.get("content", "")
        print(f"[{role}]")
        print(content)
        print()


# ---------------------------------------------------------------------------
# Command: tail
# ---------------------------------------------------------------------------

def cmd_tail(args):
    """Print a summary of the last N sessions."""
    conn = get_conn()

    rows = conn.execute(
        """
        SELECT session_id, updated_at, turn_count, transcript
        FROM sessions
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (args.n,),
    ).fetchall()
    conn.close()

    if not rows:
        print("No sessions stored yet.")
        return

    print(f"=== Last {len(rows)} session(s) ===\n")
    for i, row in enumerate(rows, 1):
        # Extract the first user message as a preview.
        try:
            turns = json.loads(row["transcript"])
            first_user = next((t["content"] for t in turns if t.get("role") == "user"), "")
            preview = first_user[:80].replace("\n", " ")
        except Exception:
            preview = ""

        print(f"[{i}] {row['session_id']}")
        print(f"     Updated : {row['updated_at']}")
        print(f"     Turns   : {row['turn_count']}")
        print(f"     Preview : {preview}")
        print()


# ---------------------------------------------------------------------------
# Command: dashboard
# ---------------------------------------------------------------------------

def cmd_dashboard(_args):
    """Generate dashboard.html and open it in the browser."""
    conn = get_conn()

    # --- Gather data ---

    total_sessions   = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    total_turns      = conn.execute("SELECT COALESCE(SUM(turn_count), 0) FROM sessions").fetchone()[0]
    total_retrievals = conn.execute("SELECT COUNT(*) FROM retrievals").fetchone()[0]
    char_count       = conn.execute("SELECT COALESCE(SUM(LENGTH(transcript)), 0) FROM sessions").fetchone()[0]
    estimated_tokens = char_count // 4

    # Sessions per day — for the timeline chart.
    # strftime('%Y-%m-%d', updated_at) extracts just the date portion from the timestamp.
    daily_rows = conn.execute(
        """
        SELECT strftime('%Y-%m-%d', updated_at) AS day, COUNT(*) AS count
        FROM sessions
        GROUP BY day
        ORDER BY day
        """
    ).fetchall()

    # Recent sessions list.
    recent_rows = conn.execute(
        """
        SELECT session_id, updated_at, turn_count
        FROM sessions
        ORDER BY updated_at DESC
        LIMIT 10
        """
    ).fetchall()

    # Retrieval breakdown by tool.
    retrieval_rows = conn.execute(
        """
        SELECT tool, COUNT(*) AS count
        FROM retrievals
        GROUP BY tool
        ORDER BY count DESC
        """
    ).fetchall()

    # Recent retrievals.
    recent_retrievals = conn.execute(
        """
        SELECT tool, query, result_size, called_at
        FROM retrievals
        ORDER BY called_at DESC
        LIMIT 10
        """
    ).fetchall()

    # Token economics — gathered here while the connection is still open.
    # We wrap in try/except so a missing or empty retrievals table never
    # crashes the dashboard generation.
    try:
        dash_econ = _get_token_economics(conn)
    except Exception:
        dash_econ = None

    conn.close()

    # --- Build token economics strings for the dashboard card ---
    # Convert the economics dict to display-ready strings so the HTML template
    # stays clean (no conditionals inside the f-string).
    if dash_econ is not None:
        econ_injection_total  = f"{dash_econ['injection_total']:,}"
        econ_retrieval_total  = f"{dash_econ['retrieval_total']:,}"
        if dash_econ['coverage_ratio'] is not None:
            econ_coverage_ratio_str = f"{dash_econ['coverage_ratio']:.2f}x"
        else:
            # No injections have been logged yet.
            econ_coverage_ratio_str = "N/A"
    else:
        # Economics data unavailable — show dashes so the card still renders.
        econ_injection_total     = "—"
        econ_retrieval_total     = "—"
        econ_coverage_ratio_str  = "—"

    # --- Build chart data as JSON strings for embedding in HTML ---

    # Labels = list of date strings; values = list of counts.
    chart_labels = json.dumps([r["day"] for r in daily_rows])
    chart_values = json.dumps([r["count"] for r in daily_rows])

    # --- Build HTML table rows ---

    # Sessions table
    session_rows_html = ""
    for r in recent_rows:
        session_rows_html += (
            f"<tr>"
            f"<td title='{r['session_id']}'>{r['session_id'][:16]}…</td>"
            f"<td>{r['updated_at'][:19]}</td>"
            f"<td>{r['turn_count']}</td>"
            f"</tr>\n"
        )

    # Retrieval breakdown table
    retrieval_breakdown_html = ""
    for r in retrieval_rows:
        retrieval_breakdown_html += (
            f"<tr><td>{r['tool']}</td><td>{r['count']}</td></tr>\n"
        )

    # Recent retrievals table
    recent_retrieval_rows_html = ""
    for r in recent_retrievals:
        query_display = (r["query"] or "—")[:40]
        retrieval_rows_html = (
            f"<tr>"
            f"<td>{r['tool']}</td>"
            f"<td>{query_display}</td>"
            f"<td>{r['result_size']:,} B</td>"
            f"<td>{r['called_at'][:19]}</td>"
            f"</tr>\n"
        )
        recent_retrieval_rows_html += retrieval_rows_html

    # --- Write the HTML file ---

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Memory Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #0f1117; color: #e2e8f0; padding: 2rem; }}
  h1   {{ font-size: 1.5rem; margin-bottom: 0.25rem; color: #f8fafc; }}
  .sub {{ color: #64748b; font-size: 0.875rem; margin-bottom: 2rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem; margin-bottom: 2rem; }}
  .stat  {{ background: #1e2532; border-radius: 10px; padding: 1.25rem; }}
  .stat .value {{ font-size: 2rem; font-weight: 700; color: #7c3aed; }}
  .stat .label {{ font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem; }}
  .card  {{ background: #1e2532; border-radius: 10px; padding: 1.25rem;
            margin-bottom: 1.5rem; }}
  .card h2 {{ font-size: 1rem; margin-bottom: 1rem; color: #cbd5e1; }}
  table  {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th     {{ text-align: left; color: #64748b; padding: 0.4rem 0.6rem;
            border-bottom: 1px solid #2d3748; }}
  td     {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid #1a2030;
            color: #cbd5e1; font-family: monospace; }}
  tr:last-child td {{ border-bottom: none; }}
  .chart-wrap {{ position: relative; height: 220px; }}
</style>
</head>
<body>

<h1>Memory Dashboard</h1>
<p class="sub">Generated from ~/.memory/memory.db</p>

<div class="stats">
  <div class="stat">
    <div class="value">{total_sessions}</div>
    <div class="label">Sessions stored</div>
  </div>
  <div class="stat">
    <div class="value">{total_turns}</div>
    <div class="label">Turns stored</div>
  </div>
  <div class="stat">
    <div class="value">{estimated_tokens:,}</div>
    <div class="label">Tokens (estimated)</div>
  </div>
  <div class="stat">
    <div class="value">{total_retrievals}</div>
    <div class="label">Retrievals logged</div>
  </div>
</div>

<div class="card">
  <h2>Sessions over time</h2>
  <div class="chart-wrap">
    <canvas id="sessionChart"></canvas>
  </div>
</div>

<div class="card">
  <h2>Recent sessions</h2>
  <table>
    <thead><tr><th>Session ID</th><th>Updated</th><th>Turns</th></tr></thead>
    <tbody>{session_rows_html}</tbody>
  </table>
</div>

<div class="card">
  <h2>Retrieval activity by tool</h2>
  <table>
    <thead><tr><th>Tool</th><th>Calls</th></tr></thead>
    <tbody>{retrieval_breakdown_html or '<tr><td colspan="2">No retrievals yet</td></tr>'}</tbody>
  </table>
</div>

<div class="card">
  <h2>Recent retrievals</h2>
  <table>
    <thead><tr><th>Tool</th><th>Query</th><th>Result size</th><th>Called at</th></tr></thead>
    <tbody>{recent_retrieval_rows_html or '<tr><td colspan="4">No retrievals yet</td></tr>'}</tbody>
  </table>
</div>

<div class="card">
  <h2>Token Economics</h2>
  <div class="stats">
    <div class="stat">
      <div class="value">{econ_injection_total}</div>
      <div class="label">Est. tokens injected</div>
    </div>
    <div class="stat">
      <div class="value">{econ_retrieval_total}</div>
      <div class="label">Est. tokens retrieved</div>
    </div>
    <div class="stat">
      <div class="value">{econ_coverage_ratio_str}</div>
      <div class="label">Coverage ratio</div>
    </div>
  </div>
  <p style="font-size:0.8rem;color:#64748b;margin-top:0.75rem;">
    Coverage ratio &ge; 1.0 means retrievals returned more context than was injected.
    It does not measure tokens saved vs. a baseline &mdash; that would require a control group.
  </p>
</div>

<script>
const ctx = document.getElementById('sessionChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {chart_labels},
    datasets: [{{
      label: 'Sessions',
      data: {chart_values},
      borderColor: '#7c3aed',
      backgroundColor: 'rgba(124, 58, 237, 0.15)',
      borderWidth: 2,
      pointRadius: 3,
      pointBackgroundColor: '#7c3aed',
      fill: true,
      tension: 0.3,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#2d3748' }} }},
      y: {{ ticks: {{ color: '#64748b', stepSize: 1 }}, grid: {{ color: '#2d3748' }}, beginAtZero: true }}
    }}
  }}
}});
</script>

</body>
</html>
"""

    with open(DASHBOARD_PATH, "w") as f:
        f.write(html)

    print(f"Dashboard written to: {DASHBOARD_PATH}")

    # webbrowser.open() opens the file in whatever browser the user has set as default.
    # "file://" prefix is required for local HTML files.
    webbrowser.open(f"file://{DASHBOARD_PATH}")


# ---------------------------------------------------------------------------
# Command: logs
# ---------------------------------------------------------------------------

# Paths for the two structured log files written by memory.logger.
ACTIVITY_LOG_PATH = os.path.expanduser("~/.memory/activity.log")
ERROR_LOG_PATH    = os.path.expanduser("~/.memory/error.log")


def cmd_logs(args):
    """Print the last N lines of activity.log or error.log."""
    # Choose which log file to read based on the --errors flag.
    path = ERROR_LOG_PATH if args.errors else ACTIVITY_LOG_PATH
    label = "error" if args.errors else "activity"

    if not os.path.exists(path):
        print(f"No {label} log found at {path}")
        print("The log is created automatically once the memory system runs.")
        return

    # Read all lines and slice to the last N.
    with open(path, "r") as f:
        lines = f.readlines()

    tail = lines[-args.tail:]  # last N lines

    print(f"=== {label}.log (last {len(tail)} lines) ===\n")
    print("".join(tail), end="")


# ---------------------------------------------------------------------------
# Command: consolidate (Phase 12)
# ---------------------------------------------------------------------------

def cmd_consolidate(args):
    """Summarise sessions older than --days days using local ollama."""
    # Open the database (exits with a clear message if it doesn't exist).
    conn = get_conn()

    # Call the consolidation function which does the heavy lifting.
    # dry_run=True means nothing is written — just printed.
    result = consolidate_old_sessions(conn, days_threshold=args.days, dry_run=args.dry_run)
    conn.close()

    # Print a human-readable summary of what was done.
    # result["summarised"] is the count of sessions that got a new summary.
    # result["skipped"] covers dry-run entries, errors, and empty transcripts.
    print(f"Summarised {result['summarised']} sessions. "
          f"Skipped {result['skipped']} (already done, empty, or error).")


# ---------------------------------------------------------------------------
# Command: prune (Phase 12)
# ---------------------------------------------------------------------------

def cmd_prune(args):
    """Null out raw transcripts for summarised sessions older than --days days."""
    # Open the database (exits with a clear message if it doesn't exist).
    conn = get_conn()

    # Call the prune function. Only sessions with an existing summary are pruned.
    result = prune_old_transcripts(conn, days_threshold=args.days, dry_run=args.dry_run)
    conn.close()

    # Print how many transcripts were cleared.
    print(f"Pruned transcripts for {result['pruned']} sessions.")


# ---------------------------------------------------------------------------
# Command: daemon (Phase 13)
# ---------------------------------------------------------------------------

# Paths for the daemon PID file and log file.
_DAEMON_PID_PATH = os.path.expanduser("~/.memory/daemon.pid")
_DAEMON_LOG_CLI_PATH = os.path.expanduser("~/.memory/daemon.log")

# Absolute path to the daemon script — constructed from this file's location.
_DAEMON_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "memory", "daemon.py"
)

# ---------------------------------------------------------------------------
# Command: ingest-server (Phase 16)
# ---------------------------------------------------------------------------

# Paths for the ingest server PID file and log file.
_INGEST_PID_PATH = os.path.expanduser("~/.memory/ingest.pid")
_INGEST_LOG_PATH  = os.path.expanduser("~/.memory/ingest.log")

# Absolute path to the ingest server script — constructed from this file's location.
_INGEST_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "memory", "ingest_server.py"
)

# Port the ingest server listens on — must match ingest_server.py default.
_INGEST_PORT = int(os.environ.get("MEMORY_INGEST_PORT", "7747"))


def _read_ingest_pid() -> int | None:
    """
    Read the ingest server PID from ~/.memory/ingest.pid.

    Returns the PID as an integer, or None if the file doesn't exist or
    contains an invalid value (e.g. leftover from a previous crash).
    """
    if not os.path.exists(_INGEST_PID_PATH):
        return None
    try:
        with open(_INGEST_PID_PATH, "r") as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        # File exists but content is not a valid integer — treat as absent.
        return None


def cmd_ingest_server(args):
    """
    Manage the HTTP ingest server (Phase 16).

    Subcommands:
      start   — launch the server in the background, write PID to ~/.memory/ingest.pid
      stop    — read PID file and send SIGTERM to the server
      status  — print whether the server is running, its PID, and the port
    """
    import signal as sig_mod  # imported locally to avoid shadowing the built-in signal

    # Ensure the ~/.memory directory exists before writing PID/log files.
    os.makedirs(os.path.expanduser("~/.memory"), exist_ok=True)

    # Read the positional subcommand (start | stop | status).
    sub = getattr(args, "ingest_sub", None)

    if sub == "start":
        # --- Check if already running ---
        pid = _read_ingest_pid()
        if pid is not None and _is_process_running(pid):
            print(f"Ingest server is already running (PID {pid}, port {_INGEST_PORT}).")
            return

        # Open the log file in append mode so previous logs are preserved.
        log_file = open(_INGEST_LOG_PATH, "a")

        # Popen launches the server as a detached background process.
        # stdout and stderr both go to the shared log file.
        proc = subprocess.Popen(
            [sys.executable, _INGEST_SCRIPT],
            stdout=log_file,
            stderr=log_file,
            # start_new_session=True detaches the server from the current terminal
            # so it keeps running after the shell exits.
            start_new_session=True,
        )

        # Write the PID so stop/status can find the process later.
        with open(_INGEST_PID_PATH, "w") as f:
            f.write(str(proc.pid))

        print(f"Ingest server started (PID {proc.pid}, port {_INGEST_PORT}).")
        print(f"Log: {_INGEST_LOG_PATH}")

    elif sub == "stop":
        # --- Stop the server by sending SIGTERM ---
        pid = _read_ingest_pid()
        if pid is None:
            print("Ingest server is not running (no PID file found).")
            return

        if not _is_process_running(pid):
            print(f"Ingest server PID {pid} is not running. Cleaning up stale PID file.")
            os.unlink(_INGEST_PID_PATH)
            return

        try:
            # SIGTERM asks the server to shut down gracefully.
            os.kill(pid, sig_mod.SIGTERM)
            print(f"Sent SIGTERM to ingest server (PID {pid}).")
            # Remove the PID file — we expect the process to exit shortly.
            os.unlink(_INGEST_PID_PATH)
        except OSError as exc:
            print(f"Failed to stop ingest server (PID {pid}): {exc}")

    elif sub == "status":
        # --- Show whether the server is running ---
        pid = _read_ingest_pid()
        if pid is None:
            print("Ingest server status: stopped (no PID file).")
            return

        if _is_process_running(pid):
            print(f"Ingest server status: running (PID {pid}, port {_INGEST_PORT}).")
        else:
            print(f"Ingest server status: stopped (PID {pid} is no longer alive).")
            # Clean up the stale PID file.
            os.unlink(_INGEST_PID_PATH)

    else:
        # Unknown or missing subcommand — print usage help.
        print("Usage: python3 cli.py ingest-server <start|stop|status>")


def _read_pid() -> int | None:
    """
    Read the daemon PID from ~/.memory/daemon.pid.

    Returns the PID as an integer, or None if the file doesn't exist or
    is not a valid integer (e.g. if it was left over from a crash).
    """
    # If the PID file doesn't exist, the daemon is not running (or was never started).
    if not os.path.exists(_DAEMON_PID_PATH):
        return None
    try:
        with open(_DAEMON_PID_PATH, "r") as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        # File exists but content is not a valid integer — treat as absent.
        return None


def _is_process_running(pid: int) -> bool:
    """
    Check whether a process with the given PID is currently alive.

    Uses os.kill(pid, 0) which sends no signal but raises OSError if
    the process does not exist or we lack permission to signal it.

    Args:
        pid — the process ID to check

    Returns:
        True if the process is running, False otherwise.
    """
    try:
        # Signal 0 checks process existence without sending a real signal.
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def cmd_daemon(args):
    """
    Manage the background relearning daemon.

    Subcommands:
      start   — launch the daemon in the background, write PID to ~/.memory/daemon.pid
      stop    — read PID file and send SIGTERM to the daemon
      status  — print whether the daemon is running and its PID
      --once  — run one processing pass and exit (useful for testing)
    """
    import signal as sig_mod  # imported locally to avoid shadowing built-in signal

    # Ensure the ~/.memory directory exists before writing PID/log files.
    os.makedirs(os.path.expanduser("~/.memory"), exist_ok=True)

    sub = getattr(args, "daemon_sub", None)

    if sub == "start" or getattr(args, "daemon_once", False):
        # --- Handle --once mode: run directly in this process and exit ---
        if getattr(args, "daemon_once", False):
            # Import and run the daemon synchronously — useful for CI and manual tests.
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from memory.daemon import run as daemon_run
            print("Running daemon in --once mode…")
            daemon_run(once=True)
            print("Done.")
            return

        # --- Start the daemon as a background process ---
        pid = _read_pid()
        if pid is not None and _is_process_running(pid):
            print(f"Daemon is already running (PID {pid}).")
            return

        # Open the log file in append mode for stdout and stderr.
        log_file = open(_DAEMON_LOG_CLI_PATH, "a")

        # Popen launches the daemon as a separate process.
        # The daemon's stdout and stderr both go to the shared log file.
        proc = subprocess.Popen(
            [sys.executable, _DAEMON_SCRIPT],
            stdout=log_file,
            stderr=log_file,
            # start_new_session=True detaches the daemon from the current terminal
            # so it keeps running after the shell exits.
            start_new_session=True,
        )

        # Write the PID so stop/status can find the process later.
        with open(_DAEMON_PID_PATH, "w") as f:
            f.write(str(proc.pid))

        print(f"Daemon started (PID {proc.pid}). Log: {_DAEMON_LOG_CLI_PATH}")

    elif sub == "stop":
        # --- Stop the daemon by sending SIGTERM ---
        pid = _read_pid()
        if pid is None:
            print("Daemon is not running (no PID file found).")
            return

        if not _is_process_running(pid):
            print(f"Daemon PID {pid} is not running. Cleaning up stale PID file.")
            os.unlink(_DAEMON_PID_PATH)
            return

        try:
            # Send SIGTERM — the daemon's signal handler will set _shutdown=True
            # and the loop will exit cleanly after the current session.
            os.kill(pid, sig_mod.SIGTERM)
            print(f"Sent SIGTERM to daemon (PID {pid}).")
            # Remove the PID file since we expect the daemon to exit shortly.
            os.unlink(_DAEMON_PID_PATH)
        except OSError as exc:
            print(f"Failed to stop daemon (PID {pid}): {exc}")

    elif sub == "status":
        # --- Show whether the daemon is running ---
        pid = _read_pid()
        if pid is None:
            print("Daemon status: stopped (no PID file).")
            return

        if _is_process_running(pid):
            print(f"Daemon status: running (PID {pid}).")
            # Show how many sessions were processed today by counting processed sessions
            # from the database.
            if os.path.exists(DB_PATH):
                conn = init_db(DB_PATH)
                today = __import__("datetime").date.today().isoformat()
                count = conn.execute(
                    """
                    SELECT COUNT(*) FROM sessions
                    WHERE daemon_processed_at IS NOT NULL
                      AND daemon_processed_at >= ?
                    """,
                    (today,),
                ).fetchone()[0]
                conn.close()
                print(f"Sessions processed today: {count}")
        else:
            print(f"Daemon status: stopped (PID {pid} is no longer alive).")
            # Clean up the stale PID file.
            os.unlink(_DAEMON_PID_PATH)

    else:
        # Unknown or missing subcommand — print usage help.
        print("Usage: python3 cli.py daemon <start|stop|status|--once>")


# ---------------------------------------------------------------------------
# Command: compact
# ---------------------------------------------------------------------------

def cmd_compact(_args):
    """Manually run one daemon compaction pass (episodic, working memory, compact, procedural)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from memory.daemon import run as daemon_run
    print("Running daemon compaction pass…")
    daemon_run(once=True)
    print("Done.")


# ---------------------------------------------------------------------------
# Command: install
# ---------------------------------------------------------------------------

# Path to the global Claude Code settings file.
_CLAUDE_SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")


def cmd_install(_args):
    """Wire wake_up.py and save_hook.py into the global ~/.claude/settings.json."""
    # Compute absolute paths to the hook scripts based on this file's location.
    repo_root  = os.path.dirname(os.path.abspath(__file__))
    wake_up    = os.path.join(repo_root, "hooks", "wake_up.py")
    save_hook  = os.path.join(repo_root, "hooks", "save_hook.py")

    wake_cmd = f"python3 {wake_up}"
    save_cmd = f"python3 {save_hook}"

    # Load existing settings or start fresh.
    if os.path.exists(_CLAUDE_SETTINGS_PATH):
        with open(_CLAUDE_SETTINGS_PATH) as f:
            settings = json.load(f)
    else:
        settings = {}

    hooks = settings.setdefault("hooks", {})

    def _has_command(hook_list: list, cmd: str) -> bool:
        # Check whether any entry in the hook list already contains this command.
        for group in hook_list:
            for h in group.get("hooks", []):
                if h.get("command") == cmd:
                    return True
        return False

    def _add_hook(event: str, cmd: str) -> bool:
        # Returns True if the hook was added, False if it was already present.
        groups = hooks.setdefault(event, [])
        if _has_command(groups, cmd):
            return False
        groups.append({"matcher": "", "hooks": [{"type": "command", "command": cmd}]})
        return True

    added_wake = _add_hook("UserPromptSubmit", wake_cmd)
    added_save = _add_hook("Stop", save_cmd)

    if not added_wake and not added_save:
        print("Hooks already registered in ~/.claude/settings.json — nothing to do.")
        return

    os.makedirs(os.path.dirname(_CLAUDE_SETTINGS_PATH), exist_ok=True)
    with open(_CLAUDE_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")

    if added_wake:
        print(f"  + UserPromptSubmit → {wake_cmd}")
    if added_save:
        print(f"  + Stop             → {save_cmd}")
    print("Done. Restart Claude Code for the hooks to take effect.")


# ---------------------------------------------------------------------------
# Command: sync-identity
# ---------------------------------------------------------------------------

IDENTITY_PATH = os.path.expanduser("~/.memory/identity.md")


def cmd_sync_identity(_args):
    # Parse each "- **Key**: Value" bullet in identity.md into a natural-language fact,
    # replacing any stale identity facts in the DB so FTS5 search can find them.
    if not os.path.exists(IDENTITY_PATH):
        print(f"No identity file found at {IDENTITY_PATH}")
        print("Create it with your name, role, and preferences first.")
        return

    with open(IDENTITY_PATH) as f:
        text = f.read()

    conn = get_conn()

    # Remove all previously synced identity facts so stale data doesn't accumulate.
    old_ids = [r[0] for r in conn.execute(
        "SELECT id FROM facts WHERE source = 'identity'"
    ).fetchall()]
    for fact_id in old_ids:
        delete_fact(conn, fact_id)

    # Parse bullet lines into facts.  Handles both "- **Key**: Value" (bold)
    # and "- Key: Value" (plain) so the format doesn't need to be exact.
    facts_added = 0
    for line in text.splitlines():
        m = re.match(r'-\s+\*{0,2}(.+?)\*{0,2}\s*:\s*(.+)', line.strip())
        if not m:
            continue
        key   = m.group(1).strip()
        value = m.group(2).strip()
        content = f"User's {key.lower()} is {value}"
        insert_fact(conn, content, tags=["identity"], source="identity")
        print(f"  + {content}")
        facts_added += 1

    # Rebuild the FTS5 index to fix any rowid-mismatch corruption from past edits.
    conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
    conn.commit()

    conn.close()
    print(f"\nSynced {facts_added} fact(s) from identity.md (removed {len(old_ids)} stale).")


# ---------------------------------------------------------------------------
# Argument parsing + dispatch
# ---------------------------------------------------------------------------

def main():
    # argparse builds a command-line interface from the definitions below.
    # Each subcommand (status, search, etc.) becomes a separate parser.
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Inspect the agentic memory database.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Show session count, turn count, date range")

    # search
    p_search = sub.add_parser("search", help="FTS5 keyword search")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)

    # semantic
    p_semantic = sub.add_parser("semantic", help="Semantic vector search")
    p_semantic.add_argument("query")
    p_semantic.add_argument("--limit", type=int, default=5)

    # get-session
    p_get = sub.add_parser("get-session", help="Print full transcript for one session")
    p_get.add_argument("session_id")

    # tail
    p_tail = sub.add_parser("tail", help="Show last N sessions")
    p_tail.add_argument("n", type=int, nargs="?", default=10,
                        help="Number of sessions to show (default 10)")

    # dashboard
    sub.add_parser("dashboard", help="Generate and open dashboard.html")

    # logs
    p_logs = sub.add_parser("logs", help="Print recent activity or error log lines")
    p_logs.add_argument(
        "--errors", action="store_true",
        help="Show error.log instead of activity.log",
    )
    p_logs.add_argument(
        "--tail", type=int, default=20,
        help="Number of lines to show (default 20)",
    )

    # consolidate — summarise old sessions with local ollama (Phase 12)
    p_consolidate = sub.add_parser(
        "consolidate",
        help="Summarise sessions older than --days days using local ollama",
    )
    p_consolidate.add_argument(
        "--days", type=int, default=30,
        help="Sessions updated more than this many days ago are candidates (default 30)",
    )
    p_consolidate.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing anything",
    )

    # prune — delete raw transcripts that already have a summary (Phase 12)
    p_prune = sub.add_parser(
        "prune",
        help="Null out raw transcripts for summarised sessions older than --days days",
    )
    p_prune.add_argument(
        "--days", type=int, default=90,
        help="Sessions updated more than this many days ago are candidates (default 90)",
    )
    p_prune.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be pruned without writing anything",
    )

    # daemon — manage the background relearning daemon (Phase 13)
    p_daemon = sub.add_parser(
        "daemon",
        help="Start, stop, or check the status of the background relearning daemon",
    )
    # Positional sub-subcommand: start, stop, status (optional; --once is a flag).
    p_daemon.add_argument(
        "daemon_sub",
        choices=["start", "stop", "status"],
        nargs="?",                    # optional — omitted when --once is used
        help="start | stop | status",
    )
    p_daemon.add_argument(
        "--once",
        dest="daemon_once",
        action="store_true",
        help="Run one processing pass and exit (for testing or manual runs)",
    )

    # compact — manually trigger one daemon compaction pass
    sub.add_parser(
        "compact",
        help="Run one daemon compaction pass: episodic entries, working memory, compacted sessions, procedural patterns",
    )

    # install — wire hooks into the global ~/.claude/settings.json
    sub.add_parser(
        "install",
        help="Add wake_up.py and save_hook.py to ~/.claude/settings.json (global hooks)",
    )

    # sync-identity — load identity.md fields into the facts table
    sub.add_parser(
        "sync-identity",
        help="Parse ~/.memory/identity.md and upsert its fields as searchable facts",
    )

    # ingest-server — manage the HTTP ingest server (Phase 16)
    p_ingest = sub.add_parser(
        "ingest-server",
        help="Start, stop, or check the status of the HTTP ingest server (port 7747)",
    )
    # Positional sub-subcommand: start, stop, or status.
    p_ingest.add_argument(
        "ingest_sub",
        choices=["start", "stop", "status"],
        nargs="?",
        help="start | stop | status",
    )

    args = parser.parse_args()

    # Dispatch to the right function based on which subcommand was typed.
    dispatch = {
        "status":        cmd_status,
        "search":        cmd_search,
        "semantic":      cmd_semantic,
        "get-session":   cmd_get_session,
        "tail":          cmd_tail,
        "dashboard":     cmd_dashboard,
        "logs":          cmd_logs,
        "consolidate":   cmd_consolidate,
        "prune":         cmd_prune,
        "daemon":        cmd_daemon,
        "ingest-server":   cmd_ingest_server,
        "compact":         cmd_compact,
        "install":         cmd_install,
        "sync-identity":   cmd_sync_identity,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
