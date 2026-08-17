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
import sys
import webbrowser    # opens the dashboard HTML in the default browser

# Add the project root to the module search path so we can import memory.db.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.db import init_db, search, semantic_search


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

    conn.close()

    print("=== Memory Status ===")
    print(f"  Sessions stored   : {total_sessions}")
    print(f"  Turns stored      : {total_turns}")
    print(f"  Embeddings stored : {total_vecs}")
    print(f"  Tokens (estimated): {estimated_tokens:,}")
    print(f"  Retrievals logged : {total_retrievals}")
    print(f"  Oldest session    : {date_row[0] or 'none'}")
    print(f"  Newest session    : {date_row[1] or 'none'}")


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

    conn.close()

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

<script>
const ctx = document.getElementById('sessionChart').getContext('2d');
new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: {chart_labels},
    datasets: [{{
      label: 'Sessions',
      data: {chart_values},
      backgroundColor: '#7c3aed',
      borderRadius: 4,
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

    args = parser.parse_args()

    # Dispatch to the right function based on which subcommand was typed.
    dispatch = {
        "status":      cmd_status,
        "search":      cmd_search,
        "semantic":    cmd_semantic,
        "get-session": cmd_get_session,
        "tail":        cmd_tail,
        "dashboard":   cmd_dashboard,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
