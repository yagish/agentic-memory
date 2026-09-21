from __future__ import annotations

import os
import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id           TEXT PRIMARY KEY,
  agent                TEXT NOT NULL DEFAULT 'claude',
  started_at           TEXT,
  updated_at           TEXT,
  turn_count           INTEGER,
  transcript           TEXT,
  metadata             TEXT,
  project_id           TEXT,
  repo_root            TEXT,
  cwd                  TEXT,
  git_remote           TEXT,
  git_branch           TEXT,
  daemon_processed_at  TEXT
);

CREATE TABLE IF NOT EXISTS facts (
  id               TEXT PRIMARY KEY,
  entity           TEXT NOT NULL,
  attribute        TEXT NOT NULL,
  value            TEXT NOT NULL,
  semantic_content TEXT NOT NULL,
  tags             TEXT,
  source           TEXT,
  session_id       TEXT,
  created_at       TEXT,
  updated_at       TEXT,
  embedding        BLOB
);

CREATE TABLE IF NOT EXISTS episodic_memory (
  id          TEXT PRIMARY KEY,
  session_id  TEXT NOT NULL,
  title       TEXT NOT NULL,
  abstract    TEXT NOT NULL,
  happened_at TEXT NOT NULL,
  details     TEXT,
  embedding   BLOB
);

CREATE TABLE IF NOT EXISTS procedural_memory (
  id          TEXT PRIMARY KEY,
  session_id  TEXT NOT NULL,
  title       TEXT NOT NULL,
  summary     TEXT NOT NULL,
  updated_at  TEXT NOT NULL,
  details     TEXT,
  embedding   BLOB
);

CREATE TABLE IF NOT EXISTS working_memory (
  id            TEXT PRIMARY KEY,
  session_id    TEXT NOT NULL UNIQUE,
  current_goal  TEXT NOT NULL,
  current_focus TEXT NOT NULL,
  next_step     TEXT NOT NULL,
  status        TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  details       TEXT,
  embedding     BLOB
);

CREATE TABLE IF NOT EXISTS session_memory (
  id           TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL UNIQUE,
  title        TEXT NOT NULL,
  summary      TEXT NOT NULL,
  left_off_at  TEXT NOT NULL,
  updated_at   TEXT NOT NULL,
  details      TEXT,
  embedding    BLOB
);

CREATE TABLE IF NOT EXISTS telemetry_recall_events (
  id                                  TEXT PRIMARY KEY,
  request_id                          TEXT NOT NULL,
  event_type                          TEXT NOT NULL,
  created_at                          TEXT NOT NULL,
  session_id                          TEXT,
  agent                               TEXT,
  prompt_chars                        INTEGER,
  prompt_tokens_estimate              INTEGER,
  include_working_memory              INTEGER,
  project_id                          TEXT,
  repo_root                           TEXT,
  cwd                                 TEXT,
  git_remote                          TEXT,
  git_branch                          TEXT,
  action                              TEXT,
  warnings_count                      INTEGER,
  facts_count                         INTEGER,
  episodic_count                      INTEGER,
  procedural_count                    INTEGER,
  session_memory_count                INTEGER,
  working_memory_count                INTEGER,
  answer_tokens_estimate              INTEGER,
  injection_tokens_estimate           INTEGER,
  recalled_context_tokens_estimate    INTEGER,
  compression_gain_tokens_estimate    INTEGER,
  tokens_saved_estimate               INTEGER,
  error_message                       TEXT,
  details                             TEXT
);

CREATE TABLE IF NOT EXISTS telemetry_retrieval_lane_metrics (
  id              TEXT PRIMARY KEY,
  request_id      TEXT NOT NULL,
  lane            TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  candidate_count INTEGER,
  filtered_count  INTEGER,
  selected_count  INTEGER,
  hit_count       INTEGER,
  duration_ms     REAL,
  details         TEXT
);

CREATE TABLE IF NOT EXISTS telemetry_latency_breakdowns (
  id           TEXT PRIMARY KEY,
  request_id   TEXT,
  created_at   TEXT NOT NULL,
  component    TEXT NOT NULL,
  operation    TEXT NOT NULL,
  stage        TEXT NOT NULL,
  duration_ms  REAL NOT NULL,
  session_id   TEXT,
  details      TEXT
);

CREATE TABLE IF NOT EXISTS telemetry_system_stats (
  id           TEXT PRIMARY KEY,
  created_at   TEXT NOT NULL,
  component    TEXT NOT NULL,
  process_id   INTEGER,
  session_id   TEXT,
  metric_name  TEXT NOT NULL,
  metric_value REAL NOT NULL,
  unit         TEXT,
  details      TEXT
);

CREATE INDEX IF NOT EXISTS idx_telemetry_recall_events_request_id ON telemetry_recall_events(request_id, created_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_recall_events_event_type ON telemetry_recall_events(event_type, created_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_lane_metrics_request_id ON telemetry_retrieval_lane_metrics(request_id, lane, created_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_latency_component ON telemetry_latency_breakdowns(component, operation, stage, created_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_latency_request_id ON telemetry_latency_breakdowns(request_id, created_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_system_stats_metric ON telemetry_system_stats(component, metric_name, created_at);
"""

_MIGRATIONS = [
    "ALTER TABLE sessions ADD COLUMN compacted_text TEXT",
    "ALTER TABLE sessions ADD COLUMN project_id TEXT",
    "ALTER TABLE sessions ADD COLUMN repo_root TEXT",
    "ALTER TABLE sessions ADD COLUMN cwd TEXT",
    "ALTER TABLE sessions ADD COLUMN git_remote TEXT",
    "ALTER TABLE sessions ADD COLUMN git_branch TEXT",
    "DELETE FROM episodic_memory WHERE rowid NOT IN (SELECT MIN(rowid) FROM episodic_memory GROUP BY session_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_episodic_memory_session_id_unique ON episodic_memory(session_id)",
    "DELETE FROM procedural_memory WHERE rowid NOT IN (SELECT MIN(rowid) FROM procedural_memory GROUP BY session_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_procedural_memory_session_id_unique ON procedural_memory(session_id)",
    "CREATE TABLE IF NOT EXISTS telemetry_recall_events (id TEXT PRIMARY KEY, request_id TEXT NOT NULL, event_type TEXT NOT NULL, created_at TEXT NOT NULL, session_id TEXT, agent TEXT, prompt_chars INTEGER, prompt_tokens_estimate INTEGER, include_working_memory INTEGER, project_id TEXT, repo_root TEXT, cwd TEXT, git_remote TEXT, git_branch TEXT, action TEXT, warnings_count INTEGER, facts_count INTEGER, episodic_count INTEGER, procedural_count INTEGER, session_memory_count INTEGER, working_memory_count INTEGER, answer_tokens_estimate INTEGER, injection_tokens_estimate INTEGER, recalled_context_tokens_estimate INTEGER, compression_gain_tokens_estimate INTEGER, tokens_saved_estimate INTEGER, error_message TEXT, details TEXT)",
    "CREATE TABLE IF NOT EXISTS telemetry_retrieval_lane_metrics (id TEXT PRIMARY KEY, request_id TEXT NOT NULL, lane TEXT NOT NULL, created_at TEXT NOT NULL, candidate_count INTEGER, filtered_count INTEGER, selected_count INTEGER, hit_count INTEGER, duration_ms REAL, details TEXT)",
    "CREATE TABLE IF NOT EXISTS telemetry_latency_breakdowns (id TEXT PRIMARY KEY, request_id TEXT, created_at TEXT NOT NULL, component TEXT NOT NULL, operation TEXT NOT NULL, stage TEXT NOT NULL, duration_ms REAL NOT NULL, session_id TEXT, details TEXT)",
    "CREATE TABLE IF NOT EXISTS telemetry_system_stats (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, component TEXT NOT NULL, process_id INTEGER, session_id TEXT, metric_name TEXT NOT NULL, metric_value REAL NOT NULL, unit TEXT, details TEXT)",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_recall_events_request_id ON telemetry_recall_events(request_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_recall_events_event_type ON telemetry_recall_events(event_type, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_lane_metrics_request_id ON telemetry_retrieval_lane_metrics(request_id, lane, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_latency_component ON telemetry_latency_breakdowns(component, operation, stage, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_latency_request_id ON telemetry_latency_breakdowns(request_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_system_stats_metric ON telemetry_system_stats(component, metric_name, created_at)",
]


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    if path != ":memory:":
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Run forward-only schema migrations. Each statement is tried once;
    OperationalError (column already exists) is silently ignored."""
    for sql in _MIGRATIONS:
        try:
            conn.execute(sql)
            conn.commit()
        except sqlite3.OperationalError:
            pass


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()
    _apply_migrations(conn)


def bootstrap_db(path: str) -> sqlite3.Connection:
    if path != ":memory:":
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    conn = open_db(path)
    try:
        ensure_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


def init_db(path: str) -> sqlite3.Connection:
    return bootstrap_db(path)
