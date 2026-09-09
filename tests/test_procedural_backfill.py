import json
import os
import tempfile
import unittest

from memory.contracts import ExtractedProcedure
from memory.db import init_db, insert_procedural, open_db, upsert_session
from memory.procedural_backfill import (
    _transcript_json_to_text,
    backfill_procedural_memory,
    list_backfill_candidate_sessions,
)
from memory.procedural_repository import list_session_procedures


class TestProceduralBackfill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.conn = init_db(self.tmp.name)

    def tearDown(self):
        self.conn.close()
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def _store_session(self, session_id: str, transcript: list[dict], updated_at: str = "2026-01-01T00:00:00Z"):
        upsert_session(
            self.conn,
            session_id=session_id,
            agent="claude",
            transcript=transcript,
            started_at=updated_at,
            updated_at=updated_at,
        )

    def test_transcript_json_to_text_formats_turns(self):
        text = _transcript_json_to_text(json.dumps([
            {"role": "user", "content": "First step"},
            {"role": "assistant", "content": "Second step"},
            {"role": "tool", "content": ""},
        ]))
        self.assertEqual(text, "user: First step\nassistant: Second step")

    def test_list_candidates_excludes_sessions_with_existing_procedure_by_default(self):
        self._store_session("session-a", [{"role": "user", "content": "Deploy the service"}], "2026-01-01T00:00:00Z")
        self._store_session("session-b", [{"role": "user", "content": "Backfill the data"}], "2026-01-02T00:00:00Z")
        insert_procedural(
            self.conn,
            session_id="session-a",
            title="Existing procedure",
            summary="Already stored.",
            updated_at="2026-01-01T00:00:00Z",
            details={"steps": ["Do it"]},
        )

        rows = list_backfill_candidate_sessions(self.conn)
        self.assertEqual([row["session_id"] for row in rows], ["session-b"])

        rows = list_backfill_candidate_sessions(self.conn, include_existing=True)
        self.assertEqual([row["session_id"] for row in rows], ["session-a", "session-b"])

    def test_backfill_creates_procedure_for_missing_session(self):
        self._store_session(
            "session-procedural",
            [{"role": "user", "content": "The deploy workflow is build the image, run alembic upgrade, then verify staging."}],
        )
        extracted = ExtractedProcedure(
            title="Web deploy workflow",
            summary="Use this when deploying the web service.",
            steps=("Build the image", "Run alembic upgrade", "Verify staging"),
            trigger_phrases=("deploy web service",),
            tools=("alembic", "staging"),
            confidence=0.91,
        )

        result = backfill_procedural_memory(
            self.tmp.name,
            extract_fn=lambda session_text, **kwargs: extracted,
            embed_fn=lambda text: [1.0, 0.0],
        )

        self.assertEqual(result["scanned"], 1)
        self.assertEqual(result["created"], 1)
        self.assertEqual(result["replaced"], 0)
        verify_conn = open_db(self.tmp.name)
        try:
            rows = list_session_procedures(verify_conn, session_id="session-procedural")
        finally:
            verify_conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Web deploy workflow")
        self.assertEqual(rows[0]["steps"], ["Build the image", "Run alembic upgrade", "Verify staging"])

    def test_backfill_dry_run_reports_without_writing(self):
        self._store_session("session-procedural", [{"role": "user", "content": "Deploy by building the image."}])
        extracted = ExtractedProcedure(
            title="Web deploy workflow",
            summary="Use this when deploying the web service.",
            steps=("Build the image",),
        )

        result = backfill_procedural_memory(
            self.tmp.name,
            dry_run=True,
            extract_fn=lambda session_text, **kwargs: extracted,
            embed_fn=lambda text: [1.0, 0.0],
        )

        self.assertEqual(result["would_create"], 1)
        verify_conn = open_db(self.tmp.name)
        try:
            rows = list_session_procedures(verify_conn, session_id="session-procedural")
        finally:
            verify_conn.close()
        self.assertEqual(rows, [])

    def test_backfill_force_replaces_existing_procedure(self):
        self._store_session("session-procedural", [{"role": "user", "content": "Deploy by building the image then verify staging."}])
        insert_procedural(
            self.conn,
            session_id="session-procedural",
            title="Old workflow",
            summary="Old summary.",
            updated_at="2026-01-01T00:00:00Z",
            details={"steps": ["Old step"]},
        )
        extracted = ExtractedProcedure(
            title="New workflow",
            summary="New summary.",
            steps=("New step",),
        )

        result = backfill_procedural_memory(
            self.tmp.name,
            force=True,
            extract_fn=lambda session_text, **kwargs: extracted,
            embed_fn=lambda text: [1.0, 0.0],
        )

        self.assertEqual(result["replaced"], 1)
        verify_conn = open_db(self.tmp.name)
        try:
            rows = list_session_procedures(verify_conn, session_id="session-procedural")
        finally:
            verify_conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "New workflow")
        self.assertEqual(rows[0]["steps"], ["New step"])

    def test_backfill_tracks_no_procedure_and_errors(self):
        self._store_session("session-none", [{"role": "user", "content": "We fixed a one-off bug today."}], "2026-01-01T00:00:00Z")
        self._store_session("session-error", [{"role": "user", "content": "Deploy the service."}], "2026-01-02T00:00:00Z")

        def fake_extract(session_text, **kwargs):
            if "one-off bug" in session_text:
                return None
            raise RuntimeError("boom")

        result = backfill_procedural_memory(
            self.tmp.name,
            extract_fn=fake_extract,
            embed_fn=lambda text: [1.0, 0.0],
        )

        self.assertEqual(result["scanned"], 2)
        self.assertEqual(result["no_procedure"], 1)
        self.assertEqual(result["errors"], 1)


if __name__ == "__main__":
    unittest.main()
