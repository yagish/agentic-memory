# test_token_economics.py — tests for the _get_token_economics helper in cli.py.
#
# These tests verify that the economics query correctly aggregates injection
# and retrieval token counts from the retrievals table, and computes the
# coverage ratio.
#
# Run with:  python3 -m pytest tests/test_token_economics.py -v

import os
import sys
import tempfile
import unittest

# Add the project root to the path so we can import cli and memory.db.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import init_db, log_retrieval
from cli import _get_token_economics


class TestTokenEconomicsEmptyDb(unittest.TestCase):
    """Test _get_token_economics against a brand-new, empty database."""

    def setUp(self):
        # Create a temporary file-backed SQLite database with all schema applied.
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._conn = init_db(self._tmp.name)

    def tearDown(self):
        # Close the connection and remove the temp file.
        self._conn.close()
        os.unlink(self._tmp.name)

    def test_economics_empty_db(self):
        """
        A freshly initialised database has no retrievals rows.
        All totals should be zero and coverage_ratio should be None
        (undefined when there are no injections).
        """
        # Call the helper on the empty database.
        result = _get_token_economics(self._conn)

        # Totals must be zero — COALESCE(SUM(...), 0) ensures this.
        self.assertEqual(result["injection_total"], 0)
        self.assertEqual(result["retrieval_total"], 0)
        self.assertEqual(result["injection_count"], 0)

        # Coverage ratio is None because there are no injections to divide by.
        self.assertIsNone(result["coverage_ratio"])


class TestTokenEconomicsInjectionOnly(unittest.TestCase):
    """Test _get_token_economics when only wake-up injection rows exist."""

    def setUp(self):
        # Temporary database with schema applied.
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._conn = init_db(self._tmp.name)

    def tearDown(self):
        self._conn.close()
        os.unlink(self._tmp.name)

    def test_economics_injection_only(self):
        """
        When a wake_up_injection row exists but no MCP retrieval rows,
        injection_total should be positive, retrieval_total should be zero,
        and coverage_ratio should be 0.0 (retrieval / injection = 0 / N).
        """
        # Insert one injection row: 800 estimated tokens.
        # log_retrieval signature: (conn, tool, query, result_size)
        log_retrieval(self._conn, "wake_up_injection", None, 800)

        result = _get_token_economics(self._conn)

        # Injection total must reflect the inserted row.
        self.assertEqual(result["injection_total"], 800)
        self.assertEqual(result["injection_count"], 1)

        # No MCP tool rows were inserted, so retrieval total is zero.
        self.assertEqual(result["retrieval_total"], 0)

        # Coverage ratio is 0 / 800 = 0.0 — injections exist so ratio is defined.
        self.assertIsNotNone(result["coverage_ratio"])
        self.assertAlmostEqual(result["coverage_ratio"], 0.0)


class TestTokenEconomicsWithRetrievals(unittest.TestCase):
    """Test _get_token_economics when both injection and retrieval rows exist."""

    def setUp(self):
        # Temporary database with schema applied.
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._conn = init_db(self._tmp.name)

    def tearDown(self):
        self._conn.close()
        os.unlink(self._tmp.name)

    def test_economics_with_retrievals(self):
        """
        With both injection rows and MCP retrieval rows, coverage_ratio should
        equal retrieval_total / injection_total.
        """
        # Insert one injection row: 400 estimated tokens.
        log_retrieval(self._conn, "wake_up_injection", None, 400)

        # Insert two MCP retrieval rows: 200 + 400 = 600 total retrieval tokens.
        log_retrieval(self._conn, "memory_search",  "auth bug",    200)
        log_retrieval(self._conn, "memory_search",  "deploy issue", 400)

        result = _get_token_economics(self._conn)

        # Injection total: one row with result_size=400.
        self.assertEqual(result["injection_total"], 400)

        # Retrieval total: 200 + 400 = 600.
        self.assertEqual(result["retrieval_total"], 600)

        # Coverage ratio: 600 / 400 = 1.5.
        self.assertIsNotNone(result["coverage_ratio"])
        self.assertAlmostEqual(result["coverage_ratio"], 1.5)

        # Injection count: one injection event.
        self.assertEqual(result["injection_count"], 1)


if __name__ == "__main__":
    unittest.main()
