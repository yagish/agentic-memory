import unittest

from memory.db import init_db, log_retrieval


class TestRemovedRetrievalLogging(unittest.TestCase):
    def test_log_retrieval_is_noop_and_retrievals_table_absent(self):
        conn = init_db(":memory:")
        try:
            log_retrieval(conn, "wake_up_injection", None, 123)
            row = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='retrievals'"
            ).fetchone()
            self.assertEqual(row[0], 0)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
