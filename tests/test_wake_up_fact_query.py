import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks.wake_up import _build_fact_query
from memory.db import init_db, insert_fact, search_facts


class TestWakeUpFactQuery(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")
        insert_fact(self.conn, "User's name is Yagish", tags=["identity"], source="identity")
        insert_fact(self.conn, "User's role is Tech Lead", tags=["identity"], source="identity")

    def tearDown(self):
        self.conn.close()

    def test_name_question_with_punctuation_still_matches_identity_fact(self):
        query = _build_fact_query("what is my name?")
        results = search_facts(self.conn, query, limit=5)
        self.assertTrue(any("name is Yagish" in r["content"] for r in results))

    def test_whos_my_name_variant_with_apostrophe_is_sanitized(self):
        query = _build_fact_query("what's my name?")
        results = search_facts(self.conn, query, limit=5)
        self.assertTrue(any("name is Yagish" in r["content"] for r in results))

    def test_who_am_i_expands_to_identity_terms(self):
        query = _build_fact_query("who am i?")
        self.assertIn("name", query)
        self.assertIn("role", query)
        results = search_facts(self.conn, query, limit=5)
        contents = [r["content"] for r in results]
        self.assertTrue(any("name is Yagish" in c for c in contents))
        self.assertTrue(any("role is Tech Lead" in c for c in contents))



if __name__ == "__main__":
    unittest.main()
