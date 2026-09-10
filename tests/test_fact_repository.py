import unittest
from unittest.mock import patch

from memory.inference import GenerationResult

from memory.contracts import ExtractedFact
from memory.db import init_db
from memory.fact_repository import (
    build_fact_content,
    build_fact_semantic_content,
    build_fact_tags,
    list_session_facts,
    save_extracted_facts,
)


class TestFactRepository(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    @patch("memory.facts.text.generate_text", return_value=GenerationResult(text="My name is Yash. What is my name? Yash.", model="test-model"))
    def test_build_fact_content_and_tags_are_deterministic(self, _mock_generate):
        fact = ExtractedFact(
            entity="User",
            attribute="Name",
            value="Yash",
            confidence=0.9,
            source_quote="My name is Yash.",
        )

        self.assertEqual(build_fact_content(fact), "user.name = Yash")
        self.assertEqual(build_fact_semantic_content(fact), "My name is Yash. What is my name? Yash.")
        self.assertEqual(
            build_fact_tags(fact),
            [
                "memory_type:fact",
                "origin:llm_extractor",
                "entity:user",
                "attribute:name",
                "has:source_quote",
                "has:confidence",
            ],
        )

    @patch("memory.facts.text.generate_text", return_value=GenerationResult(text="My name is Yash. What is my name? Yash.", model="test-model"))
    @patch("memory.inference.embed_text", return_value=[0.1, 0.2, 0.3])
    def test_save_extracted_facts_persists_one_fact(self, _mock_embed, _mock_generate):
        fact = ExtractedFact(entity="user", attribute="name", value="Yash")

        saved_ids = save_extracted_facts(
            self.conn,
            [fact],
            session_id="session-123",
        )

        self.assertEqual(len(saved_ids), 1)
        rows = list_session_facts(self.conn, session_id="session-123")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "user.name = Yash")
        self.assertEqual(rows[0]["semantic_content"], "My name is Yash. What is my name? Yash.")
        self.assertEqual(rows[0]["entity"], "user")
        self.assertEqual(rows[0]["attribute"], "name")
        self.assertEqual(rows[0]["value"], "Yash")
        self.assertEqual(rows[0]["source"], "fact_extractor")
        self.assertIn("entity:user", rows[0]["tags"])
        self.assertIn("attribute:name", rows[0]["tags"])

    @patch("memory.facts.text.generate_text", return_value=GenerationResult(text="My name is Yash. What is my name? Yash.", model="test-model"))
    @patch("memory.inference.embed_text", return_value=[0.1, 0.2, 0.3])
    def test_save_extracted_facts_dedupes_semantic_duplicates(self, _mock_embed, _mock_generate):
        saved_ids = save_extracted_facts(
            self.conn,
            [
                ExtractedFact(entity="user", attribute="name", value="Yash", confidence=0.4),
                ExtractedFact(entity="User", attribute="Name", value=" Yash ", source_quote="My name is Yash."),
            ],
            session_id="session-dup",
        )

        self.assertEqual(len(saved_ids), 1)
        rows = list_session_facts(self.conn, session_id="session-dup")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "user.name = Yash")

    @patch(
        "memory.facts.text.generate_text",
        side_effect=[
            GenerationResult(text="My timezone is PST. What is my timezone? PST.", model="test-model"),
            GenerationResult(text="My timezone is EST. What is my timezone? EST.", model="test-model"),
        ],
    )
    @patch("memory.inference.embed_text", return_value=[0.1, 0.2, 0.3])
    def test_save_extracted_facts_merges_conflicts_last_value_wins(self, _mock_embed, _mock_generate):
        # When the same (entity, attribute) appears twice in one extraction pass,
        # upsert_fact merges them: the second value overwrites the first in place,
        # keeping exactly one row. Both inputs return the same row id.
        saved_ids = save_extracted_facts(
            self.conn,
            [
                ExtractedFact(entity="user", attribute="timezone", value="PST"),
                ExtractedFact(entity="user", attribute="timezone", value="EST"),
            ],
            session_id="session-conflict",
        )

        self.assertEqual(len(saved_ids), 2)
        self.assertEqual(saved_ids[0], saved_ids[1])
        rows = list_session_facts(self.conn, session_id="session-conflict")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "user.timezone = EST")


if __name__ == "__main__":
    unittest.main()
