import unittest

from memory.db import init_db
from tests.retrieval_eval_harness import (
    DEFAULT_SUITE_PATH,
    load_eval_suite,
    run_eval_suite,
    run_eval_suite_from_path,
)


class TestOfflineRetrievalEvalScaffold(unittest.TestCase):
    def test_default_fixture_suite_passes(self):
        result = run_eval_suite_from_path(DEFAULT_SUITE_PATH)

        self.assertEqual(result.total, 2)
        self.assertEqual(result.failed, 0)
        self.assertTrue(all(case.passed for case in result.cases))

    def test_fixture_format_is_project_labeled_for_future_regression_gating(self):
        suite = load_eval_suite(DEFAULT_SUITE_PATH)

        self.assertEqual(suite["schema_version"], 1)
        self.assertIn("anchors", suite["embedding_space"])
        self.assertGreaterEqual(len(suite["cases"]), 2)
        self.assertTrue(all("project_context" in session for session in suite["sessions"]))
        self.assertTrue(all("expect" in case for case in suite["cases"]))
        self.assertTrue(all("max_wrong_project_hits" in case["expect"] for case in suite["cases"]))

    def test_harness_flags_wrong_project_contamination(self):
        suite = {
            "schema_version": 1,
            "embedding_space": {
                "type": "anchor_terms",
                "anchors": ["shared", "deploy"]
            },
            "sessions": [
                {
                    "session_id": "sess-alpha",
                    "project_context": {"project_id": "project-alpha"},
                    "procedural": [
                        {
                            "title": "Alpha deploy flow",
                            "summary": "Deploy shared service for alpha.",
                            "steps": ["Deploy alpha"],
                            "embedding_terms": ["shared", "deploy"]
                        }
                    ]
                },
                {
                    "session_id": "sess-beta",
                    "project_context": {"project_id": "project-beta"},
                    "procedural": [
                        {
                            "title": "Beta deploy flow",
                            "summary": "Deploy shared service for beta.",
                            "steps": ["Deploy beta"],
                            "embedding_terms": ["shared", "deploy"]
                        }
                    ]
                }
            ],
            "cases": [
                {
                    "id": "detect_cross_project_procedural_contamination",
                    "prompt": "run the shared deploy",
                    "project_context": {"project_id": "project-alpha"},
                    "expect": {
                        "action": "inject",
                        "min_counts": {"procedural": 1},
                        "allowed_project_ids": ["project-alpha"],
                        "forbidden_project_ids": ["project-beta"],
                        "max_wrong_project_hits": 0
                    }
                }
            ]
        }

        conn = init_db(":memory:")
        try:
            result = run_eval_suite(conn, suite)
        finally:
            conn.close()

        self.assertEqual(result.total, 1)
        self.assertEqual(result.failed, 1)
        case = result.cases[0]
        self.assertFalse(case.passed)
        self.assertEqual(case.wrong_project_hits, 1)
        self.assertIn("project-beta", case.projects_hit)
        self.assertTrue(any("max_wrong_project_hits" in error or "forbidden project_ids" in error for error in case.errors))


if __name__ == "__main__":
    unittest.main()
