import unittest
from unittest.mock import MagicMock, patch

from memory.db import (
    init_db,
    insert_episodic,
    insert_fact,
    insert_procedural,
    pack_vector,
    upsert_session,
    upsert_session_memory,
    upsert_working_memory,
)
from memory.retrieval import (
    WakeUpContext,
    build_wake_up_injection,
    retrieve_wake_up_context,
)
from memory.retrieval._hybrid import fuse_ranked_lanes


class TestBuildWakeUpInjection(unittest.TestCase):
    def test_returns_empty_when_no_sections_exist(self):
        result = build_wake_up_injection(
            WakeUpContext(None, None, [], [], [], [])
        )
        self.assertEqual(result, "")

    def test_includes_episodic_fact_and_procedural_sections(self):
        result = build_wake_up_injection(
            WakeUpContext(
                {"id": "cs-1", "similarity": 0.98, "content": "Task: Fix auth middleware"},
                {"id": "wm-1", "current_goal": "Current task is cleaning up auth middleware", "current_focus": "Regression coverage", "active_tasks": ["Add regression coverage"], "next_step": "Write the regression coverage", "status": "ready_to_resume", "similarity": 0.88},
                [{"id": "cs-2", "similarity": 0.83, "content": "Task: Add JWT refresh flow"}],
                [{
                    "id": "ep-1",
                    "title": "Resolved auth bug",
                    "abstract": "Fixed the login loop.",
                    "decisions": ["Move token validation into shared middleware"],
                    "outcomes": ["Login loop stopped reproducing"],
                    "follow_ups": ["Add regression coverage"],
                    "similarity": 0.79,
                }],
                [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.97}],
                [{"id": "proc-1", "title": "Deploy service", "summary": "Use this when releasing the auth service.", "steps": ["Build the image", "Ship to staging"], "similarity": 0.86}],
                session_memory=[{"id": "sm-1", "title": "Auth middleware refactor", "summary": "Moved token validation into shared middleware.", "left_off_at": "Regression coverage still needs to be written", "next_steps": ["Add regression coverage"], "similarity": 0.9}],
            )
        )
        self.assertTrue(result.startswith("[Memory context: "))
        self.assertIn("Current working goal: Current task is cleaning up auth middleware.", result)
        self.assertIn("Relevant prior session: Auth middleware refactor. Moved token validation into shared middleware.", result)
        self.assertIn("Left off at: Regression coverage still needs to be written.", result)
        self.assertIn("Recent related episode: Resolved auth bug. Fixed the login loop.", result)
        self.assertIn("Decision: Move token validation into shared middleware.", result)
        self.assertIn("Outcome: Login loop stopped reproducing.", result)
        self.assertIn("Follow-up: Add regression coverage.", result)
        self.assertIn("Relevant how-to pattern: Deploy service. Use this when releasing the auth service.", result)
        self.assertIn("Steps: Build the image; Ship to staging.", result)
        self.assertIn("Remembered fact: user.name = Yash.", result)
        self.assertNotIn("Related prior session", result)


class TestRetrieveWakeUpContext(unittest.TestCase):
    def test_retrieval_fetches_episodic_facts_and_procedural_memory(self):
        conn = MagicMock()

        with patch("memory.retrieval._fetch.retrieve_episodic_memories", return_value=[
            {"id": "ep-1", "title": "Resolved auth bug", "abstract": "Fixed the login loop.", "similarity": 0.79}
        ]) as retrieve_episodic, \
             patch("memory.retrieval._fetch.search_episodic_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_facts_semantic", return_value=[
                 {"id": "fact-1", "content": "user.name = Yash", "similarity": 0.45},
                 {"id": "fact-2", "content": "user.timezone = EST", "similarity": 0.24},
             ]) as search_facts_semantic, \
             patch("memory.retrieval._fetch.retrieve_procedural_memories", return_value=[
                 {"id": "proc-1", "title": "Deploy service", "summary": "Use this when releasing the auth service.", "steps": ["Build the image"], "similarity": 0.88}
             ]) as retrieve_procedural, \
             patch("memory.retrieval._fetch.search_procedural_fts", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_session_memories", return_value=[]) as retrieve_session, \
             patch("memory.retrieval._fetch.search_session_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_session_memory_fts", return_value=[]):
            context = retrieve_wake_up_context(
                conn,
                "how do i deploy the auth service?",
                include_working_memory=True,
                embed_fn=lambda prompt: [0.1, 0.2],
            )

        self.assertIsNone(context.cache_hit)
        self.assertIsNone(context.working_mem)
        self.assertEqual(context.enrichment, [])
        self.assertEqual([item["id"] for item in context.episodic], ["ep-1"])
        self.assertEqual([item["id"] for item in context.facts], ["fact-1"])
        self.assertEqual([item["id"] for item in context.procedural], ["proc-1"])
        self.assertEqual(context.warnings, [])
        retrieve_episodic.assert_called_once_with(
            conn,
            "how do i deploy the auth service?",
            query_vector=[0.1, 0.2],
            embed_fn=unittest.mock.ANY,
            min_similarity=0.58,
            limit=8,
            source="wake_up",
            session_ids=None,
        )
        search_facts_semantic.assert_called_once_with(conn, [0.1, 0.2], limit=20)
        retrieve_session.assert_called_once_with(
            conn,
            "how do i deploy the auth service?",
            query_vector=[0.1, 0.2],
            embed_fn=unittest.mock.ANY,
            min_similarity=0.6,
            limit=4,
            source="wake_up",
            exclude_session_id=None,
            session_ids=None,
        )
        retrieve_procedural.assert_called_once_with(
            conn,
            "how do i deploy the auth service?",
            query_vector=[0.1, 0.2],
            embed_fn=unittest.mock.ANY,
            min_similarity=0.58,
            limit=4,
            source="wake_up",
        )

    def test_fact_retrieval_returns_no_hits_when_semantic_search_misses(self):
        conn = MagicMock()

        with patch("memory.retrieval._fetch.retrieve_episodic_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_episodic_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_facts_semantic", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_procedural_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_procedural_fts", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_session_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_session_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_session_memory_fts", return_value=[]):
            context = retrieve_wake_up_context(
                conn,
                "what is my favorite language?",
                include_working_memory=True,
                embed_fn=lambda prompt: [0.1],
            )

        self.assertEqual(context.facts, [])

    def test_recent_episode_fallback_returns_substantive_recent_work_items(self):
        conn = MagicMock()

        recent_rows = [
            {
                "id": "ep-ignored",
                "title": "Name inquiry",
                "abstract": "The user asked about their name, but it was not stored in the assistant's memory.",
                "decisions": [],
                "outcomes": [],
                "follow_ups": [],
            },
            {
                "id": "ep-1",
                "title": "YAML error correction",
                "abstract": "The user encountered a YAML error and fixed it by correcting the nesting.",
                "decisions": ["Fix the YAML nesting"],
                "outcomes": ["Corrected the YAML file"],
                "follow_ups": [],
            },
            {
                "id": "ep-2",
                "title": "Token validation move",
                "abstract": "Moved token validation into shared auth middleware and fixed the login loop.",
                "decisions": ["Move token validation into shared auth middleware"],
                "outcomes": ["Login loop fixed"],
                "follow_ups": ["Add regression tests"],
            },
        ]

        # The test uses a 1-dim mock embedding ([0.1]). The intent classifier
        # (_get_resume_exemplar_vecs) is also patched to return a matching 1-dim
        # vector so that cosine similarity equals 1.0 and the fallback triggers.
        # This tests that the FALLBACK MECHANISM works correctly; a separate unit
        # test of _prompt_requests_recent_episode_summary covers the classifier itself.
        with patch("memory.retrieval._fetch.retrieve_episodic_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_episodic_fts", return_value=[]), \
             patch("memory.retrieval._fetch.list_recent_episodic_memories", return_value=recent_rows) as list_recent, \
             patch("memory.retrieval._fetch.search_facts_semantic", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_procedural_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_procedural_fts", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_session_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_session_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_session_memory_fts", return_value=[]), \
             patch("memory.retrieval._intent._get_resume_exemplar_vecs", return_value=([0.1],)):
            context = retrieve_wake_up_context(
                conn,
                "what was i working on last",
                include_working_memory=True,
                embed_fn=lambda prompt: [0.1],
            )

        self.assertEqual([item["id"] for item in context.episodic], ["ep-1", "ep-2"])
        list_recent.assert_called_once_with(conn, limit=5, source="wake_up_recent")

    def test_retrieval_fetches_working_and_session_memory_when_prompt_requests_resume_context(self):
        conn = MagicMock()

        with patch("memory.retrieval._fetch._retrieve_wm", return_value={
            "id": "wm-1",
            "session_id": "session-123",
            "current_goal": "Finish auth middleware refactor",
            "current_focus": "Regression coverage for refresh-token flows",
            "active_tasks": ["Add refresh-token regression tests"],
            "constraints": ["Keep branch fix/auth-middleware"],
            "next_step": "Write the refresh-token regression test",
            "status": "ready_to_resume",
            "similarity": 1.0,
        }) as retrieve_working, \
             patch("memory.retrieval._fetch.retrieve_episodic_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_episodic_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_facts_semantic", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_procedural_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_procedural_fts", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_session_memories", return_value=[
                 {
                     "id": "sm-1",
                     "session_id": "session-old",
                     "title": "Auth middleware refactor",
                     "summary": "Moved token validation into shared middleware and fixed the login redirect loop locally.",
                     "left_off_at": "Regression coverage is still missing for refresh-token and expired-session flows",
                     "next_steps": ["Add regression coverage"],
                     "similarity": 0.91,
                 }
             ]) as retrieve_session, \
             patch("memory.retrieval._fetch.search_session_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_session_memory_fts", return_value=[]):
            context = retrieve_wake_up_context(
                conn,
                "pick up where i left off on the auth middleware work",
                include_working_memory=False,
                session_id="session-123",
                embed_fn=lambda prompt: [0.1, 0.2],
            )

        self.assertIsNotNone(context.working_mem)
        self.assertEqual(context.working_mem["id"], "wm-1")
        self.assertEqual([item["id"] for item in context.session_memory], ["sm-1"])
        retrieve_working.assert_called_once_with(conn, session_id="session-123", source="wake_up")  # patched as _retrieve_wm
        retrieve_session.assert_called_once_with(
            conn,
            "pick up where i left off on the auth middleware work",
            query_vector=[0.1, 0.2],
            embed_fn=unittest.mock.ANY,
            min_similarity=0.6,
            limit=4,
            source="wake_up",
            exclude_session_id="session-123",
            session_ids=None,
        )

    def test_retrieval_searches_broadly_without_prompt_gating(self):
        conn = MagicMock()

        with patch("memory.retrieval._fetch.retrieve_episodic_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_episodic_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_facts_semantic", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_procedural_memories", return_value=[]) as retrieve_procedural, \
             patch("memory.retrieval._fetch.search_procedural_fts", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_session_memories", return_value=[]) as retrieve_session, \
             patch("memory.retrieval._fetch.search_session_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_session_memory_fts", return_value=[]):
            retrieve_wake_up_context(
                conn,
                "what did we decide about auth middleware?",
                include_working_memory=False,
                embed_fn=lambda prompt: [0.1],
            )

        retrieve_procedural.assert_called_once()
        retrieve_session.assert_called_once()

    def test_ranked_episodic_results_filter_missing_memory_artifacts(self):
        conn = MagicMock()

        with patch("memory.retrieval._fetch.retrieve_episodic_memories", return_value=[
            {
                "id": "ep-noise",
                "title": "Name inquiry",
                "abstract": "The user asked about their name, but it was not stored in the assistant's memory.",
                "decisions": [],
                "outcomes": [],
                "follow_ups": [],
                "similarity": 0.95,
            },
            {
                "id": "ep-1",
                "title": "Auth middleware decision",
                "abstract": "Moved token validation into shared middleware.",
                "decisions": ["Move token validation into shared middleware"],
                "outcomes": ["Login loop fixed"],
                "follow_ups": ["Add regression coverage"],
                "similarity": 0.74,
            },
            {
                "id": "ep-2",
                "title": "Refresh token regression",
                "abstract": "Added regression coverage for expired-session flows.",
                "decisions": ["Cover expired-session flows"],
                "outcomes": ["Regression test added"],
                "follow_ups": [],
                "similarity": 0.73,
            },
        ]), \
             patch("memory.retrieval._fetch.search_episodic_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_facts_semantic", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_procedural_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_procedural_fts", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_session_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_session_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_session_memory_fts", return_value=[]):
            context = retrieve_wake_up_context(
                conn,
                "what did we decide about auth middleware?",
                include_working_memory=False,
                embed_fn=lambda prompt: [0.1],
            )

        self.assertEqual([item["id"] for item in context.episodic], ["ep-1", "ep-2"])

    def test_non_fatal_layer_errors_become_warnings(self):
        conn = MagicMock()

        with patch("memory.retrieval._fetch._retrieve_wm", side_effect=RuntimeError("working boom")), \
             patch("memory.retrieval._fetch.retrieve_episodic_memories", side_effect=RuntimeError("episodic boom")), \
             patch("memory.retrieval._fetch.search_episodic_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_facts_semantic", side_effect=RuntimeError("facts boom")), \
             patch("memory.retrieval._fetch.retrieve_procedural_memories", side_effect=RuntimeError("procedural boom")), \
             patch("memory.retrieval._fetch.search_procedural_fts", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_session_memories", side_effect=RuntimeError("session boom")), \
             patch("memory.retrieval._fetch.search_session_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_session_memory_fts", return_value=[]):
            context = retrieve_wake_up_context(
                conn,
                "pick up where i left off and how do i deploy?",
                include_working_memory=True,
                session_id="session-123",
                embed_fn=lambda prompt: [0.1],
            )

        self.assertEqual(context.cache_hit, None)
        self.assertEqual(context.working_mem, None)
        self.assertEqual(context.enrichment, [])
        self.assertEqual(context.episodic, [])
        self.assertEqual(context.facts, [])
        self.assertEqual(context.procedural, [])
        self.assertEqual(context.session_memory, [])
        self.assertEqual(
            [warning.stage for warning in context.warnings],
            ["working_memory", "episodic", "facts", "procedural", "session_memory"],
        )


class TestHybridLaneFusion(unittest.TestCase):
    def test_fuse_ranked_lanes_deduplicates_rows_and_accumulates_rrf_scores(self):
        fused = fuse_ranked_lanes(
            (
                "semantic",
                [
                    {"id": "mem-a", "similarity": 0.71},
                    {"id": "mem-b", "similarity": 0.92},
                ],
            ),
            (
                "keyword",
                [
                    {"id": "mem-a", "keyword_hit": True, "keyword_score": 1.0, "keyword_rank": -1.2},
                    {"id": "mem-c", "keyword_hit": True, "keyword_score": 0.5, "keyword_rank": -0.6},
                ],
            ),
            (
                "session",
                [
                    {"id": "mem-a", "session_hit": True, "session_hit_score": 1.0, "session_hit_rank": -0.9},
                ],
            ),
        )

        self.assertEqual([row["id"] for row in fused], ["mem-a", "mem-b", "mem-c"])
        self.assertEqual(set(fused[0]["rrf_sources"]), {"semantic", "keyword", "session"})
        self.assertAlmostEqual(fused[0]["rrf_score"], round((1 / 61) + (1 / 61) + (1 / 61), 6), places=6)
        self.assertAlmostEqual(fused[1]["rrf_score"], round(1 / 62, 6), places=6)
        self.assertGreater(fused[0]["rrf_score"], fused[1]["rrf_score"])
        self.assertTrue(fused[0]["keyword_hit"])
        self.assertTrue(fused[0]["session_hit"])
        self.assertEqual(fused[0]["similarity"], 0.71)


class TestProjectAwareRetrieval(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")
        self.embed_fn = self._build_anchor_embed_fn(
            [
                "alpha",
                "beta",
                "auth",
                "deploy",
                "resume",
                "checklist",
                "database",
                "engine",
                "aurora",
                "timezone",
            ]
        )

    def tearDown(self):
        self.conn.close()

    @staticmethod
    def _build_anchor_embed_fn(anchors: list[str]):
        tokens = {anchor: idx for idx, anchor in enumerate(anchors)}

        def embed_fn(text: str) -> list[float]:
            vector = [0.0] * len(anchors)
            for token in (text or "").lower().replace("-", " ").split():
                idx = tokens.get(token)
                if idx is not None:
                    vector[idx] += 1.0
            return vector

        return embed_fn

    def _project_context(self, project_id: str) -> dict[str, str]:
        return {
            "project_id": project_id,
            "repo_root": f"/tmp/{project_id}",
            "cwd": f"/tmp/{project_id}/app",
            "git_remote": f"git@github.com:example/{project_id}.git",
            "git_branch": "main",
        }

    def _seed_session(self, session_id: str, project_id: str, *, updated_at: str = "2026-02-01T00:00:00Z") -> None:
        upsert_session(
            self.conn,
            session_id,
            "claude",
            [{"role": "user", "content": f"seed {session_id}"}],
            "2026-02-01T00:00:00Z",
            updated_at,
            project_context=self._project_context(project_id),
        )

    def test_fetch_time_scoping_passes_same_project_session_ids_to_retrievers(self):
        conn = MagicMock()
        ambient = self._project_context("project-alpha")
        session_projects = {"alpha-current": ambient, "alpha-prior": ambient}

        with patch("memory.retrieval._fetch.resolve_ambient_project_context", return_value=ambient), \
             patch("memory.retrieval._fetch.same_project_session_ids", return_value={"alpha-current", "alpha-prior"}), \
             patch("memory.retrieval._fetch.load_session_project_contexts", return_value=session_projects), \
             patch("memory.retrieval._fetch._retrieve_wm", return_value=None), \
             patch("memory.retrieval._fetch.retrieve_episodic_memories", return_value=[] ) as retrieve_episodic, \
             patch("memory.retrieval._fetch.search_episodic_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_facts_semantic", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_procedural_memories", return_value=[]), \
             patch("memory.retrieval._fetch.search_procedural_fts", return_value=[]), \
             patch("memory.retrieval._fetch.retrieve_session_memories", return_value=[] ) as retrieve_session, \
             patch("memory.retrieval._fetch.search_session_fts", return_value=[]), \
             patch("memory.retrieval._fetch.search_session_memory_fts", return_value=[]):
            retrieve_wake_up_context(
                conn,
                "resume alpha auth deploy",
                include_working_memory=False,
                session_id="alpha-current",
                project_context=ambient,
                embed_fn=lambda prompt: [0.1, 0.2],
            )

        retrieve_episodic.assert_called_once_with(
            conn,
            "resume alpha auth deploy",
            query_vector=[0.1, 0.2],
            embed_fn=unittest.mock.ANY,
            min_similarity=0.58,
            limit=8,
            source="wake_up",
            session_ids={"alpha-current", "alpha-prior"},
        )
        retrieve_session.assert_called_once_with(
            conn,
            "resume alpha auth deploy",
            query_vector=[0.1, 0.2],
            embed_fn=unittest.mock.ANY,
            min_similarity=0.6,
            limit=4,
            source="wake_up",
            exclude_session_id="alpha-current",
            session_ids={"alpha-current", "alpha-prior"},
        )

    def test_same_project_scoping_filters_cross_project_session_and_episodic_memory(self):
        self._seed_session("alpha-current", "project-alpha", updated_at="2026-02-01T10:00:00Z")
        self._seed_session("alpha-prior", "project-alpha", updated_at="2026-02-01T09:00:00Z")
        self._seed_session("beta-prior", "project-beta", updated_at="2026-02-01T11:00:00Z")

        upsert_working_memory(
            self.conn,
            session_id="alpha-current",
            current_goal="Ship alpha auth deploy safely",
            current_focus="Run alpha deploy checklist",
            next_step="Resume alpha deploy",
            status="ready_to_resume",
            updated_at="2026-02-01T10:00:00Z",
            details={},
            embedding=self.embed_fn("alpha auth deploy resume checklist"),
        )
        upsert_session_memory(
            self.conn,
            session_id="alpha-prior",
            title="Alpha auth handoff",
            summary="Stopped before the alpha deploy checklist.",
            left_off_at="Resume alpha auth deploy checklist.",
            updated_at="2026-02-01T09:00:00Z",
            details={"next_steps": ["Resume alpha deploy"]},
            embedding=self.embed_fn("alpha auth deploy resume checklist"),
        )
        upsert_session_memory(
            self.conn,
            session_id="beta-prior",
            title="Beta auth handoff",
            summary="Stopped before the beta deploy checklist.",
            left_off_at="Resume beta auth deploy checklist.",
            updated_at="2026-02-01T11:00:00Z",
            details={"next_steps": ["Resume beta deploy"]},
            embedding=self.embed_fn("beta auth deploy resume checklist"),
        )
        insert_episodic(
            self.conn,
            session_id="alpha-prior",
            title="Alpha deploy decision",
            abstract="Keep the alpha auth deploy behind the checklist.",
            happened_at="2026-02-01T09:00:00Z",
            details={"decisions": ["Run alpha deploy checklist"], "outcomes": ["Ready to resume"]},
            embedding=self.embed_fn("alpha auth deploy checklist"),
        )
        insert_episodic(
            self.conn,
            session_id="beta-prior",
            title="Beta deploy decision",
            abstract="Keep the beta auth deploy behind the checklist.",
            happened_at="2026-02-01T11:00:00Z",
            details={"decisions": ["Run beta deploy checklist"], "outcomes": ["Ready to resume"]},
            embedding=self.embed_fn("beta auth deploy checklist"),
        )

        context = retrieve_wake_up_context(
            self.conn,
            "resume auth deploy checklist",
            include_working_memory=True,
            session_id="alpha-current",
            project_context=self._project_context("project-alpha"),
            embed_fn=self.embed_fn,
        )

        self.assertEqual(context.working_mem["session_id"], "alpha-current")
        self.assertEqual([row["session_id"] for row in context.session_memory], ["alpha-prior"])
        self.assertEqual([row["session_id"] for row in context.episodic], ["alpha-prior"])
        self.assertNotIn("beta-prior", [row["session_id"] for row in context.session_memory])
        self.assertNotIn("beta-prior", [row["session_id"] for row in context.episodic])

    def test_project_facts_are_scoped_but_user_facts_remain_global(self):
        self._seed_session("alpha-facts", "project-alpha")
        self._seed_session("beta-facts", "project-beta")

        alpha_fact_id = insert_fact(
            self.conn,
            entity="project",
            attribute="database_engine",
            value="sqlite",
            semantic_content="Alpha billing uses sqlite as its database engine.",
            session_id="alpha-facts",
        )
        beta_fact_id = insert_fact(
            self.conn,
            entity="project",
            attribute="database_engine",
            value="aurora",
            semantic_content="Beta billing uses aurora as its database engine.",
            session_id="beta-facts",
        )
        timezone_fact_id = insert_fact(
            self.conn,
            entity="user",
            attribute="timezone",
            value="EST",
            semantic_content="The user timezone is EST.",
            session_id="beta-facts",
        )
        self.conn.execute("UPDATE facts SET embedding = ? WHERE id = ?", (pack_vector(self.embed_fn("alpha database engine")), alpha_fact_id))
        self.conn.execute("UPDATE facts SET embedding = ? WHERE id = ?", (pack_vector(self.embed_fn("beta database engine aurora")), beta_fact_id))
        self.conn.execute("UPDATE facts SET embedding = ? WHERE id = ?", (pack_vector(self.embed_fn("timezone")), timezone_fact_id))
        self.conn.commit()

        project_context = self._project_context("project-alpha")
        project_query = retrieve_wake_up_context(
            self.conn,
            "what database engine does alpha use",
            include_working_memory=False,
            project_context=project_context,
            embed_fn=self.embed_fn,
        )
        timezone_query = retrieve_wake_up_context(
            self.conn,
            "what is my timezone",
            include_working_memory=False,
            project_context=project_context,
            embed_fn=self.embed_fn,
        )

        self.assertEqual([row["content"] for row in project_query.facts], ["project.database_engine = sqlite"])
        self.assertEqual([row["fact_scope"] for row in project_query.facts], ["project"])
        self.assertEqual([row["content"] for row in timezone_query.facts], ["user.timezone = EST"])
        self.assertEqual([row["fact_scope"] for row in timezone_query.facts], ["global"])

    def test_uncertain_facts_default_to_project_scope(self):
        self._seed_session("alpha-uncertain", "project-alpha")
        self._seed_session("beta-uncertain", "project-beta")

        alpha_fact_id = insert_fact(
            self.conn,
            entity="team",
            attribute="owner",
            value="platform",
            semantic_content="The alpha database owner is the platform team.",
            session_id="alpha-uncertain",
        )
        beta_fact_id = insert_fact(
            self.conn,
            entity="team",
            attribute="owner",
            value="reliability",
            semantic_content="The beta database owner is the reliability team.",
            session_id="beta-uncertain",
        )
        self.conn.execute("UPDATE facts SET embedding = ? WHERE id = ?", (pack_vector(self.embed_fn("alpha database owner")), alpha_fact_id))
        self.conn.execute("UPDATE facts SET embedding = ? WHERE id = ?", (pack_vector(self.embed_fn("beta database owner")), beta_fact_id))
        self.conn.commit()

        context = retrieve_wake_up_context(
            self.conn,
            "who owns the alpha database",
            include_working_memory=False,
            project_context=self._project_context("project-alpha"),
            embed_fn=self.embed_fn,
        )

        self.assertEqual([row["content"] for row in context.facts], ["team.owner = platform"])
        self.assertEqual([row["fact_scope"] for row in context.facts], ["project"])

    def test_same_project_procedural_is_ranked_ahead_of_cross_project_procedural(self):
        self._seed_session("alpha-proc", "project-alpha", updated_at="2026-02-01T10:00:00Z")
        self._seed_session("beta-proc", "project-beta", updated_at="2026-02-01T11:00:00Z")

        insert_procedural(
            self.conn,
            session_id="alpha-proc",
            title="Alpha deploy runbook",
            summary="Deploy the alpha auth service.",
            updated_at="2026-02-01T10:00:00Z",
            details={"steps": ["Deploy alpha"]},
            embedding=self.embed_fn("auth deploy checklist"),
        )
        insert_procedural(
            self.conn,
            session_id="beta-proc",
            title="Beta deploy runbook",
            summary="Deploy the beta auth service.",
            updated_at="2026-02-01T11:00:00Z",
            details={"steps": ["Deploy beta"]},
            embedding=self.embed_fn("auth deploy checklist"),
        )

        context = retrieve_wake_up_context(
            self.conn,
            "deploy auth service",
            include_working_memory=False,
            project_context=self._project_context("project-alpha"),
            embed_fn=self.embed_fn,
        )

        self.assertEqual([row["session_id"] for row in context.procedural], ["alpha-proc", "beta-proc"])
        self.assertEqual(context.procedural[0]["title"], "Alpha deploy runbook")

    def test_keyword_only_queries_surface_typed_memories_without_semantic_signal(self):
        self._seed_session("alpha-current", "project-alpha", updated_at="2026-02-01T10:00:00Z")
        self._seed_session("alpha-prior", "project-alpha", updated_at="2026-02-01T09:00:00Z")
        self._seed_session("beta-prior", "project-beta", updated_at="2026-02-01T11:00:00Z")

        upsert_session_memory(
            self.conn,
            session_id="alpha-prior",
            title="Alpha handoff",
            summary="Paused after the deploy prep.",
            left_off_at="Waiting on the next session.",
            updated_at="2026-02-01T09:00:00Z",
            details={"next_steps": ["Resume after lunarflag approval"]},
            embedding=self.embed_fn("alpha auth deploy"),
        )
        upsert_session_memory(
            self.conn,
            session_id="beta-prior",
            title="Beta handoff",
            summary="Paused after the deploy prep.",
            left_off_at="Waiting on the next session.",
            updated_at="2026-02-01T11:00:00Z",
            details={"next_steps": ["Resume after sunflag approval"]},
            embedding=self.embed_fn("beta auth deploy"),
        )
        insert_episodic(
            self.conn,
            session_id="alpha-prior",
            title="Alpha release decision",
            abstract="Captured the deploy decision.",
            happened_at="2026-02-01T09:00:00Z",
            details={"decisions": ["Enable frostbyte gate before rollout"]},
            embedding=self.embed_fn("alpha auth deploy"),
        )
        insert_episodic(
            self.conn,
            session_id="beta-prior",
            title="Beta release decision",
            abstract="Captured the deploy decision.",
            happened_at="2026-02-01T11:00:00Z",
            details={"decisions": ["Enable embergate before rollout"]},
            embedding=self.embed_fn("beta auth deploy"),
        )
        insert_procedural(
            self.conn,
            session_id="alpha-prior",
            title="Alpha rollout runbook",
            summary="Run the rollout safely.",
            updated_at="2026-02-01T09:00:00Z",
            details={"steps": ["Run quartzsync before cutover"]},
            embedding=self.embed_fn("alpha auth deploy"),
        )

        context = retrieve_wake_up_context(
            self.conn,
            "lunarflag frostbyte quartzsync",
            include_working_memory=False,
            session_id="alpha-current",
            project_context=self._project_context("project-alpha"),
            embed_fn=self.embed_fn,
        )

        self.assertEqual([row["session_id"] for row in context.session_memory], ["alpha-prior"])
        self.assertEqual([row["session_id"] for row in context.episodic], ["alpha-prior"])
        self.assertEqual([row["title"] for row in context.procedural], ["Alpha rollout runbook"])
        self.assertTrue(context.session_memory[0].get("keyword_hit"))
        self.assertTrue(context.episodic[0].get("keyword_hit"))
        self.assertTrue(context.procedural[0].get("keyword_hit"))

    def test_session_fts_expands_matching_sessions_into_typed_memory_candidates(self):
        self._seed_session("alpha-current", "project-alpha", updated_at="2026-02-01T10:00:00Z")
        upsert_session(
            self.conn,
            "alpha-prior",
            "claude",
            [
                {"role": "user", "content": "Remember the starlock blocker for the rollout."},
                {"role": "assistant", "content": "We paused after investigating it."},
            ],
            "2026-02-01T09:00:00Z",
            "2026-02-01T09:30:00Z",
            project_context=self._project_context("project-alpha"),
        )

        upsert_session_memory(
            self.conn,
            session_id="alpha-prior",
            title="Alpha handoff",
            summary="Paused after the deploy prep.",
            left_off_at="Waiting for approval.",
            updated_at="2026-02-01T09:30:00Z",
            details={"next_steps": ["Resume rollout review"]},
            embedding=self.embed_fn("alpha deploy resume"),
        )
        insert_episodic(
            self.conn,
            session_id="alpha-prior",
            title="Alpha rollout checkpoint",
            abstract="Captured the current rollout status.",
            happened_at="2026-02-01T09:15:00Z",
            details={"decisions": ["Pause rollout"], "outcomes": ["Waiting for approval"]},
            embedding=self.embed_fn("alpha deploy resume"),
        )
        insert_procedural(
            self.conn,
            session_id="alpha-prior",
            title="Alpha rollout runbook",
            summary="Run the rollout safely.",
            updated_at="2026-02-01T09:20:00Z",
            details={"steps": ["Resume rollout review"]},
            embedding=self.embed_fn("alpha deploy resume"),
        )

        context = retrieve_wake_up_context(
            self.conn,
            "starlock",
            include_working_memory=False,
            session_id="alpha-current",
            project_context=self._project_context("project-alpha"),
            embed_fn=self.embed_fn,
        )

        self.assertEqual([row["session_id"] for row in context.session_memory], ["alpha-prior"])
        self.assertEqual([row["session_id"] for row in context.episodic], ["alpha-prior"])
        self.assertEqual([row["session_id"] for row in context.procedural], ["alpha-prior"])
        self.assertTrue(context.session_memory[0].get("session_hit"))
        self.assertTrue(context.episodic[0].get("session_hit"))
        self.assertTrue(context.procedural[0].get("session_hit"))
        self.assertFalse(context.session_memory[0].get("keyword_hit", False))
        self.assertFalse(context.episodic[0].get("keyword_hit", False))
        self.assertFalse(context.procedural[0].get("keyword_hit", False))

    def test_rrf_fusion_promotes_multi_lane_typed_memory_candidates(self):
        self._seed_session("alpha-current", "project-alpha", updated_at="2026-02-01T10:00:00Z")
        upsert_session(
            self.conn,
            "alpha-hybrid",
            "claude",
            [{"role": "user", "content": "Investigate the starlock blocker before rollout."}],
            "2026-02-01T09:00:00Z",
            "2026-02-01T09:15:00Z",
            project_context=self._project_context("project-alpha"),
        )
        upsert_session(
            self.conn,
            "alpha-semantic",
            "claude",
            [{"role": "user", "content": "General rollout follow-up without the keyword."}],
            "2026-02-01T08:00:00Z",
            "2026-02-01T08:15:00Z",
            project_context=self._project_context("project-alpha"),
        )

        upsert_session_memory(
            self.conn,
            session_id="alpha-hybrid",
            title="Checklist handoff",
            summary="Paused after prep work.",
            left_off_at="Waiting for approval.",
            updated_at="2026-02-01T09:15:00Z",
            details={"next_steps": ["Review checklist"]},
            embedding=self.embed_fn("deploy"),
        )
        upsert_session_memory(
            self.conn,
            session_id="alpha-semantic",
            title="Opaque handoff",
            summary="Paused after prep work.",
            left_off_at="Waiting for approval.",
            updated_at="2026-02-01T08:15:00Z",
            details={"next_steps": ["Wait for approval"]},
            embedding=self.embed_fn("deploy checklist auth"),
        )

        insert_episodic(
            self.conn,
            session_id="alpha-hybrid",
            title="Checklist checkpoint",
            abstract="Captured rollout status.",
            happened_at="2026-02-01T09:10:00Z",
            details={"decisions": ["Review checklist"], "outcomes": ["Paused rollout"]},
            embedding=self.embed_fn("deploy"),
        )
        insert_episodic(
            self.conn,
            session_id="alpha-semantic",
            title="Opaque checkpoint",
            abstract="Captured rollout status.",
            happened_at="2026-02-01T08:10:00Z",
            details={"decisions": ["Wait for approval"], "outcomes": ["Paused rollout"]},
            embedding=self.embed_fn("deploy checklist auth"),
        )

        insert_procedural(
            self.conn,
            session_id="alpha-hybrid",
            title="Checklist runbook",
            summary="Use before rollout.",
            updated_at="2026-02-01T09:12:00Z",
            details={"steps": ["Review checklist"]},
            embedding=self.embed_fn("deploy"),
        )
        insert_procedural(
            self.conn,
            session_id="alpha-semantic",
            title="Opaque runbook",
            summary="Use before rollout.",
            updated_at="2026-02-01T08:12:00Z",
            details={"steps": ["Wait for approval"]},
            embedding=self.embed_fn("deploy checklist auth"),
        )

        context = retrieve_wake_up_context(
            self.conn,
            "deploy checklist starlock",
            include_working_memory=False,
            session_id="alpha-current",
            project_context=self._project_context("project-alpha"),
            embed_fn=self.embed_fn,
        )

        self.assertEqual(context.session_memory[0]["session_id"], "alpha-hybrid")
        self.assertEqual(context.episodic[0]["session_id"], "alpha-hybrid")
        self.assertEqual(context.procedural[0]["session_id"], "alpha-hybrid")
        self.assertEqual(set(context.session_memory[0]["rrf_sources"]), {"semantic", "keyword", "session"})
        self.assertEqual(set(context.episodic[0]["rrf_sources"]), {"semantic", "keyword", "session"})
        self.assertEqual(set(context.procedural[0]["rrf_sources"]), {"semantic", "keyword", "session"})
        self.assertGreater(context.session_memory[0]["rrf_score"], context.session_memory[1]["rrf_score"])
        self.assertGreater(context.episodic[0]["rrf_score"], context.episodic[1]["rrf_score"])
        self.assertGreater(context.procedural[0]["rrf_score"], context.procedural[1]["rrf_score"])
        self.assertNotIn("search_text", context.session_memory[0])
        self.assertNotIn("snippet", context.session_memory[0])

    def test_session_fts_lane_returns_nothing_when_matching_session_has_no_typed_memory(self):
        self._seed_session("alpha-current", "project-alpha", updated_at="2026-02-01T10:00:00Z")
        upsert_session(
            self.conn,
            "alpha-empty",
            "claude",
            [{"role": "user", "content": "Investigate starlock in the transcript only."}],
            "2026-02-01T08:00:00Z",
            "2026-02-01T08:05:00Z",
            project_context=self._project_context("project-alpha"),
        )

        context = retrieve_wake_up_context(
            self.conn,
            "starlock",
            include_working_memory=False,
            session_id="alpha-current",
            project_context=self._project_context("project-alpha"),
            embed_fn=self.embed_fn,
        )

        self.assertEqual(context.session_memory, [])
        self.assertEqual(context.episodic, [])
        self.assertEqual(context.procedural, [])


if __name__ == "__main__":
    unittest.main()
