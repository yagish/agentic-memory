import unittest

from memory.inference import GenerationResult
from memory.procedural import (
    build_procedural_extraction_prompt,
    extract_procedure_from_session_text,
    parse_extracted_procedure,
    procedure_to_semantic_core,
)


class TestProceduralExtractionHelpers(unittest.TestCase):
    def test_extract_procedure_parses_wrapped_json_and_normalizes_lists(self):
        def fake_generate(_request):
            return GenerationResult(
                text='''Here is the procedure:\n{\n  "title": " Web deploy workflow ",\n  "summary": " Deploy the web service through staging before production. ",\n  "steps": [" Build the Docker image ", "Run alembic upgrade", "Build the Docker image"],\n  "trigger_phrases": [" how do i deploy the web service ", "deploy workflow", "deploy workflow"],\n  "tools": [" Docker ", "alembic", "Docker"],\n  "confidence": 0.93\n}\n''',
                model="fake-model",
            )

        procedure = extract_procedure_from_session_text(
            "User: The deploy workflow is build the Docker image, run alembic upgrade, then roll out in staging before production.",
            generate_fn=fake_generate,
            source="test-harness",
            session_id="session-123",
        )

        self.assertEqual(
            procedure_to_semantic_core(procedure),
            {
                "title": "Web deploy workflow",
                "summary": "Deploy the web service through staging before production.",
                "steps": ["Build the Docker image", "Run alembic upgrade"],
                "trigger_phrases": ["how do i deploy the web service", "deploy workflow"],
                "tools": ["Docker", "alembic", "staging", "production"],
            },
        )
        self.assertEqual(procedure.confidence, 0.93)

    def test_empty_json_means_no_procedural_memory(self):
        self.assertIsNone(parse_extracted_procedure("{}"))

    def test_extract_procedure_returns_none_when_model_finds_no_procedure(self):
        def fake_generate(_request):
            return GenerationResult(text="{}", model="fake-model")

        procedure = extract_procedure_from_session_text(
            "User: We fixed the auth bug today.",
            generate_fn=fake_generate,
            source="test-harness",
            session_id="session-123",
        )

        self.assertIsNone(procedure)

    def test_parse_repairs_unescaped_quotes_inside_json_strings(self):
        procedure = parse_extracted_procedure(
            '''{
  "title": "Admin-web flaky test triage workflow",
  "summary": "Use this flaky test triage workflow for admin-web.",
  "steps": ["Rerun the suite with PYTEST_ADDOPTS="-x"", "File the Jira ticket"],
  "trigger_phrases": ["flaky test triage workflow for admin-web"],
  "tools": ["PYTEST_ADDOPTS="-x"", "Jira"],
  "confidence": 0.95
}'''
        )

        self.assertEqual(procedure.steps[0], 'Rerun the suite with PYTEST_ADDOPTS="-x"')
        self.assertEqual(procedure.tools[0], 'PYTEST_ADDOPTS="-x"')

    def test_parse_backfills_stage_literals_into_tools(self):
        procedure = parse_extracted_procedure(
            '''{
  "title": "Docs-site release cutoff checklist",
  "summary": "Use this release cutoff checklist for docs-site.",
  "steps": ["Run smoke tests in preview"],
  "trigger_phrases": ["release cutoff checklist for docs-site"],
  "tools": ["release/delta", "Slack"],
  "confidence": 0.95
}'''
        )

        self.assertIn("preview", procedure.tools)

    def test_prompt_includes_transcript_and_output_shape(self):
        prompt = build_procedural_extraction_prompt(
            "User: Deploy by building the image, then run alembic upgrade.\nAssistant: Verify staging before production."
        )

        self.assertIn("User: Deploy by building the image, then run alembic upgrade.", prompt)
        self.assertIn("Assistant: Verify staging before production.", prompt)
        self.assertIn('"steps": ["ordered repeatable steps"]', prompt)
        self.assertIn("Return only the JSON object.", prompt)

    def test_prompt_pushes_repeatable_project_specific_workflows(self):
        prompt = build_procedural_extraction_prompt(
            "User: When the checkout flag misbehaves, disable it, clear the edge cache, then retry with an employee account."
        )

        self.assertIn("durable repeatable how-to, workflow, checklist, or operating pattern", prompt)
        self.assertIn("Prefer project-specific workflows over generic advice.", prompt)
        self.assertIn("Do not change the spelling or casing of env vars", prompt)
        self.assertIn("Do not invent missing steps, tools, or trigger phrases.", prompt)


if __name__ == "__main__":
    unittest.main()
