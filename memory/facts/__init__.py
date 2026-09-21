"""Facts subpackage — LLM extraction, text helpers, persistence, and rendering."""

from memory.facts.extractor import (
    GenerationFn,
    build_fact_extraction_prompt,
    discover_fact_fixture_cases,
    extract_facts_from_session_text,
    facts_to_semantic_core,
    log_fact_event,
    normalize_extracted_facts,
    parse_extracted_facts,
)
from memory.facts.renderer import (
    build_fact_render_prompt,
    render_fact_answer,
)
from memory.facts.repository import (
    build_fact_content,
    build_fact_semantic_content,
    build_fact_tags,
    list_session_facts,
    save_extracted_facts,
)
from memory.facts.scope import (
    classify_fact_scope,
    classify_fact_scope_by_rules,
    ensure_scope_tag,
)
from memory.facts.text import (
    build_canonical_fact_content,
    build_canonical_fact_from_parts,
    build_semantic_fact_text,
    generate_semantic_fact_text,
)

__all__ = [
    "GenerationFn",
    "build_canonical_fact_content",
    "build_canonical_fact_from_parts",
    "build_fact_content",
    "build_fact_extraction_prompt",
    "build_fact_render_prompt",
    "build_fact_semantic_content",
    "build_fact_tags",
    "build_semantic_fact_text",
    "classify_fact_scope",
    "classify_fact_scope_by_rules",
    "discover_fact_fixture_cases",
    "extract_facts_from_session_text",
    "facts_to_semantic_core",
    "ensure_scope_tag",
    "generate_semantic_fact_text",
    "list_session_facts",
    "log_fact_event",
    "normalize_extracted_facts",
    "parse_extracted_facts",
    "render_fact_answer",
    "save_extracted_facts",
]
