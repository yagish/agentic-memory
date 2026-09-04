#!/usr/bin/env python3
"""Debug one fact-extraction case end to end.

Examples:
  python3 scripts/debug_fact_extraction.py --case natural_location
  MEMORY_DEBUG_FACTS=1 python3 scripts/debug_fact_extraction.py --case user_profile
  python3 scripts/debug_fact_extraction.py --session-file /tmp/session.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.facts import (
    _FACTS_DEBUG_LOG_PATH,
    build_fact_extraction_prompt,
    facts_to_semantic_core,
    parse_extracted_facts,
)
from memory.inference import GenerationRequest, generate_text


_FIXTURES_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "facts"


def _load_case(case_name: str) -> tuple[str, Path | None]:
    session_path = _FIXTURES_ROOT / "sessions" / f"{case_name}.txt"
    expected_path = _FIXTURES_ROOT / "expected" / f"{case_name}.json"
    return session_path.read_text(), expected_path if expected_path.exists() else None


def _maybe_write_debug_file(*, case_label: str, model: str, session_text: str, prompt: str, raw_output: str, parsed_facts: list[dict], semantic_core: list[dict], expected: list[dict] | None) -> None:
    if str(os.environ.get("MEMORY_DEBUG_FACTS", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        return

    sections = [
        f"=== SCRIPT DEBUG: case ===\n{case_label}\n",
        f"=== SCRIPT DEBUG: model ===\n{model}\n",
        f"=== SCRIPT DEBUG: session_text ===\n{session_text}\n",
        f"=== SCRIPT DEBUG: prompt ===\n{prompt}\n",
        f"=== SCRIPT DEBUG: raw_model_output ===\n{raw_output}\n",
        "=== SCRIPT DEBUG: parsed_facts ===\n" + json.dumps(parsed_facts, indent=2) + "\n",
        "=== SCRIPT DEBUG: semantic_core ===\n" + json.dumps(semantic_core, indent=2) + "\n",
    ]
    if expected is not None:
        sections.append("=== SCRIPT DEBUG: expected ===\n" + json.dumps(expected, indent=2) + "\n")

    debug_path = Path(_FACTS_DEBUG_LOG_PATH)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_path, "a") as f:
        f.write("\n" + "\n".join(sections))


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug fact extraction for one session")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", help="Fixture case basename under tests/fixtures/facts")
    group.add_argument("--session-file", help="Path to a session text file")
    parser.add_argument("--model", help="Override model name; defaults to MEMORY_OLLAMA_MODEL/current default")
    parser.add_argument("--timeout", type=int, default=180, help="Ollama timeout seconds")
    args = parser.parse_args()

    if args.case:
        session_text, expected_path = _load_case(args.case)
        expected = json.loads(expected_path.read_text()) if expected_path is not None else None
        case_label = args.case
    else:
        session_text = Path(args.session_file).read_text()
        expected = None
        case_label = args.session_file

    prompt = build_fact_extraction_prompt(session_text)
    result = generate_text(
        GenerationRequest(
            prompt=prompt,
            model=args.model,
            timeout_seconds=args.timeout,
            temperature=0.0,
        )
    )
    facts = parse_extracted_facts(result.text)

    parsed_facts = [fact.model_dump(mode="json") for fact in facts]
    semantic_core = facts_to_semantic_core(facts)

    _maybe_write_debug_file(
        case_label=case_label,
        model=result.model,
        session_text=session_text,
        prompt=prompt,
        raw_output=result.text,
        parsed_facts=parsed_facts,
        semantic_core=semantic_core,
        expected=expected,
    )

    print(f"=== CASE ===\n{case_label}")
    print(f"\n=== MODEL ===\n{result.model}")
    print(f"\n=== SESSION TEXT ===\n{session_text}")
    print(f"\n=== PROMPT ===\n{prompt}")
    print(f"\n=== RAW MODEL OUTPUT ===\n{result.text}")
    print("\n=== PARSED FACTS ===\n" + json.dumps(parsed_facts, indent=2))
    print("\n=== SEMANTIC CORE ===\n" + json.dumps(semantic_core, indent=2))
    if expected is not None:
        print("\n=== EXPECTED ===\n" + json.dumps(expected, indent=2))
        print(
            "\n=== MATCH ===\n"
            + str(semantic_core == facts_to_semantic_core(expected))
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
