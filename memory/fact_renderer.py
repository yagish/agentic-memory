"""Deterministic rendering for retrieved facts.

Wake-up recall should stay model-free. This module turns canonical stored fact
strings such as ``user.name = Yagish`` into short direct answers without calling
Ollama.
"""

from __future__ import annotations

import os
import re


_RENDER_MAX_CHARS = int(os.environ.get("MEMORY_FACT_RENDER_MAX_CHARS", "240"))
_FACT_PATTERN = re.compile(r"^\s*([a-z0-9_]+)\.([a-z0-9_]+)\s*=\s*(.+?)\s*$", re.IGNORECASE)

_ATTRIBUTE_ALIASES = {
    "name": {"name", "called", "who"},
    "location": {"location", "live", "where", "from", "based"},
    "company": {"company", "work", "employer"},
    "role": {"role", "job", "title", "position"},
    "timezone": {"timezone", "zone", "time"},
    "editor": {"editor", "ide"},
    "shell": {"shell", "terminal"},
    "favorite_language": {"favorite", "language", "languages", "prefer", "preferred"},
    "preferred_language": {"preferred", "prefer", "language", "languages", "favorite"},
    "response_style": {"response", "style", "answers", "answer", "concise", "verbose", "brief"},
}


def _canonical_fallback(fact_contents: list[str]) -> str:
    """Render a deterministic non-LLM fallback from retrieved fact strings."""
    cleaned = [fact.strip() for fact in fact_contents if fact.strip()]
    if not cleaned:
        return "I don't have that fact stored."
    return "\n".join(cleaned)



def build_fact_render_prompt(*, user_prompt: str, fact_contents: list[str]) -> str:
    """Legacy helper retained for compatibility in tests/callers."""
    rendered_facts = "\n".join(f"- {fact.strip()}" for fact in fact_contents if fact.strip())
    return f"Question: {user_prompt}\nFacts:\n{rendered_facts}\nAnswer:"



def _normalize_rendered_answer(text: str) -> str:
    """Clean a renderer response into one short plain-text answer."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()

    cleaned = " ".join(line.strip() for line in cleaned.splitlines() if line.strip())
    return cleaned.strip()



def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))



def _humanize_identifier(value: str) -> str:
    return value.strip().replace("_", " ")



def _parse_fact(content: str) -> dict | None:
    match = _FACT_PATTERN.fullmatch(content)
    if not match:
        return None
    entity, attribute, value = match.groups()
    return {
        "entity": entity.strip().lower(),
        "attribute": attribute.strip().lower(),
        "value": value.strip(),
        "content": content.strip(),
    }



def _attribute_tokens(attribute: str) -> set[str]:
    humanized = _humanize_identifier(attribute)
    tokens = set(humanized.split())
    tokens |= _ATTRIBUTE_ALIASES.get(attribute, set())
    return {token for token in tokens if token}



def _fact_match_score(prompt: str, fact: dict) -> int:
    prompt_tokens = _tokenize(prompt)
    attribute = fact["attribute"]
    score = len(prompt_tokens & _attribute_tokens(attribute))

    if attribute == "location" and prompt_tokens & {"where", "live", "from"}:
        score += 2
    elif attribute == "name" and prompt_tokens & {"name", "who", "called"}:
        score += 2
    elif attribute == "company" and prompt_tokens & {"company", "work", "employer"}:
        score += 2
    elif attribute == "role" and prompt_tokens & {"role", "job", "title", "position"}:
        score += 2
    elif attribute == "timezone" and prompt_tokens & {"timezone", "zone", "time"}:
        score += 2
    elif attribute in {"favorite_language", "preferred_language"} and prompt_tokens & {"language", "languages", "favorite", "prefer", "preferred"}:
        score += 2

    if score > 0 and fact["entity"] == "user" and prompt_tokens & {"i", "me", "my"}:
        score += 1

    return score



def _render_user_fact(attribute: str, value: str) -> str:
    templates = {
        "name": f"Your name is {value}.",
        "location": f"You live in {value}.",
        "company": f"You work at {value}.",
        "role": f"Your role is {value}.",
        "timezone": f"Your timezone is {value}.",
        "editor": f"Your editor is {value}.",
        "shell": f"Your shell is {value}.",
        "favorite_language": f"Your favorite language is {value}.",
        "preferred_language": f"Your preferred language is {value}.",
        "response_style": f"Your preferred response style is {value}.",
    }
    if attribute in templates:
        return templates[attribute]
    return f"Your {_humanize_identifier(attribute)} is {value}."



def _render_fact_sentence(fact: dict) -> str:
    entity = fact["entity"]
    attribute = fact["attribute"]
    value = fact["value"]
    if entity == "user":
        return _render_user_fact(attribute, value)
    return f"The {_humanize_identifier(entity)} {_humanize_identifier(attribute)} is {value}."



def _compact_answer(sentences: list[str], *, max_chars: int) -> str:
    joined = " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()
    if len(joined) <= max_chars:
        return joined

    kept: list[str] = []
    for sentence in sentences:
        candidate = (" ".join(kept + [sentence])).strip()
        if len(candidate) <= max_chars:
            kept.append(sentence)
            continue
        if not kept:
            return sentence[:max_chars].rstrip()
        break
    return " ".join(kept).strip()



def render_fact_answer(user_prompt: str, fact_contents: list[str]) -> str:
    """Render a direct answer from canonical stored facts without an LLM.

    Returns an empty string when the prompt does not appear to directly ask for
    one of the retrieved facts.
    """
    parsed_facts = [parsed for parsed in (_parse_fact(content) for content in fact_contents) if parsed is not None]
    if not parsed_facts:
        return ""

    scored = [
        (index, _fact_match_score(user_prompt, fact), fact)
        for index, fact in enumerate(parsed_facts)
    ]
    matched = [(index, score, fact) for index, score, fact in scored if score > 0]
    if not matched:
        return ""

    matched.sort(key=lambda item: item[0])
    sentences: list[str] = []
    seen: set[str] = set()
    for _index, _score, fact in matched:
        sentence = _render_fact_sentence(fact)
        if sentence in seen:
            continue
        seen.add(sentence)
        sentences.append(sentence)

    return _compact_answer(sentences, max_chars=_RENDER_MAX_CHARS)
