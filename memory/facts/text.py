"""Shared fact text helpers."""

from __future__ import annotations

import re

from memory.llm.inference import GenerationRequest, InferenceError, generate_text


_SEMANTIC_TEXT_TIMEOUT_SECONDS = 20



def build_canonical_fact_content(entity: str, attribute: str, value: str) -> str:
    return f"{entity.strip().lower()}.{attribute.strip().lower()} = {value.strip()}"



def _humanize_identifier(value: str) -> str:
    return value.strip().replace("_", " ")



def _split_sentences(text: str) -> list[str]:
    collapsed = " ".join((text or "").split())
    if not collapsed:
        return []

    sentences: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"(?<=[.!?])\s+", collapsed):
        sentence = part.strip().strip('"').strip("'")
        if not sentence:
            continue
        lowered = sentence.lower()
        if any(marker in lowered for marker in ("entity=", "attribute=", "value=", "requirements:", "examples:")):
            continue
        if "->" in sentence:
            continue
        if sentence.startswith(("- ", "* ")):
            continue
        if sentence not in seen:
            sentences.append(sentence)
            seen.add(sentence)
    return sentences



def _normalize_semantic_text(text: str) -> str:
    return " ".join(_split_sentences(text)[:3]).strip()



def _is_valid_semantic_text(text: str, value: str) -> bool:
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return False
    exact_value = value.strip()
    if not exact_value:
        return False
    if text.count(exact_value) < 2:
        return False

    for index, sentence in enumerate(sentences):
        if "?" not in sentence:
            continue
        if exact_value in sentence:
            return True
        if index + 1 < len(sentences) and exact_value in sentences[index + 1]:
            return True
    return False



def build_semantic_fact_text(entity: str, attribute: str, value: str) -> str:
    entity_text = _humanize_identifier(entity.strip().lower())
    attribute_text = _humanize_identifier(attribute.strip().lower())
    value_text = value.strip()
    if entity_text.endswith("s"):
        owner = f"{entity_text}'"
    else:
        owner = f"{entity_text}'s"

    first_sentence = f"The {owner} {attribute_text} is {value_text}."
    if entity_text == "user":
        second_sentence = f"My {attribute_text} is {value_text}."
        third_sentence = f"What is my {attribute_text}? {value_text}."
    else:
        second_sentence = f"The {entity_text} {attribute_text} is {value_text}."
        third_sentence = f"What is the {owner} {attribute_text}? {value_text}."
    return f"{first_sentence} {second_sentence} {third_sentence}"



def _build_semantic_fact_prompt(entity: str, attribute: str, value: str, source_text: str | None = None) -> str:
    entity = entity.strip().lower()
    attribute = attribute.strip().lower()
    value = value.strip()
    source_block = ""
    if source_text and source_text.strip():
        source_block = f"\nConversation wording for tone only:\n{source_text.strip()}\n"

    return (
        "You are writing semantic retrieval text for one structured memory fact.\n\n"
        f"Fact:\n- entity: {entity}\n- attribute: {attribute}\n- value: {value}\n"
        f"{source_block}\n"
        "Return only plain text.\n"
        "Write 2 or 3 short sentences that help semantic search retrieve this fact from natural-language user prompts.\n"
        "Requirements:\n"
        "- Preserve the fact exactly.\n"
        "- Do not add any new facts, guesses, or explanations.\n"
        "- Use natural wording, not schema jargon.\n"
        "- Use the exact value string verbatim every time you mention the value. Do not expand, shorten, translate, or paraphrase it.\n"
        "- Mention the exact value in at least two sentences.\n"
        "- If the entity is user, first-person wording is allowed.\n"
        "- Include at least one likely natural user phrasing or question whose answer is the value.\n"
        "- If more natural wording exists than the literal attribute name, prefer the natural wording.\n"
        "- Do not echo these instructions. Do not repeat the input fields. Do not include example text.\n"
        "- No bullets, no JSON, no markdown, no commentary.\n"
    )



def generate_semantic_fact_text(
    entity: str,
    attribute: str,
    value: str,
    *,
    source_text: str | None = None,
    model: str | None = None,
    timeout_seconds: int = _SEMANTIC_TEXT_TIMEOUT_SECONDS,
) -> str:
    fallback = build_semantic_fact_text(entity, attribute, value)
    try:
        result = generate_text(
            GenerationRequest(
                prompt=_build_semantic_fact_prompt(entity, attribute, value, source_text=source_text),
                model=model,
                timeout_seconds=timeout_seconds,
                temperature=0.0,
            )
        )
    except InferenceError:
        return fallback

    semantic_text = _normalize_semantic_text(result.text)
    if not semantic_text:
        return fallback
    if not _is_valid_semantic_text(semantic_text, value):
        return fallback
    return semantic_text



def build_canonical_fact_from_parts(entity: str | None, attribute: str | None, value: str | None) -> str:
    if not entity or not attribute or value is None:
        return ""
    return build_canonical_fact_content(entity, attribute, value)
