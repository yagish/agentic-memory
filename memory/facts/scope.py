from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

GLOBAL_FACT_SCOPE = "global"
PROJECT_FACT_SCOPE = "project"
_SCOPE_TAG_PREFIX = "scope:"

_PROJECT_ENTITIES = {
    "app",
    "application",
    "codebase",
    "database",
    "environment",
    "package",
    "project",
    "repo",
    "repository",
    "service",
    "workspace",
}

_PROJECT_ATTRIBUTES = {
    "api_base",
    "api_url",
    "branch",
    "database_engine",
    "database_name",
    "default_branch",
    "deploy_target",
    "deployment_target",
    "entrypoint",
    "git_branch",
    "git_remote",
    "port",
    "project_id",
    "project_name",
    "repo_name",
    "repository_name",
    "root_path",
    "service_name",
}

_GLOBAL_USER_ATTRIBUTES = {
    "alias",
    "company",
    "editor",
    "email",
    "employer",
    "favorite_language",
    "first_name",
    "full_name",
    "github_username",
    "ide",
    "last_name",
    "location",
    "name",
    "occupation",
    "operating_system",
    "os",
    "package_manager",
    "preferred_language",
    "pronouns",
    "response_style",
    "role",
    "shell",
    "terminal",
    "timezone",
    "title",
}

_PROJECT_TEXT_MARKERS = (
    "this project",
    "this repo",
    "this repository",
    "this codebase",
    "this workspace",
    "for this project",
    "for this repo",
    "for this repository",
    "on this project",
    "in this repo",
    "in this repository",
    "repo root",
    "default branch",
    "git remote",
    "git branch",
)


def normalize_fact_scope(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", "-")
    if not normalized:
        return None
    if normalized in {"global", "portable", "user"}:
        return GLOBAL_FACT_SCOPE
    if normalized in {"project", "project-specific", "repo", "repository"}:
        return PROJECT_FACT_SCOPE
    return None


def scope_tag(scope: str) -> str:
    return f"{_SCOPE_TAG_PREFIX}{scope}"


def scope_from_tags(tags: Iterable[str] | None) -> str | None:
    if not tags:
        return None
    for tag in tags:
        normalized = str(tag).strip().lower()
        if normalized.startswith(_SCOPE_TAG_PREFIX):
            return normalize_fact_scope(normalized[len(_SCOPE_TAG_PREFIX):])
    return None


def _normalize_identifier(value: str | None) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _combined_fact_text(*parts: str | None) -> str:
    return " ".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())


def classify_fact_scope_by_rules(
    *,
    entity: str | None,
    attribute: str | None,
    value: str | None,
    semantic_content: str | None = None,
    source_quote: str | None = None,
) -> str | None:
    entity_key = _normalize_identifier(entity)
    attribute_key = _normalize_identifier(attribute)
    combined = _combined_fact_text(entity, attribute, value, semantic_content, source_quote)

    if combined.startswith(("repo.", "project.")):
        return PROJECT_FACT_SCOPE
    if combined.startswith("user."):
        return GLOBAL_FACT_SCOPE
    if entity_key in _PROJECT_ENTITIES:
        return PROJECT_FACT_SCOPE
    if attribute_key in _PROJECT_ATTRIBUTES:
        return PROJECT_FACT_SCOPE
    if any(marker in combined for marker in _PROJECT_TEXT_MARKERS):
        return PROJECT_FACT_SCOPE

    if entity_key == "user":
        if attribute_key in _GLOBAL_USER_ATTRIBUTES:
            return GLOBAL_FACT_SCOPE
        if attribute_key.startswith(("preferred_", "favorite_")):
            return GLOBAL_FACT_SCOPE
        if attribute_key.endswith(("_preference", "_style")):
            return GLOBAL_FACT_SCOPE

    return None


def classify_fact_scope(
    *,
    entity: str | None,
    attribute: str | None,
    value: str | None,
    tags: Iterable[str] | None = None,
    semantic_content: str | None = None,
    source_quote: str | None = None,
    llm_scope: str | None = None,
) -> str:
    tagged_scope = scope_from_tags(tags)
    if tagged_scope:
        return tagged_scope

    rule_scope = classify_fact_scope_by_rules(
        entity=entity,
        attribute=attribute,
        value=value,
        semantic_content=semantic_content,
        source_quote=source_quote,
    )
    if rule_scope:
        return rule_scope

    hinted_scope = normalize_fact_scope(llm_scope)
    if hinted_scope:
        return hinted_scope

    return PROJECT_FACT_SCOPE


def classify_fact_scope_from_mapping(row: Mapping[str, Any]) -> str:
    return classify_fact_scope(
        entity=row.get("entity"),
        attribute=row.get("attribute"),
        value=row.get("value"),
        tags=row.get("tags"),
        semantic_content=row.get("semantic_content") or row.get("content"),
        source_quote=row.get("source_quote"),
        llm_scope=row.get("scope") or row.get("fact_scope"),
    )


def ensure_scope_tag(
    tags: Iterable[str] | None,
    *,
    entity: str | None,
    attribute: str | None,
    value: str | None,
    semantic_content: str | None = None,
    source_quote: str | None = None,
    llm_scope: str | None = None,
) -> list[str]:
    normalized_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    final_scope = classify_fact_scope(
        entity=entity,
        attribute=attribute,
        value=value,
        tags=normalized_tags,
        semantic_content=semantic_content,
        source_quote=source_quote,
        llm_scope=llm_scope,
    )
    scope_marker = scope_tag(final_scope)
    if not any(str(tag).strip().lower() == scope_marker for tag in normalized_tags):
        normalized_tags.append(scope_marker)
    return normalized_tags
