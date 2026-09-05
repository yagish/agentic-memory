"""Canonical validated memory contracts.

Start narrow with FactMemory so the facts pipeline can be built test-first,
then extend this module with additional memory types later.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _normalize_identifier(value: str) -> str:
    """Normalize entity/attribute keys to one stable comparison format.

    Example:
    - "User Name" -> "user_name"
    - "default-branch" -> "default_branch"
    """
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _strip_text(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip()


def _normalize_text_list(value: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ()

    deduped: list[str] = []
    for item in value:
        normalized = str(item).strip()
        if not normalized or normalized in deduped:
            continue
        deduped.append(normalized)
    return tuple(deduped)


class ExtractedFact(BaseModel):
    """One fact emitted by the extractor before persistence/provenance is added.

    This is the schema for raw extraction output coming back from the LLM.
    It intentionally stays smaller than FactMemory because persistence-specific
    fields like session ids and validity timestamps are added later.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = None
    source_quote: str | None = None

    @field_validator("entity", "attribute", mode="before")
    @classmethod
    def _normalize_keys(cls, value: str) -> str:
        # Make fixture comparisons stable even if the model varies casing or separators.
        return _normalize_identifier(str(value))

    @field_validator("value", "evidence", "source_quote", mode="before")
    @classmethod
    def _strip_text_fields(cls, value: str | None) -> str | None:
        # Trim harmless outer whitespace but preserve the actual semantic text.
        return _strip_text(value)


class FactMemory(BaseModel):
    """A durable structured fact extracted from conversation history.

    This is the storage-facing contract. Compared with ExtractedFact, it carries
    provenance and temporal fields required for persistence and later retrieval.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source_session_id: str = Field(min_length=1)
    valid_from: datetime
    valid_to: datetime | None = None

    @field_validator("entity", "attribute", mode="before")
    @classmethod
    def _normalize_fact_memory_keys(cls, value: str) -> str:
        # Persist normalized keys so retrieval/update logic can rely on one shape.
        return _normalize_identifier(str(value))

    @field_validator("value", "source_session_id", mode="before")
    @classmethod
    def _strip_fact_memory_text_fields(cls, value: str) -> str:
        return str(value).strip()


class ExtractedEpisode(BaseModel):
    """One episodic memory emitted by the extractor before persistence fields.

    This is the extractor-facing contract for event/discussion/decision memory.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1, max_length=200)
    abstract: str = Field(min_length=1, max_length=600)
    participants: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    follow_ups: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source_quote: str | None = None

    @field_validator("title", "abstract", "source_quote", mode="before")
    @classmethod
    def _strip_episode_text_fields(cls, value: str | None) -> str | None:
        return _strip_text(value)

    @field_validator("participants", "decisions", "outcomes", "follow_ups", mode="before")
    @classmethod
    def _normalize_episode_lists(cls, value: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        return _normalize_text_list(value)


class EpisodicMemory(BaseModel):
    """A durable structured episodic memory stored with provenance.

    Compared with ExtractedEpisode, this adds the source session and timestamp.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1, max_length=200)
    abstract: str = Field(min_length=1, max_length=600)
    participants: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    follow_ups: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source_quote: str | None = None
    source_session_id: str = Field(min_length=1)
    happened_at: datetime

    @field_validator("title", "abstract", "source_quote", "source_session_id", mode="before")
    @classmethod
    def _strip_episodic_memory_text_fields(cls, value: str | None) -> str | None:
        return _strip_text(value)

    @field_validator("participants", "decisions", "outcomes", "follow_ups", mode="before")
    @classmethod
    def _normalize_episodic_memory_lists(cls, value: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        return _normalize_text_list(value)
