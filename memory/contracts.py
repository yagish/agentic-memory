"""Canonical validated memory contracts.

Start narrow with FactMemory so the facts pipeline can be built test-first,
then extend this module with additional memory types later.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FactMemory(BaseModel):
    """A durable structured fact extracted from conversation history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source_session_id: str = Field(min_length=1)
    valid_from: datetime
    valid_to: datetime | None = None
