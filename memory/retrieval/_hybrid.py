from __future__ import annotations

from collections.abc import Iterable

from memory.retrieval._models import MemoryRow, RRF_K


def _merge_lane_metadata(current: MemoryRow, incoming: MemoryRow) -> None:
    similarity = max(float(current.get("similarity", 0.0) or 0.0), float(incoming.get("similarity", 0.0) or 0.0))
    if similarity > 0.0:
        current["similarity"] = round(similarity, 4)

    if incoming.get("keyword_hit"):
        current["keyword_hit"] = True
        current["keyword_score"] = max(
            float(current.get("keyword_score", 0.0) or 0.0),
            float(incoming.get("keyword_score", 0.0) or 0.0),
        )
        current_rank = current.get("keyword_rank")
        incoming_rank = incoming.get("keyword_rank")
        if current_rank is None or (incoming_rank is not None and incoming_rank < current_rank):
            current["keyword_rank"] = incoming_rank

    if incoming.get("session_hit"):
        current["session_hit"] = True
        current["session_hit_score"] = max(
            float(current.get("session_hit_score", 0.0) or 0.0),
            float(incoming.get("session_hit_score", 0.0) or 0.0),
        )
        current_rank = current.get("session_hit_rank")
        incoming_rank = incoming.get("session_hit_rank")
        if current_rank is None or (incoming_rank is not None and incoming_rank < current_rank):
            current["session_hit_rank"] = incoming_rank


def fuse_ranked_lanes(*lane_groups: tuple[str, Iterable[MemoryRow] | None]) -> list[MemoryRow]:
    merged: dict[str, MemoryRow] = {}

    for lane_name, rows in lane_groups:
        for rank, row in enumerate(rows or [], start=1):
            row_id = str(row.get("id") or "")
            if not row_id:
                continue
            payload = merged.setdefault(row_id, {**dict(row), "rrf_score": 0.0, "rrf_sources": []})
            _merge_lane_metadata(payload, row)
            payload["rrf_score"] = float(payload.get("rrf_score", 0.0) or 0.0) + (1.0 / (RRF_K + rank))
            sources = payload.setdefault("rrf_sources", [])
            if lane_name not in sources:
                sources.append(lane_name)

    ranked = sorted(
        merged.values(),
        key=lambda row: (
            float(row.get("rrf_score", 0.0) or 0.0),
            len(row.get("rrf_sources", [])),
            float(row.get("similarity", 0.0) or 0.0),
            float(row.get("session_hit_score", 0.0) or 0.0),
            float(row.get("keyword_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    for row in ranked:
        row["rrf_score"] = round(float(row.get("rrf_score", 0.0) or 0.0), 6)
    return ranked
