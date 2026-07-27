"""Helpers for reporting selected context."""

from __future__ import annotations

from typing import Any, Iterable


def selected_context_text(docs: Iterable[Any]) -> str:
    return "\n\n".join(
        segment.text
        for doc in docs
        for segment in (getattr(doc, "cacheables", []) or [])
    )


def selected_context_segment_count(docs: Iterable[Any]) -> int:
    return sum(len(getattr(doc, "cacheables", []) or []) for doc in docs)
