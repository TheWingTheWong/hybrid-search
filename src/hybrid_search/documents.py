"""Core data model: source documents and the chunks we actually index."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """A logical source document (a file, or one JSON Lines record)."""

    doc_id: str
    title: str
    text: str
    source: str
    format: str  # "markdown" | "text" | "jsonl"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """A retrievable unit: a slice of a document with its section context.

    ``heading`` is a breadcrumb such as ``"Leader election > Randomized timeouts"``.
    Chunks carry three text fields (title, heading, body) so BM25F can weight them
    independently.
    """

    chunk_id: str
    doc_id: str
    title: str
    heading: str
    text: str
    source: str
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def fields(self) -> dict[str, str]:
        return {"title": self.title, "heading": self.heading, "body": self.text}

    def embedding_text(self) -> str:
        """Text fed to the dense encoder: context first, then the body."""
        parts = [self.title]
        if self.heading:
            parts.append(self.heading)
        parts.append(self.text)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)
