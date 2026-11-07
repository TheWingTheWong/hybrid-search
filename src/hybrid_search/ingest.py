"""Document loading (Markdown, plain text, JSON Lines) and chunking.

Chunking is two-staged:

1. **Structural** - Markdown is split on ATX headings (``#`` .. ``######``),
   ignoring ``#`` lines inside fenced code blocks. Each section remembers its
   heading breadcrumb, e.g. ``"Log replication > Commit index"``.
2. **Token windows** - sections longer than ``max_tokens`` whitespace tokens
   are cut into overlapping windows (stride ``max_tokens - overlap``). Windows
   are sliced from the original text by character offsets, so code blocks and
   line breaks survive intact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .documents import Chunk, Document

MARKDOWN_EXT = {".md", ".markdown"}
TEXT_EXT = {".txt", ".text"}
JSONL_EXT = {".jsonl", ".ndjson"}
SUPPORTED_EXT = MARKDOWN_EXT | TEXT_EXT | JSONL_EXT

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_tokens: int = 160
    overlap: int = 32
    split_headings: bool = True

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.overlap < self.max_tokens:
            raise ValueError("overlap must satisfy 0 <= overlap < max_tokens")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def iter_files(paths: Iterable[str | Path]) -> Iterator[tuple[Path, Path]]:
    """Yield ``(file, root)`` pairs for every supported file under ``paths``."""
    for p in map(Path, paths):
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if (
                    f.is_file()
                    and f.suffix.lower() in SUPPORTED_EXT
                    and not any(part.startswith(".") for part in f.relative_to(p).parts)
                ):
                    yield f, p
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED_EXT:
                raise ValueError(f"unsupported file type: {p}")
            yield p, p.parent
        else:
            raise FileNotFoundError(p)


def load_documents(paths: Iterable[str | Path]) -> list[Document]:
    docs: list[Document] = []
    seen: set[str] = set()
    for f, root in iter_files(paths):
        for doc in load_file(f, root):
            if doc.doc_id in seen:
                raise ValueError(f"duplicate document id {doc.doc_id!r} ({doc.source})")
            seen.add(doc.doc_id)
            docs.append(doc)
    return docs


def load_file(path: Path, root: Path | None = None) -> list[Document]:
    root = root or path.parent
    rel = path.relative_to(root).as_posix()
    base_id = rel.rsplit(".", 1)[0]
    ext = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")
    if ext in MARKDOWN_EXT:
        return [parse_markdown(raw, doc_id=base_id, source=rel)]
    if ext in TEXT_EXT:
        return [parse_text(raw, doc_id=base_id, source=rel)]
    if ext in JSONL_EXT:
        return list(parse_jsonl(raw, base_id=base_id, source=rel))
    raise ValueError(f"unsupported file type: {path}")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return [_parse_scalar(v) for v in value[1:-1].split(",") if v.strip()]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d*", value):
        return float(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parse a minimal ``key: value`` YAML front matter block.

    Supports scalars, quoted strings and inline lists (``tags: [a, b]``),
    which covers what notes typically carry without a YAML dependency.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines(keepends=True)
    if lines[0].strip() != "---":
        return {}, text
    meta: dict[str, Any] = {}
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            return meta, "".join(lines[i + 1 :])
        if ":" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition(":")
            meta[key.strip()] = _parse_scalar(value)
    return {}, text  # unterminated block: treat as content


def parse_markdown(raw: str, doc_id: str, source: str) -> Document:
    meta, body = parse_front_matter(raw)
    title = meta.pop("title", None)
    if not title:
        for line in body.splitlines():
            m = _HEADING_RE.match(line)
            if m and len(m.group(1)) == 1:
                title = m.group(2)
                break
    title = str(title or Path(source).stem.replace("-", " ").replace("_", " ").title())
    return Document(doc_id, title, body, source, "markdown", meta)


def parse_text(raw: str, doc_id: str, source: str) -> Document:
    lines = raw.strip().splitlines()
    first = lines[0].strip() if lines else ""
    if first and len(first) <= 80 and (len(lines) == 1 or not lines[1].strip()):
        title, body = first, "\n".join(lines[1:]).strip()
    else:
        title, body = Path(source).stem.replace("-", " ").replace("_", " ").title(), raw
    return Document(doc_id, title, body, source, "text", {})


def parse_jsonl(raw: str, base_id: str, source: str) -> Iterator[Document]:
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{source}:{lineno}: invalid JSON: {e.msg}") from e
        if not isinstance(rec, dict) or not isinstance(rec.get("text"), str):
            raise ValueError(f"{source}:{lineno}: each record needs a string 'text' field")
        rid = str(rec.pop("id", lineno))
        text = rec.pop("text")
        title = str(rec.pop("title", rid))
        yield Document(f"{base_id}:{rid}", title, text, f"{source}:{lineno}", "jsonl", rec)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def split_markdown_sections(body: str, title: str = "") -> list[tuple[str, str]]:
    """Split Markdown into ``(breadcrumb, text)`` sections on ATX headings."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "".join(buf).strip()
        if text:
            crumb = " > ".join(h for lvl, h in stack if not (lvl == 1 and h == title))
            sections.append((crumb, text))
        buf.clear()

    for line in body.splitlines(keepends=True):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line.rstrip("\n"))
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        else:
            buf.append(line)
    flush()
    return sections


def window_spans(text: str, max_tokens: int, overlap: int) -> list[tuple[int, int]]:
    """Character spans of overlapping whitespace-token windows over ``text``."""
    toks = [m.span() for m in _TOKEN_RE.finditer(text)]
    if not toks:
        return []
    if len(toks) <= max_tokens:
        return [(toks[0][0], toks[-1][1])]
    stride = max_tokens - overlap
    spans = []
    for start in range(0, len(toks), stride):
        end = min(start + max_tokens, len(toks))
        spans.append((toks[start][0], toks[end - 1][1]))
        if end == len(toks):
            break
    return spans


def chunk_document(doc: Document, config: ChunkingConfig | None = None) -> list[Chunk]:
    config = config or ChunkingConfig()
    if doc.format == "markdown" and config.split_headings:
        sections = split_markdown_sections(doc.text, doc.title)
    else:
        sections = [("", doc.text.strip())]

    meta = dict(doc.metadata)
    meta["format"] = doc.format
    chunks: list[Chunk] = []
    for heading, text in sections:
        for s, e in window_spans(text, config.max_tokens, config.overlap):
            n = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{n}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    heading=heading,
                    text=text[s:e],
                    source=doc.source,
                    position=n,
                    metadata=dict(meta),
                )
            )
    return chunks


def chunk_documents(docs: Iterable[Document], config: ChunkingConfig | None = None) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, config))
    return out
