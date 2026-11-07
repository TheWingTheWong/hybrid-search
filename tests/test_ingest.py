import json
from pathlib import Path

import pytest

from hybrid_search.ingest import (
    ChunkingConfig,
    chunk_document,
    load_documents,
    parse_front_matter,
    parse_jsonl,
    parse_markdown,
    split_markdown_sections,
    window_spans,
)


def test_front_matter_types() -> None:
    meta, body = parse_front_matter("---\ntitle: 'X'\ntags: [a, b]\nyear: 2021\nok: true\n---\nbody")
    assert meta == {"title": "X", "tags": ["a", "b"], "year": 2021, "ok": True}
    assert body == "body"


def test_unterminated_front_matter_is_content() -> None:
    meta, body = parse_front_matter("---\ntitle: x\nno end")
    assert meta == {} and body.startswith("---")


def test_title_falls_back_to_h1_then_filename() -> None:
    assert parse_markdown("# Hello\ntext", "d", "d.md").title == "Hello"
    assert parse_markdown("plain text", "my-note", "my-note.md").title == "My Note"


def test_sections_build_breadcrumbs_and_skip_fenced_hashes() -> None:
    body = "# T\nintro\n## A\na text\n```\n# not heading\n```\n### B\nb text\n## C\nc text\n"
    sections = split_markdown_sections(body, title="T")
    assert [h for h, _ in sections] == ["", "A", "A > B", "C"]
    assert "# not heading" in sections[1][1]


def test_window_spans_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(25))
    spans = window_spans(text, max_tokens=10, overlap=4)
    windows = [text[s:e].split() for s, e in spans]
    assert [len(w) for w in windows] == [10, 10, 10, 7]
    assert windows[0][-4:] == windows[1][:4]
    assert windows[-1][-1] == "w24"


def test_window_spans_short_and_empty() -> None:
    assert window_spans("a b c", 10, 2) == [(0, 5)]
    assert window_spans("   ", 10, 2) == []


def test_chunking_config_validation() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_tokens=10, overlap=10)


def test_chunks_carry_metadata_and_ids() -> None:
    doc = parse_markdown("---\ntags: [x]\n---\n# T\n## S\n" + "word " * 50, "d", "d.md")
    chunks = chunk_document(doc, ChunkingConfig(max_tokens=20, overlap=5))
    assert [c.chunk_id for c in chunks] == [f"d#{i}" for i in range(len(chunks))]
    assert all(c.heading == "S" and c.metadata == {"tags": ["x"], "format": "markdown"} for c in chunks)


def test_load_documents_all_formats(tiny_corpus: Path) -> None:
    docs = {d.doc_id: d for d in load_documents([tiny_corpus])}
    assert set(docs) == {"notes/raft", "bloom", "handbook", "glossary:quorum", "glossary:lsm"}
    assert docs["notes/raft"].metadata["tags"] == ["consensus", "raft"]
    assert docs["handbook"].title == "Pager Handbook"
    assert docs["glossary:lsm"].metadata == {"tags": ["storage"]}
    assert docs["glossary:lsm"].source == "glossary.jsonl:2"


def test_jsonl_requires_text() -> None:
    with pytest.raises(ValueError, match="text"):
        list(parse_jsonl(json.dumps({"id": 1}), "g", "g.jsonl"))
    with pytest.raises(ValueError, match="invalid JSON"):
        list(parse_jsonl("{oops", "g", "g.jsonl"))


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "a.txt").write_text("y")
    with pytest.raises(ValueError, match="duplicate"):
        load_documents([tmp_path])


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents([tmp_path / "nope"])
