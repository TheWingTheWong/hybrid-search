import numpy as np
import pytest

from hybrid_search.documents import Chunk
from hybrid_search.filters import Filter, FilterError, filter_mask, parse_filters
from hybrid_search.highlight import make_snippet


def chunk(**meta) -> Chunk:
    return Chunk("raft#0", "raft", "Raft", "Leader election", "text", "notes/raft.md", 0, meta)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("tags=raft", True),
        ("tags=RAFT", True),
        ("tags=paxos", False),
        ("tags=paxos,raft", True),
        ("tags!=paxos", True),
        ("tags!=raft", False),
        ("year>=2021", True),
        ("year>2021", False),
        ("year<2022", True),
        ("source=notes/*.md", True),
        ("source=*.txt", False),
        ("title~af", True),
        ("heading~ELECTION", True),
        ("doc_id=raft", True),
        ("missing=x", False),
        ("missing!=x", True),
        ("title>=3", False),  # non-numeric field values never match numeric ops
    ],
)
def test_filter_semantics(expr: str, expected: bool) -> None:
    assert Filter.parse(expr).matches(chunk(tags=["consensus", "raft"], year=2021)) is expected


@pytest.mark.parametrize("bad", ["tags", "=raft", "tags=", "year >> 3", "year>=recent"])
def test_invalid_filters(bad: str) -> None:
    with pytest.raises(FilterError):
        Filter.parse(bad)


def test_filter_mask_ands_filters() -> None:
    chunks = [chunk(tags=["raft"], year=2020), chunk(tags=["raft"], year=2023), chunk(year=2023)]
    mask = filter_mask(chunks, parse_filters(["tags=raft", "year>=2022"]))
    assert mask.tolist() == [False, True, False]
    assert filter_mask(chunks, []) is None
    assert isinstance(mask, np.ndarray)


def test_snippet_picks_window_with_most_distinct_terms() -> None:
    text = ("filler " * 40) + "the leader election uses a randomized timeout" + (" filler" * 40)
    snip = make_snippet(text, "how is a leader elected after a timeout", width=12)
    assert snip.text.startswith("... ") and snip.text.endswith(" ...")
    marked = [snip.text[s:e] for s, e in snip.spans]
    assert marked == ["leader", "election", "timeout"]


def test_snippet_render_and_whitespace() -> None:
    snip = make_snippet("Bloom\n\n   filters   use bits.", "bloom filter")
    assert snip.text == "Bloom filters use bits."
    assert snip.render("<b>", "</b>") == "<b>Bloom</b> <b>filters</b> use bits."


def test_snippet_without_matches_shows_beginning() -> None:
    snip = make_snippet("alpha beta gamma delta", "zeta", width=2)
    assert snip.text == "alpha beta ..." and snip.spans == []
