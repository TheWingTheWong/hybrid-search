import pytest

from hybrid_search.text import Analyzer, porter_stem

# Pairs from Porter's paper and the reference vocabulary.
PORTER_CASES = {
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat",
    "feed": "feed", "agreed": "agre", "plastered": "plaster", "motoring": "motor",
    "sing": "sing", "conflated": "conflat", "troubled": "troubl", "sized": "size",
    "hopping": "hop", "tanned": "tan", "falling": "fall", "hissing": "hiss", "fizzed": "fizz",
    "failing": "fail", "filing": "file", "happy": "happi", "sky": "sky",
    "relational": "relat", "conditional": "condit", "rational": "ration",
    "digitizer": "digit", "generalization": "gener", "electrical": "electr",
    "hopeful": "hope", "goodness": "good", "revival": "reviv", "allowance": "allow",
    "adjustment": "adjust", "adoption": "adopt", "controll": "control", "roll": "roll",
    "probate": "probat", "rate": "rate", "cease": "ceas",
}  # fmt: skip


@pytest.mark.parametrize(("word", "stem"), sorted(PORTER_CASES.items()))
def test_porter_reference_pairs(word: str, stem: str) -> None:
    assert porter_stem(word) == stem


def test_inflections_share_a_stem() -> None:
    a = Analyzer()
    assert a.analyze("elections elected electing") == ["elect"] * 3


def test_analyzer_drops_stopwords_and_possessives() -> None:
    assert Analyzer().analyze("How does the leader's term work?") == ["leader", "term", "work"]


def test_analyzer_strips_diacritics_and_case() -> None:
    assert Analyzer(stem=False).analyze("Café NAÏVE") == ["cafe", "naive"]


def test_tokens_keep_character_offsets() -> None:
    text = "Raft elects leaders"
    toks = Analyzer().tokens(text)
    assert [(t.term, text[t.start : t.end]) for t in toks] == [
        ("raft", "Raft"), ("elect", "elects"), ("leader", "leaders"),
    ]  # fmt: skip


def test_numbers_are_kept() -> None:
    assert Analyzer().analyze("HTTP 503 after 30s") == ["http", "503", "30s"]
