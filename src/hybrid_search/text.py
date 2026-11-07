"""Text analysis: normalization, tokenization, stopwords and a Porter stemmer.

The same analyzer is used at index time and query time so that terms line up.
Everything here is pure Python with no third-party dependencies.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

# Unicode letters/digits (not underscore), allowing one inner apostrophe: "leader's".
_APOSTROPHES = "'’"
_WORD_RE = re.compile(rf"[^\W_]+(?:[{_APOSTROPHES}][^\W_]+)?")

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him himself
    his how i if in into is it its itself just me more most my myself no nor not of off
    on once only or other our ours ourselves out over own same she should so some such
    than that the their theirs them themselves then there these they this those through
    to too under until up very was we were what when where which while who whom why will
    with would you your yours yourself yourselves also via per may might must shall
    """.split()
)


@dataclass(frozen=True, slots=True)
class Token:
    """A token with its analyzed term and character span in the original text."""

    term: str
    surface: str
    start: int
    end: int


def normalize(text: str) -> str:
    """Unicode-normalize and casefold text, stripping diacritics."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def iter_words(text: str) -> Iterator[re.Match[str]]:
    """Yield regex matches for word-like spans in ``text`` (original casing kept)."""
    return _WORD_RE.finditer(text)


class Analyzer:
    """Lowercase -> split -> drop stopwords -> Porter stem.

    ``analyze`` returns the list of index terms; ``tokens`` additionally keeps
    character offsets, which the snippet highlighter needs.
    """

    def __init__(self, stem: bool = True, stopwords: frozenset[str] | None = STOPWORDS):
        self.stem = stem
        self.stopwords = stopwords or frozenset()

    def term(self, word: str) -> str | None:
        w = normalize(word).replace("’", "'")
        if w.endswith("'s"):
            w = w[:-2]
        w = w.replace("'", "")
        if not w or w in self.stopwords:
            return None
        return porter_stem(w) if self.stem else w

    def analyze(self, text: str) -> list[str]:
        out = []
        for m in iter_words(text):
            t = self.term(m.group())
            if t:
                out.append(t)
        return out

    def tokens(self, text: str) -> list[Token]:
        out = []
        for m in iter_words(text):
            t = self.term(m.group())
            if t:
                out.append(Token(t, m.group(), m.start(), m.end()))
        return out


DEFAULT_ANALYZER = Analyzer()


# ---------------------------------------------------------------------------
# Porter stemmer (M.F. Porter, 1980), implemented from the original paper.
# ---------------------------------------------------------------------------

_VOWELS = frozenset("aeiou")


def _is_cons(w: str, i: int) -> bool:
    ch = w[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_cons(w, i - 1)
    return True


def _measure(stem: str) -> int:
    """Number of VC sequences in ``stem`` ([C](VC)^m[V])."""
    m = 0
    prev_vowel = False
    for i in range(len(stem)):
        cons = _is_cons(stem, i)
        if cons and prev_vowel:
            m += 1
        prev_vowel = not cons
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_cons(stem, i) for i in range(len(stem)))


def _ends_double_cons(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _is_cons(w, len(w) - 1)


def _cvc(w: str) -> bool:
    if len(w) < 3:
        return False
    return (
        _is_cons(w, len(w) - 3)
        and not _is_cons(w, len(w) - 2)
        and _is_cons(w, len(w) - 1)
        and w[-1] not in "wxy"
    )


def _replace(w: str, suffix: str, repl: str, min_m: int) -> str | None:
    """If ``w`` ends with ``suffix`` and m(stem) > min_m, swap the suffix."""
    if w.endswith(suffix):
        stem = w[: len(w) - len(suffix)]
        return stem + repl if _measure(stem) > min_m else w
    return None


_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"),
    ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
    ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
    ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log"),
]  # fmt: skip
_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]  # fmt: skip
_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]  # fmt: skip


@lru_cache(maxsize=65536)
def porter_stem(word: str) -> str:
    """Return the Porter stem of a lowercase ASCII word."""
    w = word
    if len(w) <= 2 or not w.isalpha():
        return w

    # Step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]

    # Step 1b
    step1b_extra = False
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w = w[:-2]
        step1b_extra = True
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w = w[:-3]
        step1b_extra = True
    if step1b_extra:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _ends_double_cons(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _measure(w) == 1 and _cvc(w):
            w += "e"

    # Step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"

    # Step 2
    for suf, rep in _STEP2:
        r = _replace(w, suf, rep, 0)
        if r is not None:
            w = r
            break

    # Step 3
    for suf, rep in _STEP3:
        r = _replace(w, suf, rep, 0)
        if r is not None:
            w = r
            break

    # Step 4
    for suf in _STEP4:
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if _measure(stem) > 1 and (suf != "ion" or stem.endswith(("s", "t"))):
                w = stem
            break

    # Step 5a
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            w = stem
    # Step 5b
    if w.endswith("ll") and _measure(w) > 1:
        w = w[:-1]
    return w
