"""Query-biased snippet extraction with highlight spans.

We slide a window of ``width`` words over the chunk and keep the window that
covers the most *distinct* query terms (ties broken by total matches, then by
position), then collapse whitespace and record the character spans of every
matching word so callers can render them as ANSI bold, ``<mark>`` or ``**``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import DEFAULT_ANALYZER, Analyzer, iter_words

_WS = re.compile(r"\s+")


@dataclass(slots=True)
class Snippet:
    text: str
    spans: list[tuple[int, int]] = field(default_factory=list)

    def render(self, start: str = "**", end: str = "**") -> str:
        out, pos = [], 0
        for s, e in self.spans:
            out += [self.text[pos:s], start, self.text[s:e], end]
            pos = e
        out.append(self.text[pos:])
        return "".join(out)

    def to_dict(self) -> dict:
        return {"text": self.text, "highlights": [list(s) for s in self.spans]}


def make_snippet(
    text: str, query: str, width: int = 36, analyzer: Analyzer | None = None
) -> Snippet:
    analyzer = analyzer or DEFAULT_ANALYZER
    qterms = set(analyzer.analyze(query))
    words = list(iter_words(text))
    if not words:
        return Snippet(_WS.sub(" ", text).strip())
    terms = [analyzer.term(m.group()) for m in words]
    hits = [t if t in qterms else None for t in terms]

    best_start, best_key = 0, (-1, -1)
    if qterms:
        for start in range(0, max(1, len(words) - width + 1)):
            window = [h for h in hits[start : start + width] if h]
            key = (len(set(window)), len(window))
            if key > best_key:
                best_start, best_key = start, key
    end = min(len(words), best_start + width)

    pieces: list[tuple[str, bool]] = []
    cursor = words[best_start].start()
    for i in range(best_start, end):
        m = words[i]
        if hits[i]:
            pieces.append((text[cursor : m.start()], False))
            pieces.append((m.group(), True))
            cursor = m.end()
    pieces.append((text[cursor : words[end - 1].end()], False))

    out, spans = [], []
    length = 0
    if best_start > 0:
        out.append("... ")
        length = 4
    for chunk, is_hit in pieces:
        s = chunk if is_hit else _WS.sub(" ", chunk)
        if is_hit:
            spans.append((length, length + len(s)))
        out.append(s)
        length += len(s)
    if end < len(words):
        out.append(" ...")
    return Snippet("".join(out), spans)
