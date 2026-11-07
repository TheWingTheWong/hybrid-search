"""Query-time metadata filters, e.g. ``tags=raft``, ``year>=2022``, ``source=*.md``.

Syntax: ``FIELD OP VALUE`` where OP is one of ``= != > >= < <= ~``.

* ``=`` / ``!=`` compare case-insensitively; values containing ``*``, ``?`` or
  ``[`` are treated as globs; ``a,b,c`` means "any of".
* For list-valued metadata (``tags``), ``=`` means "contains any of".
* ``> >= < <=`` compare numerically.
* ``~`` is a case-insensitive substring match.

Fields: ``doc_id``, ``title``, ``heading``, ``source``, plus any document metadata
(front matter keys, JSON Lines attributes, ``format``). Multiple filters AND.
"""

from __future__ import annotations

import fnmatch
import operator
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .documents import Chunk

_FILTER_RE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*(!=|>=|<=|=|>|<|~)\s*(.*?)\s*$")
_NUMERIC: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}
_BUILTIN_FIELDS = ("doc_id", "title", "heading", "source")


class FilterError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Filter:
    field: str
    op: str
    value: str

    @classmethod
    def parse(cls, expr: str) -> Filter:
        m = _FILTER_RE.match(expr)
        if not m or not m.group(3):
            raise FilterError(
                f"invalid filter {expr!r}; expected FIELD OP VALUE, e.g. tags=raft or year>=2022"
            )
        field, op, value = m.groups()
        if op in _NUMERIC:
            try:
                float(value)
            except ValueError:
                raise FilterError(f"filter {expr!r}: {op} needs a number, got {value!r}") from None
        return cls(field, op, value)

    def _lookup(self, chunk: Chunk) -> Any:
        if self.field in _BUILTIN_FIELDS:
            return getattr(chunk, self.field)
        return chunk.metadata.get(self.field)

    def matches(self, chunk: Chunk) -> bool:
        actual = self._lookup(chunk)
        if self.op in _NUMERIC:
            try:
                return _NUMERIC[self.op](float(actual), float(self.value))
            except (TypeError, ValueError):
                return False
        values = actual if isinstance(actual, (list, tuple, set)) else [actual]
        values = [str(v).casefold() for v in values if v is not None]
        if self.op == "~":
            needle = self.value.casefold()
            return any(needle in v for v in values)
        wanted = [w.strip().casefold() for w in self.value.split(",") if w.strip()]
        hit = any(_eq(v, w) for v in values for w in wanted)
        return hit if self.op == "=" else not hit

    def __str__(self) -> str:
        return f"{self.field}{self.op}{self.value}"


def _eq(actual: str, pattern: str) -> bool:
    if any(c in pattern for c in "*?["):
        return fnmatch.fnmatchcase(actual, pattern)
    return actual == pattern


def parse_filters(exprs: Iterable[str | Filter] | None) -> list[Filter]:
    return [e if isinstance(e, Filter) else Filter.parse(e) for e in (exprs or [])]


def filter_mask(chunks: Sequence[Chunk], filters: Sequence[Filter]) -> np.ndarray | None:
    """Boolean allow-mask over ``chunks``, or ``None`` when there are no filters."""
    if not filters:
        return None
    return np.fromiter(
        (all(f.matches(c) for f in filters) for c in chunks), dtype=bool, count=len(chunks)
    )
