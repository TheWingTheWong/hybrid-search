"""Minimal ANSI styling and word wrapping for terminal output (no dependencies).

Color is enabled only on a TTY and is disabled by ``NO_COLOR``; ``FORCE_COLOR``
turns it on regardless (useful for screenshots and ``less -R``).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from typing import TextIO

_CODES = {
    "bold": "1", "dim": "2", "italic": "3", "underline": "4",
    "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36",
    "gray": "90", "hl": "1;33",
}  # fmt: skip
_WORD = re.compile(r"\S+")


def color_enabled(stream: TextIO = sys.stdout) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


class Style:
    def __init__(self, enabled: bool | None = None):
        self.enabled = color_enabled() if enabled is None else enabled

    def __call__(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        codes = ";".join(_CODES[s] for s in styles)
        return f"\x1b[{codes}m{text}\x1b[0m"


def terminal_width(default: int = 100) -> int:
    return max(40, min(shutil.get_terminal_size((default, 24)).columns, 120))


def wrap_highlighted(
    text: str,
    spans: list[tuple[int, int]],
    width: int,
    style: Style,
    indent: str = "",
    hl: tuple[str, ...] = ("hl",),
) -> list[str]:
    """Word-wrap ``text`` by visible width, styling characters inside ``spans``."""
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    avail = max(10, width - len(indent))
    for m in _WORD.finditer(text):
        s, e = m.span()
        styled, pos = [], s
        for hs, he in spans:
            if he <= s or hs >= e:
                continue
            a, b = max(hs, s), min(he, e)
            styled += [text[pos:a], style(text[a:b], *hl)]
            pos = b
        styled.append(text[pos:e])
        word, wlen = "".join(styled), e - s
        if cur and cur_len + 1 + wlen > avail:
            lines.append(indent + " ".join(cur))
            cur, cur_len = [], 0
        cur_len += wlen + (1 if cur else 0)
        cur.append(word)
    if cur:
        lines.append(indent + " ".join(cur))
    return lines
