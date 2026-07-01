"""Canonical [n] numbered-hit text format shared by the search tool (writer) and
the evidence collector (reader). Keeping both here keeps them in lockstep.

One hit = three lines, hits separated by a blank line:

    [1] Eiffel Tower
        source: encyclopedia.example | 2019-04-11 | https://encyclopedia.example/e
        The Eiffel Tower was completed in 1889 ...
"""
import re
from dataclasses import dataclass

from backends.base import SearchHit  # type: ignore[import-not-found]

_HEADER_RE = re.compile(r"^\[(\d+)\]\s+(.*)$")
_SOURCE_PREFIX = "source: "


@dataclass(frozen=True)
class ParsedHit:
    """A hit recovered from formatted text."""

    id: int
    title: str
    domain: str
    published: str | None
    url: str
    content: str


def _one_line(text: str) -> str:
    return " ".join(text.split())


def format_hits(hits: list[SearchHit], start: int = 1) -> str:
    """Render hits as numbered [n] blocks starting at `start`."""
    blocks: list[str] = []
    for i, h in enumerate(hits, start):
        published = h.published or "?"
        blocks.append(
            f"[{i}] {_one_line(h.title)}\n"
            f"    {_SOURCE_PREFIX}{h.domain} | {published} | {h.url}\n"
            f"    {_one_line(h.content)}"
        )
    return "\n\n".join(blocks)


def parse_hits(text: str) -> list[ParsedHit]:
    """Parse numbered [n] blocks back into ParsedHit records; malformed blocks skipped."""
    parsed: list[ParsedHit] = []
    for block in text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        m = _HEADER_RE.match(lines[0].strip())
        if not m:
            continue
        meta = lines[1].strip()
        if meta.startswith(_SOURCE_PREFIX):
            meta = meta[len(_SOURCE_PREFIX):]
        parts = [p.strip() for p in meta.split("|", 2)]
        domain = parts[0] if len(parts) > 0 else ""
        published_raw = parts[1] if len(parts) > 1 else "?"
        url = parts[2] if len(parts) > 2 else ""
        content = " ".join(ln.strip() for ln in lines[2:])
        parsed.append(
            ParsedHit(
                id=int(m.group(1)),
                title=m.group(2).strip(),
                domain=domain,
                published=None if published_raw in ("", "?") else published_raw,
                url=url,
                content=content,
            )
        )
    return parsed
