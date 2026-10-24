"""
Epistemic Stress Harness — Checkpoint Extraction

Deterministic parser for the checkpoint grammar. Spec: spec.md section 1.

v0.2 behavior (all deviations from the v0.1 regex are bug fixes):
  * Tag types are matched case-sensitively by default, as the spec requires.
    Pass strict=False to also accept mixed case.
  * Checkpoint content may contain balanced square brackets, e.g.
    ``[CLAIM: arr[0] is the pivot]``. v0.1 truncated content at the first ``]``.
  * Tags inside fenced code blocks or inline code are ignored.
  * A tag opened but never closed, or interrupted by another tag (the grammar
    forbids nesting), is counted in ``malformed_count`` and not emitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .schema import Checkpoint

_TYPES = "ASSUME|CLAIM|BRANCH|SELECT|CONCLUDE"
_OPEN_STRICT = re.compile(rf"\[({_TYPES})(?=[:\]])")
_OPEN_LENIENT = re.compile(rf"\[({_TYPES})(?=[:\]])", re.IGNORECASE)

_FENCE = re.compile(r"(```|~~~).*?(?:\1|\Z)", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")


@dataclass
class ParseResult:
    checkpoints: List[Checkpoint]
    malformed_count: int


def _mask_code(text: str) -> str:
    """Blank out code regions, preserving length and newlines."""
    def blank(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    return _INLINE_CODE.sub(blank, _FENCE.sub(blank, text))


def parse_checkpoints_detailed(text: str, strict: bool = True) -> ParseResult:
    """Parse checkpoints and report malformed tags."""
    opener = _OPEN_STRICT if strict else _OPEN_LENIENT
    masked = _mask_code(text)

    checkpoints: List[Checkpoint] = []
    malformed = 0
    pos = 0
    n = len(masked)

    while True:
        m = opener.search(masked, pos)
        if not m:
            break

        start = m.start()
        j = m.end()
        if masked[j] == ":":
            content_start = j + 1
        else:  # "]"
            content_start = j

        depth = 1
        k = content_start
        end = -1
        interrupted_at = -1
        while k < n:
            ch = masked[k]
            if ch == "[":
                if opener.match(masked, k):
                    interrupted_at = k
                    break
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = k
                    break
            k += 1

        if end == -1:
            malformed += 1
            # Resume at the interrupting tag, or just past this opener.
            pos = interrupted_at if interrupted_at != -1 else m.end()
            continue

        content = text[content_start:end].strip() if end > content_start else ""
        checkpoints.append({
            "index": len(checkpoints),
            "type": m.group(1).upper(),  # type: ignore[typeddict-item]
            "text": content,
            "start": start,
            "end": end + 1,
        })
        pos = end + 1

    return ParseResult(checkpoints=checkpoints, malformed_count=malformed)


def parse_checkpoints(text: str, strict: bool = True) -> List[Checkpoint]:
    """
    Extract checkpoints from annotated text.

    Pattern: [TYPE: content] or [TYPE]
    Types: ASSUME, CLAIM, BRANCH, SELECT, CONCLUDE
    """
    return parse_checkpoints_detailed(text, strict=strict).checkpoints
