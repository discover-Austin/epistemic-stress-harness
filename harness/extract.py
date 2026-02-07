"""
Epistemic Stress Harness — Checkpoint Extraction

Deterministic regex parser for checkpoint grammar.
Spec: spec.md section 1.
"""

import re
from typing import List

from .schema import Checkpoint


def parse_checkpoints(text: str) -> List[Checkpoint]:
    """
    Extract checkpoints from annotated text.

    Pattern: [TYPE: content] or [TYPE]
    Types: ASSUME, CLAIM, BRANCH, SELECT, CONCLUDE
    """
    pattern = r"\[(ASSUME|CLAIM|BRANCH|SELECT|CONCLUDE):?\s*([^\]]*)\]"
    matches = re.finditer(pattern, text, re.IGNORECASE)

    checkpoints: List[Checkpoint] = []
    for idx, match in enumerate(matches):
        checkpoints.append({
            "index": idx,
            "type": match.group(1).upper(),  # type: ignore[typeddict-item]
            "text": match.group(2).strip(),
        })

    return checkpoints
