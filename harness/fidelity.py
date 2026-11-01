"""
Epistemic Stress Harness — Fidelity Metrics

The structural metrics count tags the model chose to write. A tag is not
evidence of the behavior it names: a model (or a script) can emit
``[BRANCH: x]`` without weighing anything. These checks test whether each tag
is internally consistent with what it claims to be.

They are heuristics over text, deliberately simple and deterministic. They
detect *malformed* annotation; they cannot prove genuine reasoning. For that
use an independent judge (judge.py) and objective outcomes (outcomes.py).
Spec: spec.md section 4.2.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

from .schema import Checkpoint, FidelityMetrics

_ALT_SPLIT = re.compile(r"\s+(?:vs\.?|versus|or)\s+|\s+/\s+", re.IGNORECASE)
_BECAUSE = re.compile(r"\bbecause\s*:?\s*(.*)$", re.IGNORECASE | re.DOTALL)
_WORD = re.compile(r"[a-z0-9]+")

# "a" is deliberately not a stopword: "A vs B" are legitimate option labels.
_STOP = frozenset(
    "an the of to in on for and or vs versus with without by as at is are be "
    "use using include includes via".split()
)


def _tokens(text: str) -> Set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def split_alternatives(text: str) -> List[str]:
    """Split a BRANCH body into its alternatives."""
    return [p.strip() for p in _ALT_SPLIT.split(text) if p.strip()]


def branch_is_well_formed(text: str) -> bool:
    """>= 2 non-empty alternatives whose content differs."""
    alts = split_alternatives(text)
    if len(alts) < 2:
        return False
    keys = {frozenset(_tokens(a)) or a.strip().lower() for a in alts}
    return len(keys) >= 2


def select_split(text: str):
    """Return (chosen, justification) for a SELECT body."""
    chosen, _, rest = text.partition("|")
    m = _BECAUSE.search(rest)
    justification = m.group(1).strip() if m else ""
    return chosen.strip(), justification


def select_is_justified(text: str) -> bool:
    """SELECT carries a justification of at least 3 words."""
    _, why = select_split(text)
    return len(why.split()) >= 3


def _frac(hits: int, total: int) -> Optional[float]:
    return hits / total if total else None


def compute_fidelity(checkpoints: List[Checkpoint]) -> FidelityMetrics:
    """
    Compute tag-text consistency.

    branch_well_formed      fraction of BRANCH tags naming >= 2 distinct alternatives
    select_justified        fraction of SELECT tags with a >= 3-word justification
    select_resolves_branch  fraction of SELECTs that follow an unresolved BRANCH
                            and whose chosen option overlaps (>= 50% of its
                            content words) one of that BRANCH's alternatives
    nonempty_rate           fraction of tags with non-empty content (CONCLUDE excluded
                            from the denominator only when empty is allowed by spec)
    fidelity_score          mean of the components that are defined
    """
    if not checkpoints:
        return FidelityMetrics()

    branches = [cp for cp in checkpoints if cp["type"] == "BRANCH"]
    selects = [cp for cp in checkpoints if cp["type"] == "SELECT"]

    bwf = _frac(sum(branch_is_well_formed(b["text"]) for b in branches), len(branches))
    sj = _frac(sum(select_is_justified(s["text"]) for s in selects), len(selects))

    # SELECT resolves the most recent open BRANCH.
    open_alts: Optional[List[Set[str]]] = None
    resolved_hits = resolved_total = 0
    for cp in checkpoints:
        if cp["type"] == "BRANCH":
            open_alts = [_tokens(a) for a in split_alternatives(cp["text"])]
        elif cp["type"] == "SELECT" and open_alts is not None:
            chosen, _ = select_split(cp["text"])
            ct = _tokens(chosen)
            resolved_total += 1
            if ct and any(
                alt and len(ct & alt) / len(ct) >= 0.5 for alt in open_alts
            ):
                resolved_hits += 1
            open_alts = None
    srb = _frac(resolved_hits, resolved_total)

    nonempty = _frac(sum(bool(cp["text"].strip()) for cp in checkpoints), len(checkpoints))

    parts = [v for v in (bwf, sj, srb, nonempty) if v is not None]
    score = sum(parts) / len(parts) if parts else None

    return FidelityMetrics(
        branch_well_formed=bwf,
        select_justified=sj,
        select_resolves_branch=srb,
        nonempty_rate=nonempty,
        fidelity_score=score,
    )
