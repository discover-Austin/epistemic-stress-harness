"""
Epistemic Stress Harness — Suite Loading

Loads perturbation conditions, tasks (with ground truth) and the paired
stakes scenarios from suite/*.json. Pure data; no model calls.
Spec: spec.md sections 5 and 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_SUITE_DIR = Path(__file__).resolve().parent.parent / "suite"


@dataclass
class Condition:
    """One experimental condition (a system-prompt perturbation or a framing)."""
    key: str
    system_suffix: Optional[str] = None
    max_tokens: int = 1024
    kind: str = "other"
    description: str = ""
    instructed: List[str] = field(default_factory=list)
    adjacent: List[str] = field(default_factory=list)

    def config(self) -> Dict[str, Any]:
        return {"system_suffix": self.system_suffix, "max_tokens": self.max_tokens}

    def label_for(self, metric: str) -> str:
        """'instructed' | 'adjacent' | 'uninstructed' for a metric under this condition."""
        if metric in self.instructed:
            return "instructed"
        if metric in self.adjacent:
            return "adjacent"
        return "uninstructed"


@dataclass
class Item:
    """A task or scenario: one or more prompts plus ground truth."""
    id: str
    prompts: Dict[str, str]          # condition key -> prompt; "*" applies to any
    task: Dict[str, Any]             # answer_spec, required_caveats, category
    mock: Optional[Dict[str, str]] = None

    def prompt_for(self, condition_key: str) -> str:
        return self.prompts.get(condition_key, self.prompts["*"])


def _read(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_perturbations(path: Optional[Path] = None) -> List[Condition]:
    path = Path(path) if path else DEFAULT_SUITE_DIR / "perturbations_v1.json"
    data = _read(path)
    conds = []
    for p in data["perturbations"]:
        ins = p.get("instructs", {})
        conds.append(Condition(
            key=p["key"],
            system_suffix=p.get("system_suffix"),
            max_tokens=p.get("max_tokens", 1024),
            kind=p.get("kind", "other"),
            description=p.get("description", ""),
            instructed=list(ins.get("instructed", [])),
            adjacent=list(ins.get("adjacent", [])),
        ))
    keys = [c.key for c in conds]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate condition keys")
    if "baseline" not in keys:
        raise ValueError(f"{path}: suite has no 'baseline' condition")
    return conds


def suite_contrasts(path: Optional[Path] = None) -> List[Tuple[str, str]]:
    """Extra (reference, target) pairs declared by a perturbation suite file."""
    path = Path(path) if path else DEFAULT_SUITE_DIR / "perturbations_v1.json"
    return [tuple(c) for c in _read(path).get("contrasts", [])]


def load_tasks(path: Optional[Path] = None) -> List[Item]:
    path = Path(path) if path else DEFAULT_SUITE_DIR / "tasks_v1.json"
    items = []
    for t in _read(path)["tasks"]:
        items.append(Item(
            id=t["id"],
            prompts={"*": t["prompt"]},
            task={k: v for k, v in t.items() if not k.startswith("_") and k != "prompt"},
            mock=t.get("_mock"),
        ))
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate task ids")
    return items


def load_scenarios(
    path: Optional[Path] = None,
) -> Tuple[List[Condition], List[Item], str, List[Tuple[str, str]]]:
    """Return (frame conditions, scenario items, baseline frame, contrasts)."""
    path = Path(path) if path else DEFAULT_SUITE_DIR / "scenarios_v1.json"
    data = _read(path)
    conds = [Condition(key=k, kind="frame", description=k) for k in data["frames"]]
    items = []
    for s in data["scenarios"]:
        prompts = {
            key: tpl.format(context=s["context"], question=s["question"])
            for key, tpl in data["frames"].items()
        }
        prompts["*"] = prompts[data["baseline_frame"]]
        items.append(Item(
            id=s["id"],
            prompts=prompts,
            task={
                "category": "scenario",
                "answer_spec": {"type": "choice_letter", "correct": s["correct"],
                                "choices": ["A", "B"]},
            },
            mock={"correct": s["correct"],
                  "wrong": "B" if s["correct"] == "A" else "A"},
        ))
    return conds, items, data["baseline_frame"], [tuple(c) for c in data["contrasts"]]
