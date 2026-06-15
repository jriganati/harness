"""State data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SubTask:
    id: str
    description: str
    acceptance_criteria: str
    dependencies: list[str] = field(default_factory=list)
    status: Literal["pending", "in_progress", "passed", "failed"] = "pending"
    hint: str | None = None  # injected from last evaluator retry
    role: Literal["executor", "inspector"] = "executor"


@dataclass
class EvalResult:
    verdict: Literal["pass", "retry_with_hint", "escalate"]
    reason: str
    hint: str | None = None
    score: float = 0.0


@dataclass
class HarnessState:
    goal: str
    subtasks: list[SubTask] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)    # task_id → text output
    eval_scores: dict[str, EvalResult] = field(default_factory=dict)
    iteration: dict[str, int] = field(default_factory=dict)    # retry counts per task


def state_from_dict(d: dict[str, Any]) -> HarnessState:
    """Rehydrate a HarnessState from its serialized JSON form (for resume)."""
    state = HarnessState(goal=d["goal"])
    state.subtasks = [SubTask(**t) for t in d.get("subtasks", [])]
    state.artifacts = dict(d.get("artifacts", {}))
    state.eval_scores = {k: EvalResult(**v) for k, v in d.get("eval_scores", {}).items()}
    state.iteration = dict(d.get("iteration", {}))
    return state
