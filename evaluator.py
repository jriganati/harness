"""Evaluator — judges executor output against acceptance criteria."""

from __future__ import annotations

import textwrap

from anthropic import AnthropicVertex

from config import Config
from run_log import RunLog
from state import EvalResult, SubTask


_EVAL_TOOL: dict = {
    "name": "submit_verdict",
    "description": "Submit an evaluation verdict for the executor's output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "retry_with_hint", "escalate"],
            },
            "score":  {"type": "number", "description": "Quality score 0.0–1.0"},
            "reason": {"type": "string"},
            "hint": {
                "type": "string",
                "description": "Required for retry_with_hint: specific guidance for the next attempt",
            },
        },
        "required": ["verdict", "score", "reason"],
    },
}


def run_evaluator(
    subtask: SubTask,
    output: str,
    cfg: Config,
    log: RunLog,
) -> EvalResult:
    """Evaluate executor output. Returns a verdict with optional hint."""
    client = AnthropicVertex(
        project_id=cfg.vertex_project,
        region=cfg.evaluator_region,
    )

    prompt = textwrap.dedent(f"""
        Evaluate whether this executor output satisfies the subtask.

        Task [{subtask.id}]: {subtask.description}
        Acceptance criteria: {subtask.acceptance_criteria}

        Executor output (truncated to 6000 chars):
        {output[:6000]}

        Verdict rules:
        - pass             → acceptance criteria are clearly met
        - retry_with_hint  → close but fixable; include a specific, actionable hint
        - escalate         → fundamentally wrong approach; needs re-planning
    """).strip()

    attempt = f"attempt{ (subtask.hint is not None) + 1 }"  # crude attempt counter for filename
    timing_label = f"evaluator.{subtask.id}"

    with log.time(timing_label):
        msg = client.messages.create(
            model=cfg.evaluator_model,
            max_tokens=cfg.evaluator_max_tokens,
            tools=[_EVAL_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )

    log.write_json(log.task_path(subtask.id, f"evaluator_{attempt}.json"), {
        "model":    cfg.evaluator_model,
        "region":   cfg.evaluator_region,
        "prompt":   prompt,
        "response": [b.model_dump() for b in msg.content],
        "usage":    msg.usage.model_dump() if msg.usage else None,
    })

    for block in msg.content:
        if block.type == "tool_use" and block.name == "submit_verdict":
            inp = block.input
            return EvalResult(
                verdict=inp["verdict"],
                reason=inp["reason"],
                hint=inp.get("hint"),
                score=float(inp.get("score", 0.0)),
            )

    return EvalResult(
        verdict="retry_with_hint",
        reason="evaluator parse error — could not extract verdict",
        hint="try again",
    )
