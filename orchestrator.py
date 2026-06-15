"""Orchestrator — decomposes goals into subtasks."""

from __future__ import annotations

import textwrap

from anthropic import AnthropicVertex

from config import Config
from run_log import RunLog
from state import EvalResult, HarnessState, SubTask


_PLAN_TOOL: dict = {
    "name": "plan_tasks",
    "description": "Decompose the coding goal into an ordered list of subtasks.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":                  {"type": "string"},
                        "description":         {"type": "string"},
                        "acceptance_criteria": {"type": "string"},
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of subtasks that must pass before this one runs",
                        },
                        "role": {
                            "type": "string",
                            "enum": ["executor", "inspector"],
                            "description": (
                                "Which agent runs this subtask. 'executor' (default) is a "
                                "local open-weight coding model with full write/test/browser "
                                "tools — use for building, modifying, and verifying code. "
                                "'inspector' is a Claude Sonnet read-only reviewer — use for "
                                "diagnostic 'read these files and explain what's broken' "
                                "subtasks, typically inserted before a fix step after a "
                                "verification failure."
                            ),
                        },
                    },
                    "required": ["id", "description", "acceptance_criteria", "dependencies", "role"],
                },
            },
        },
        "required": ["subtasks"],
    },
}


def run_orchestrator(
    state: HarnessState,
    cfg: Config,
    log: RunLog,
    escalation: EvalResult | None = None,
    call_label: str = "initial",
) -> list[SubTask]:
    """Ask the orchestrator to (re)plan. Returns a fresh list of pending SubTasks."""
    client = AnthropicVertex(
        project_id=cfg.vertex_project,
        region=cfg.orchestrator_region,
    )

    completed_summary = "\n".join(
        f"  [{t.id}] {t.description}: {t.status}"
        for t in state.subtasks
    ) or "  none"

    escalation_section = ""
    if escalation:
        escalation_section = (
            f"\n\nEscalation from evaluator:\n"
            f"  reason: {escalation.reason}\n"
            f"  hint:   {escalation.hint or 'n/a'}"
        )

    prompt = textwrap.dedent(f"""
        Goal: {state.goal}

        Work directory: {cfg.work_dir}

        Subtask status so far:
        {completed_summary}
        {escalation_section}

        Call plan_tasks with the remaining subtasks needed to fully accomplish the goal.
        - Only include subtasks that are NOT yet marked passed.
        - Each acceptance_criteria must be verifiable programmatically.
        - Use dependency IDs to express ordering constraints.
        - Pick a `role` for each subtask:
            * "executor" (default) — for building, modifying, or running/verifying
              code. Has bash, read_file, write_file, run_tests, and Playwright
              browser tools.
            * "inspector" — for diagnostic "read and analyze" subtasks. Stronger
              at finding subtle bugs across files. Has read_file + read-only bash
              only (no writes, no tests, no browser). Use this BEFORE a fix step
              when a verification has failed and you want a careful diagnosis.
    """).strip()

    log.say(f"[orchestrator] planning ({call_label}) — model={cfg.orchestrator_model} region={cfg.orchestrator_region}")

    with log.time(f"orchestrator.{call_label}"):
        with client.messages.stream(
            model=cfg.orchestrator_model,
            max_tokens=cfg.orchestrator_max_tokens,
            thinking={"type": "adaptive"},
            tools=[_PLAN_TOOL],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            msg = stream.get_final_message()

    log.write_json(f"orchestrator/{call_label}.json", {
        "model":    cfg.orchestrator_model,
        "region":   cfg.orchestrator_region,
        "prompt":   prompt,
        "response": [b.model_dump() for b in msg.content],
        "usage":    msg.usage.model_dump() if msg.usage else None,
        "stop_reason": msg.stop_reason,
    })

    for block in msg.content:
        if block.type == "tool_use" and block.name == "plan_tasks":
            subtasks = [
                SubTask(
                    id=t["id"],
                    description=t["description"],
                    acceptance_criteria=t["acceptance_criteria"],
                    dependencies=t.get("dependencies", []),
                    role=t.get("role", "executor"),
                )
                for t in block.input["subtasks"]
            ]
            log.write_json(f"orchestrator/{call_label}_plan.json", subtasks)
            log.say(f"[orchestrator] plan: {len(subtasks)} subtask(s) in {log.timings[f'orchestrator.{call_label}']:.1f}s")
            for t in subtasks:
                deps = f" (after: {', '.join(t.dependencies)})" if t.dependencies else ""
                role_tag = f" <{t.role}>" if t.role != "executor" else ""
                log.say(f"  [{t.id}]{role_tag}{deps} {t.description}")
                log.say(f"       acceptance: {t.acceptance_criteria}")
            return subtasks

    raise RuntimeError(f"Orchestrator did not call plan_tasks.\nResponse: {msg}")
