#!/usr/bin/env python3
"""
Three-role agentic coding harness.

  Orchestrator  — claude-opus-4-8 with adaptive thinking
                  Called once per plan (not in a loop). Decomposes goal →
                  SubTask list via tool_use. Re-plans on escalation.

  Executor      — open-weight model via llama.cpp on port 8001 (OpenAI-compat).
                  Runs its own internal tool loop: bash, read_file, write_file,
                  run_tests.

  Evaluator     — claude-haiku-4-5 (fast/cheap, called on every executor output).
                  Returns structured verdict: pass, retry_with_hint, or escalate.
                  Hint feeds back into executor's next prompt; escalate triggers
                  a full re-plan.

Configuration is read from config.toml in the same directory as this file.
Per-run artifacts (plan, conversations, timings, state) are written under runs/.
"""

from __future__ import annotations

import json
import os

import mcp_client
from config import Config, load_config
from evaluator import run_evaluator
from executor import EXHAUSTED_PREFIX, run_executor
from inspector import run_inspector
from orchestrator import run_orchestrator
from run_log import RunLog
from state import EvalResult, HarnessState, state_from_dict


def _checkpoint(log: RunLog, state: HarnessState) -> None:
    """Write the full state.json so a crash leaves a recoverable snapshot."""
    log.write_json("state.json", state)


def run_harness(goal: str, cfg: Config, resume_from: str | None = None) -> HarnessState:
    """Main harness loop. Returns final state.

    If `resume_from` is a run-id string, reuse that run's directory: load
    state.json, reset any in_progress tasks back to pending, and continue.
    """
    os.makedirs(cfg.work_dir, exist_ok=True)
    log = RunLog(run_id=resume_from)

    if resume_from:
        with open(log.dir / "state.json") as f:
            state = state_from_dict(json.load(f))
        # A task that was mid-flight when we crashed should be retried.
        for t in state.subtasks:
            if t.status == "in_progress":
                t.status = "pending"
        passed = sum(1 for t in state.subtasks if t.status == "passed")
        log.say(f"[harness] resumed: {passed}/{len(state.subtasks)} tasks already passed")
    else:
        state = HarnessState(goal=goal)
        log.write_json("goal.txt", goal)
        log.write_json("config.json", cfg)

    _checkpoint(log, state)
    log.say(f"\n[harness] goal: {goal[:120]}{'...' if len(goal) > 120 else ''}\n")

    try:
        return _run_harness_inner(goal, cfg, log, state)
    finally:
        mcp_client.shutdown()


def _run_harness_inner(goal: str, cfg: Config, log: RunLog, state: HarnessState) -> HarnessState:
    # Only call the orchestrator if we don't already have a plan (resume case).
    if not state.subtasks:
        state.subtasks = run_orchestrator(state, cfg, log, call_label="initial")
        _checkpoint(log, state)
    log.say("")

    replan_count = 0

    while True:
        runnable = next(
            (
                t for t in state.subtasks
                if t.status == "pending"
                and all(
                    any(s.id == dep and s.status == "passed" for s in state.subtasks)
                    for dep in t.dependencies
                )
            ),
            None,
        )

        if runnable is None:
            still_open = [t for t in state.subtasks if t.status in ("pending", "in_progress")]
            if not still_open:
                break  # everything resolved
            log.say("[harness] WARNING: no runnable task found (possible dep cycle); stopping")
            break

        t = runnable
        t.status = "in_progress"
        retries = state.iteration.get(t.id, 0)
        role_tag = f" <{t.role}>" if t.role != "executor" else ""
        log.say(f"[harness] ▶ [{t.id}]{role_tag} attempt {retries + 1}: {t.description}")
        _checkpoint(log, state)

        if t.role == "inspector":
            output = run_inspector(t, state, cfg, log)
        else:
            output = run_executor(t, state, cfg, log)
        state.artifacts[t.id] = output

        # Round-exhaustion is a "task too large" signal — skip the evaluator
        # and any further retries, escalate directly so the orchestrator can
        # decompose the work using the executor's progress summary.
        if output.startswith(EXHAUSTED_PREFIX):
            result = EvalResult(
                verdict="escalate",
                reason=f"executor exhausted {cfg.executor_max_tool_rounds} tool rounds; partial-progress summary attached",
                hint=output,
                score=0.0,
            )
            state.eval_scores[t.id] = result
            log.say(f"  [{t.id}] exhausted → skipping retries, escalating immediately")
        else:
            result = run_evaluator(t, output, cfg, log)
            state.eval_scores[t.id] = result
            eval_time = log.timings.get(f"evaluator.{t.id}", 0)
            log.say(f"  [{t.id}] eval → {result.verdict} (score={result.score:.2f}, {eval_time:.1f}s): {result.reason}")

        if result.verdict == "pass":
            t.status = "passed"
            log.say(f"  [{t.id}] ✓ passed\n")

        elif result.verdict == "retry_with_hint":
            state.iteration[t.id] = retries + 1
            if state.iteration[t.id] >= cfg.max_retries:
                result = EvalResult(
                    verdict="escalate",
                    reason=f"max retries ({cfg.max_retries}) exhausted",
                    hint=result.hint,
                    score=result.score,
                )
                state.eval_scores[t.id] = result
                log.say(f"  [{t.id}] max retries reached, escalating")
            else:
                t.status = "pending"
                t.hint = result.hint
                log.say(f"  [{t.id}] retry with hint: {result.hint}\n")
                _checkpoint(log, state)
                continue

        if result.verdict == "escalate":
            t.status = "failed"
            if replan_count >= cfg.max_replans:
                log.say(f"  [{t.id}] ✗ max re-plans reached; marking failed\n")
            else:
                replan_count += 1
                log.say(f"  [{t.id}] escalating to orchestrator (replan #{replan_count})...")
                new_tasks = run_orchestrator(
                    state, cfg, log,
                    escalation=result,
                    call_label=f"replan{replan_count}",
                )
                state.subtasks = [s for s in state.subtasks if s.status == "passed"] + new_tasks
                log.say(f"  replanned: {len(new_tasks)} new subtask(s)\n")

        _checkpoint(log, state)

    passed = sum(1 for t in state.subtasks if t.status == "passed")
    total = len(state.subtasks)
    log.say(f"\n[harness] finished: {passed}/{total} subtasks passed")
    log.say(f"[harness] artifacts in: {log.dir}")
    _checkpoint(log, state)
    return state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Three-role agentic coding harness")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--resume", metavar="RUN_ID", help="Resume from a specific run directory")
    g.add_argument("--resume-latest", action="store_true", help="Resume from the most recent run directory")
    args = parser.parse_args()

    resume_id: str | None = args.resume
    if args.resume_latest:
        resume_id = RunLog.latest()
        if resume_id is None:
            raise SystemExit("--resume-latest: no prior runs found under runs/")

    cfg = load_config()
    run_harness(cfg.goal.strip(), cfg, resume_from=resume_id)
