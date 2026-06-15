"""Inspector — read-only code analysis using Claude Sonnet.

Used by the orchestrator for "look at the code and explain what's wrong" steps,
which the local executor model has been weak at. The inspector has read-only
tools (read_file, bash for grep/ls/cat) — no write_file, no test runner.
Returns a text analysis that downstream executor tasks can use as a hint.
"""

from __future__ import annotations

import json
import textwrap

from anthropic import AnthropicVertex

from config import Config
from run_log import RunLog
from state import HarnessState, SubTask
from tools import _tool_bash, _tool_read_file

# Same sentinel as the executor — harness short-circuits both to escalate.
from executor import EXHAUSTED_PREFIX


# Read-only tool subset for the inspector (Anthropic tool format, not OpenAI)
_INSPECTOR_TOOLS: list[dict] = [
    {
        "name": "read_file",
        "description": "Read a text file. Path is relative to workspace unless absolute.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "bash",
        "description": "Run a read-only shell command (grep, ls, cat, find, wc, etc.). "
                       "Do NOT use for file modification.",
        "input_schema": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
]


def _dispatch(name: str, args: dict, work_dir: str) -> str:
    if name == "read_file":
        return _tool_read_file(args.get("path", ""), work_dir)
    if name == "bash":
        return _tool_bash(args.get("cmd", ""), work_dir)
    return f"error: unknown tool '{name}'"


def run_inspector(
    subtask: SubTask,
    state: HarnessState,
    cfg: Config,
    log: RunLog,
) -> str:
    """Run the inspector's tool loop. Returns its final text analysis."""
    client = AnthropicVertex(
        project_id=cfg.vertex_project,
        region=cfg.inspector_region,
    )

    hint_note = f"\n\nPREVIOUS ATTEMPT FEEDBACK: {subtask.hint}" if subtask.hint else ""

    system = textwrap.dedent(f"""
        You are a careful code reviewer. Your workspace is {cfg.work_dir}.

        Use read_file and bash (read-only commands like grep/ls/cat/find) to
        examine the code described in the task. Then produce a concise,
        prioritized analysis.

        SCOPE DISCIPLINE — IMPORTANT:
        Be efficient. Aim to produce your analysis in 3-6 tool calls total:
          - Read the relevant file(s) once
          - Do AT MOST one or two targeted verifications (grep, small simulation)
          - Then write your analysis
        Do NOT exhaustively enumerate every possible bug. Focus on the SINGLE
        MOST LIKELY root cause of the reported symptom. If you find it on the
        first read, stop reading and write up.

        ANTI-PATTERNS that waste rounds:
          - Multiple `node -e` simulations to "trace through the code" — at most
            ONE such trace is allowed, only if static reading was inconclusive
          - "Now let me check one more thing..." — don't. Write up what you have.
          - Speculating about unrelated bugs in the same area. Stay focused on
            the task's stated symptom.

        Your final response (assistant text, no tool call) must include:
          1. The ONE most likely root cause (file:line + brief reasoning)
          2. A concrete suggested fix
          3. Optionally: a short list of secondary concerns (max 3, one line each)

        Do NOT modify any files — you only read and analyze.
    """).strip()

    user = textwrap.dedent(f"""
        Task [{subtask.id}]: {subtask.description}

        Acceptance criteria: {subtask.acceptance_criteria}
        {hint_note}
    """).strip()

    messages: list[dict] = [{"role": "user", "content": user}]

    jsonl_path = log.task_path(subtask.id, "inspector.jsonl")
    log.append_jsonl(jsonl_path, {"role": "system", "content": system})
    log.append_jsonl(jsonl_path, {"role": "user", "content": user})

    log.say(f"[inspector] reading code — model={cfg.inspector_model} region={cfg.inspector_region}")

    tool_call_count = 0
    timing_label = f"inspector.{subtask.id}.attempt{state.iteration.get(subtask.id, 0) + 1}"

    with log.time(timing_label):
        for turn_idx in range(cfg.inspector_max_tool_rounds):
            msg = client.messages.create(
                model=cfg.inspector_model,
                max_tokens=cfg.inspector_max_tokens,
                system=system,
                tools=_INSPECTOR_TOOLS,
                messages=messages,
            )

            log.append_jsonl(jsonl_path, {
                "turn": turn_idx,
                "stop_reason": msg.stop_reason,
                "content": [b.model_dump() for b in msg.content],
                "usage": msg.usage.model_dump() if msg.usage else None,
            })

            # Collect tool calls and final text
            assistant_content: list[dict] = []
            text_parts: list[str] = []
            tool_uses: list = []
            for block in msg.content:
                assistant_content.append(block.model_dump())
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            messages.append({"role": "assistant", "content": assistant_content})

            if msg.stop_reason == "end_turn" or not tool_uses:
                final = "\n".join(text_parts).strip() or "(inspector produced no text)"
                log.say(f"  [{subtask.id}] inspector done — {tool_call_count} tool call(s), {turn_idx + 1} turn(s)")
                return final

            # Execute tool calls and feed results back
            tool_results: list[dict] = []
            for tu in tool_uses:
                tool_call_count += 1
                log.say(f"  [{subtask.id}]   → {tu.name}({_summarize_args(tu.name, tu.input)})")
                result = _dispatch(tu.name, tu.input, cfg.work_dir)
                log.say(f"  [{subtask.id}]     ← {result.splitlines()[0][:120] if result else ''}"
                        f"  ({len(result.splitlines())} lines)")
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tu.id,
                    "content":     result,
                })
                log.append_jsonl(jsonl_path, {
                    "role": "tool", "tool_use_id": tu.id, "name": tu.name, "content": result,
                })
            messages.append({"role": "user", "content": tool_results})

    log.say(f"  [{subtask.id}] inspector hit max tool rounds ({cfg.inspector_max_tool_rounds}) — requesting partial analysis")
    messages.append({
        "role": "user",
        "content": textwrap.dedent("""
            You have exhausted your tool round budget. Do NOT call any more tools.
            Without further tools, produce your best-guess analysis based on what
            you've seen so far, in this exact format:

            MOST LIKELY ROOT CAUSE:
              - <file:line + brief reasoning, even if uncertain>

            SUGGESTED FIX:
              - <concrete change>

            UNCERTAINTY:
              - <what you'd want to verify if you had more rounds>
        """).strip(),
    })
    try:
        wrap = client.messages.create(
            model=cfg.inspector_model,
            max_tokens=cfg.inspector_max_tokens,
            system=system,
            messages=messages,
        )
        summary = "\n".join(b.text for b in wrap.content if b.type == "text").strip() \
                  or "(inspector produced no summary)"
    except Exception as e:
        summary = f"(summary call failed: {e})"
    log.append_jsonl(jsonl_path, {"role": "assistant", "content": summary, "exhaustion_summary": True})
    return f"{EXHAUSTED_PREFIX}\n{summary}"


def _summarize_args(name: str, args: dict) -> str:
    if name == "read_file":
        return args.get("path", "")
    if name == "bash":
        cmd = args.get("cmd", "")
        return f'"{cmd[:80]}{"..." if len(cmd) > 80 else ""}"'
    return json.dumps(args)[:80]
