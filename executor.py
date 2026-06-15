"""Executor — coding agent tool loop.

Two providers supported, selected by cfg.executor_provider:

  "openai-compat"  — local llama.cpp / Qwen-Coder via the OpenAI chat-completions
                     protocol on cfg.executor_base_url. Free per-token but
                     less reliable at tool-call formatting and final summaries.
  "anthropic"      — Claude (typically Haiku) via Vertex AI. Costs money per
                     token but much stronger at multi-step tool use, can see
                     screenshots, and rarely malforms tool arguments.

Both providers share: prompt construction, tool schemas, MCP integration,
local tool dispatch, and the executor.jsonl log format.
"""

from __future__ import annotations

import json
import textwrap

from anthropic import AnthropicVertex
from openai import OpenAI

import mcp_client
from config import Config
from run_log import RunLog
from state import HarnessState, SubTask
from tools import EXECUTOR_TOOLS, dispatch_tool


# Sentinel prefix on the artifact when the executor hit its round budget.
# The harness detects this and short-circuits to escalation (no retries),
# passing the summary that follows the prefix to the orchestrator's replan.
EXHAUSTED_PREFIX = "[EXECUTOR_EXHAUSTED]"

_EXHAUSTION_PROMPT = textwrap.dedent("""
    You have exhausted your tool round budget for this subtask. Do NOT call any
    more tools. Without further tool calls, produce a concise progress report
    so that a planner can decide how to break this work into smaller pieces.

    Use this exact format (3 short sections, each 1-5 bullet points):

    ACCOMPLISHED:
      - <files written/modified, tests run, observations confirmed>

    INCOMPLETE:
      - <what remains; specific subtasks or fixes still needed>

    NEXT STEPS:
      - <what you would do next if you had more rounds, in order>
""").strip()


# ---------------------------------------------------------------- shared


def _summarize_args(name: str, args: dict) -> str:
    if name == "bash":
        cmd = args.get("cmd", "")
        return f'"{cmd[:80]}{"..." if len(cmd) > 80 else ""}"'
    if name == "read_file":
        return args.get("path", "")
    if name == "write_file":
        path = args.get("path", "")
        size = len(args.get("content", ""))
        return f"{path} ({size} bytes)"
    if name == "run_tests":
        return args.get("pattern", "")
    if name.startswith("browser_"):
        key = next(iter(args), None)
        if key:
            return f"{key}={str(args[key])[:80]}"
        return ""
    return json.dumps(args)[:80]


def _summarize_result(result: str) -> str:
    if not result:
        return ""
    if result.startswith("error:") or "Error" in result.splitlines()[0]:
        return result[:400].replace("\n", " ⏎ ")
    lines = result.splitlines()
    first = lines[0][:120]
    if len(lines) > 1:
        return f"{first}  ({len(lines)} lines, {len(result)} chars total)"
    return first


def _build_prompts(subtask: SubTask, state: HarnessState, cfg: Config) -> tuple[str, str]:
    prior_artifacts = "\n\n".join(
        f"=== {tid} ===\n{code}"
        for tid, code in state.artifacts.items()
        if any(t.id == tid and t.status == "passed" for t in state.subtasks)
    ) or "none"

    hint_note = f"\n\nPREVIOUS ATTEMPT FEEDBACK: {subtask.hint}" if subtask.hint else ""

    browser_hint = (
        " Browser automation tools (browser_navigate, browser_click, browser_type, "
        "browser_snapshot, browser_take_screenshot, etc.) are also available via "
        "Playwright MCP for testing web UIs."
        if cfg.mcp_playwright_enabled else ""
    )

    system = textwrap.dedent(f"""
        You are an expert coding agent. Your workspace is {cfg.work_dir}.
        Use the provided tools (bash, read_file, write_file, run_tests) to
        implement the assigned task.{browser_hint}

        SCOPE DISCIPLINE — IMPORTANT:
        Do exactly what the acceptance criteria require, and no more. The
        following are ANTI-PATTERNS that waste rounds and pollute the workspace:
          - Writing extra test files, summary documents, validation scripts,
            integration tests, comprehensive demos, or "final verification"
            artifacts that weren't explicitly requested.
          - Repeating the same verification multiple ways ("Perfect! Now let me
            create one more comprehensive test...").
          - Writing markdown summaries or completion reports as files.
        ONE quick verification call is enough (e.g., a single `node --check` or
        `grep` to confirm the change took). After that, write your final summary
        and STOP. If your acceptance criteria are met and you're tempted to "do
        one more thing for completeness" — don't. That's the anti-pattern.

        IMPORTANT — HOW TO PRODUCE YOUR FINAL OUTPUT:
        Your final summary MUST be written as your own assistant text message
        (no tool call). Do NOT use bash/echo/cat to "print" your summary —
        bash output is private tool data, not your response. The harness only
        captures the text you write yourself in your final assistant message.

        End-of-task checklist (in your final text message, no tools):
          1. Describe what you did (files created/modified, tools used)
          2. Confirm each acceptance criterion is met, with brief evidence
          3. List any issues, defects, or partial completions

        Never end your turn with empty content. If you cannot complete the task,
        say so explicitly in text and explain why. A response with no text after
        tool use will be marked as failed.

        Previously completed artifacts (already in workspace):
        {prior_artifacts}
    """).strip()

    user = textwrap.dedent(f"""
        Task [{subtask.id}]: {subtask.description}

        Acceptance criteria: {subtask.acceptance_criteria}
        {hint_note}

        Use your tools, verify your work, then stop when the criteria are met.
    """).strip()

    return system, user


def _assemble_tools_openai(cfg: Config) -> tuple[list[dict], set[str]]:
    """Build the OpenAI-format tool list and return (tools, mcp_names)."""
    tools = list(EXECUTOR_TOOLS)
    mcp_names: set[str] = set()
    if cfg.mcp_playwright_enabled:
        client = mcp_client.get_client(cfg.mcp_playwright_command, cfg.mcp_playwright_args)
        mcp_tools = client.list_tools()
        tools.extend(mcp_tools)
        mcp_names = {t["function"]["name"] for t in mcp_tools}
    return tools, mcp_names


def _to_anthropic_tools(openai_tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool schemas to Anthropic format."""
    return [
        {
            "name":         t["function"]["name"],
            "description":  t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for t in openai_tools
    ]


def _dispatch(name: str, args: dict, cfg: Config, mcp_names: set[str]) -> str:
    """Route a tool call to local handler or MCP."""
    if name in mcp_names:
        return mcp_client.get_client(
            cfg.mcp_playwright_command, cfg.mcp_playwright_args
        ).call_tool(name, args)
    return dispatch_tool(name, args, cfg.work_dir)


# -------------------------------------------------------------- dispatch


def run_executor(subtask: SubTask, state: HarnessState, cfg: Config, log: RunLog) -> str:
    """Run the executor for one subtask; dispatches to the configured provider."""
    if cfg.executor_provider == "anthropic":
        return _run_executor_anthropic(subtask, state, cfg, log)
    return _run_executor_openai(subtask, state, cfg, log)


# ---------------------------------------------------------- openai-compat


def _run_executor_openai(subtask: SubTask, state: HarnessState, cfg: Config, log: RunLog) -> str:
    client = OpenAI(base_url=cfg.executor_base_url, api_key="none")
    tools, mcp_names = _assemble_tools_openai(cfg)
    system, user = _build_prompts(subtask, state, cfg)

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    jsonl_path = log.task_path(subtask.id, "executor.jsonl")
    log.append_jsonl(jsonl_path, {"role": "system", "content": system})
    log.append_jsonl(jsonl_path, {"role": "user", "content": user})

    tool_call_count = 0
    timing_label = f"executor.{subtask.id}.attempt{state.iteration.get(subtask.id, 0) + 1}"

    with log.time(timing_label):
        for turn_idx in range(cfg.executor_max_tool_rounds):
            response = client.chat.completions.create(
                model=cfg.executor_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=cfg.executor_temperature,
                max_tokens=cfg.executor_max_tokens,
            )
            choice = response.choices[0]

            assistant_msg: dict = {"role": "assistant", "content": choice.message.content}
            if choice.message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ]
            messages.append(assistant_msg)
            log.append_jsonl(jsonl_path, {"turn": turn_idx, **assistant_msg})

            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                final_text = choice.message.content
                if not final_text and tool_call_count > 0:
                    log.say(f"  [{subtask.id}] executor stopped with empty content — requesting summary")
                    messages.append({
                        "role": "user",
                        "content": "Please now produce the required final summary: "
                                   "describe what you did, confirm each acceptance "
                                   "criterion with evidence, and list any issues found.",
                    })
                    followup = client.chat.completions.create(
                        model=cfg.executor_model,
                        messages=messages,
                        temperature=cfg.executor_temperature,
                        max_tokens=cfg.executor_max_tokens,
                    )
                    final_text = followup.choices[0].message.content or "(no text output after follow-up)"
                    log.append_jsonl(jsonl_path, {"role": "assistant", "content": final_text, "followup": True})
                log.say(f"  [{subtask.id}] executor done — {tool_call_count} tool call(s), {turn_idx + 1} turn(s)")
                return final_text or "(no text output)"

            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_call_count += 1
                log.say(f"  [{subtask.id}]   → {tc.function.name}({_summarize_args(tc.function.name, args)})")
                result = _dispatch(tc.function.name, args, cfg, mcp_names)
                log.say(f"  [{subtask.id}]     ← {_summarize_result(result)}")

                tool_msg = {
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         tc.function.name,
                    "content":      result,
                }
                messages.append({k: v for k, v in tool_msg.items() if k != "name"})
                log.append_jsonl(jsonl_path, tool_msg)

    log.say(f"  [{subtask.id}] executor hit max tool rounds ({cfg.executor_max_tool_rounds}) — requesting progress summary")
    messages.append({"role": "user", "content": _EXHAUSTION_PROMPT})
    try:
        wrap = client.chat.completions.create(
            model=cfg.executor_model,
            messages=messages,
            temperature=cfg.executor_temperature,
            max_tokens=cfg.executor_max_tokens,
        )
        summary = wrap.choices[0].message.content or "(model produced no summary)"
    except Exception as e:
        summary = f"(summary call failed: {e})"
    log.append_jsonl(jsonl_path, {"role": "assistant", "content": summary, "exhaustion_summary": True})
    return f"{EXHAUSTED_PREFIX}\n{summary}"


# ------------------------------------------------------------- anthropic


def _run_executor_anthropic(subtask: SubTask, state: HarnessState, cfg: Config, log: RunLog) -> str:
    client = AnthropicVertex(project_id=cfg.vertex_project, region=cfg.executor_region)
    openai_tools, mcp_names = _assemble_tools_openai(cfg)
    tools = _to_anthropic_tools(openai_tools)
    system, user = _build_prompts(subtask, state, cfg)

    messages: list[dict] = [{"role": "user", "content": user}]

    jsonl_path = log.task_path(subtask.id, "executor.jsonl")
    log.append_jsonl(jsonl_path, {"role": "system", "content": system})
    log.append_jsonl(jsonl_path, {"role": "user", "content": user})

    tool_call_count = 0
    timing_label = f"executor.{subtask.id}.attempt{state.iteration.get(subtask.id, 0) + 1}"

    with log.time(timing_label):
        for turn_idx in range(cfg.executor_max_tool_rounds):
            msg = client.messages.create(
                model=cfg.executor_model,
                max_tokens=cfg.executor_max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )

            log.append_jsonl(jsonl_path, {
                "turn": turn_idx,
                "stop_reason": msg.stop_reason,
                "content": [b.model_dump() for b in msg.content],
                "usage": msg.usage.model_dump() if msg.usage else None,
            })

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
                final = "\n".join(text_parts).strip()
                if not final and tool_call_count > 0:
                    log.say(f"  [{subtask.id}] executor stopped with empty content — requesting summary")
                    messages.append({
                        "role": "user",
                        "content": "Please now produce the required final summary: "
                                   "describe what you did, confirm each acceptance "
                                   "criterion with evidence, and list any issues found.",
                    })
                    followup = client.messages.create(
                        model=cfg.executor_model,
                        max_tokens=cfg.executor_max_tokens,
                        system=system,
                        messages=messages,
                    )
                    final = "\n".join(
                        b.text for b in followup.content if b.type == "text"
                    ).strip() or "(no text output after follow-up)"
                    log.append_jsonl(jsonl_path, {
                        "role": "assistant", "content": final, "followup": True,
                    })
                log.say(f"  [{subtask.id}] executor done — {tool_call_count} tool call(s), {turn_idx + 1} turn(s)")
                return final or "(no text output)"

            # Execute tool calls and feed results back as a single user message
            tool_results: list[dict] = []
            for tu in tool_uses:
                tool_call_count += 1
                log.say(f"  [{subtask.id}]   → {tu.name}({_summarize_args(tu.name, tu.input)})")
                result = _dispatch(tu.name, tu.input, cfg, mcp_names)
                log.say(f"  [{subtask.id}]     ← {_summarize_result(result)}")
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tu.id,
                    "content":     result,
                })
                log.append_jsonl(jsonl_path, {
                    "role": "tool", "tool_use_id": tu.id, "name": tu.name, "content": result,
                })
            messages.append({"role": "user", "content": tool_results})

    log.say(f"  [{subtask.id}] executor hit max tool rounds ({cfg.executor_max_tool_rounds}) — requesting progress summary")
    messages.append({"role": "user", "content": _EXHAUSTION_PROMPT})
    try:
        wrap = client.messages.create(
            model=cfg.executor_model,
            max_tokens=cfg.executor_max_tokens,
            system=system,
            messages=messages,
        )
        summary = "\n".join(
            b.text for b in wrap.content if b.type == "text"
        ).strip() or "(model produced no summary)"
    except Exception as e:
        summary = f"(summary call failed: {e})"
    log.append_jsonl(jsonl_path, {"role": "assistant", "content": summary, "exhaustion_summary": True})
    return f"{EXHAUSTED_PREFIX}\n{summary}"
