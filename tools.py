"""Executor tool implementations and schema."""

import json
import os
import subprocess


def _tool_bash(cmd: str, work_dir: str) -> str:
    # Detect a backgrounded command and run it fully detached so subprocess.run
    # doesn't hang waiting for the bg child to close its inherited stdout/stderr.
    stripped = cmd.rstrip()
    if stripped.endswith("&") and not stripped.endswith("&&"):
        bg_cmd = stripped[:-1].rstrip()
        try:
            subprocess.Popen(
                bg_cmd, shell=True, cwd=work_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return f"started in background: {bg_cmd}"
        except Exception as e:
            return f"error: {e}"

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            cwd=work_dir, timeout=30,
        )
        out = result.stdout[-4000:] if result.stdout else ""
        err = result.stderr[-2000:] if result.stderr else ""
        return f"exit={result.returncode}\n{out}{err}".strip()
    except subprocess.TimeoutExpired:
        return "error: command timed out after 30s"
    except Exception as e:
        return f"error: {e}"


def _tool_read_file(path: str, work_dir: str) -> str:
    target = path if os.path.isabs(path) else os.path.join(work_dir, path)
    try:
        with open(target) as f:
            return f.read()
    except Exception as e:
        return f"error: {e}"


def _tool_write_file(path: str, content: str, work_dir: str) -> str:
    target = path if os.path.isabs(path) else os.path.join(work_dir, path)
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    try:
        with open(target, "w") as f:
            f.write(content)
        return f"wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"error: {e}"


def _tool_run_tests(pattern: str, work_dir: str) -> str:
    return _tool_bash(
        f"python -m pytest {pattern} -v --tb=short 2>&1 | tail -80",
        work_dir,
    )


def dispatch_tool(name: str, args: dict, work_dir: str) -> str:
    """Route a tool call to its implementation. Returns an error string
    (rather than raising) when required arguments are missing or malformed,
    so the executor model can see the mistake and self-correct."""
    def _pick(*keys: str) -> str | None:
        """Return the first non-empty arg value from `keys`, else None."""
        for k in keys:
            v = args.get(k)
            if v not in (None, ""):
                return v
        return None

    def _missing(label: str, *keys: str) -> str:
        sent = json.dumps(args)[:200] if args else "(empty)"
        return (
            f"error: tool '{name}' requires arg '{label}' "
            f"(accepted: {', '.join(keys)}). You sent: {sent}"
        )

    if name == "bash":
        # Accept both 'command' (Anthropic convention) and 'cmd' (legacy / Qwen)
        cmd = _pick("command", "cmd")
        return _tool_bash(cmd, work_dir) if cmd else _missing("command", "command", "cmd")

    if name == "read_file":
        path = _pick("path", "file_path", "filename")
        return _tool_read_file(path, work_dir) if path else _missing("path", "path", "file_path", "filename")

    if name == "write_file":
        path = _pick("path", "file_path", "filename")
        content = args.get("content")
        if content is None:
            content = args.get("text")  # accept 'text' as fallback
        if not path:
            return _missing("path", "path", "file_path", "filename")
        if content is None:
            return _missing("content", "content", "text")
        return _tool_write_file(path, content, work_dir)

    if name == "run_tests":
        return _tool_run_tests(args.get("pattern", ""), work_dir)

    return f"error: unknown tool '{name}'"


# Tool schema sent to the executor (OpenAI tool_call format)
EXECUTOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command. Returns exit code, stdout, and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file. Path is relative to workspace unless absolute.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file, creating parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest on a file or glob pattern. Returns test output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "pytest target, e.g. 'test_calc.py' or 'calc/'",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]
