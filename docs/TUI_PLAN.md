# Multi-pane TUI for the Harness — Design Plan

Status: not started. Captured 2026-06-15 from conversation.

## Why

The single-stream console output hides too much when many things happen in
parallel: the orchestrator's plan, which task is active, what tool calls it's
making, retry counts, per-role stats. A multi-pane TUI lets you see the full
picture at a glance.

## Layout (4 panes)

```
┌─ Plan ─────────────────────┬─ Active task ────────────────────┐
│ ✓ setup_dir       0.5s     │ ▶ inspect_code <inspector>       │
│ ✓ create_html     8.2s     │   model: claude-sonnet-4-6       │
│ ✓ create_css      6.1s     │   elapsed: 4.2s / max 60s        │
│ ✓ create_js     45.0s [2×] │   current tool: read_file(...)   │
│ ✗ playwright_v  120s [3×]  │   tool calls so far: 3           │
│ ▶ inspect_code  4.2s       │                                  │
│ ⋯ fix_defects              │                                  │
│ ⋯ pw_reverify              │                                  │
├─ Conversation ─────────────┴─ Stats ───────────────────────────┤
│ → read_file(calculator.js) │ Orchestrator   Opus      2 calls │
│   ← 414 lines, 13KB        │ Inspector      Sonnet    1 call  │
│ → bash("grep handleClick") │ Executor       local     8 calls │
│   ← exit=0  (12 lines)     │ Evaluator      Haiku     6 calls │
│ → read_file(index.html)    │ Wall clock: 4:32                 │
│   ← 87 lines, 2KB          │ Tokens (Claude): 28K in / 9K out │
│                            │ Tool calls total: 24             │
└────────────────────────────┴──────────────────────────────────┘
```

### Pane responsibilities

| Pane | Shows | Updates on |
|---|---|---|
| **Plan** | All subtasks with status, role, duration, retry count, dependencies | `plan_received`, any task state change |
| **Active task** | Current task ID, role, model, elapsed time, current tool call | `task_started`, `tool_called`, `tool_returned` |
| **Conversation** | Live stream of `→ tool(...)` / `← result` lines for the active task | `tool_called`, `tool_returned` |
| **Stats** | Per-role call counts, wall-clock total, token totals, tool counts | Every event (cheap to recompute) |

## Library choice

**Textual** (https://textual.textualize.io)

- Modern Python TUI framework, reactive widgets
- Built on Rich for formatting
- Built-in widgets for tables (Plan, Stats), live log (Conversation), label panels (Active task)
- Runs over SSH, supports mouse and keyboard
- No JS toolchain — `pip install textual` is enough

## Architecture decision

### Event bus refactor (prerequisite, useful on its own)

Refactor `RunLog` to emit structured events to a list of listeners. Default
listener writes to stdout + JSONL (current behavior). TUI is another listener.

```python
class RunLog:
    def __init__(self, ..., listeners: list[Callable[[Event], None]] = None):
        self.listeners = listeners or []

    def emit(self, event_type: str, **data):
        for fn in self.listeners:
            fn(Event(type=event_type, ts=time.time(), data=data))
```

Convert existing `log.say` and `log.write_json` call sites to emit typed events.
~30 lines of changes. Cleans up the implicit "print to stdout" coupling even if
the TUI never ships.

### Event types

About a dozen, all currently implicit in `log.say` calls:

- `plan_received` — orchestrator returned a plan
- `task_started` — main loop picked a runnable task
- `tool_called` — executor/inspector invoked a tool
- `tool_returned` — tool produced a result
- `task_eval_passed` / `task_eval_retry` / `task_eval_escalate`
- `replan_triggered` — escalation caused a new orchestrator call
- `harness_finished`

### Same-process vs out-of-process TUI

**Recommended: out-of-process viewer.**

- Harness writes events to a unix socket or JSONL file in the run dir
- TUI is a separate `viewer.py` that tails it
- Run with: `python viewer.py runs/2026-06-15T13-07-55` (live or replay)

Benefits:
- Event log on disk is useful for its own sake (post-mortem, replay)
- TUI doesn't have to be running for harness to work — attach/detach freely
- Watch a run on another terminal or even another machine (over SSH)
- Replay old runs by tailing their event log

The alternative (same-process Textual app with harness in a background thread)
is simpler but couples the two and blocks the harness if the TUI is slow.

## Implementation phases

### Phase 1: Event bus refactor (standalone, ~1 hr)

- Add `Event` dataclass and `RunLog.emit(event_type, **data)` method
- Add a `JsonlFileListener` writing one event per line to `events.jsonl` in the
  run dir
- Convert existing `log.say` and explicit `write_json` calls to emit events,
  with a `ConsoleListener` reproducing today's output for backward compat
- Test: existing harness runs produce identical console output, plus a new
  `events.jsonl` per run

### Phase 2: Minimal viewer (~2 hrs)

- New `viewer.py` script — single-file Textual app
- Takes a run dir path; tails its `events.jsonl`
- Implements all four panes
- Keyboard: `q` quit, `r` restart from beginning of log (replay)
- Supports both live (still-running) and finished runs

### Phase 3: Polish (~1 hr)

- Color the Plan pane by status (green/yellow/red/grey)
- Token usage tracking (need to thread `msg.usage` into events from
  orchestrator, inspector, evaluator)
- Cost estimate per role based on model pricing
- Highlight current tool call in Active task pane
- `viewer.py runs/latest` shortcut for the most recent run

### Phase 4 (optional): Multi-run dashboard

- `viewer.py` without args shows a list of all runs and lets you select one
- Filter by status, goal substring
- Compare two runs side by side

## Open questions

1. Should the harness auto-spawn the viewer on launch? (Probably not — keep
   concerns separated, but a config flag could be nice.)
2. Should we keep the existing console output once events are in place, or
   trim it to just headlines and rely on the viewer for detail?
3. Token accounting requires capturing `msg.usage` from every Claude call.
   Worth threading through now or defer to Phase 3?

## Files that will change

- `run_log.py` — add Event, emit(), listener support
- `harness.py`, `orchestrator.py`, `executor.py`, `inspector.py`, `evaluator.py`
  — convert `log.say` calls to typed `log.emit(...)` calls
- New: `viewer.py` — Textual app
- New: `events.py` — Event dataclass + type constants (or keep in `run_log.py`)
