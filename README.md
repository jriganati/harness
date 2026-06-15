# Harness

A three-role agentic coding harness for understanding how agent frameworks work
under the hood. Decomposes a natural-language goal into subtasks, executes them
with appropriate models, verifies output, and re-plans on failure.

This is a learning / experimentation tool, not production code.

---

## Architecture

Four model roles, each chosen for its job, plus a deterministic state machine
that wires them together.

```
                ┌────────────────────────────────────────┐
                │           run_harness loop             │
                │  (state machine, no model intelligence)│
                └────────────────────────────────────────┘
                   │           │            │           │
       orchestrate │   execute │    inspect │  evaluate │
                   ▼           ▼            ▼           ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
            │   Opus   │ │ Haiku /  │ │  Sonnet  │ │  Haiku   │
            │ adaptive │ │ Qwen /   │ │ read-only│ │  verdict │
            │ thinking │ │ local    │ │  review  │ │  tool    │
            └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### Roles

| Role | Default model | When called | Why this tier |
|---|---|---|---|
| **Orchestrator** | `claude-opus-4-7` | Once per plan; again on each escalation | Needs adaptive thinking and judgment to decompose a goal and to re-plan after failure. Called rarely, so the cost is amortized. |
| **Executor** | `claude-haiku-4-5` (or local Qwen-Coder) | Once per subtask, plus once per retry | Highest-volume role. Lots of turns per task. Provider is swappable: `anthropic` (reliable, costs $) or `openai-compat` (local, free, less reliable). |
| **Inspector** | `claude-sonnet-4-6` | Only when the orchestrator marks a subtask `role: inspector` (usually a diagnostic step after a verification failure) | Read-only code review needs cross-file reasoning. Stronger than Haiku, cheaper than Opus. Has no `write_file` — analysis only. |
| **Evaluator** | `claude-haiku-4-5` | Once per executor/inspector attempt | Constrained classification: does the output meet the acceptance criteria? `pass` / `retry_with_hint` / `escalate`. |

The state machine in `harness.py` is pure Python — no model intelligence in the
control flow. It picks dependency-ready subtasks, dispatches to the right role,
processes the evaluator verdict, and decides when to escalate.

---

## The main loop

```
load config → instantiate run log (or resume existing)
  ↓
orchestrator → returns ordered SubTask list
  ↓
LOOP forever:
  pick next pending subtask whose dependencies all passed
  if none → break (done)

  dispatch by t.role:
    "executor"  → run_executor   (local or Anthropic)
    "inspector" → run_inspector  (Sonnet, read-only)

  if output begins with [EXECUTOR_EXHAUSTED] sentinel:
    skip evaluator → build escalate verdict → replan

  else:
    run_evaluator → pass | retry_with_hint | escalate

  on pass:    mark passed, continue
  on retry:   bump iteration, restore to pending with hint
              (if iteration >= max_retries → promote to escalate)
  on escalate: mark failed, call orchestrator again with state + reason
               state.subtasks = passed tasks + new plan
```

The orchestrator is called **once initially** plus **once per escalation**, up
to `max_replans`. Everything else is the inner loop.

---

## Configuration (`config.toml`)

```toml
[task]
goal = """
    <free-text description of what to build, with acceptance criteria>
"""

[vertex]
project = "your-gcp-project-id"             # or env $GOOGLE_CLOUD_PROJECT

[orchestrator]
model      = "claude-opus-4-7"
region     = "us"
max_tokens = 16000

[evaluator]
model      = "claude-haiku-4-5"
region     = "us-east5"
max_tokens = 1024

[inspector]
model           = "claude-sonnet-4-6"
region          = "us-east5"
max_tokens      = 4000
max_tool_rounds = 10

[executor]
provider        = "anthropic"              # or "openai-compat"
region          = "us-east5"               # for "anthropic" path
base_url        = "http://localhost:8001/v1"  # for "openai-compat" path
model           = "claude-haiku-4-5"       # use "local" for openai-compat
temperature     = 0.2
max_tokens      = 4096
max_tool_rounds = 20

[harness]
work_dir    = "/tmp/harness_workspace"     # where the agents write files
max_retries = 3                             # per-subtask retry budget
max_replans = 3                             # max times we'll re-call Opus

[mcp.playwright]
enabled = true
command = "npx"
args    = ["@playwright/mcp@latest", "--headless"]
```

Each role has its own model and region — different Claude models are
available in different Vertex regions, so they're independent.

---

## Modules

| File | Responsibility |
|---|---|
| `harness.py` | Main loop. Owns dependency-ordered subtask selection, role dispatch, retry/escalate logic, state checkpointing, MCP lifecycle. |
| `orchestrator.py` | Calls Opus with adaptive thinking to produce a SubTask list via the `plan_tasks` tool. Re-called on escalation with the failure reason and current state. |
| `executor.py` | Two providers behind one `run_executor()` entry point. Builds prompts, assembles tools (local + MCP), runs the tool loop, handles round-exhaustion. |
| `inspector.py` | Sonnet with a read-only tool subset (`read_file`, `bash`). Returns a text analysis used as input for downstream fix steps. |
| `evaluator.py` | Haiku, single non-loop call. Forced to call `submit_verdict` tool returning `pass` / `retry_with_hint` / `escalate` + reason + optional hint. |
| `tools.py` | Local tool implementations (`bash`, `read_file`, `write_file`, `run_tests`) and the OpenAI-format `EXECUTOR_TOOLS` schema. Dispatch is **lenient**: accepts arg-name synonyms across providers (`command`/`cmd` for bash, `path`/`file_path`/`filename` for file tools, `content`/`text` for writes). When a required arg is missing, the error message echoes back the arguments the model sent and lists accepted names, so the next turn can self-correct. `bash` also auto-detaches commands ending in `&` so a backgrounded server doesn't hang the tool call. |
| `mcp_client.py` | Sync wrapper around an MCP stdio server (currently Playwright). Runs a background thread with its own asyncio event loop so the sync executor can call into MCP. One persistent connection per harness run; browser stays open across all subtasks. |
| `state.py` | `SubTask`, `EvalResult`, `HarnessState` dataclasses plus `state_from_dict()` for resume. |
| `run_log.py` | `RunLog` — creates `runs/<timestamp>/` and writes structured artifacts. Tracks timings via `time(label)` context manager. Supports resume into an existing run dir. |
| `config.py` | Loads `config.toml` into a `Config` dataclass. |

---

## Data flow per subtask

```
SubTask(id, description, acceptance_criteria, dependencies, status, hint, role)
        │
        ▼
  dispatch by role:                 returns text string
   ─────────────────                 ────────────────────
   executor  →  Haiku/local model
                + bash/read/write/test
                + Playwright MCP tools          → state.artifacts[t.id]

   inspector →  Sonnet
                + read_file + read-only bash    → state.artifacts[t.id]

        │
        ▼
  evaluator (Haiku, sees only the text artifact + acceptance criteria)
        │
        ▼
  EvalResult(verdict, reason, hint, score)  →  state.eval_scores[t.id]
        │
        ▼
  branch on verdict:
   pass            → mark passed, continue
   retry_with_hint → restore to pending, attach hint, increment iteration
   escalate        → mark failed, call orchestrator for replan
```

Persistent communication between subtasks happens via two channels:

1. **`state.artifacts`** — the text output of every passed subtask is included
   in the next executor's system prompt as `prior_artifacts`.
2. **The filesystem** — files in `work_dir` written by one subtask remain on
   disk for any subsequent subtask to read.

---

## MCP / Playwright

Browser automation is exposed to the executor via the official `@playwright/mcp`
server (Node, spawned by npx). On the first executor call of a run:

1. `mcp_client.get_client()` spawns `npx @playwright/mcp@latest --headless`
2. A background thread owns an asyncio event loop that holds the stdio session
3. `list_tools()` fetches the MCP tool catalog (~15 tools: `browser_navigate`,
   `browser_click`, `browser_type`, `browser_snapshot`, `browser_take_screenshot`,
   etc.)
4. Tools are converted from MCP schema → OpenAI format and merged with
   `EXECUTOR_TOOLS`
5. The dispatcher routes by name: anything in the MCP tool set goes to
   `client.call_tool()`; everything else to local `dispatch_tool()`

The Playwright browser stays open across all subtasks in a single run, so the
executor can navigate once and reuse the page. `mcp_client.shutdown()` runs in
a `try/finally` in `run_harness` to guarantee cleanup even on crash.

---

## Logging — what's in `runs/<id>/`

```
runs/2026-06-15T13-07-55/
├── config.json                       # snapshot of the loaded Config
├── goal.txt                          # the input goal
├── state.json                        # full HarnessState, rewritten after every transition
├── timings.json                      # {label: seconds} for every model/tool call
├── orchestrator/
│   ├── initial.json                  # full request/response/usage for the first plan
│   ├── initial_plan.json             # parsed SubTask list
│   ├── replan1.json                  # on escalation #1
│   ├── replan1_plan.json
│   └── ...
└── tasks/
    ├── <subtask_id>/
    │   ├── executor.jsonl            # every turn of the tool loop, one JSON per line
    │   │                             # OR inspector.jsonl for inspector tasks
    │   └── evaluator_attemptN.json   # one per attempt
    └── ...
```

`state.json` is the source of truth for resume. It's rewritten after every
state transition (task start, retry, escalation, pass/fail). If the process
crashes, the most recent `state.json` reflects exactly where it stopped.

---

## Failure handling

### Three failure modes, three responses

| Failure type | Detection | Response |
|---|---|---|
| **Agent produced wrong output** | Evaluator returns `retry_with_hint` | Restart same agent with the hint injected. Up to `max_retries` times. |
| **Agent is fundamentally broken on this task** | Evaluator returns `escalate`, OR retries exhausted | Call orchestrator for a replan with current state + failure reason. Up to `max_replans` times. |
| **Agent ran out of tool rounds** (executor or inspector) | Output starts with `[EXECUTOR_EXHAUSTED]` sentinel | Skip evaluator, skip retries, escalate immediately with the agent's own progress summary embedded as the hint. |

### Round exhaustion (the trickiest case)

When an agent hits its `max_tool_rounds` without finishing, retrying with no
memory just hits the same wall. So when this happens (in either the executor
or the inspector):

1. The agent makes **one final non-tool call** asking itself to summarize
   what it did, what's incomplete, and what it would do next, in a fixed format.
   - Executor format: `ACCOMPLISHED / INCOMPLETE / NEXT STEPS`
   - Inspector format: `MOST LIKELY ROOT CAUSE / SUGGESTED FIX / UNCERTAINTY`
2. That summary is returned as the artifact with the prefix
   `[EXECUTOR_EXHAUSTED]` (shared sentinel for both agents).
3. The harness detects the prefix, skips the evaluator, skips remaining
   retries, and goes straight to `escalate`.
4. The orchestrator's replan prompt includes the summary as the escalation
   hint, so Opus can decompose the task using actual progress information
   instead of guessing.

Net effect: round-exhausted tasks save dozens of wasted rounds and produce
sharper replans.

### Prompt discipline (anti-padding, anti-paralysis)

Each agent's system prompt includes specific anti-pattern instructions
discovered empirically during development:

**Executor — "scope discipline":**
- Do exactly what acceptance criteria require, no more
- Do NOT write extra test files, summary docs, validation scripts,
  integration tests, or completion reports unless explicitly requested
- ONE verification call is enough — after that, write the final summary and STOP
- No "Perfect! Now let me create one more comprehensive test..." padding
- Final summary must be assistant text (not bash echo output)

**Inspector — "focus discipline":**
- Aim for 3-6 tool calls total
- Find the SINGLE most likely root cause; don't enumerate every possibility
- At most ONE `node -e` simulation, only if static reading was inconclusive
- Output must identify THE root cause (singular), not a list
- No "let me check one more thing" loops

These exist because Claude models (especially Sonnet for inspection, Haiku for
execution) are trained to be thorough — without explicit scope guards they
default to writing extensive verification artifacts and exploratory traces.
Tightening the prompts cuts typical task durations 3-5x.

### Replan semantics

When the orchestrator is re-called, it replaces (does not insert) the unfinished
tail of the task list:

```python
state.subtasks = [t for t in state.subtasks if t.status == "passed"] + new_tasks
```

Everything `pending`/`in_progress`/`failed` is discarded. Opus sees the full
state (what passed, what failed, the escalation reason) and emits a fresh
forward plan. The main loop then picks up where it left off.

---

## Resume

```bash
python harness.py --resume-latest          # reopen most recent run
python harness.py --resume <run-id>        # reopen a specific run
```

What happens:

1. `RunLog(run_id=...)` reopens the existing dir instead of creating a new one
2. `state.json` is loaded and rehydrated via `state_from_dict()`
3. Any task left in `in_progress` is reset to `pending` (so the crashed-mid-task
   gets retried)
4. The orchestrator call is **skipped** if a plan is already loaded — saves an
   Opus call
5. The main loop resumes; logs append to the same files

What persists across resume:
- Completed task artifacts (no re-execution)
- Evaluator verdicts and retry counts
- Timings (new ones append to the same `timings.json`)

What doesn't:
- The Playwright browser session — spawned fresh, executor re-navigates as
  needed (idempotent)

---

## CLI

```bash
# Fresh run (uses task.goal from config.toml)
python harness.py

# Resume the most recent run after a crash
python harness.py --resume-latest

# Resume a specific run
python harness.py --resume 2026-06-15T13-07-55
```

---

## Setup

```bash
cd /path/to/dir/harness
python3 -m venv .venv
source .venv/bin/activate
pip install anthropic[vertex] openai mcp

# Vertex auth (one-time)
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="gcp-project-name"

# Playwright (one-time)
brew install node
npx playwright install chromium

# Optional — local llama.cpp for executor_provider="openai-compat"
# Run llama-server on port 8001 with Qwen2.5-Coder or similar
```

---

## Glossary

- **Subtask** — one indivisible unit of work the orchestrator emits. Has a
  description, acceptance criteria, dependencies on other subtask IDs, a
  status, and a role.
- **Role** — which agent runs a subtask. `executor` (default) or `inspector`.
  Selected by the orchestrator per subtask.
- **Provider** — which backend the executor uses: `anthropic` (Vertex Claude)
  or `openai-compat` (local llama.cpp). Set in `[executor] provider`.
- **Verdict** — the evaluator's judgment: `pass` (criteria met), `retry_with_hint`
  (close but fixable), or `escalate` (fundamentally broken, replan needed).
- **Iteration** — retry count for a single subtask. Resets on replan.
- **Replan** — when the orchestrator is called again mid-run due to escalation.
- **Round exhaustion** — when the executor used all `max_tool_rounds` without
  producing a final text response.

---

## See also

- `docs/TUI_PLAN.md` — design plan for a multi-pane TUI viewer of live and
  historical runs (not yet implemented).
