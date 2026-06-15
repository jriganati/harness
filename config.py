"""Configuration management."""

from __future__ import annotations

import os
try:
    import tomllib
except ModuleNotFoundError:
    raise SystemExit("Python 3.11+ is required for built-in tomllib support.")
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # project
    goal: str
    # vertex
    vertex_project: str
    # orchestrator
    orchestrator_model: str
    orchestrator_region: str
    orchestrator_max_tokens: int
    # evaluator
    evaluator_model: str
    evaluator_region: str
    evaluator_max_tokens: int
    # inspector
    inspector_model: str
    inspector_region: str
    inspector_max_tokens: int
    inspector_max_tool_rounds: int
    # executor
    executor_provider: str   # "openai-compat" or "anthropic"
    executor_base_url: str
    executor_region: str
    executor_model: str
    executor_temperature: float
    executor_max_tokens: int
    executor_max_tool_rounds: int
    # harness
    work_dir: str
    max_retries: int
    max_replans: int
    # mcp.playwright
    mcp_playwright_enabled: bool
    mcp_playwright_command: str
    mcp_playwright_args: list[str]


def load_config(path: str | Path | None = None) -> Config:
    """Load config from TOML. Falls back to GOOGLE_CLOUD_PROJECT env var if needed."""
    config_path = Path(path) if path else Path(__file__).parent / "config.toml"
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    t = raw.get("task", {})
    v = raw.get("vertex", {})
    o = raw.get("orchestrator", {})
    e = raw.get("evaluator", {})
    i = raw.get("inspector", {})
    x = raw.get("executor", {})
    h = raw.get("harness", {})
    mp = raw.get("mcp", {}).get("playwright", {})

    project = v.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit(
            "No GCP project set. Add it to config.toml [vertex] project = '...' "
            "or export GOOGLE_CLOUD_PROJECT."
        )

    return Config(
        goal=t.get("goal", "no goal set. check with the boss."),
        vertex_project=v.get("project", "default"),
        orchestrator_model=o.get("model", "claude-opus-4-7"),
        orchestrator_region=v.get("orchestrator_region", "us"),
        orchestrator_max_tokens=o.get("max_tokens", 16000),
        evaluator_model=e.get("model", "claude-haiku-4-5"),
        evaluator_region=v.get("evaluator_region", "us-east5"),
        evaluator_max_tokens=e.get("max_tokens", 1024),
        inspector_model=i.get("model", "claude-sonnet-4-6"),
        inspector_region=i.get("region", "us-east5"),
        inspector_max_tokens=i.get("max_tokens", 4000),
        inspector_max_tool_rounds=i.get("max_tool_rounds", 10),
        executor_provider=x.get("provider", "anthropic"),
        executor_base_url=x.get("base_url", "http://localhost:8001/v1"),
        executor_region=x.get("region", "us-east5"),
        executor_model=x.get("model", "claude-haiku-4-5"),
        executor_temperature=x.get("temperature", 0.2),
        executor_max_tokens=x.get("max_tokens", 8192),
        executor_max_tool_rounds=x.get("max_tool_rounds", 20),
        work_dir=h.get("work_dir", "/tmp/harness_workspace"),
        max_retries=h.get("max_retries", 3),
        max_replans=h.get("max_replans", 3),
        mcp_playwright_enabled=mp.get("enabled", False),
        mcp_playwright_command=mp.get("command", "npx"),
        mcp_playwright_args=mp.get("args", ["@playwright/mcp@latest", "--headless"]),
    )
