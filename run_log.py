"""Per-run logging: structured artifacts on disk + live console output."""

from __future__ import annotations

import dataclasses
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


def _default(obj: Any) -> Any:
    """JSON serializer fallback — handles dataclasses, pydantic models, datetimes."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "model_dump"):       # pydantic / anthropic / openai response objects
        return obj.model_dump()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


class RunLog:
    """Writes structured artifacts under runs/<timestamp>/ and prints live updates."""

    def __init__(self, root: str = "runs", run_id: str | None = None):
        if run_id is None:
            self.run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            self.dir = Path(root) / self.run_id
            (self.dir / "tasks").mkdir(parents=True, exist_ok=True)
            self.timings: dict[str, float] = {}
            print(f"[harness] run log: {self.dir}")
        else:
            self.run_id = run_id
            self.dir = Path(root) / run_id
            if not self.dir.exists():
                raise SystemExit(f"cannot resume: {self.dir} does not exist")
            (self.dir / "tasks").mkdir(parents=True, exist_ok=True)
            timings_path = self.dir / "timings.json"
            if timings_path.exists():
                with open(timings_path) as f:
                    self.timings = json.load(f)
            else:
                self.timings = {}
            print(f"[harness] resuming run log: {self.dir}")

    @staticmethod
    def latest(root: str = "runs") -> str | None:
        """Find the most recent run-id under `root`, or None if none exists."""
        rp = Path(root)
        if not rp.exists():
            return None
        runs = sorted(p.name for p in rp.iterdir() if p.is_dir())
        return runs[-1] if runs else None

    # ------------------------------------------------------------------ writes

    def write_json(self, relpath: str, obj: Any) -> None:
        path = self.dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=_default)

    def append_jsonl(self, relpath: str, obj: Any) -> None:
        path = self.dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(obj, default=_default) + "\n")

    def task_path(self, task_id: str, filename: str) -> str:
        return f"tasks/{task_id}/{filename}"

    # ----------------------------------------------------------------- timing

    @contextmanager
    def time(self, label: str):
        t0 = time.time()
        try:
            yield
        finally:
            self.timings[label] = round(time.time() - t0, 2)
            self.write_json("timings.json", self.timings)

    # ----------------------------------------------------------- live console

    def say(self, msg: str) -> None:
        print(msg)
