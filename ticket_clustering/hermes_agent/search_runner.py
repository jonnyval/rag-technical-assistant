"""Stateless Hermes one-shot runner used by the OpenAI-compatible API."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = Path(__file__).resolve().with_name("SEARCH_AGENT_PROMPT.md")


class HermesSearchRunner:
    def __init__(self, timeout_seconds: int = 900) -> None:
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        self.executable = Path(local_app_data) / "hermes" / "hermes-agent" / "venv" / "Scripts" / "hermes.exe"
        self.timeout_seconds = max(60, int(timeout_seconds))
        self.instructions = PROMPT_PATH.read_text(encoding="utf-8").strip()

    def run(self, query: str) -> tuple[str, dict[str, Any]]:
        normalized = " ".join(str(query).split()).strip()
        if not normalized:
            raise ValueError("Empty Hermes search query")
        if len(normalized) > 12_000:
            raise ValueError("Hermes search query is longer than 12000 characters")
        if not self.executable.is_file():
            raise RuntimeError(f"Hermes executable not found: {self.executable}")

        prompt = f"{self.instructions}\n\n# Вопрос пользователя\n\n{normalized}"
        descriptor, usage_name = tempfile.mkstemp(prefix="reglab_hermes_usage_", suffix=".json")
        os.close(descriptor)
        usage_path = Path(usage_name)
        try:
            completed = subprocess.run(
                [
                    str(self.executable),
                    "--oneshot",
                    prompt,
                    "--toolsets",
                    "reglab_search",
                    "--ignore-rules",
                    "--usage-file",
                    str(usage_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown Hermes error").strip()
                raise RuntimeError(f"Hermes search failed with exit code {completed.returncode}: {detail[-2000:]}")
            answer = completed.stdout.strip()
            if not answer:
                raise RuntimeError("Hermes returned an empty answer")
            usage: dict[str, Any] = {}
            if usage_path.is_file() and usage_path.stat().st_size:
                usage = json.loads(usage_path.read_text(encoding="utf-8"))
            return answer, usage
        finally:
            usage_path.unlink(missing_ok=True)
