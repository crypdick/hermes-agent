"""Regression: startup context-file reads must not wedge agent initialization."""

import time
from pathlib import Path

from agent.prompt_builder import build_context_files_prompt


def test_blocking_startup_agents_md_read_does_not_hang_prompt_build(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "AGENTS.md").write_text("slow project instructions", encoding="utf-8")
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_CONTEXT_FILE_READ_TIMEOUT", "0.2")

    real_read_text = Path.read_text

    def slow_read_text(self, *args, **kwargs):
        if self.name.lower() == "agents.md":
            time.sleep(5)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", slow_read_text)

    t0 = time.monotonic()
    result = build_context_files_prompt(cwd=str(tmp_path))
    elapsed = time.monotonic() - t0

    assert elapsed < 2.0, (
        f"build_context_files_prompt blocked {elapsed:.1f}s on a slow AGENTS.md "
        "read; startup context reads must be time-bounded"
    )
    assert "slow project instructions" not in result
