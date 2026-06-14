"""Regression: a blocking hint-file read must not wedge the agent turn.

Root cause (mac-mini diary 2026-06-13): the daily body-debugger cron hung for
600s every run. A thread dump pinned it to ``subdirectory_hints``'s
``_load_hints_for_directory`` doing an UNBOUNDED ``read_text()`` on a hint file
(AGENTS.md / CLAUDE.md / .cursorrules) for a directory the next tool was about
to touch. The file's ``is_file()`` stat returned fine but the ``open()/read()``
blocked indefinitely (a half-synced / stale / provider-backed vault path), with
no timeout and no activity heartbeat — so the cron inactivity killer (600s) was
the only thing that ever stopped it.

This best-effort context hook must be time-bounded: a slow/stuck hint read is
skipped, never blocking the turn.
"""
import time
from pathlib import Path

from agent.subdirectory_hints import SubdirectoryHintTracker


def test_blocking_hint_read_does_not_hang_check_tool_call(tmp_path, monkeypatch):
    sub = tmp_path / "area"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("backend hint")
    target = sub / "file.py"
    target.write_text("x = 1\n")

    monkeypatch.setenv("HERMES_HINT_READ_TIMEOUT", "1")

    real_read_text = Path.read_text

    def slow_read_text(self, *args, **kwargs):
        # Simulate a hint file whose read blocks (stale/slow FS path).
        # Match case-insensitively: macOS tmp dirs are case-insensitive, so the
        # tracker probes both "AGENTS.md" and "agents.md" against the same file.
        if self.name.lower() == "agents.md":
            time.sleep(10)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", slow_read_text)

    tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))

    t0 = time.monotonic()
    result = tracker.check_tool_call("read_file", {"path": str(target)})
    elapsed = time.monotonic() - t0

    assert elapsed < 5.0, (
        f"check_tool_call blocked {elapsed:.1f}s on a slow hint read; "
        "the hint read must be time-bounded"
    )
    # The stuck hint is skipped, so no hint text is returned.
    assert result is None


def test_normal_hint_read_still_works(tmp_path, monkeypatch):
    """Bounding must not break the normal fast path."""
    sub = tmp_path / "backend"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("Use FastAPI; add type hints")
    target = sub / "main.py"
    target.write_text("print('hi')\n")

    monkeypatch.setenv("HERMES_HINT_READ_TIMEOUT", "3")
    tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
    result = tracker.check_tool_call("read_file", {"path": str(target)})
    assert result is not None
    assert "FastAPI" in result
