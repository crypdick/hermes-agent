"""Best-effort filesystem helpers for optional context files.

Some provider-backed or half-synced paths can make ``open()``/``read()`` hang
even after a stat succeeds. These helpers keep optional context discovery from
wedging an agent turn.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def _bounded_call(
    operation: str,
    path: Path,
    timeout: float,
    fn: Callable[[], T],
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[T]:
    result: dict[str, object] = {}

    def _worker() -> None:
        try:
            result["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - best-effort optional IO
            result["err"] = exc

    thread = threading.Thread(
        target=_worker,
        daemon=True,
        name=f"bounded-file-io:{operation}",
    )
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        if logger:
            logger.warning(
                "%s exceeded %.1fs, skipping (possible stale/slow path): %s",
                operation,
                timeout,
                path,
            )
        return None
    if "err" in result:
        if logger:
            logger.debug("Could not %s %s: %s", operation.lower(), path, result["err"])
        return None
    return result.get("value")  # type: ignore[return-value]


def read_text_bounded(
    path: Path,
    timeout: float,
    *,
    encoding: str = "utf-8",
    logger: Optional[logging.Logger] = None,
    operation: str = "File read",
) -> Optional[str]:
    return _bounded_call(
        operation,
        path,
        timeout,
        lambda: path.read_text(encoding=encoding),
        logger=logger,
    )


def path_exists_bounded(
    path: Path,
    timeout: float,
    *,
    logger: Optional[logging.Logger] = None,
    operation: str = "Path exists check",
) -> Optional[bool]:
    return _bounded_call(operation, path, timeout, path.exists, logger=logger)


def path_is_file_bounded(
    path: Path,
    timeout: float,
    *,
    logger: Optional[logging.Logger] = None,
    operation: str = "Path file check",
) -> Optional[bool]:
    return _bounded_call(operation, path, timeout, path.is_file, logger=logger)


def path_is_dir_bounded(
    path: Path,
    timeout: float,
    *,
    logger: Optional[logging.Logger] = None,
    operation: str = "Path directory check",
) -> Optional[bool]:
    return _bounded_call(operation, path, timeout, path.is_dir, logger=logger)


def glob_bounded(
    path: Path,
    pattern: str,
    timeout: float,
    *,
    logger: Optional[logging.Logger] = None,
    operation: str = "Directory glob",
) -> Optional[list[Path]]:
    return _bounded_call(
        operation,
        path,
        timeout,
        lambda: list(path.glob(pattern)),
        logger=logger,
    )
