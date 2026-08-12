"""Subprocess supervision helpers for GUI-launched workers."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any, Sequence


def worker_popen(args: Sequence[str], **kwargs: Any) -> subprocess.Popen:
    """Launch a worker in its own process group/session when the OS supports it."""
    if os.name == "nt":
        flags = int(kwargs.pop("creationflags", 0) or 0)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs.setdefault("start_new_session", True)
    return subprocess.Popen(args, **kwargs)


def open_in_file_manager(path: Any) -> None:
    """Open a folder (or file) in the platform's file manager.

    Best effort: raises when the platform launcher itself cannot be started,
    so callers surface the failure in their own UI/log.
    """
    import sys

    target = str(path)
    if sys.platform.startswith("win"):
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


def terminate_process_tree(proc: "subprocess.Popen | None", *,
                           timeout: float = 1.5) -> None:
    """Terminate ``proc`` and any children it spawned, returning quickly."""
    if proc is None or proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=min(max(float(timeout), 0.1), 0.75))
            _close_stdout(proc)
            return
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(float(timeout), 0.1),
                check=False,
            )
            if result.returncode != 0 and proc.poll() is None:
                proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        _wait_briefly(proc, timeout)
        _close_stdout(proc)
        return

    pgid = None
    try:
        pgid = os.getpgid(proc.pid)
        if pgid != os.getpgrp():
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=max(float(timeout), 0.1))
        _close_stdout(proc)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        if pgid is not None and pgid != os.getpgrp():
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    _wait_briefly(proc, 0.5)
    _close_stdout(proc)


def _wait_briefly(proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + max(float(timeout), 0.0)
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)


def _close_stdout(proc: subprocess.Popen) -> None:
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except Exception:
        pass
