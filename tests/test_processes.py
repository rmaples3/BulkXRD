"""Worker subprocess cancellation helpers."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seriesxrd.core.processes import terminate_process_tree, worker_popen


def _pid_running(pid: int) -> bool:
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
        except Exception:
            return False
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_terminate_process_tree_kills_stdout_child_quickly():
    with tempfile.TemporaryDirectory() as td:
        child_script = Path(td) / "child.py"
        parent_script = Path(td) / "parent.py"
        child_script.write_text(
            "import time\n"
            "print('child-started', flush=True)\n"
            "time.sleep(60)\n",
            encoding="utf-8",
        )
        parent_script.write_text(
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, sys.argv[1]])\n"
            "print(f'child-pid {child.pid}', flush=True)\n"
            "time.sleep(60)\n",
            encoding="utf-8",
        )

        proc = worker_popen(
            [sys.executable, str(parent_script), str(child_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        child_pid = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            line = proc.stdout.readline().strip()
            if line.startswith("child-pid "):
                child_pid = int(line.split()[1])
                break
        assert child_pid is not None

        t0 = time.monotonic()
        terminate_process_tree(proc, timeout=2.0)
        elapsed = time.monotonic() - t0

        assert proc.poll() is not None
        assert elapsed < 3.0
        deadline = time.monotonic() + 2.0
        while _pid_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_running(child_pid)


def test_open_in_file_manager_uses_platform_launcher(monkeypatch, tmp_path):
    from seriesxrd.core import processes

    launched = []
    monkeypatch.setattr(
        processes.subprocess, "Popen",
        lambda args, **_kwargs: launched.append(list(args)),
    )
    if sys.platform.startswith("win"):
        monkeypatch.setattr(
            processes.os, "startfile",
            lambda target: launched.append(["startfile", target]),
            raising=False,
        )
    processes.open_in_file_manager(tmp_path)
    assert launched and str(tmp_path) in launched[0][-1]


def main() -> None:
    test_terminate_process_tree_kills_stdout_child_quickly()
    print("PROCESS TEST OK")


if __name__ == "__main__":
    main()
