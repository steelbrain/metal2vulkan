"""The matrix's sizing of its jobs, on recorded runs whose answers are known.

    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/matrix
"""

import importlib.machinery
import importlib.util
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

_loader = importlib.machinery.SourceFileLoader("matrix", str(Path(__file__).resolve().parent / "matrix"))
_spec = importlib.util.spec_from_loader("matrix", _loader)
matrix = importlib.util.module_from_spec(_spec)
_loader.exec_module(matrix)


def run_of(stats):
    """A matrix run holding STATS, {kind@node: [runs]}, and nothing else it needs to size jobs."""
    run = object.__new__(matrix.Run)
    run.stats = stats
    run.lock = threading.Lock()
    run.spaces = {}
    run.nodes = {"host": {"os": "windows"}}
    run.short = "abc"
    run.jobs = []
    return run


def recorded(ram, peak, minutes=10, killed=False, full=False):
    return dict(ram=ram, peak=peak, minutes=minutes, killed=killed, full=full)


class SizeTests(unittest.TestCase):
    def test_a_build_is_sized_from_whole_builds_alone(self):
        # Incremental builds peaked near 2 GiB: their peaks size nothing, so the default stands.
        incremental = [recorded(3, 2.1), recorded(3, 2.0), recorded(3, 2.2)]
        self.assertEqual(run_of({"winbuild@host": incremental}).size("winbuild", "host"), (16, 60))
        # A whole build that peaked at 9.5 GiB: 1.3 times it, rounded up.
        whole = incremental + [recorded(16, 9.5, full=True)]
        self.assertEqual(run_of({"winbuild@host": whole}).size("winbuild", "host")[0], 13)
        # One that ended for want of its 16 GiB: twice that.
        killed = whole + [recorded(16, 16, killed=True, full=True)]
        self.assertEqual(run_of({"winbuild@host": killed}).size("winbuild", "host")[0], 32)

    def test_minutes_follow_the_last_three_runs_of_either_form(self):
        runs = [recorded(16, 9, minutes=50, full=True), recorded(3, 2, minutes=8), recorded(3, 2, minutes=9)]
        self.assertEqual(run_of({"build@n": runs}).size("build", "n")[1], 75)
        self.assertEqual(run_of({"build@n": runs[1:]}).size("build", "n")[1], 30)


class RetryTests(unittest.TestCase):
    def test_a_job_that_wanted_ram_runs_again_with_twice_its_room_up_to_the_default(self):
        run = run_of({"winbuild@host": [recorded(3, 2.0, full=True)]})
        asked = []

        def potluck(*args, cwd=None):
            if args[0] == "dispatch":
                asked.append(int(args[args.index("--ram") + 1]))
                reply = {"job": f"j{len(asked)}", "state": "queued"}
            else:
                ram = asked[-1]
                reply = {"recent": [{"id": f"j{len(asked)}", "started_at": 1, "ended_at": 60001,
                                     "exit_code": 137, "workspace": None,
                                     "usage": {"peak_mem_bytes": ram * 2**30}}]}
            return subprocess.CompletedProcess(args, 0, json.dumps(reply), "")

        waited = subprocess.CompletedProcess([], 137, "", "rustc-LLVM ERROR: out of memory")
        with tempfile.TemporaryDirectory() as rdir, \
                mock.patch.object(matrix, "potluck", potluck), \
                mock.patch.object(matrix.subprocess, "run", return_value=waited), \
                mock.patch.object(matrix, "log"):
            run.rdir = Path(rdir)
            code, _, _ = run.job("winbuild", "host", "true", "winbuild-leg", space="winbuild")
        # Sized from its whole build's 2 GiB peak, then doubled up to the default, 16 GiB.
        self.assertEqual(asked, [4, 8, 16])
        self.assertEqual(code, 137)


if __name__ == "__main__":
    unittest.main()
