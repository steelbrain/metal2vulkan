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


class PlanTests(unittest.TestCase):
    """The legs a run plans from the nodes' GPUs and the config."""

    nodes = {
        "a": {"os": "linux", "gpus": [{"vendor": "10de", "device": "2204", "id": "0000:01:00.0"},
                                      {"vendor": "8086", "device": "a7a0", "id": "0000:00:02.0"}]},
        "b": {"os": "linux", "gpus": [{"vendor": "10de", "device": "2204", "id": "0000:05:00.0"}]},
    }

    def legs(self, cfg):
        with mock.patch.object(matrix, "log"):
            return sorted((leg.id, leg.node) for leg in matrix.plan(self.nodes, cfg, None)
                          if leg.id not in matrix.CPU_LEGS)

    def test_a_card_on_two_nodes_has_one_leg_unless_the_config_names_its_node_of_record(self):
        self.assertEqual(len([leg for leg, _ in self.legs({}) if leg.startswith("10de:2204")]), 1)
        self.assertEqual(self.legs({"record_nodes": {"10de:2204": "b"}}),
                         [("10de:2204", "b"), ("10de:2204@a", "a"), ("8086:a7a0", "a")])
        # Its node of record away: the other runs as the extra leg, and none is of record.
        self.assertEqual(self.legs({"record_nodes": {"10de:2204": "c"}}),
                         [("10de:2204@a", "a"), ("10de:2204@b", "b"), ("8086:a7a0", "a")])

    def test_an_extra_leg_selects_its_device_by_its_card(self):
        leg = matrix.Leg("1002:744c@a", "a", "0000:03:00.0", "radeon", True)
        script = matrix.gpu_script("abc", "", [leg], "POTLUCK_MIRROR_x", "0" * 40, "rt", False)
        self.assertIn("leg 1002:744c@a radeon '1002:744c!'", script)


class LayerTests(unittest.TestCase):
    """The shell a device-test job runs to unpack the carried validation layer, run for real."""

    def run_lines(self, layer, cwd, env=None):
        lines = matrix.layer_lines(layer, "linux", "leg")
        script = lines + '\necho "path $VK_LAYER_PATH"'
        return subprocess.run(["bash", "-c", script], cwd=cwd, env=env, capture_output=True, text=True).stdout

    def test_a_job_unpacks_its_os_directory_and_points_the_loader_at_it(self):
        with tempfile.TemporaryDirectory() as carry, tempfile.TemporaryDirectory() as work:
            (Path(carry) / "linux").mkdir()
            (Path(carry) / "linux/VERSION").write_text("1.2.3 (from somewhere)\n")
            for args in (["init", "-q"], ["add", "linux"],
                         ["-c", "user.name=t", "-c", "user.email=t@localhost", "commit", "-q", "-m", "layer"]):
                subprocess.run(["git", *args], cwd=carry, check=True)
            rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=carry, capture_output=True, text=True).stdout.strip()
            env = {"PATH": "/usr/bin:/bin", "POTLUCK_MIRROR_layer": str(Path(carry) / ".git")}
            out = self.run_lines(("POTLUCK_MIRROR_layer", rev), work, env)
            self.assertIn("M2V-DEVICE leg validationLayer 1.2.3 (from somewhere)", out)
            pointed = Path(out.split("path ", 1)[1].strip())
            self.assertEqual(pointed.resolve(), (Path(work) / ".validation-layer/linux").resolve())

    def test_a_job_with_no_layer_says_so_and_finds_none(self):
        with tempfile.TemporaryDirectory() as work:
            for layer in (None, ("POTLUCK_MIRROR_absent", "0" * 40)):
                out = self.run_lines(layer, work, {"PATH": "/usr/bin:/bin"})
                self.assertIn("M2V-DEVICE leg validationLayer missing: ", out)
                self.assertIn("/.no-validation-layer", out)


#: A stand-in for the harness command: it logs its `--runtime` and `--only`, and prints a time
#: per runtime it names, a row of the creation workloads under `--only`, else a timed row and a
#: first-use record.
FAKE_HARNESS = r"""#!/bin/bash
runtimes= only=
while [ $# -gt 0 ]; do
  case $1 in --runtime) runtimes=$2; shift ;; --only) only=$2; shift ;; esac
  shift
done
echo "$runtimes ${only:-all}" >> "$LOG"
for runtime in ${runtimes//,/ }; do
  case $only in
    create_*) echo "M2V-MATRIX time harness-$runtime/create_state/first 1000.000" ;;
    '') echo "M2V-MATRIX time harness-$runtime/alu 10.000"
        echo "M2V-MATRIX time harness-$runtime/authored/first_use_wait_total 5.000" ;;
  esac
done
"""


class HarnessRunTests(unittest.TestCase):
    def test_both_runtimes_are_timed_in_one_process_and_so_are_the_creation_rows_in_each_of_theirs(self):
        with tempfile.TemporaryDirectory() as work:
            fake, log = Path(work) / "harness", Path(work) / "log"
            fake.write_text(FAKE_HARNESS)
            fake.chmod(0o755)
            script = matrix.harness_runs(f'"{work}"', f'"{fake}"', "a")
            run = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                                 env={"PATH": "/usr/bin:/bin", "LOG": str(log), "TMPDIR": work})
            self.assertEqual(run.returncode, 0, run.stderr)
            calls = log.read_text().splitlines()
        # One process times both runtimes, the new runtime's pass first in each workload; the warm
        # run is the new runtime's; each of the five creation processes times both too.
        self.assertEqual(calls, ["new,old all", "new all"] + ["new,old create_*"] * 5)
        legs = matrix.results.parse_job("M2V-LEG a begin\n" + run.stdout)
        suites = legs["a"]["suites"]
        self.assertEqual(list(suites), [matrix.HARNESS_SUITE])
        times = suites[matrix.HARNESS_SUITE]["records"]["time"]
        self.assertEqual(sorted(key for key in times if not key.startswith(matrix.results.HARNESS_FIRST)), [
            "harness-new-warm/authored/first_use_wait_total", "harness-new/alu",
            "harness-new/authored/first_use_wait_total", "harness-new/create_state/first",
            "harness-old/alu", "harness-old/authored/first_use_wait_total", "harness-old/create_state/first"])
        self.assertEqual(sorted(key for key in times if key.startswith(matrix.results.HARNESS_FIRST)),
                         [f"{matrix.results.HARNESS_FIRST}create_state/first/{n}/{runtime}"
                          for n in range(2, 6) for runtime in ("new", "old")])
        self.assertEqual(suites[matrix.HARNESS_SUITE]["duplicated"], set())


if __name__ == "__main__":
    unittest.main()
