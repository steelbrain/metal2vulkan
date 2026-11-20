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
    run.failures = {}
    run.carried = {}
    return run


def recorded(ram, peak, minutes=10, killed=False, full=False):
    return dict(ram=ram, peak=peak, minutes=minutes, killed=killed, full=full)


class SizeTests(unittest.TestCase):
    def test_a_build_is_sized_from_whole_builds_alone(self):
        # Incremental builds peaked near 2 GiB: their peaks size nothing, so the default stands.
        incremental = [recorded(3, 2.1), recorded(3, 2.0), recorded(3, 2.2)]
        self.assertEqual(run_of({"winbuild@host": incremental}).size("winbuild", "host"), (10, 15))
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
        # Sized from its whole build's 2 GiB peak, then doubled up to the default, 10 GiB.
        self.assertEqual(asked, [4, 8, 10])
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
    *) echo "M2V-MATRIX time harness-$runtime/$only 20.000" ;;
  esac
done
"""


class TimedAgainTests(unittest.TestCase):
    def test_a_harness_past_the_old_runtime_s_spread_is_timed_once_more_after_the_node_s_other_jobs(self):
        run = run_of({})
        run.harness, run.layer, run.cfg = True, None, {}
        legs = [matrix.Leg("1002:744c", "n", "0000:06:00.0", "radeon"),
                matrix.Leg("10de:1e87", "n", "0000:09:00.0", "nvidia")]
        # Each timing's old runtime runs of its one re-timed row: the main run's, then its pairs'.
        olds = {"1002:744c": [[100.0, 115.0], [100.0, 104.0], [100.0, 140.0]], "10de:1e87": [[100.0, 101.0]]}
        asked = []

        def job(kind, node, script, name, gpus=(), carries=(), space=None, exclusive=False, room=None,
                hold_line=False):
            asked.append((name, 'rm -rf "$S"' in script))
            if name.startswith("rt-"):
                return 0, "".join(f"M2V-LEG {leg.id} begin\nM2V-SUITE {leg.id} rt exit 0\n" for leg in legs), {"id": name}
            leg_id = name.split("-", 1)[1].removesuffix("-again")
            lines = [f"M2V-LEG {leg_id} begin", "M2V-MATRIX time harness-old/alu 100.0",
                     "M2V-MATRIX time harness-new/alu 110.0"]
            for pair, old in enumerate(olds[leg_id].pop(0), 1):
                lines += [f"M2V-MATRIX time harness-retime/alu/{pair}/old {old}",
                          f"M2V-MATRIX time harness-retime/alu/{pair}/new {old * 1.1:.1f}"]
            return 0, "\n".join(lines + [f"M2V-SUITE {leg_id} harness exit 0"]) + "\n", {"id": name}

        with mock.patch.object(run, "job", job, create=True), mock.patch.object(matrix, "log"):
            run.gpu_node("n", legs, lambda: (("carry", "mirror", "rev"), None))
        # The XTX's old runtime spread 15% in its first timing: it is timed once more, after the
        # 2080's, and that last job is the node's last, which tidies; once more only.
        self.assertEqual(asked, [("rt-n", False), ("harness-1002:744c", False), ("harness-10de:1e87", False),
                                 ("harness-1002:744c-again", True)])
        self.assertEqual([(round(spread, 2), row) for spread, row in legs[0].timings], [(0.15, "alu"), (0.04, "alu")])
        self.assertEqual([(round(spread, 2), row) for spread, row in legs[1].timings], [(0.01, "alu")])
        self.assertEqual(run.carried, {"n": ("carry", "mirror", "rev")})


class OutlierRetimeTests(unittest.TestCase):
    def test_each_leg_is_timed_once_more_alone_its_records_taking_the_first_pass_s(self):
        run = run_of({})
        run.harness, run.layer, run.cfg = True, None, {}
        run.nodes = {"n": {"os": "linux"}, "w": {"os": "windows"}, "away": {"os": "linux"}}
        run.carried = {"n": ("carry", "mirror", "rev")}
        legs = [matrix.Leg("1002:744c", "n", "0000:06:00.0", "radeon"),
                matrix.Leg("10de:1e87", "n", "0000:09:00.0", "nvidia"),
                matrix.Leg("8086:3e9b", "w", "8086:3e9b"),
                matrix.Leg("1002:15bf", "away", "0000:c7:00.0", "radeon")]
        for leg in legs:
            leg.parsed["suites"][matrix.HARNESS_SUITE] = {"records": {"time": {"harness-new/alu": 200.0}}}
        asked = []

        def job(kind, node, script, name, gpus=(), carries=(), space=None, exclusive=False, room=None):
            asked.append((kind, node, name, list(gpus), list(carries), space, exclusive, 'rm -rf "$S"' in script))
            leg_id = name.split("-", 1)[1].removesuffix("-outlier")
            return 0, (f"M2V-LEG {leg_id} begin\nM2V-MATRIX time harness-old/alu 100.0\n"
                       f"M2V-MATRIX time harness-new/alu 101.0\nM2V-SUITE {leg_id} harness exit 0\n"), {"id": name}

        with mock.patch.object(run, "job", job, create=True), mock.patch.object(matrix, "log"):
            run.retime(legs)
        # A Linux node's legs in turn with its carry, its last job tidying; the Windows leg in its
        # workspace; each holding its node alone. A node whose legs never started has no carry.
        self.assertEqual(sorted(asked), [
            ("time", "n", "harness-1002:744c-outlier", ["0000:06:00.0"], ["carry"], "gpu", True, False),
            ("time", "n", "harness-10de:1e87-outlier", ["0000:09:00.0"], ["carry"], "gpu", True, True),
            ("wintime", "w", "winharness-8086:3e9b-outlier", ["8086:3e9b"], [], "win", True, False)])
        self.assertEqual([name for _, node, name, *_ in asked if node == "n"],
                         ["harness-1002:744c-outlier", "harness-10de:1e87-outlier"])
        for leg, name in zip(legs, ("harness-1002:744c-outlier", "harness-10de:1e87-outlier",
                                    "winharness-8086:3e9b-outlier")):
            self.assertEqual(leg.parsed["suites"][matrix.HARNESS_SUITE]["records"]["time"],
                             {"harness-old/alu": 100.0, "harness-new/alu": 101.0})
            self.assertEqual(leg.jobs, [name])
        self.assertEqual(run.failures, {"1002:15bf": ["its cross-GPU outlier's re-time found no carry on away"]})

    def test_a_re_time_that_raises_fails_its_leg_not_the_run(self):
        run = run_of({})
        run.cfg = {}
        run.nodes = {"w": {"os": "windows"}}
        leg = matrix.Leg("8086:3e9b", "w", "8086:3e9b")

        def job(*args, **kwargs):
            raise OSError("no potluck")

        with mock.patch.object(run, "job", job, create=True), mock.patch.object(matrix, "log"):
            run.retime([leg])
        self.assertEqual(run.failures, {"8086:3e9b": ["its cross-GPU outlier's re-time failed: OSError('no potluck')"]})


class HarnessRunTests(unittest.TestCase):
    def test_both_runtimes_are_timed_in_one_process_and_each_runtime_s_creation_rows_in_its_own(self):
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
        # run is the new runtime's; each of the five creation rounds times each runtime in a
        # process of its own, cold against cold, the order alternating.
        self.assertEqual(calls, ["new,old all", "new all"] + ["new create_*", "old create_*",
                                                             "old create_*", "new create_*"] * 2
                         + ["new create_*", "old create_*"])
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

    def test_a_cross_gpu_outlier_s_rows_are_timed_alone_in_fresh_processes_in_its_second_pass(self):
        with tempfile.TemporaryDirectory() as work:
            fake, log = Path(work) / "harness", Path(work) / "log"
            fake.write_text(FAKE_HARNESS)
            fake.chmod(0o755)
            script = matrix.harness_runs(f'"{work}"', f'"{fake}"', "a")
            run = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                                 env={"PATH": "/usr/bin:/bin", "LOG": str(log), "TMPDIR": work, "M2V_ALONE": "alu"})
            self.assertEqual(run.returncode, 0, run.stderr)
            calls = log.read_text().splitlines()
        # After the first pass's runs, the row alone, in pairs, old then new, then new then old.
        self.assertEqual(calls[-4:], ["old alu", "new alu", "new alu", "old alu"])
        times = matrix.results.parse_job("M2V-LEG a begin\n" + run.stdout)["a"]["suites"][matrix.HARNESS_SUITE]["records"]["time"]
        self.assertEqual(sorted(key for key in times if key.startswith(matrix.results.HARNESS_ALONE)),
                         [f"{matrix.results.HARNESS_ALONE}alu/{pair}/{runtime}" for pair in (1, 2) for runtime in ("new", "old")])
        # A leg's alone rows reach its job's shell; a first pass names none.
        leg = matrix.Leg("1002:15bf", "n", "0000:c7:00.0", "radeon")
        self.assertIn("leg 1002:15bf radeon '' ''", matrix.gpu_script("abc", "", [leg], "POTLUCK_MIRROR_x", "0" * 40,
                                                                     "harness", False))
        leg.alone = ["alu", "create_64MiB/device_first"]
        self.assertIn("leg 1002:15bf radeon '' 'alu create_64MiB/device_first'",
                      matrix.gpu_script("abc", "", [leg], "POTLUCK_MIRROR_x", "0" * 40, "harness", False))


class JudgingRuleTests(unittest.TestCase):
    def test_a_node_whose_cpu_clock_follows_the_load_re_times_no_frame_row(self):
        cpu = next(iter(matrix.results.CLOCK_FOLLOWS_LOAD))
        listed = matrix.harness_runs('"w"', '"h"', "a", cpu=cpu)
        self.assertIn(f"-v skipre='{matrix.results.retime_skip(cpu)}'", listed)
        self.assertIn("-v skipre='' ", matrix.harness_runs('"w"', '"h"', "a"))
        script = matrix.gpu_script("abc", "", [matrix.Leg("1002:15bf", "n", "0000:c7:00.0", "radeon")],
                                   "POTLUCK_MIRROR_x", "0" * 40, "harness", False, cpu=cpu)
        self.assertIn(f"-v skipre='{matrix.results.retime_skip(cpu)}'", script)


class DeviceTestTests(unittest.TestCase):
    """The GPU legs' archive holds the replay player's stream device tests beside the runtime's,
    and a leg runs exactly those two sets."""

    def test_the_archive_holds_the_validation_crate_beside_the_runtime(self):
        script = matrix.build_script("abc", "", False)
        self.assertIn("cargo nextest archive -p metal2vulkan-rt -p metal2vulkan-validation --all-features", script)

    def test_a_leg_runs_the_runtime_s_tests_and_the_two_stream_modules(self):
        leg = matrix.Leg("1002:15bf", "n", "0000:c7:00.0", "radeon")
        script = matrix.gpu_script("abc", "", [leg], "POTLUCK_MIRROR_x", "0" * 40, "rt", False)
        self.assertIn(f"-E '{matrix.DEVICE_TEST_FILTER}'", script)
        self.assertIn("test(/^runtime_lane::(replay_streams|rt_replay)::/)", matrix.DEVICE_TEST_FILTER)

    def test_the_tree_carries_metal_s_observations_and_no_other_backend_s(self):
        head = matrix.linux_head("abc", "")
        self.assertIn(f"git archive HEAD -- {matrix.METAL_OBSERVATIONS} | tar -x", head)
        self.assertIn('":(exclude)crates/metal2vulkan-validation/corpus/observations"', head)

    def test_the_harness_suite_takes_no_test_filter(self):
        leg = matrix.Leg("1002:15bf", "n", "0000:c7:00.0", "radeon")
        script = matrix.gpu_script("abc", "", [leg], "POTLUCK_MIRROR_x", "0" * 40, "harness", False)
        self.assertNotIn(matrix.DEVICE_TEST_FILTER, script)


class UncountedTests(unittest.TestCase):
    """A lock budget test on a node that names no event passes counting nothing: its leg says so."""

    def test_a_lock_budget_test_that_counted_nothing_shows_its_leg_as_not_counted(self):
        # nextest's output of a GPU leg, as `--success-output final` prints it: the passing lock
        # budget test's captured lines come after the summary.
        out = "\n".join([
            "    Starting 2 tests across 1 binary",
            "        PASS [   0.027s] (1/2) metal2vulkan-rt tests::draw::on_device::every_gpu_adds_no_locked_instruction_per_bind",
            "        PASS [   0.010s] (2/2) metal2vulkan-rt tests::pass::on_device::every_device_clears",
            "     Summary [   0.030s] 2 tests run: 2 passed, 0 skipped",
            "        PASS [   0.027s] (1/2) metal2vulkan-rt tests::draw::on_device::every_gpu_adds_no_locked_instruction_per_bind",
            "--- STDOUT:              metal2vulkan-rt tests::draw::on_device::every_gpu_adds_no_locked_instruction_per_bind ---",
            "LOCKS bind_locks: not counted, the node names no M2V_LOCK_EVENT",
            "M2V-MATRIX uncounted lock_budget/bound_draw",
            "",
        ])
        kept = subprocess.run(["awk", matrix.KEEP], input=out, capture_output=True, text=True, check=True).stdout
        self.assertIn("M2V-MATRIX uncounted lock_budget/bound_draw", kept)
        legs = matrix.results.parse_job("M2V-LEG a begin\n" + kept + "M2V-SUITE a rt exit 0\n")
        suite = legs["a"]["suites"]["rt"]
        self.assertEqual((suite["run"], suite["passed"], suite["problems"]), (2, 2, []))
        uncounted = matrix.results.leg_uncounted(legs["a"])
        self.assertEqual(uncounted, ["lock_budget/bound_draw"])
        leg = dict(suites={"rt": suite}, records=suite["records"], verdict="pass", problems=[],
                   uncounted=uncounted)
        line = matrix.results.verdict_line("a", leg)
        self.assertTrue(line.startswith("PASS"), line)
        self.assertIn("not counted: lock_budget/bound_draw", line)
        # A leg whose budget counted shows no such words.
        leg["uncounted"] = []
        self.assertNotIn("not counted", matrix.results.verdict_line("a", leg))


class ArgumentTests(unittest.TestCase):
    def test_an_argument_the_usage_does_not_name_prints_it_and_starts_no_run(self):
        for argv in (["run", "--help"], ["run", "-h"], ["run", "--leg", "lavapipe"], ["run", "--legs"],
                     ["run", "--legs", "--dry-run"], ["run", "a", "b"], ["show", "--dry-run"]):
            with self.subTest(argv=argv), mock.patch.object(matrix, "run_cmd") as run, \
                    mock.patch.object(matrix, "show_cmd") as show, mock.patch.object(matrix, "git") as git:
                with self.assertRaises(SystemExit) as exit:
                    matrix.main(argv)
                self.assertIn("matrix run [SHA]", str(exit.exception.code))
                run.assert_not_called()
                show.assert_not_called()
                git.assert_not_called()

    def test_a_named_run_passes_its_sha_legs_and_dry_run(self):
        with mock.patch.object(matrix, "run_cmd", return_value=0) as run, \
                mock.patch.object(matrix, "git", side_effect=lambda *a: a[-1]):
            matrix.main(["run", "abc", "--legs", "lavapipe,1002:744c", "--dry-run"])
        run.assert_called_once_with("abc^{commit}", ["lavapipe", "1002:744c"], True)


if __name__ == "__main__":
    unittest.main()
