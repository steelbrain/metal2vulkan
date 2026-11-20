"""The matrix's reading of job output and its rules, on inputs whose answers are known.

    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/matrix

The job output below is nextest's own layout (0.9.140 and 0.9.151 print it alike): status lines,
the summary, then each test's captured output, indented, after a header naming the test.
"""

import subprocess
import unittest
from pathlib import Path
from unittest import mock

import results

NEXTEST = """\
    Starting 3 tests across 1 binary
        FAIL [   0.002s] (1/3) metal2vulkan-runtime tests::breaks
        PASS [   0.003s] (2/3) metal2vulkan-runtime tests::counts
        PASS [   0.003s] (3/3) metal2vulkan-runtime tests::loops
     Summary [   0.003s] 3 tests run: 2 passed, 1 failed, 0 skipped
        PASS [   0.003s] (2/3) metal2vulkan-runtime tests::counts
  stdout ───
    M2V-MATRIX metric compute/submissions 3
    M2V-MATRIX instructions compute/per_dispatch 2000
        PASS [   0.003s] (3/3) metal2vulkan-runtime tests::loops
  stdout ───
    M2V-MATRIX gauge loop/buffers@100 12
    M2V-MATRIX gauge loop/buffers@200 12
    M2V-MATRIX gauge loop/buffers@drop 0
    M2V-MATRIX time loop/gpu 1000.250
        FAIL [   0.002s] (1/3) metal2vulkan-runtime tests::breaks
  stdout ───
    test tests::breaks ... FAILED
    test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 2 filtered out
"""


def job(*legs):
    """A job's output holding LEGS: (leg, suite, exit, output) each."""
    out = []
    for leg, suite, code, text in legs:
        out += [f"M2V-LEG {leg} begin", f"M2V-DEVICE {leg} deviceName AMD Radeon RX 7900 XTX (RADV NAVI31)",
                f"M2V-DEVICE {leg} driverInfo Mesa 26.2.6", text, f"M2V-SUITE {leg} {suite} exit {code}"]
    return "\n".join(out) + "\n"


class ParseTests(unittest.TestCase):
    def test_a_suite_gives_its_counts_failures_and_records(self):
        legs = results.parse_job(job(("1002:744c", "rt", 100, NEXTEST)))
        suite = legs["1002:744c"]["suites"]["rt"]
        self.assertEqual((suite["run"], suite["passed"], suite["exit"]), (3, 2, 100))
        self.assertEqual(suite["failed"], ["metal2vulkan-runtime tests::breaks"])
        self.assertEqual(suite["records"]["metric"], {"compute/submissions": 3})
        self.assertEqual(suite["records"]["instructions"], {"compute/per_dispatch": 2000})
        self.assertEqual(suite["records"]["gauge"], {"loop/buffers@100": 12, "loop/buffers@200": 12,
                                                     "loop/buffers@drop": 0})
        self.assertEqual(suite["records"]["time"], {"loop/gpu": 1000.25})
        self.assertEqual(suite["problems"], [])
        self.assertEqual(legs["1002:744c"]["device"]["driverInfo"], "Mesa 26.2.6")

    def test_output_goes_to_the_leg_and_suite_it_came_in(self):
        text = job(("1002:744c", "rt", 0, "    M2V-MATRIX metric a/x 1"),
                   ("10de:1e87", "rt", 0, "    M2V-MATRIX metric a/x 2"))
        legs = results.parse_job(text)
        self.assertEqual(legs["1002:744c"]["suites"]["rt"]["records"]["metric"], {"a/x": 1})
        self.assertEqual(legs["10de:1e87"]["suites"]["rt"]["records"]["metric"], {"a/x": 2})

    def test_a_doctest_suite_counts_libtest_results(self):
        text = job(("lavapipe", "doctests", 101, "test result: ok. 10 passed; 0 failed; 1 ignored\n"
                    "test src/lib.rs - f (line 3) ... FAILED\n"
                    "test result: FAILED. 4 passed; 1 failed; 0 ignored"))
        suite = results.parse_job(text)["lavapipe"]["suites"]["doctests"]
        self.assertEqual((suite["run"], suite["passed"]), (15, 14))
        self.assertEqual(suite["failed"], ["src/lib.rs - f (line 3)"])

    def test_malformed_records_are_problems(self):
        suite = results.parse_suite([
            "M2V-MATRIX colour a/x 1",
            "M2V-MATRIX metric a/y 1.5",
            "M2V-MATRIX time a/t 1.5us",
            "M2V-MATRIX gauge a/buffers 3",
            "M2V-MATRIX metric a/z 1",
            "M2V-MATRIX metric a/z 2",
        ])
        self.assertEqual(len(suite["problems"]), 5, suite["problems"])
        self.assertEqual(suite["records"]["metric"], {"a/z": 1})


def leg_of(records, code=0):
    suite = dict(run=1, passed=1 if code == 0 else 0, failed=[], records=records, problems=[], exit=code)
    return dict(device={}, suites={"rt": suite})


def recs(**kinds):
    out = results.empty_records()
    out.update(kinds)
    return out


class RuleTests(unittest.TestCase):
    def problems(self, now, base):
        leg = leg_of(now)
        records, problems = results.leg_records(leg)
        return problems + results.leg_problems(leg, records, base)

    def test_a_node_naming_the_lock_event_fails_a_leg_without_the_budget_s_records(self):
        self.assertTrue(results.names_lock_event("export M2V_LOCK_EVENT=cpu_core/r21d0"))
        self.assertTrue(results.names_lock_event("renice -n 19 $$; export M2V_LOCK_EVENT=r1e25"))
        self.assertFalse(results.names_lock_event("export CARGO_BUILD_JOBS=4"))
        self.assertFalse(results.names_lock_event(None))
        measured = recs(locks={"lock_budget/draw": 0, "lock_budget/bound_draw": 1500,
                               "lock_budget/dispatch": 4000})
        self.assertEqual(results.lock_budget_missing(measured, True), [])
        partial = recs(locks={"lock_budget/draw": 0})
        self.assertEqual(len(results.lock_budget_missing(partial, True)), 2)
        self.assertEqual(results.lock_budget_missing(recs(), False), [])
        suite = results.parse_suite(["    M2V-MATRIX locks lock_budget/draw 120"])
        self.assertEqual((suite["records"]["locks"], suite["problems"]), ({"lock_budget/draw": 120}, []))

    def test_a_metric_fails_on_any_increase_and_passes_a_decrease(self):
        base = recs(metric={"a/barriers": 4, "a/passes": 2})
        self.assertEqual(self.problems(recs(metric={"a/barriers": 4, "a/passes": 1}), base), [])
        self.assertEqual(self.problems(recs(metric={"a/barriers": 5, "a/passes": 2}), base),
                         ["metric a/barriers: 4 -> 5"])

    def test_a_replay_metric_fails_on_any_change_and_where_the_leg_lacks_it(self):
        # The replay player's judged totals (judged_records): equal to the base's, and printed
        # only by a play that finished its range, so a short play fails for the records it lacks.
        key = "replay-rt/s/through-356/barriers"
        base = recs(metric={key: 4, "a/passes": 2})
        self.assertEqual(self.problems(recs(metric={key: 4, "a/passes": 1}), base), [])
        self.assertEqual(self.problems(recs(metric={key: 3, "a/passes": 2}), base),
                         [f"metric {key}: 4 -> 3"])
        self.assertEqual(self.problems(recs(metric={key: 5, "a/passes": 2}), base),
                         [f"metric {key}: 4 -> 5"])
        self.assertEqual(self.problems(recs(metric={"a/passes": 2}), base),
                         [f"metric {key}: 4 -> none: the play did not finish its range"])
        self.assertEqual(self.problems(recs(metric={key: 4}), base), [])
        self.assertEqual(self.problems(recs(metric={key: 9}), None), [])

    def test_the_harness_s_instructions_are_not_compared_to_the_base(self):
        # The instructions gate judges them (harness_budget); a driver's threads swing the whole count.
        base = recs(instructions={"harness-new/frame/per_run": 50000,
                                  "harness-new/frame/own_per_run": 30000, "a/per_draw": 10000})
        now = recs(instructions={"harness-new/frame/per_run": 71000,
                                 "harness-new/frame/own_per_run": 31000, "a/per_draw": 10200})
        self.assertEqual(self.problems(now, base), ["instructions a/per_draw: 10000 -> 10200 (+2.0%)"])

    def test_instructions_fail_past_one_percent(self):
        base = recs(instructions={"a/per_draw": 10000})
        self.assertEqual(self.problems(recs(instructions={"a/per_draw": 10100}), base), [])
        self.assertEqual(len(self.problems(recs(instructions={"a/per_draw": 10101}), base)), 1)

    def test_no_base_compares_nothing_but_the_gauges(self):
        now = recs(metric={"a/x": 9}, gauge={"l/b@100": 1, "l/b@200": 2, "l/b@drop": 1})
        self.assertEqual(self.problems(now, None),
                         ["gauge l/b: 1 at iteration 100, 2 at 200", "gauge l/b: 1 live at device drop"])

    def test_flat_gauges_that_reach_zero_pass(self):
        now = recs(gauge={"l/b@100": 7, "l/b@200": 7, "l/b@drop": 0})
        self.assertEqual(self.problems(now, None), [])

    def test_a_pool_gauge_may_fall_between_iterations_but_not_rise_and_must_reach_zero(self):
        fell = {"harness-new/long_4cb/blocks@100": 11, "harness-new/long_4cb/blocks@200": 8,
                "harness-new/long_4cb/buffers@100": 2, "harness-new/long_4cb/buffers@200": 1}
        self.assertEqual(self.problems(recs(gauge=fell), None),
                         ["gauge harness-new/long_4cb/buffers: 2 at iteration 100, 1 at 200"])
        rose = {"harness-new/long_4cb/blocks@100": 8, "harness-new/long_4cb/blocks@200": 9,
                "harness-new/creates/device/blocks@drop": 1}
        self.assertEqual(self.problems(recs(gauge=rose), None), [
            "gauge harness-new/creates/device/blocks: 1 live at device drop",
            "gauge harness-new/long_4cb/blocks: 8 at iteration 100, 9 at 200"])

    def test_a_failed_suite_fails_the_leg(self):
        leg = leg_of(recs(), code=100)
        self.assertEqual(results.leg_problems(leg, recs(), None), ["suite rt exited 100"])

    def test_a_parked_gauge_moves_between_iterations_but_must_reach_zero(self):
        prefix = results.PARKED_RECORDS[0]
        now = recs(gauge={f"{prefix}w/b@100": 3, f"{prefix}w/b@200": 2, f"{prefix}w/b@drop": 1})
        self.assertEqual(self.problems(now, None), [f"gauge {prefix}w/b: 1 live at device drop"])
        self.assertEqual(results.leg_parked(leg_of(now), now),
                         [f"gauge {prefix}w/b: 3 at iteration 100, 2 at 200"])

    def test_two_suites_recording_one_key_is_a_problem(self):
        leg = leg_of(recs(metric={"a/x": 1}))
        leg["suites"]["other"] = dict(leg["suites"]["rt"])
        self.assertEqual(results.leg_records(leg)[1], ["record metric a/x: recorded by two suites"])

    def test_time_fails_only_as_a_cross_gpu_outlier(self):
        bases = {"a": {"t": 100}, "b": {"t": 200}, "c": {"t": 50}, "d": {"t": 10}}
        # Every GPU 10% slower: no outlier.
        times = {"a": {"t": 110}, "b": {"t": 220}, "c": {"t": 55}, "d": {"t": 11}}
        self.assertEqual(results.time_outliers(times, bases), ({}, {}))
        # One GPU 50% slower while the others hold: it alone fails.
        times["d"] = {"t": 15}
        self.assertEqual(list(results.time_outliers(times, bases)[0]), ["d"])

    def test_a_parked_time_outlier_is_named_and_fails_no_leg(self):
        key = results.PARKED_RECORDS[0] + "frame"
        bases = {leg: {key: 100} for leg in "abcd"}
        times = {"a": {key: 100}, "b": {key: 101}, "c": {key: 99}, "d": {key: 150}}
        failing, parked = results.time_outliers(times, bases)
        self.assertEqual(failing, {})
        self.assertEqual(list(parked), ["d"])

    def test_a_warm_first_use_outlier_is_parked_and_a_cold_one_fails(self):
        # The harness's second run on the cache its first wrote keeps its first-use records warm.
        cold = "harness-new/authored/first_use_wait_total"
        warm = "harness-new-warm/authored/first_use_wait_total"
        bases = {leg: {cold: 5000, warm: 400} for leg in "abcd"}
        times = {leg: {cold: 5000, warm: 400} for leg in "abc"}
        times["d"] = {cold: 20000, warm: 4000}
        failing, parked = results.time_outliers(times, bases)
        self.assertEqual((list(failing), list(parked)), (["d"], ["d"]))
        self.assertEqual([cold in line for line in failing["d"]], [True])
        self.assertEqual([warm in line for line in parked["d"]], [True])

    def test_a_harness_row_s_outlier_in_one_pass_fails_only_where_a_second_pass_agrees(self):
        alu, frame, bandwidth = (f"{results.HARNESS_NEW}{row}" for row in ("alu", "frame", "bandwidth"))
        bases = {leg: {alu: 100, frame: 100, bandwidth: 100, "loop/gpu": 100} for leg in "abcd"}
        times = {leg: dict(bases[leg]) for leg in "abcd"}
        times["d"].update({alu: 150, frame: 150})
        times["c"]["loop/gpu"] = 150
        found = results.outlier_keys(times, bases)
        self.assertEqual(found, {"c": {"loop/gpu": (1.5, 1.0, 4)}, "d": {alu: (1.5, 1.0, 4), frame: (1.5, 1.0, 4)}})
        # Only a harness row's outlier is timed once more; another fails at once, as before.
        first = results.outliers_to_retime(found)
        self.assertEqual(first, {"d": found["d"]})
        # The second pass: alu again, frame back with the others, bandwidth new.
        times["d"].update({frame: 100, bandwidth: 150})
        failing, parked = results.time_outliers(times, bases, first)
        self.assertEqual(failing, {
            "c": ["time loop/gpu: 1.50x its base, against a median of 1.00x over 4 GPUs"],
            "d": [f"time {alu}: 1.50x its base, against a median of 1.00x over 4 GPUs, and 1.50x against 1.00x "
                  "in the first pass: both passes agree; its lowest of 2 processes, 150.0 us, over the base's lowest "
                  "of 1, 100.0 us, is 1.50x (samples 150.0, 150.0 us): high in every process"]})
        self.assertEqual(parked, {"d": [
            f"time {bandwidth}: 1.50x its base, against a median of 1.00x over 4 GPUs when timed once more, but "
            "not in the first pass: the passes disagree, so it fails nothing",
            f"time {frame}: an outlier in the first pass, 1.50x its base against a median of 1.00x, but not when "
            "timed once more: the passes disagree, so it fails nothing"]})
        # Without a second pass, every outlier fails, as one pass did before.
        self.assertEqual(sorted(results.time_outliers(times, bases)[0]["d"]),
                         [f"time {key}: 1.50x its base, against a median of 1.00x over 4 GPUs" for key in (alu, bandwidth)])

    def test_a_harness_row_out_in_both_passes_is_judged_by_its_lowest_level(self):
        # The RTX 3090 at homelab-um870-1, round 6b535f55c9: about 37 or about 80 us a process, two
        # passes both high, the alone runs low.
        row = "sampling_deep_4x16x252"
        key = f"{results.HARNESS_NEW}{row}"
        alone = [f"{results.HARNESS_ALONE}{row}/{pair}/new" for pair in (1, 2)]
        bases = {leg: {key: 38.8} for leg in "abcd"}
        times = {leg: {key: 38.8} for leg in "abcd"}
        times["d"][key] = 81.4
        first = results.outliers_to_retime(results.outlier_keys(times, bases))
        self.assertEqual(results.alone_rows(first["d"]), [row])
        # The alone runs' keys judge nothing on their own, whatever their spread across GPUs.
        times = {leg: {**values, **{name: 9.0 for name in alone}} for leg, values in times.items()}
        times["d"].update({key: 81.2, alone[0]: 37.9, alone[1]: 80.6})
        failing, parked = results.time_outliers(times, bases, first)
        self.assertEqual(failing, {})
        self.assertEqual(parked, {"d": [
            f"time {key}: 2.09x its base, against a median of 1.00x over 4 GPUs, and 2.10x against 1.00x in the "
            "first pass: both passes agree; its lowest of 4 processes, 37.9 us, over the base's lowest of 1, 38.8 us, "
            "is 0.98x (samples 81.4, 81.2, 37.9, 80.6 us): levels, no verdict"]})
        # High in every process, the alone runs included: it fails as before.
        times["d"].update({alone[0]: 79.9})
        failing, parked = results.time_outliers(times, bases, first)
        self.assertEqual(parked, {})
        self.assertEqual(failing, {"d": [
            f"time {key}: 2.09x its base, against a median of 1.00x over 4 GPUs, and 2.10x against 1.00x in the "
            "first pass: both passes agree; its lowest of 4 processes, 79.9 us, over the base's lowest of 1, 38.8 us, "
            "is 2.06x (samples 81.4, 81.2, 79.9, 80.6 us): high in every process"]})
        # The base's own alone runs count among its levels: a base that was high in its pass and low
        # alone compares low with low.
        bases["d"].update({alone[0]: 10.0})
        self.assertEqual(results.time_outliers(times, bases, first)[0]["d"][0].split("; ")[1],
                         "its lowest of 4 processes, 79.9 us, over the base's lowest of 2, 10.0 us, is 7.99x "
                         "(samples 81.4, 81.2, 79.9, 80.6 us): high in every process")

    def test_the_old_path_moving_fails_unless_a_commit_since_the_bless_is_its_fix(self):
        # The golden test's two panics, as its captured output prints them.
        changed = ("    tests/golden_hashes.tsv: output changed under an unchanged key: 333 cell(s) "
                   "changed while their fixture source and profile fingerprint did not, so a")
        stale = ("    tests/golden_hashes.tsv is stale: 4 cell(s) are new, gone, or under a changed "
                 "fixture source or profile (their keys changed, so no SEMANTICS_VERSION bump is "
                 "needed): 0 new, 0 gone, 0 under a changed fixture source, 4 under a changed "
                 "profile fingerprint. Bless it with")
        new_rows = stale.replace("0 new", "4 new").replace("4 under a changed profile", "0 under a changed profile")
        for text, moved in ((changed, True), (stale, True), (new_rows, False), (NEXTEST, False)):
            leg = results.parse_job(job(("cpu", "workspace", 100, text)))["cpu"]
            problems = results.old_path_moved(leg, ["translator: one stage bound on every device"])
            self.assertEqual(len(problems), int(moved), text)
            # A message that only quotes the phrase, as the check's own commit did, exempts nothing.
            quoting = ['matrix: the old path moving fails a leg\n\nunless a commit says '
                       '"old-runtime regression fix"']
            self.assertEqual(len(results.old_path_moved(leg, quoting)), int(moved), text)
            self.assertEqual(results.old_path_moved(
                leg, ["Old-runtime regression fix: the stage bound returns to Device::new"]), [])
        leg = results.parse_job(job(("cpu", "workspace", 100, changed)))["cpu"]
        self.assertIn("333 cell(s) changed under an unchanged key and 0 under a changed profile",
                      results.old_path_moved(leg, [])[0])

    def test_a_changed_harness_leaves_its_own_counts_uncompared(self):
        base = recs(metric={"harness-new/authored/pipelines_created": 8, "rt/submissions": 3})
        now = recs(metric={"harness-new/authored/pipelines_created": 10, "rt/submissions": 4})
        self.assertEqual(self.problems(now, results.base_records(base, True)),
                         ["metric harness-new/authored/pipelines_created: 8 -> 10",
                          "metric rt/submissions: 3 -> 4"])
        self.assertEqual(self.problems(now, results.base_records(base, False)),
                         ["metric rt/submissions: 3 -> 4"])

    def test_time_needs_three_gpus(self):
        bases = {"a": {"t": 100}, "b": {"t": 100}}
        self.assertEqual(results.time_outliers({"a": {"t": 100}, "b": {"t": 300}}, bases), ({}, {}))


class HarnessBudgetTests(unittest.TestCase):
    def budget(self, leg_id, counts, times=None):
        times = times or {"harness-new/frame": 101.0, "harness-new/alu": 101.0}
        return results.harness_budget(leg_id, leg_of(recs(time=times, instructions=counts)))

    def test_a_time_recorded_twice_has_no_one_time_to_show(self):
        refused = ["harness budget frame: its time is recorded more than once, so it has no one time to show"]
        # Within one suite, the later and slower time would otherwise go unseen: the first is kept.
        once = "    M2V-MATRIX time harness-new/frame 101\n"
        text = job(("10de:1e87", "harness", 0, once + "    M2V-MATRIX time harness-new/frame 300"))
        leg = results.parse_job(text)["10de:1e87"]
        self.assertEqual(results.harness_budget("10de:1e87", leg), (refused, []))
        # Two suites that each record it.
        text = job(("10de:1e87", "harness", 0, once), ("10de:1e87", "other", 0, once))
        leg = results.parse_job(text)["10de:1e87"]
        self.assertEqual(results.harness_budget("10de:1e87", leg), (refused, []))
        # Recorded once, it is shown and judged by nothing here.
        leg = results.parse_job(job(("10de:1e87", "harness", 0, once)))["10de:1e87"]
        self.assertEqual(results.harness_budget("10de:1e87", leg), ([], []))

    def test_the_creation_process_records_each_key_once_beside_the_other_rows(self):
        # Each harness process prints its device's totals and gauges at drop, the creation process
        # as well as the other rows'.
        others = ["M2V-MATRIX time harness-new/alu 10.000",
                  "M2V-MATRIX metric harness-new/device/stamps_not_available 0",
                  "M2V-MATRIX gauge harness-new/device/blocks@drop 0"]
        creates = ["M2V-MATRIX time harness-new/create_64MiB 700.000",
                   "M2V-MATRIX time harness-new/create_64MiB/first_take 900.000",
                   "M2V-MATRIX metric harness-new/create_64MiB/first_allocations 1",
                   "M2V-MATRIX gauge harness-new/create_64MiB/buffers@100 0",
                   "M2V-MATRIX metric harness-new/device/stamps_not_available 0",
                   "M2V-MATRIX gauge harness-new/device/blocks@drop 0", "a table row"]
        run = subprocess.run(["sed", results.CREATES_APART], input="\n".join(creates) + "\n",
                             capture_output=True, text=True, check=True)
        suite = results.parse_suite(others + run.stdout.splitlines())
        self.assertEqual((suite["problems"], suite["duplicated"]), ([], set()))
        self.assertEqual(suite["records"]["gauge"], {"harness-new/device/blocks@drop": 0,
                                                     "harness-new/creates/device/blocks@drop": 0,
                                                     "harness-new/create_64MiB/buffers@100": 0})
        self.assertEqual(suite["records"]["metric"], {"harness-new/device/stamps_not_available": 0,
                                                      "harness-new/creates/device/stamps_not_available": 0,
                                                      "harness-new/create_64MiB/first_allocations": 1})
        # Without the program, the device's totals and gauges would be recorded twice.
        unrenamed = results.parse_suite(others + creates)
        self.assertIn(("gauge", "harness-new/device/blocks@drop"), unrenamed["duplicated"])
        self.assertIn(("metric", "harness-new/device/stamps_not_available"), unrenamed["duplicated"])
        self.assertIn("record metric harness-new/device/stamps_not_available: recorded twice", unrenamed["problems"])

    def test_every_row_shows_its_time_gpu_time_and_instructions(self):
        times = {"harness-new/frame": 120.0, "harness-gpu-new/frame": 81.0,
                 "harness-new/create_64MiB/first_take": 50.0, "loop/gpu": 1000.0}
        counted = {"harness-new/frame/per_run": 61253, "harness-new/frame/own_per_run": 40001,
                   "harness-new/frame/others_per_run": 21252}
        self.assertEqual(results.harness_rows(recs(time=times, instructions=counted)), [
            "create_64MiB/first_take                    50.0 us, GPU p50          - us",
            "frame                                     120.0 us, GPU p50       81.0 us, 61253 instructions "
            "(40001 own)"])

    def test_the_frame_and_burst_rows_pass_the_instructions_gate(self):
        counted = {"harness-new/frame/per_run": 161000, "harness-new/frame/own_per_run": 61000,
                   "harness-new/frame/others_per_run": 100000, "harness-new/alu/per_run": 9000}
        gate = "harness budget frame: instructions new own {} per run, {}"
        # alu, not a frame or burst row, has no gate. The whole count, a driver's threads in it,
        # judges nothing: the runtime's own count is the one gated.
        with mock.patch.object(results, "STORED_INSTRUCTIONS", {}):
            self.assertEqual(self.budget("10de:1e87", counted), ([], [gate.format(61000, "none stored")]))
        with mock.patch.object(results, "STORED_INSTRUCTIONS", {"all": {"frame": 60000}}):
            # Within 1.03 times the stored count: nothing to say.
            self.assertEqual(self.budget("10de:1e87", counted), ([], []))
            over = dict(counted, **{"harness-new/frame/own_per_run": 61900})
            self.assertEqual(self.budget("10de:1e87", over), ([
                gate.format(61900, "stored 60000, is 1.032") + ": past 1.03 times the stored count"], []))
            under = dict(counted, **{"harness-new/frame/own_per_run": 59000})
            self.assertEqual(self.budget("10de:1e87", under), ([], [
                gate.format(59000, "stored 60000, is 0.983") + ": more than 1% under the stored count, "
                "which ratchets down to it"]))
            # A whole count with no count of the runtime's own threads: the gate could not run.
            no_own = {key: value for key, value in counted.items() if "own" not in key and "others" not in key}
            self.assertEqual(self.budget("10de:1e87", no_own)[0], [
                "harness budget frame: instructions new 161000 per run, but no count of its own threads"])
        # A driver family's count before every leg's.
        with mock.patch.object(results, "STORED_INSTRUCTIONS", {"all": {"frame": 60000},
                                                                "1002": {"frame": 70000}}):
            self.assertEqual(results.stored_count("1002:744c", "frame"), 70000)
            self.assertEqual(results.stored_count("10de:2204", "frame"), 60000)
        # A leg with no counts (Windows, or a refused counter) has no gate.
        self.assertEqual(self.budget("8086:3e9b", {}), ([], []))

    def test_the_stored_counts_table_reads_per_family_and_row(self):
        text = "# a comment\nrow  all  1002\nframe  61253  -\nburst_160  -  90000\n"
        self.assertEqual(results.read_counts(text), {"all": {"frame": 61253}, "1002": {"burst_160": 90000}})
        for family, rows in results.STORED_INSTRUCTIONS.items():
            self.assertRegex(family, r"^(all|[0-9a-f]{4})$")
            self.assertTrue(all(row.startswith(results.WALL_ROWS) for row in rows), family)

    def test_an_extra_leg_takes_its_card_s_stored_counts(self):
        extra = results.extra_leg_id("1002:744c", "node-b")
        self.assertEqual(results.card_of(extra), "1002:744c")
        self.assertEqual(results.card_of("1002:744c"), "1002:744c")
        with mock.patch.object(results, "STORED_INSTRUCTIONS", {"1002": {"frame": 70000}}):
            self.assertEqual(results.stored_count(extra, "frame"), 70000)

    def test_the_verdict_line_names_the_suites_records_base_and_problems(self):
        leg = dict(verdict="fail", device="AMD Radeon RX 7900 XTX", driver="radv", node="n",
                   suites={"harness": dict(exit=0, run=0, passed=0)}, records={"time": {"harness-new/frame": 1.0}},
                   base="0123456789abc", jobs=["m3-max-1"], problems=["harness budget frame: past"],
                   parked=["a parked line"])
        self.assertEqual(results.verdict_line("1002:744c", leg),
                         "FAIL 1002:744c  AMD Radeon RX 7900 XTX radv on n: harness exit 0; 1 records; base "
                         "0123456789; jobs m3-max-1 | harness budget frame: past | parked: 1 (a parked line)")


if __name__ == "__main__":
    unittest.main()
