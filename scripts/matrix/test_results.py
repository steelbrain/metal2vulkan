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

#: The cards' variance rows as stored, before a test patches them away.
CARD_VARIANCE_STORED = results.CARD_VARIANCE

NEXTEST = """\
    Starting 3 tests across 1 binary
        FAIL [   0.002s] (1/3) metal2vulkan-rt tests::breaks
        PASS [   0.003s] (2/3) metal2vulkan-rt tests::counts
        PASS [   0.003s] (3/3) metal2vulkan-rt tests::loops
     Summary [   0.003s] 3 tests run: 2 passed, 1 failed, 0 skipped
        PASS [   0.003s] (2/3) metal2vulkan-rt tests::counts
  stdout ───
    M2V-MATRIX metric compute/submissions 3
    M2V-MATRIX instructions compute/per_dispatch 2000
        PASS [   0.003s] (3/3) metal2vulkan-rt tests::loops
  stdout ───
    M2V-MATRIX gauge loop/buffers@100 12
    M2V-MATRIX gauge loop/buffers@200 12
    M2V-MATRIX gauge loop/buffers@drop 0
    M2V-MATRIX time loop/gpu 1000.250
        FAIL [   0.002s] (1/3) metal2vulkan-rt tests::breaks
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
        self.assertEqual(suite["failed"], ["metal2vulkan-rt tests::breaks"])
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

    def test_an_extra_leg_is_held_to_its_card_s_ceilings_and_floors(self):
        card = next(iter(results.HARNESS_CEILINGS))
        row = next(iter(results.HARNESS_CEILINGS[card]))
        extra = results.extra_leg_id(card, "node-b")
        self.assertEqual(results.card_of(extra), card)
        self.assertEqual(results.card_of(card), card)
        self.assertEqual(results.row_limit(extra, row), results.row_limit(card, row))
        self.assertGreater(results.row_limit(extra, row), results.HARNESS_BUDGET)
        self.assertEqual(results.harness_limits(extra), results.harness_limits(card))


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
        base = recs(instructions={"harness-old/frame/per_run": 100000, "harness-new/frame/per_run": 50000,
                                  "harness-new/frame/own_per_run": 30000, "a/per_draw": 10000})
        now = recs(instructions={"harness-old/frame/per_run": 105000, "harness-new/frame/per_run": 71000,
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

    def test_a_suite_whose_failures_are_all_parked_passes_and_names_them(self):
        parked = next(iter(results.PARKED_TESTS))
        leg = leg_of(recs(), code=100)
        leg["suites"]["rt"]["failed"] = [parked]
        self.assertEqual(results.leg_problems(leg, recs(), None), [])
        self.assertEqual(results.leg_parked(leg, recs()), [f"test {parked}"])
        leg["suites"]["rt"]["failed"].append("metal2vulkan-rt tests::breaks")
        self.assertEqual(results.leg_problems(leg, recs(), None),
                         ["suite rt exited 100, 1 failed: metal2vulkan-rt tests::breaks"])

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
        alone = [f"{results.HARNESS_ALONE}{row}/{pair}/{runtime}" for pair in (1, 2) for runtime in ("old", "new")]
        bases = {leg: {key: 38.8} for leg in "abcd"}
        times = {leg: {key: 38.8} for leg in "abcd"}
        times["d"][key] = 81.4
        first = results.outliers_to_retime(results.outlier_keys(times, bases))
        self.assertEqual(results.alone_rows(first["d"]), [row])
        # The alone runs' keys judge nothing on their own, whatever their spread across GPUs.
        times = {leg: {**values, **{name: 9.0 for name in alone}} for leg, values in times.items()}
        times["d"].update({key: 81.2, alone[1]: 37.9, alone[3]: 80.6})
        failing, parked = results.time_outliers(times, bases, first)
        self.assertEqual(failing, {})
        self.assertEqual(parked, {"d": [
            f"time {key}: 2.09x its base, against a median of 1.00x over 4 GPUs, and 2.10x against 1.00x in the "
            "first pass: both passes agree; its lowest of 4 processes, 37.9 us, over the base's lowest of 1, 38.8 us, "
            "is 0.98x (samples 81.4, 81.2, 37.9, 80.6 us): levels, no verdict"]})
        # High in every process, the alone runs included: it fails as before.
        times["d"].update({alone[1]: 79.9})
        failing, parked = results.time_outliers(times, bases, first)
        self.assertEqual(parked, {})
        self.assertEqual(failing, {"d": [
            f"time {key}: 2.09x its base, against a median of 1.00x over 4 GPUs, and 2.10x against 1.00x in the "
            "first pass: both passes agree; its lowest of 4 processes, 79.9 us, over the base's lowest of 1, 38.8 us, "
            "is 2.06x (samples 81.4, 81.2, 79.9, 80.6 us): high in every process"]})
        # The base's own alone runs count among its levels: a base that was high in its pass and low
        # alone compares low with low.
        bases["d"].update({alone[1]: 10.0})
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
        base = recs(metric={"harness-old/authored/pipelines_created": 8, "rt/submissions": 3})
        now = recs(metric={"harness-old/authored/pipelines_created": 10, "rt/submissions": 4})
        self.assertEqual(self.problems(now, results.base_records(base, True)),
                         ["metric harness-old/authored/pipelines_created: 8 -> 10",
                          "metric rt/submissions: 3 -> 4"])
        self.assertEqual(self.problems(now, results.base_records(base, False)),
                         ["metric rt/submissions: 3 -> 4"])

    def test_time_needs_three_gpus(self):
        bases = {"a": {"t": 100}, "b": {"t": 100}}
        self.assertEqual(results.time_outliers({"a": {"t": 100}, "b": {"t": 300}}, bases), ({}, {}))


class HarnessBudgetTests(unittest.TestCase):
    def setUp(self):
        # The rules below on no floors, no card's variance and no status entry unless a test gives
        # its own.
        for name, value in (("HARNESS_FLOORS", {}), ("HARNESS_GPU_FLOORS", {}), ("CARD_VARIANCE", {}),
                            ("STATUS", [])):
            patch = mock.patch.object(results, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def select(self, leg_id, lines, cpu=None):
        """The rows the leg's re-time selection chooses from LINES, as the matrix runs it on a node
        whose CPU model is CPU."""
        run = subprocess.run(["awk", "-v", f"limits={results.harness_limits(leg_id)}", "-v",
                              f"gpulimits={results.harness_limits(leg_id, gpu=True)}", "-v",
                              f"wallre={results.WALL_RE}", "-v", f"budget={results.HARNESS_BUDGET!r}",
                              "-v", f"skipre={results.retime_skip(cpu)}", results.RETIME_SELECT],
                             input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
        return sorted(line.split()[1] for line in run.stdout.splitlines())

    def budget(self, leg_id, times):
        return results.harness_budget(leg_id, leg_of(recs(time=times)))

    def test_an_unlisted_row_fails_past_the_budget_of_new_over_old_in_its_leg(self):
        times = {"harness-old/alu": 100.0, "harness-new/alu": 103.0,
                 "harness-old/frame": 100.0, "harness-new/frame": 103.5,
                 "harness-new/authored/first_use_wait_total": 900.0, "harness-old/sampling": 50.0,
                 "loop/gpu": 1000.0}
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            self.assertEqual(self.budget("10de:1e87", times),
                             (["harness budget frame: new 103.5 us over old 100.0 us is 1.035, past 1.03"], []))

    def test_a_listed_row_fails_only_past_its_ceiling_with_its_allowance_and_is_named_below_it(self):
        ceilings = {"10de:1e87": {"frame": 2.114, "alu": 1.2, "bandwidth": 1.1}}
        times = {"harness-old/frame": 100.0, "harness-new/frame": 211.0,
                 "harness-old/alu": 100.0, "harness-new/alu": 124.0,
                 "harness-old/bandwidth": 100.0, "harness-new/bandwidth": 101.0}
        with mock.patch.object(results, "HARNESS_CEILINGS", ceilings):
            problems, parked = self.budget("10de:1e87", times)
            self.assertEqual(problems, ["harness budget alu: new 124.0 us over old 100.0 us is 1.240, past its "
                                        "ceiling 1.2 with its allowance, 1.236"])
            self.assertEqual(parked, [
                "harness budget bandwidth: new 101.0 us over old 100.0 us is 1.010, within 1.03: its ceiling "
                "1.1 leaves the list",
                "harness budget frame: new 211.0 us over old 100.0 us is 2.110, its ceiling 2.114"])
            # Another GPU's ceilings are not this one's.
            self.assertEqual(len(self.budget("1002:744c", times)[0]), 2)

    def test_a_ceiling_s_allowance_is_the_budget_s_three_percent(self):
        # 2 times 1.03 and 206 over 100 are the same double.
        with mock.patch.object(results, "HARNESS_CEILINGS", {"a": {"t": 2.0}}):
            at = {"harness-old/t": 100.0, "harness-new/t": 206.0}
            past = {"harness-old/t": 100.0, "harness-new/t": 206.1}
            self.assertEqual(self.budget("a", at)[0], [])
            self.assertEqual(self.budget("a", past)[0], [
                "harness budget t: new 206.1 us over old 100.0 us is 2.061, past its ceiling 2.0 with its "
                "allowance, 2.060"])

    def test_a_row_the_old_runtime_failed_is_named_and_fails_nothing_and_the_new_runtime_s_fails(self):
        old = ("burst_2000x5: FAILED: old: the command buffer ended Error: Some(GpuRecoveryVictim)\n"
               "M2V-FAILED harness-old/burst_2000x5\n    M2V-MATRIX time harness-old/alu 100")
        new = "    M2V-MATRIX time harness-new/burst_2000x5 50\n    M2V-MATRIX time harness-new/alu 101"
        leg = results.parse_job(job(("8086:3e9b", "harness-old", 1, old), ("8086:3e9b", "harness-new", 0, new)))
        leg = leg["8086:3e9b"]
        records, problems = results.leg_records(leg)
        self.assertEqual(problems + results.leg_problems(leg, records, None), [])
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            self.assertEqual(results.harness_budget("8086:3e9b", leg),
                             ([], ["harness budget burst_2000x5: old failed, not compared"]))
        failed = "burst_2000x5: FAILED: new: why\nM2V-FAILED harness-new/burst_2000x5"
        leg = results.parse_job(job(("8086:3e9b", "harness-new", 1, failed)))["8086:3e9b"]
        self.assertEqual(results.leg_problems(leg, results.leg_records(leg)[0], None),
                         ["suite harness-new exited 1, 1 failed: harness-new/burst_2000x5"])

    def test_the_old_runtime_s_parked_records_are_still_the_budget_s_reference(self):
        prefix = results.PARKED_RECORDS[0]
        self.assertEqual(prefix, results.HARNESS_OLD)
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            self.assertEqual(len(self.budget("a", {prefix + "t": 10.0, "harness-new/t": 20.0})[0]), 1)

    def test_a_time_recorded_twice_has_no_one_time_to_judge(self):
        refused = ["harness budget frame: its time is recorded more than once, so it has no one time to judge"]
        # Within one suite, the later and slower time would otherwise go unseen: the first is kept.
        once = "    M2V-MATRIX time harness-old/frame 100\n    M2V-MATRIX time harness-new/frame 101\n"
        text = job(("10de:1e87", "harness", 0, once + "    M2V-MATRIX time harness-new/frame 300"))
        leg = results.parse_job(text)["10de:1e87"]
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            self.assertEqual(results.harness_budget("10de:1e87", leg), (refused, []))
            # Two suites that each record it.
            text = job(("10de:1e87", "harness", 0, once), ("10de:1e87", "other", 0,
                                                           "    M2V-MATRIX time harness-old/frame 300"))
            leg = results.parse_job(text)["10de:1e87"]
            self.assertEqual(results.harness_budget("10de:1e87", leg), (refused, []))
            # Recorded once each, it is judged.
            leg = results.parse_job(job(("10de:1e87", "harness", 0, once)))["10de:1e87"]
            self.assertEqual(results.harness_budget("10de:1e87", leg), ([], []))

    def test_a_sized_row_is_judged_only_where_both_runtimes_ran_it_at_one_size(self):
        times = {"harness-old/long_4cb": 10000.0, "harness-new/long_4cb": 10100.0}
        out = ("    M2V-MATRIX size harness-old/long_4cb 4096\n    M2V-MATRIX time harness-old/long_4cb 10000\n"
               "    M2V-MATRIX size harness-new/long_4cb {}\n    M2V-MATRIX time harness-new/long_4cb 10100")
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            leg = results.parse_job(job(("1002:699f", "harness", 0, out.format(4096))))["1002:699f"]
            self.assertEqual(results.leg_records(leg)[0]["size"],
                             {"harness-old/long_4cb": 4096, "harness-new/long_4cb": 4096})
            self.assertEqual(results.harness_budget("1002:699f", leg), ([], []))
            leg = results.parse_job(job(("1002:699f", "harness", 0, out.format(2048))))["1002:699f"]
            self.assertEqual(results.harness_budget("1002:699f", leg), ([
                "harness budget long_4cb: new ran at size 2048, old at 4096, so its times measure different "
                "work"], []))
            # One runtime's size alone is not the same work either.
            self.assertEqual(len(self.budget("a", times)[0]), 0)
            leg = leg_of(recs(time=times, size={"harness-new/long_4cb": 4096}))
            self.assertEqual(len(results.harness_budget("a", leg)[0]), 1)

    def test_the_old_runtime_s_own_spread_is_its_re_timed_rows_widest_on_what_each_is_judged_on(self):
        def retimed(row, main, olds, gpu=False):
            prefix = "harness-retime-gpu" if gpu else "harness-retime"
            out = {f"{'harness-gpu-' if gpu else 'harness-'}old/{row}": main}
            for pair, old in enumerate(olds, 1):
                out[f"{prefix}/{row}/{pair}/old"] = old
            return out

        # No row re-timed: no spread to read.
        self.assertIsNone(results.base_spread({"harness-old/alu": 100.0, "harness-new/alu": 101.0}))
        # The widest of the re-timed rows, on its time.
        times = {**retimed("alu", 100.0, [104.0, 98.0]), **retimed("bandwidth", 100.0, [100.0, 115.0])}
        spread, row = results.base_spread(times)
        self.assertEqual((round(spread, 3), row), (0.15, "bandwidth"))
        # A row judged on its GPU time spreads on it, its host time's spread aside.
        gpu = {"harness-gpu-new/sampling": 20.0, "harness-old/sampling": 50.0, "harness-new/sampling": 50.0,
               "harness-retime/sampling/1/old": 80.0, "harness-retime/sampling/2/old": 50.0,
               **retimed("sampling", 20.0, [20.4, 20.2], gpu=True)}
        spread, row = results.base_spread(gpu)
        self.assertEqual((round(spread, 3), row), (0.02, "sampling"))
        # A creation row's cold first samples never count.
        cold = {f"harness-old/{r}": 1000.0 for r in ("create_state/first",)}
        cold.update({f"harness-retime/create_state/first/{pair}/old": 3000.0 for pair in (1, 2)})
        self.assertIsNone(results.base_spread(cold))

    def test_a_leg_past_the_old_runtime_s_own_spread_fails_nothing_on_its_times(self):
        times = {"harness-old/alu": 100.0, "harness-new/alu": 110.0,
                 "harness-retime/alu/1/old": 100.0, "harness-retime/alu/1/new": 110.0,
                 "harness-retime/alu/2/old": 115.0, "harness-retime/alu/2/new": 126.5,
                 "harness-old/sized": 100.0, "harness-new/sized": 100.0}
        sizes = {"harness-old/sized": 1, "harness-new/sized": 2}
        past = ("harness budget alu: new 110.0 us over old 100.0 us is 1.100, past 1.03; re-timed in 2 pairs, "
                "1.100, 1.100, median 1.100")
        sized = "harness budget sized: new ran at size 2, old at 1, so its times measure different work"
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            # Its old runtime spread 15% across its passes on alu: its times judge nothing, its
            # records' problems stand.
            problems, parked = results.harness_budget("a", leg_of(recs(time=times, size=sizes)))
            self.assertEqual(problems, [sized])
            self.assertEqual(parked, [
                "harness budget VOID: the old runtime's own spread across its passes is 15.0% on alu, past "
                "10%, so the leg's times judge nothing", f"VOID: {past}"])
            # Within 10%, the same row fails.
            times["harness-retime/alu/2/old"], times["harness-retime/alu/2/new"] = 105.0, 115.5
            problems, _ = results.harness_budget("a", leg_of(recs(time=times, size=sizes)))
            self.assertEqual(problems, [sized, past])

    def test_a_row_over_its_budget_is_judged_by_the_median_of_its_re_timed_pairs(self):
        one = {"harness-old/vote_loop": 419.3, "harness-new/vote_loop": 470.8}
        past = "harness budget vote_loop: new 470.8 us over old 419.3 us is 1.123, past 1.03"

        def pairs(*news):
            out = dict(one)
            for pair, new in enumerate(news, 1):
                out[f"harness-retime/vote_loop/{pair}/old"] = 419.0
                out[f"harness-retime/vote_loop/{pair}/new"] = new
            return out

        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            # Back within in the re-time, A B B A, it passes, named with its pairs.
            self.assertEqual(self.budget("1002:1900", pairs(415.7, 425.5)), ([], [
                f"{past}; re-timed in 2 pairs, 0.992, 1.016, median 1.004, within: judged by the "
                "median"]))
            # A regression in the re-time still fails.
            self.assertEqual(self.budget("1002:1900", pairs(470.0, 466.0)), ([
                f"{past}; re-timed in 2 pairs, 1.122, 1.112, median 1.117"], []))
            # A row with a pair missing was not re-timed: its one time judges it.
            missing = pairs(415.7, 425.5)
            del missing["harness-retime/vote_loop/2/new"]
            self.assertEqual(self.budget("1002:1900", missing), ([past], []))
        # A listed row is re-timed past its ceiling's limit, and judged against that limit.
        with mock.patch.object(results, "HARNESS_CEILINGS", {"1002:1900": {"vote_loop": 1.05}}):
            self.assertEqual(self.budget("1002:1900", pairs(440.0, 441.0))[0], [])
            self.assertEqual(len(self.budget("1002:1900", pairs(455.0, 456.0))[0]), 1)

    def test_a_row_within_its_floor_passes_and_past_it_fails_only_if_its_re_time_stays_past(self):
        times = {"harness-old/frame": 100.0, "harness-new/frame": 112.0}
        with mock.patch.object(results, "HARNESS_CEILINGS", {}), \
                mock.patch.object(results, "HARNESS_FLOORS", {"a": {"frame": 1.1}}):
            # Within the old runtime's own spread with the allowance, 1.133: not past.
            self.assertEqual(self.budget("a", times), ([], []))
            # Another GPU's floor is not this one's.
            self.assertEqual(len(self.budget("b", times)[0]), 1)
            past = dict(times, **{"harness-new/frame": 120.0})
            line = "harness budget frame: new 120.0 us over old 100.0 us is 1.200, past its floor 1.1 with " \
                   "its allowance, 1.133"
            self.assertEqual(self.budget("a", past), ([line], []))
            # Past it on its one time, but back within when re-timed alone: not past.
            retimed = dict(past, **{"harness-retime/frame/1/old": 100.0, "harness-retime/frame/1/new": 110.0,
                                    "harness-retime/frame/2/old": 100.0, "harness-retime/frame/2/new": 112.0})
            self.assertEqual(self.budget("a", retimed), ([], [
                f"{line}; re-timed in 2 pairs, 1.100, 1.120, median 1.110, within: judged by the median"]))
            # Past it in the re-time too: past.
            retimed.update({"harness-retime/frame/1/new": 118.0, "harness-retime/frame/2/new": 121.0})
            self.assertEqual(len(self.budget("a", retimed)[0]), 1)
            # The re-time selection reads the floor's limit.
            self.assertEqual(results.harness_limits("a"), f"frame={1.1 * results.HARNESS_BUDGET!r}")
        # A ceiling over the floor sets the limit, and a floor over the ceiling does.
        with mock.patch.object(results, "HARNESS_CEILINGS", {"a": {"frame": 1.2}}), \
                mock.patch.object(results, "HARNESS_FLOORS", {"a": {"frame": 1.1}}):
            self.assertAlmostEqual(results.row_limit("a", "frame"), 1.236)
        with mock.patch.object(results, "HARNESS_CEILINGS", {"a": {"frame": 1.05}}), \
                mock.patch.object(results, "HARNESS_FLOORS", {"a": {"frame": 1.1}}):
            self.assertAlmostEqual(results.row_limit("a", "frame"), 1.133)

    def test_a_first_sample_is_judged_by_the_median_of_five_processes_and_never_alone(self):
        row = "create_64MiB/first_take"

        def processes(olds, news):
            out = {f"harness-old/{row}": olds[0], f"harness-new/{row}": news[0]}
            for n, (old, new) in enumerate(zip(olds[1:], news[1:]), 2):
                out[f"{results.HARNESS_FIRST}{row}/{n}/old"] = old
                out[f"{results.HARNESS_FIRST}{row}/{n}/new"] = new
            return out

        self.assertEqual(results.FIRST_RUNS, 5)
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            # One sample, however far past, is never past: it is named as not judged.
            one = {f"harness-old/{row}": 1000.0, f"harness-new/{row}": 2000.0}
            self.assertEqual(self.budget("a", one), ([], [
                f"harness budget {row}: new 2000.0 us over old 1000.0 us is 2.000, one sample: a first sample "
                "is judged by the median of 5 processes, not judged"]))
            # One slow process of five decides nothing.
            olds = [1000.0, 1010.0, 990.0, 1005.0, 995.0]
            self.assertEqual(self.budget("a", processes(olds, [1000.0, 3000.0, 1010.0, 990.0, 1020.0])),
                             ([], []))
            # A median past the limit is past.
            self.assertEqual(self.budget("a", processes(olds, [1100.0, 1200.0, 1150.0, 3000.0, 1120.0])), ([
                f"harness budget {row}: median of 5 processes, new 1150.0 us over old 1000.0 us is 1.150, "
                "past 1.03"], []))
            # Against its own floor.
            with mock.patch.object(results, "HARNESS_FLOORS", {"a": {row: 1.2}}):
                self.assertEqual(
                    self.budget("a", processes(olds, [1100.0, 1200.0, 1150.0, 3000.0, 1120.0]))[0], [])
        # The later processes' samples are parked records: the budget judges them.
        self.assertTrue(f"{results.HARNESS_FIRST}{row}/2/new".startswith(results.PARKED_RECORDS))

    def test_a_gated_first_take_is_judged_on_the_new_runtime_s_own_part_against_old_s_whole(self):
        row, driver = "create_64MiB/first_take", "create_64MiB/first_allocation_total"

        def processes(olds, news, drivers):
            out = {f"harness-old/{row}": olds[0], f"harness-new/{row}": news[0]}
            if drivers[0] is not None:
                out[f"harness-new/{driver}"] = drivers[0]
            for n, (old, new, part) in enumerate(zip(olds[1:], news[1:], drivers[1:]), 2):
                out[f"{results.HARNESS_FIRST}{row}/{n}/old"] = old
                out[f"{results.HARNESS_FIRST}{row}/{n}/new"] = new
                if part is not None:
                    out[f"{results.HARNESS_FIRST}{driver}/{n}/new"] = part
            return out

        gates = {"10de:1e87": {row: (driver, "a gap")}}
        olds = [5000.0, 5100.0, 4900.0, 5050.0, 4950.0]
        news = [12000.0, 12500.0, 11900.0, 12100.0, 12300.0]
        with mock.patch.object(results, "HARNESS_CEILINGS", {}), \
                mock.patch.object(results, "OWN_PART_GATES", gates):
            # The whole is 2.4 times old's; the own part, each process's less its driver's part,
            # is within, and the line shows both and names the gap.
            parts = [9500.0, 10000.0, 9400.0, 9600.0, 9800.0]
            self.assertEqual(self.budget("10de:1e87", processes(olds, news, parts)), ([], [
                f"harness budget {row}: median of 5 processes, the new runtime's own part 2500.0 us over "
                "old's whole 5000.0 us is 0.500; the driver's part 9600.0 us, new 12100.0 us whole is "
                "2.420: a gap"]))
            # The extra leg on another node is its card's.
            self.assertEqual(self.budget("10de:1e87@node-b", processes(olds, news, parts))[0], [])
            # An own part past the limit is a regression of its own.
            small = [5000.0, 5500.0, 4900.0, 5200.0, 5300.0]
            problems, parked = self.budget("10de:1e87", processes(olds, news, small))
            self.assertEqual(parked, [])
            self.assertEqual(len(problems), 1)
            self.assertIn("own part 7000.0 us over old's whole 5000.0 us is 1.400", problems[0])
            self.assertTrue(problems[0].endswith("past 1.03: a regression of its own"), problems[0])
            # A process without its driver's part leaves nothing to judge the gate on.
            missing = processes(olds, news, parts[:2] + [None] + parts[3:])
            problems, _ = self.budget("10de:1e87", missing)
            self.assertEqual(problems, [
                f"harness budget {row}: median of 5 processes, new 12100.0 us over old 5000.0 us is 2.420: "
                "judged on the new runtime's own part (a gap), which needs the driver's part from every "
                "process"])
            # Another card's row is judged whole.
            self.assertEqual(len(self.budget("1002:744c", processes(olds, news, parts))[0]), 1)

    def test_a_creation_call_alone_is_shown_beside_its_first_use_and_never_judged(self):
        row = "create_state/first_call"
        times = {f"harness-old/{row}": 1000.0, f"harness-new/{row}": 5000.0}
        for n in range(2, results.FIRST_RUNS + 1):
            times[f"{results.HARNESS_FIRST}{row}/{n}/old"] = 1000.0 + n
            times[f"{results.HARNESS_FIRST}{row}/{n}/new"] = 5000.0 + n
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            self.assertEqual(self.budget("a", times), ([], [
                f"harness budget {row}: median of 5 processes, new 5003.0 us over old 1003.0 us is 4.988: "
                "shown, not judged: the creation call alone, which old returns from before it makes its "
                "pipeline"]))
            # One process's sample alone is not shown, and judges nothing either.
            one = {f"harness-old/{row}": 1000.0, f"harness-new/{row}": 5000.0}
            self.assertEqual(self.budget("a", one), ([], []))

    def test_a_warm_up_is_shown_beside_the_first_it_comes_before_and_never_judged(self):
        row = "create_private_16MiB/device_first"
        times = {f"harness-old/{row}": 1000.0, f"harness-new/{row}": 3000.0}
        for n in range(2, results.FIRST_RUNS + 1):
            times[f"{results.HARNESS_FIRST}{row}/{n}/old"] = 1000.0 + n
            times[f"{results.HARNESS_FIRST}{row}/{n}/new"] = 3000.0 + n
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            self.assertEqual(self.budget("a", times), ([], [
                f"harness budget {row}: median of 5 processes, new 3003.0 us over old 1003.0 us is 2.994: "
                "shown, not judged: the warm-up, the first object of its kind on the device"]))
            # The first it comes before is judged.
            first = {key.replace("/device_first", "/first_take"): us for key, us in times.items()}
            self.assertEqual(self.budget("a", first), ([
                "harness budget create_private_16MiB/first_take: median of 5 processes, new 3003.0 us over "
                "old 1003.0 us is 2.994, past 1.03"], []))

    def test_a_first_at_one_thread_is_kept_from_every_process_shown_and_never_judged(self):
        row = "create_state/first_one_thread"
        keep = results.FIRST_KEEP.replace("$runtime", "new").replace("$n", "2")
        run = subprocess.run(["sed", "-n", keep], input=f"M2V-MATRIX time harness-new/{row} 3000.000\n",
                             capture_output=True, text=True, check=True)
        self.assertEqual(run.stdout.splitlines(), [f"M2V-MATRIX time {results.HARNESS_FIRST}{row}/2/new 3000.000"])
        times = {f"harness-old/{row}": 1000.0, f"harness-new/{row}": 3000.0}
        for n in range(2, results.FIRST_RUNS + 1):
            times[f"{results.HARNESS_FIRST}{row}/{n}/old"] = 1000.0 + n
            times[f"{results.HARNESS_FIRST}{row}/{n}/new"] = 3000.0 + n
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            self.assertEqual(self.budget("a", times), ([], [
                f"harness budget {row}: median of 5 processes, new 3003.0 us over old 1003.0 us is 2.994: "
                "shown, not judged: a first dispatched as one thread, a size of few dispatches"]))

    def test_the_creation_processes_record_each_key_once_beside_the_other_rows(self):
        # Each harness process prints its device's totals and gauges at drop, the creation process
        # as well as the other rows'.
        others = ["M2V-MATRIX time harness-new/alu 10.000",
                  "M2V-MATRIX metric harness-new/device/stamps_not_available 0",
                  "M2V-MATRIX gauge harness-new/device/blocks@drop 0"]
        creates = ["M2V-MATRIX time harness-new/create_64MiB 700.000",
                   "M2V-MATRIX time harness-new/create_64MiB/first_take 900.000",
                   "M2V-MATRIX time harness-old/create_64MiB/first_take 950.000",
                   "M2V-MATRIX time harness-gpu-old/create_64MiB/first_take 5.000",
                   "M2V-MATRIX metric harness-new/create_64MiB/first_allocations 1",
                   "M2V-MATRIX gauge harness-new/create_64MiB/buffers@100 0",
                   "M2V-MATRIX metric harness-new/device/stamps_not_available 0",
                   "M2V-MATRIX gauge harness-new/device/blocks@drop 0", "a table row"]

        def sed(program, lines, quiet=False):
            run = subprocess.run(["sed"] + (["-n"] if quiet else []) + [program], input="\n".join(lines) + "\n",
                                 capture_output=True, text=True, check=True)
            return run.stdout.splitlines()

        first = sed(results.CREATES_APART, creates)
        # The shell gives the later processes' program its process and runtime.
        keep = results.FIRST_KEEP.replace("$n", "2")
        later = sed(keep, creates, quiet=True)
        # One process times both runtimes: each line's runtime names its sample.
        self.assertEqual(later, [f"M2V-MATRIX time {results.HARNESS_FIRST}create_64MiB/first_take/2/new 900.000",
                                 f"M2V-MATRIX time {results.HARNESS_FIRST}create_64MiB/first_take/2/old 950.000"])
        suite = results.parse_suite(others + first + later)
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

    def test_the_floors_table_reads_per_leg_and_row(self):
        text = "# a comment\nrow  a  b\nframe  1.10  -\nalu  1.00  1.02\n"
        self.assertEqual(results.read_floors(text), {"a": {"frame": 1.1, "alu": 1.0}, "b": {"alu": 1.02}})
        # The stored tables: every floor is a ratio of at least 1, on a leg id; the GPU floors name no
        # row judged on its host time, and each card's variance row has a GPU floor.
        floors = results.read_floors((Path(results.__file__).parent / "floors.tsv").read_text())
        gpu_floors = results.read_floors((Path(results.__file__).parent / "gpu-floors.tsv").read_text())
        for table in (floors, gpu_floors):
            for leg, rows in table.items():
                self.assertRegex(leg, r"^[0-9a-f]{4}:[0-9a-f]{4}$")
                self.assertTrue(all(value >= 1.0 for value in rows.values()), leg)
        self.assertFalse([row for rows in gpu_floors.values() for row in rows if row.startswith(results.WALL_ROWS)])
        for leg, rows in CARD_VARIANCE_STORED.items():
            self.assertTrue(set(rows) <= set(gpu_floors[leg]), leg)

    def test_a_leg_re_times_the_rows_the_budget_would_fail_on_their_one_time(self):
        ceilings = {"a": {"listed": 2.0, "edge": 2.0}}
        times = {"alu": (100.0, 103.0), "frame": (100.0, 103.5), "listed": (100.0, 230.0),
                 "edge": (100.0, 206.0), "under": (100.0, 90.0)}
        lines = [f"M2V-MATRIX time harness-{runtime}/{row} {pair[i]:.3f}"
                 for row, pair in times.items() for i, runtime in enumerate(("old", "new"))]
        lines += ["M2V-MATRIX time harness-new/authored/first_use_wait_total 900.000",
                  "M2V-MATRIX gauge harness-new/alu/buffers@100 2", "a table row"]
        with mock.patch.object(results, "HARNESS_CEILINGS", ceilings):
            chosen = self.select("a", lines)
            records = {f"harness-{runtime}/{row}": pair[i]
                       for row, pair in times.items() for i, runtime in enumerate(("old", "new"))}
            failed = sorted(line.split()[2].rstrip(":") for line in self.budget("a", records)[0])
            # Each limit reads back as the double harness_budget compares with.
            self.assertEqual(results.harness_limits("a"), "edge=2.06 listed=2.06")
        self.assertEqual(chosen, ["frame", "listed"])
        self.assertEqual(chosen, failed)

    def test_a_frames_latency_row_takes_its_workload_s_limit(self):
        ceilings = {"a": {"frame": 2.0}}
        times = {"frame": (100.0, 150.0), f"frame{results.FROM_IDLE}": (100.0, 230.0),
                 "frame_5cb": (100.0, 102.0), f"frame_5cb{results.FROM_IDLE}": (100.0, 110.0)}
        records = {f"harness-{runtime}/{row}": pair[i]
                   for row, pair in times.items() for i, runtime in enumerate(("old", "new"))}
        lines = [f"M2V-MATRIX time {key} {value:.3f}" for key, value in records.items()]
        with mock.patch.object(results, "HARNESS_CEILINGS", ceilings):
            problems, parked = self.budget("a", records)
            # frame's latency from idle is past frame's ceiling, frame_5cb's past the budget, which
            # frame_5cb's own row has; frame's throughput is within its ceiling.
            self.assertEqual([line.split(":")[0] for line in problems],
                             ["harness budget frame_5cb_from_idle", "harness budget frame_from_idle"])
            self.assertIn("past its ceiling 2.0 with its allowance, 2.060", problems[1])
            self.assertEqual(parked, [
                "harness budget frame: new 150.0 us over old 100.0 us is 1.500, its ceiling 2.0"])
            self.assertEqual([row for row, *_ in results.over_rows("a", recs(time=records))],
                             ["frame_5cb_from_idle", "frame_from_idle"])
            # The leg re-times the rows it fails, the latency rows against their workloads' limits.
            self.assertEqual(self.select("a", lines), ["frame_5cb_from_idle", "frame_from_idle"])
        self.assertIn(f"sub(/{results.FROM_IDLE}$/", results.RETIME_SELECT)

    def test_re_timed_pairs_are_parked_records(self):
        self.assertTrue(f"{results.HARNESS_RETIME}vote_loop/1/new".startswith(results.PARKED_RECORDS))
        self.assertTrue(f"{results.HARNESS_RETIME_GPU}vote_loop/1/new".startswith(results.PARKED_RECORDS))

    def test_a_row_with_a_gpu_time_is_judged_on_it_and_a_frame_or_burst_row_on_its_time(self):
        times = {"harness-old/frame": 100.0, "harness-new/frame": 120.0,
                 "harness-gpu-old/frame": 80.25, "harness-gpu-new/frame": 81.0,
                 "harness-old/burst_160": 100.0, "harness-new/burst_160": 101.0,
                 "harness-gpu-old/burst_160": 50.0, "harness-gpu-new/burst_160": 90.0,
                 "harness-old/alu": 100.0, "harness-new/alu": 100.0,
                 "harness-gpu-old/alu": 10.0, "harness-gpu-new/alu": 900.0,
                 "harness-old/simd_reduce": 45.0, "harness-new/simd_reduce": 55.0,
                 "harness-gpu-old/simd_reduce": 37.0, "harness-gpu-new/simd_reduce": 37.1}
        frame = ("harness budget frame: new 120.0 us (GPU p50 81.0 us) over old 100.0 us (GPU p50 80.2 us) "
                 "is 1.200, past 1.03")
        alu = ("harness budget alu: GPU p50 new 900.0 us over old 10.0 us is 90.000 (time new 100.0 us over old "
               "100.0 us is 1.000), past 1.03")
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            # alu is past on its GPU time, its time level; simd_reduce is within on its GPU time, its
            # time 1.22 past; frame is judged on its time, burst_160 on its time, whatever their GPU's.
            self.assertEqual(self.budget("a", times), ([alu, frame], []))
            self.assertTrue(results.gpu_judged(times, "alu"))
            self.assertFalse(results.gpu_judged(times, "burst_160"))
            # Only one runtime's GPU time: the row is judged on its time, the other's line showing none.
            del times["harness-gpu-old/frame"], times["harness-gpu-old/alu"]
            self.assertEqual(self.budget("a", times)[0], [
                "harness budget frame: new 120.0 us (GPU p50 81.0 us) over old 100.0 us is 1.200, past 1.03"])
            # The re-time selection makes the same choice.
            times["harness-gpu-old/alu"] = 10.0
            lines = [f"M2V-MATRIX time {key} {value:.3f}" for key, value in times.items()]
            self.assertEqual(self.select("a", lines), ["alu", "frame"])
        # A GPU time's cross-GPU outlier is named and fails no leg.
        key = f"{results.HARNESS_GPU}new/frame"
        bases = {leg: {key: 100} for leg in "abcd"}
        failing, parked = results.time_outliers(
            {"a": {key: 100}, "b": {key: 101}, "c": {key: 99}, "d": {key: 150}}, bases)
        self.assertEqual((failing, list(parked)), ({}, ["d"]))

    def test_a_row_judged_on_its_gpu_time_takes_its_gpu_floor_and_its_re_timed_gpu_pairs(self):
        times = {"harness-old/sampling": 35.0, "harness-new/sampling": 36.0,
                 "harness-gpu-old/sampling": 16.0, "harness-gpu-new/sampling": 17.6}
        line = ("harness budget sampling: GPU p50 new 17.6 us over old 16.0 us is 1.100 (time new 36.0 us over "
                "old 35.0 us is 1.029), past its GPU floor 1.05 with its allowance, 1.082")
        with mock.patch.object(results, "HARNESS_CEILINGS", {}), \
                mock.patch.object(results, "HARNESS_FLOORS", {"a": {"sampling": 1.3}}):
            # Its host time's floor is not its limit.
            self.assertEqual(len(self.budget("a", times)[0]), 1)
            with mock.patch.object(results, "HARNESS_GPU_FLOORS", {"a": {"sampling": 1.1}}):
                self.assertEqual(self.budget("a", times), ([], []))
                self.assertEqual(results.harness_limits("a", gpu=True), f"sampling={1.1 * results.HARNESS_BUDGET!r}")
            with mock.patch.object(results, "HARNESS_GPU_FLOORS", {"a": {"sampling": 1.05}}):
                self.assertEqual(self.budget("a", times), ([line], []))
                lines = [f"M2V-MATRIX time {key} {value:.3f}" for key, value in times.items()]
                self.assertEqual(self.select("a", lines), ["sampling"])
                # Its re-timed GPU pairs judge it, not its re-timed times.
                pairs = dict(times, **{"harness-retime/sampling/1/old": 35.0, "harness-retime/sampling/1/new": 50.0,
                                       "harness-retime/sampling/2/old": 35.0, "harness-retime/sampling/2/new": 50.0,
                                       "harness-retime-gpu/sampling/1/old": 16.0,
                                       "harness-retime-gpu/sampling/1/new": 16.2,
                                       "harness-retime-gpu/sampling/2/old": 16.0,
                                       "harness-retime-gpu/sampling/2/new": 16.4})
                self.assertEqual(self.budget("a", pairs), ([], [
                    f"{line}; re-timed in 2 pairs, 1.012, 1.025, median 1.019, within: judged by the median"]))

    def test_a_card_s_variance_row_is_named_and_fails_nothing(self):
        times = {"harness-old/sampling": 35.0, "harness-new/sampling": 36.0,
                 "harness-gpu-old/sampling": 16.0, "harness-gpu-new/sampling": 20.0}
        variance = {"a": {"sampling": "its GPU p50 ran 16-20 us"}}
        with mock.patch.object(results, "HARNESS_CEILINGS", {}), \
                mock.patch.object(results, "CARD_VARIANCE", variance):
            self.assertEqual(self.budget("a", times), ([], [
                "harness budget sampling: GPU p50 new 20.0 us over old 16.0 us is 1.250 (time new 36.0 us over "
                "old 35.0 us is 1.029), past 1.03: the card's own variance, not judged (its GPU p50 ran 16-20 us)"]))
            # Within its limit and over the budget, it is named too.
            with mock.patch.object(results, "HARNESS_GPU_FLOORS", {"a": {"sampling": 1.3}}):
                self.assertEqual(self.budget("a", times)[1], [
                    "harness budget sampling: GPU p50 new 20.0 us over old 16.0 us is 1.250 (time new 36.0 us "
                    "over old 35.0 us is 1.029), within its limit: the card's own variance (its GPU p50 ran "
                    "16-20 us)"])
            # Another GPU's row is judged.
            self.assertEqual(len(self.budget("b", times)[0]), 1)

    def test_the_frozen_arm_voids_a_row_whose_old_time_moved_past_ten_percent_on_the_base(self):
        row = "create_state/first"
        times = {"harness-old/frame": 100.0, "harness-new/frame": 150.0,
                 "harness-old/alu": 100.0, "harness-new/alu": 150.0,
                 "harness-old/sampling": 30.0, "harness-new/sampling": 30.0,
                 "harness-gpu-old/sampling": 20.0, "harness-gpu-new/sampling": 30.0,
                 f"harness-old/{row}": 1000.0, f"harness-new/{row}": 2000.0}
        for n in range(2, results.FIRST_RUNS + 1):
            times[f"{results.HARNESS_FIRST}{row}/{n}/old"] = 1000.0
            times[f"{results.HARNESS_FIRST}{row}/{n}/new"] = 2000.0
        # The base's old: frame's time moved +25%, alu's +5%; sampling's host time held while its
        # GPU time, which it is judged on, moved -20%; the first's median moved +11.1%, one slow
        # process of five deciding nothing.
        base = dict(times, **{"harness-old/frame": 80.0, "harness-old/alu": 95.24,
                              "harness-gpu-old/sampling": 25.0, f"harness-old/{row}": 5000.0})
        for n in range(2, results.FIRST_RUNS + 1):
            base[f"{results.HARNESS_FIRST}{row}/{n}/old"] = 900.0
        void = "so the round judges nothing from it: re-time it in one process, both runtimes interleaved"
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            problems, parked = results.harness_budget("a", leg_of(recs(time=times)), base)
            self.assertEqual(problems, ["harness budget alu: new 150.0 us over old 100.0 us is 1.500, past 1.03"])
            self.assertEqual(parked, [
                f"harness budget {row}: VOID: old moved +11.1% from the base's, 900.0 us to 1000.0 us, {void}",
                f"harness budget frame: VOID: old moved +25.0% from the base's, 80.0 us to 100.0 us, {void}",
                f"harness budget sampling: VOID: old moved -20.0% from the base's, 25.0 us to 20.0 us, {void}"])
            self.assertEqual([(row, round(move, 3)) for row, _, _, move in results.frozen_void(times, base)],
                             [(row, 0.111), ("frame", 0.25), ("sampling", -0.2)])
            # No base, or a base without the row, voids nothing.
            self.assertEqual(len(self.budget("a", times)[0]), 4)
            self.assertEqual(len(results.harness_budget("a", leg_of(recs(time=times)), {"loop/gpu": 1.0})[0]), 4)
            # A base of another harness or old runtime leaves the creation rows alone compared; one
            # of another timing shape, none.
            problems, parked = results.harness_budget("a", leg_of(recs(time=times)), base, results.REBASED_TREES)
            self.assertEqual((len(problems), [line.split(":")[0] for line in parked]),
                             (3, [f"harness budget {row}"]))
            self.assertEqual([row for row, *_ in results.frozen_void(times, base, results.REBASED_TREES)], [row])
            problems, parked = results.harness_budget("a", leg_of(recs(time=times)), base, results.REBASED_SHAPE)
            self.assertEqual((len(problems), parked), (4, []))
            self.assertEqual(results.frozen_void(times, base, results.REBASED_SHAPE), [])
            self.assertEqual([line.split(" | ")[-1] for line in
                              results.harness_rows(recs(time=times), base, results.REBASED_TREES)], [
                "1.500, old rebased on the base", "2.000, old +11.1% on the base VOID",
                "1.500, old rebased on the base", "1.000, old rebased on the base"])
        lines = results.harness_rows(recs(time=times), base)
        self.assertEqual([line.split(" | ")[-1] for line in lines], [
            "1.500, old +5.0% on the base", "2.000, old +11.1% on the base VOID",
            "1.500, old +25.0% on the base VOID", "1.000, old -20.0% on the base VOID"])
        self.assertEqual(results.harness_rows(recs(time=times)), [line.split(", old ")[0] for line in lines])

    def test_a_host_time_row_past_its_limit_by_less_than_old_s_own_move_is_void(self):
        first = "create_state/first"
        times = {"harness-old/alu": 100.0, "harness-new/alu": 106.0,
                 "harness-old/frame": 100.0, "harness-new/frame": 112.0,
                 "harness-old/loop": 100.0, "harness-new/loop": 106.0,
                 "harness-old/sampling": 30.0, "harness-new/sampling": 30.0,
                 "harness-gpu-old/sampling": 20.0, "harness-gpu-new/sampling": 21.0,
                 f"harness-old/{first}": 1000.0, f"harness-new/{first}": 1040.0}
        for n in range(2, results.FIRST_RUNS + 1):
            times[f"{results.HARNESS_FIRST}{first}/{n}/old"] = 1000.0
            times[f"{results.HARNESS_FIRST}{first}/{n}/new"] = 1040.0
        for pair in range(1, results.RETIME_PAIRS + 1):
            times[f"{results.HARNESS_RETIME}loop/{pair}/old"] = 100.0
            times[f"{results.HARNESS_RETIME}loop/{pair}/new"] = 106.0
        # Old moved +5.3% on alu, frame and loop, +2.6% on sampling's GPU time and -4.8% on the
        # first's median: each under FROZEN_DRIFT, so the first rule voids none.
        base = dict(times, **{"harness-old/alu": 95.0, "harness-old/frame": 95.0, "harness-old/loop": 95.0,
                              "harness-gpu-old/sampling": 19.5, f"harness-old/{first}": 1050.0})
        for n in range(2, results.FIRST_RUNS + 1):
            base[f"{results.HARNESS_FIRST}{first}/{n}/old"] = 1050.0
        void = "so the round judges nothing from it: re-time it in one process, both runtimes interleaved"
        with mock.patch.object(results, "HARNESS_CEILINGS", {}), mock.patch.object(results, "OWN_PART_GATES", {}):
            problems, parked = results.harness_budget("a", leg_of(recs(time=times)), base)
            # alu, 2.9% past its limit, and the first, 1.0% past, are within old's own move;
            # frame, 8.7% past, is not; loop is judged by its re-timed pairs and sampling by its
            # GPU time, which the second rule never voids.
            self.assertEqual([line.split(":")[0] for line in problems],
                             ["harness budget frame", "harness budget loop", "harness budget sampling"])
            self.assertEqual(parked, [
                "harness budget alu: new 106.0 us over old 100.0 us is 1.060: VOID: past its limit 1.030 by "
                f"2.9%, less than old's own move of +5.3% from the base's, {void}",
                f"harness budget {first}: median of {results.FIRST_RUNS} processes, new 1040.0 us over old "
                f"1000.0 us is 1.040: VOID: past its limit 1.030 by 1.0%, less than old's own move of -4.8% "
                f"from the base's, {void}"])
            self.assertEqual([(row, round(ratio, 3), round(move, 3)) for row, ratio, _, move in
                              results.excess_voids("a", recs(time=times), base)],
                             [("alu", 1.06, 0.053), (first, 1.04, -0.048)])
            # A base that moved past FROZEN_DRIFT leaves the row to the first rule alone.
            far = dict(base, **{"harness-old/alu": 80.0})
            self.assertEqual([row for row, *_ in results.excess_voids("a", recs(time=times), far)], [first])
            # No base voids nothing.
            self.assertEqual(len(self.budget("a", times)[0]), 5)

    def test_a_re_timed_run_keeps_the_row_s_time_and_gpu_time(self):
        output = ["M2V-MATRIX time harness-new/sampling 36.000", "M2V-MATRIX time harness-gpu-new/sampling 17.600",
                  "M2V-MATRIX time harness-new/alu 10.000", "a table row"]
        keep = results.RETIME_KEEP.replace("$timed", "new").replace("$row", "sampling").replace("$pair", "2")
        run = subprocess.run(["sed", "-n", keep], input="\n".join(output) + "\n", capture_output=True, text=True,
                             check=True)
        self.assertEqual(run.stdout.splitlines(), [
            "M2V-MATRIX time harness-retime/sampling/2/new 36.000",
            "M2V-MATRIX time harness-retime-gpu/sampling/2/new 17.600"])

    def test_every_row_shows_its_times_gpu_times_and_instructions_on_both_runtimes(self):
        times = {"harness-old/frame": 100.0, "harness-new/frame": 120.0,
                 "harness-gpu-old/frame": 80.25, "harness-gpu-new/frame": 81.0,
                 "harness-new/create_64MiB/first_take": 50.0, "harness-old/create_64MiB/first_take": 40.0,
                 "harness-old/sampling": 50.0, "harness-first/create_64MiB/first_take/2/new": 51.0,
                 "loop/gpu": 1000.0}
        counted = {"harness-new/frame/per_run": 61253, "harness-old/frame/per_run": 149185,
                   "harness-new/frame/own_per_run": 40001, "harness-new/frame/others_per_run": 21252}
        self.assertEqual(results.harness_rows(recs(time=times, instructions=counted)), [
            "create_64MiB/first_take              new       50.0 us, GPU p50          - us | old       40.0 us, "
            "GPU p50          - us | 1.250",
            "frame                                new      120.0 us, GPU p50       81.0 us, 61253 instructions "
            "(40001 own) | old      100.0 us, GPU p50       80.2 us, 149185 instructions | 1.200, "
            "instructions 0.411, own over old 0.268",
            "sampling                             new          - us, GPU p50          - us | old       50.0 us, "
            "GPU p50          - us | -"])

    def test_the_frame_and_burst_rows_keep_their_time_s_limit_and_pass_the_instructions_gate_too(self):
        times = {"harness-old/frame": 100.0, "harness-new/frame": 101.0,
                 "harness-old/alu": 100.0, "harness-new/alu": 101.0}
        counted = {"harness-new/frame/per_run": 161000, "harness-old/frame/per_run": 149000,
                   "harness-new/frame/own_per_run": 61000, "harness-new/frame/others_per_run": 100000,
                   "harness-new/alu/per_run": 9000, "harness-old/alu/per_run": 5000}

        def budget(leg_id, counts, these=times):
            return results.harness_budget(leg_id, leg_of(recs(time=these, instructions=counts)))

        def lenient(own, ratio, others=100000, over=""):
            return (f"harness budget frame: LENIENT instructions new own {own} per run over old whole 149000 is "
                    f"{ratio}; new others {others}, old's driver threads not separable{over}")

        exact = "harness budget frame: EXACT instructions new own {} per run, {}"
        with mock.patch.object(results, "HARNESS_CEILINGS", {}), \
                mock.patch.object(results, "HARNESS_FLOORS", {}):
            # Both within the time's budget. alu, not a frame or burst row, has no gate, its count
            # over old's though it is. The whole count, a driver's threads in it, is over old's
            # and judges nothing: the runtime's own count is the one gated.
            with mock.patch.object(results, "STORED_INSTRUCTIONS", {}):
                self.assertEqual(budget("10de:1e87", counted), ([], [
                    exact.format(61000, "none stored"), lenient(61000, "0.409")]))
            stored = {"all": {"frame": 60000}}
            with mock.patch.object(results, "STORED_INSTRUCTIONS", stored):
                # Within 1.03 times the stored count: only the lenient line.
                self.assertEqual(budget("10de:1e87", counted), ([], [lenient(61000, "0.409")]))
                over = dict(counted, **{"harness-new/frame/own_per_run": 61900})
                self.assertEqual(budget("10de:1e87", over), ([
                    exact.format(61900, "stored 60000, is 1.032") + ": past 1.03 times the stored count"],
                    [lenient(61900, "0.415")]))
                under = dict(counted, **{"harness-new/frame/own_per_run": 59000})
                self.assertEqual(budget("10de:1e87", under), ([], [
                    exact.format(59000, "stored 60000, is 0.983") + ": more than 1% under the stored count, "
                    "which ratchets down to it", lenient(59000, "0.396")]))
                past_old = dict(counted, **{"harness-new/frame/own_per_run": 150000})
                self.assertEqual(budget("10de:1e87", past_old)[0], [
                    exact.format(150000, "stored 60000, is 2.500") + ": past 1.03 times the stored count",
                    lenient(150000, "1.007", over=": new own over old whole")])
                # A whole count with no count of the runtime's own threads: the gate could not run.
                no_own = {key: value for key, value in counted.items() if "own" not in key and "others" not in key}
                self.assertEqual(budget("10de:1e87", no_own)[0], [
                    "harness budget frame: instructions new 161000 per run, but no count of its own threads"])
                # A time past its limit still fails, its count within the gate though it is.
                slow = dict(times, **{"harness-new/frame": 120.0})
                self.assertEqual(budget("10de:1e87", counted, slow)[0], [
                    "harness budget frame: new 120.0 us over old 100.0 us is 1.200, past 1.03"])
            # A driver family's count before every leg's.
            with mock.patch.object(results, "STORED_INSTRUCTIONS", {"all": {"frame": 60000},
                                                                    "1002": {"frame": 70000}}):
                self.assertEqual(results.stored_count("1002:744c", "frame"), 70000)
                self.assertEqual(results.stored_count("10de:2204", "frame"), 60000)
            # A leg with no counts (Windows, or a refused counter) has no gate.
            self.assertEqual(budget("8086:3e9b", {}), ([], []))

    def test_a_frame_row_where_the_cpu_clock_follows_the_load_is_judged_by_its_instructions_alone(self):
        cpu = next(iter(results.CLOCK_FOLLOWS_LOAD))
        times = {"harness-old/frame": 100.0, "harness-new/frame": 130.0,
                 "harness-old/frame_from_idle": 100.0, "harness-new/frame_from_idle": 150.0,
                 "harness-old/burst_160": 100.0, "harness-new/burst_160": 110.0,
                 "harness-old/alu": 100.0, "harness-new/alu": 110.0}
        counted = {"harness-new/frame/per_run": 161000, "harness-old/frame/per_run": 149000,
                   "harness-new/frame/own_per_run": 61000, "harness-new/frame/others_per_run": 100000}
        lines = [f"M2V-MATRIX time {key} {value:.3f}" for key, value in times.items()]

        def budget(these=times, counts=counted, on=cpu):
            return results.harness_budget("1002:15bf", leg_of(recs(time=these, instructions=counts)), cpu=on)

        lenient = ("harness budget frame: LENIENT instructions new own {} per run over old whole 149000 is {}; "
                   "new others 100000, old's driver threads not separable")
        not_judged = ("harness budget {}: new {} us over old 100.0 us is {}, not judged: judged by {} "
                      "instructions per frame, its time only as a cross-GPU outlier, as its node's CPU clock "
                      "follows the load")
        with mock.patch.object(results, "HARNESS_CEILINGS", {}), \
                mock.patch.object(results, "STORED_INSTRUCTIONS", {}):
            # On the listed CPU the frame rows' times fail nothing and are named; their count is
            # judged; the burst row, and every other, is judged on its time as anywhere.
            problems, parked = budget()
            self.assertEqual(problems, ["harness budget alu: new 110.0 us over old 100.0 us is 1.100, past 1.03",
                                        "harness budget burst_160: new 110.0 us over old 100.0 us is 1.100, past 1.03"])
            self.assertEqual(parked, [
                "harness budget frame: EXACT instructions new own 61000 per run, none stored",
                lenient.format(61000, "0.409"), not_judged.format("frame", "130.0", "1.300", "its"),
                not_judged.format("frame_from_idle", "150.0", "1.500", "its workload's")])
            # Its count past old's whole count fails it there.
            over = dict(counted, **{"harness-new/frame/own_per_run": 150000})
            self.assertIn(lenient.format(150000, "1.007") + ": new own over old whole", budget(counts=over)[0])
            # On any other CPU, and on a leg whose note names none, the frame rows' times fail.
            for other in ("AMD Ryzen 7 8745H w/ Radeon 780M Graphics", None):
                problems, _ = budget(on=other)
                self.assertIn("harness budget frame: new 130.0 us over old 100.0 us is 1.300, past 1.03", problems)
                self.assertIn("harness budget frame_from_idle: new 150.0 us over old 100.0 us is 1.500, past 1.03",
                              problems)
            # The leg re-times none of them there, and every row the budget fails elsewhere.
            self.assertEqual(self.select("1002:15bf", lines, cpu), ["alu", "burst_160"])
            self.assertEqual(self.select("1002:15bf", lines), ["alu", "burst_160", "frame", "frame_from_idle"])
            # The old runtime's spread on a frame row voids no leg there, and does elsewhere.
            spread = dict(times, **{"harness-retime/frame/1/old": 100.0, "harness-retime/frame/2/old": 130.0,
                                    "harness-retime/frame/1/new": 130.0, "harness-retime/frame/2/new": 130.0})
            self.assertIsNone(results.base_spread(spread, cpu))
            self.assertEqual(results.base_spread(spread)[1], "frame")
            self.assertFalse(results.base_spread_void(spread, cpu))
            self.assertTrue(results.base_spread_void(spread))
            # The time stays a cross-GPU outlier check: harness-new/ records are no parked records.
            self.assertFalse("harness-new/frame".startswith(results.PARKED_RECORDS))
            # The over table lists their counts there, never their times.
            rows = results.over_rows("1002:15bf", recs(time=times, instructions=over), cpu)
            self.assertEqual([(row, kind) for row, kind, _, _ in rows],
                             [("alu", "host time"), ("burst_160", "host time"), ("frame", "count")])
            rows = results.over_rows("1002:15bf", recs(time=times, instructions=over))
            self.assertEqual([(row, kind) for row, kind, _, _ in rows],
                             [("alu", "host time"), ("burst_160", "host time"), ("frame", "count"),
                              ("frame", "host time"), ("frame_from_idle", "host time")])

    def test_the_stored_counts_table_reads_per_family_and_row(self):
        text = "# a comment\nrow  all  1002\nframe  61253  -\nburst_160  -  90000\n"
        self.assertEqual(results.read_counts(text), {"all": {"frame": 61253}, "1002": {"burst_160": 90000}})
        for family, rows in results.STORED_INSTRUCTIONS.items():
            self.assertRegex(family, r"^(all|[0-9a-f]{4})$")
            self.assertTrue(all(row.startswith(results.WALL_ROWS) for row in rows), family)

    def test_every_ceiling_is_over_the_budget(self):
        for leg, rows in results.HARNESS_CEILINGS.items():
            for row, ceiling in rows.items():
                self.assertGreater(ceiling, results.HARNESS_BUDGET, f"{leg} {row}")

    def test_an_owned_row_fails_only_past_its_owned_ratio_and_a_gap_only_past_its_ruled_ceiling(self):
        entries = [("10de:1e87", "alu", "open", "core", "abc1234", 1.2, "-"),
                   ("10de:1e87", "sampling_*", "gap", "a ruling", "-", None, "-"),
                   ("10de:1e87", "bandwidth", "gap", "its ruling", "-", None, "-"),
                   ("10de:1e87", "frame", "note", "-", "-", None, "looked at once")]
        times = {"harness-old/alu": 100.0, "harness-new/alu": 123.0,
                 "harness-old/sampling_deep": 100.0, "harness-new/sampling_deep": 150.0,
                 "harness-old/bandwidth": 100.0, "harness-new/bandwidth": 120.0,
                 "harness-old/frame": 100.0, "harness-new/frame": 110.0}
        with mock.patch.object(results, "STATUS", entries), \
                mock.patch.object(results, "HARNESS_CEILINGS", {"10de:1e87": {"bandwidth": 1.1}}):
            # alu is within 3% of the ratio it was owned at, sampling_deep a gap with no ruled
            # ceiling: both named, neither failing. bandwidth's gap is past the ceiling its ruling
            # gave it, and frame's note holds nothing: both fail.
            self.assertEqual(self.budget("10de:1e87", times), ([
                "harness budget bandwidth: new 120.0 us over old 100.0 us is 1.200, past its ceiling 1.1 with "
                "its allowance, 1.133: a known gap (its ruling), past the ceiling its ruling gave it",
                "harness budget frame: new 110.0 us over old 100.0 us is 1.100, past 1.03"], [
                "harness budget alu: new 123.0 us over old 100.0 us is 1.230, past 1.03: owned by core at 1.200, "
                "within 1.236: listed, fails nothing",
                "harness budget sampling_deep: new 150.0 us over old 100.0 us is 1.500, past 1.03: a known gap "
                "(a ruling): listed, fails nothing"]))
            # More than 3% worse than its owned ratio, alu fails its leg again.
            problems, _ = self.budget("10de:1e87", dict(times, **{"harness-new/alu": 124.0}))
            self.assertIn("harness budget alu: new 124.0 us over old 100.0 us is 1.240, past 1.03: owned by core "
                          "at 1.200, and 1.240 is more than 3% worse (1.236)", problems)
            # Re-timed, it is judged by its pairs' median: one pass at 1.300 fails nothing where the
            # pairs bring it back within.
            again = dict(times, **{"harness-new/alu": 130.0})
            for pair, new in enumerate((121.0, 123.0), 1):
                again[f"{results.HARNESS_RETIME}alu/{pair}/old"] = 100.0
                again[f"{results.HARNESS_RETIME}alu/{pair}/new"] = new
            problems, parked = self.budget("10de:1e87", again)
            self.assertNotIn("alu", " ".join(problems))
            self.assertIn("harness budget alu: new 130.0 us over old 100.0 us is 1.300, past 1.03; re-timed in 2 "
                          "pairs, 1.210, 1.230, median 1.220: owned by core at 1.200, within 1.236: listed, fails "
                          "nothing", parked)
            # The pairs agreeing with the one pass, it fails.
            for pair in (1, 2):
                again[f"{results.HARNESS_RETIME}alu/{pair}/new"] = 130.0
            self.assertIn("alu", " ".join(self.budget("10de:1e87", again)[0]))
        # Without its entry, alu fails at 1.230 as any row.
        with mock.patch.object(results, "HARNESS_CEILINGS", {}):
            self.assertEqual(self.budget("10de:1e87", {"harness-old/alu": 100.0, "harness-new/alu": 123.0}),
                             (["harness budget alu: new 123.0 us over old 100.0 us is 1.230, past 1.03"], []))

    def test_an_owned_first_take_is_judged_by_its_processes_median(self):
        row = "create_64MiB/first_take"
        times = {f"harness-old/{row}": 1000.0, f"harness-new/{row}": 1200.0}
        for n in range(2, results.FIRST_RUNS + 1):
            times[f"{results.HARNESS_FIRST}{row}/{n}/old"] = 1000.0
            times[f"{results.HARNESS_FIRST}{row}/{n}/new"] = 1200.0
        for owned, fails in ((1.17, False), (1.16, True)):
            entries = [("10de:1e87", "create_64MiB/*", "open", "core", "abc1234", owned, "-")]
            with mock.patch.object(results, "STATUS", entries), mock.patch.object(results, "OWN_PART_GATES", {}), \
                    mock.patch.object(results, "HARNESS_CEILINGS", {}):
                problems, parked = self.budget("10de:1e87", times)
            self.assertEqual((len(problems), len(parked)), (1, 0) if fails else (0, 1), owned)
            self.assertIn(f"owned by core at {owned:.3f}", (problems + parked)[0])


class OverTableTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("HARNESS_FLOORS", {}), ("HARNESS_GPU_FLOORS", {}), ("CARD_VARIANCE", {}),
                            ("HARNESS_CEILINGS", {"a:0001": {"listed": 2.0}}), ("STORED_INSTRUCTIONS", {}),
                            ("STATUS", [])):
            patch = mock.patch.object(results, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_stored_status_table_reads_and_a_malformed_entry_is_refused(self):
        stored = results.read_status((Path(results.__file__).parent / "status.tsv").read_text())
        self.assertTrue(stored)
        self.assertTrue(all(status in results.STATUSES for _, _, status, _, _, _, _ in stored))
        # Every open record names the ratio it was owned at; nothing else has one.
        self.assertTrue(all((status == "open") == (owned is not None) for _, _, status, _, _, owned, _ in stored))
        head = "card | row | status | who | since | owned | note\n"
        self.assertEqual(results.read_status(head + "10de:1e87 | frame | open | core | 140714ceae | 1.120 | -\n"
                                             "10de:1e87 | alu | gap | here | - | - | ruled\n"),
                         [("10de:1e87", "frame", "open", "core", "140714ceae", 1.12, "-"),
                          ("10de:1e87", "alu", "gap", "here", "-", None, "ruled")])
        for bad in ("10de:1e87 | frame | open | core | - | 1.120 | -",  # an open record without its first round
                    "10de:1e87 | frame | open | core | 140714ceae | - | -",  # nor the ratio it was owned at
                    "10de:1e87 | frame | open | core | 140714ceae | 1.1x | -",  # a ratio that is no number
                    "10de:1e87 | frame | gap | here | - | 1.120 | -",  # a gap is owned by no one
                    "10de:1e87 | frame | gap | - | - | - | -",  # a gap without its place
                    "10de:1e87 | frame | closed | core | - | - | -",  # no such status
                    "nvidia | frame | gap | here | - | - | -",  # not a card
                    "10de:1e87 | frame | gap | here | - | -"):  # a field short
            with self.assertRaises(ValueError, msg=bad):
                results.read_status(head + bad + "\n")

    def test_every_row_past_its_limit_is_listed_with_its_kind_and_the_ratio_it_is_judged_by(self):
        row = "create_state/first"
        times = {"harness-old/alu": 100.0, "harness-new/alu": 110.0,
                 "harness-old/frame": 100.0, "harness-new/frame": 102.0,
                 "harness-old/listed": 100.0, "harness-new/listed": 150.0,
                 "harness-old/sampling": 30.0, "harness-new/sampling": 30.0,
                 "harness-gpu-old/sampling": 20.0, "harness-gpu-new/sampling": 30.0,
                 "harness-old/bandwidth": 100.0, "harness-new/bandwidth": 120.0,
                 f"harness-old/{row}": 1000.0, f"harness-new/{row}": 3000.0,
                 "harness-old/create_state/first_call": 1000.0, "harness-new/create_state/first_call": 9000.0}
        for n in range(2, results.FIRST_RUNS + 1):
            for runtime, value in (("old", 1000.0), ("new", 1200.0)):
                times[f"{results.HARNESS_FIRST}{row}/{n}/{runtime}"] = value
                times[f"{results.HARNESS_FIRST}create_state/first_call/{n}/{runtime}"] = 9 * value
        # bandwidth's re-timed pairs bring it within; alu's keep it past, judged by their median.
        for pair, (old, new) in enumerate(((100.0, 101.0), (100.0, 102.0)), 1):
            times[f"{results.HARNESS_RETIME}bandwidth/{pair}/old"] = old
            times[f"{results.HARNESS_RETIME}bandwidth/{pair}/new"] = new
            times[f"{results.HARNESS_RETIME}alu/{pair}/old"] = old
            times[f"{results.HARNESS_RETIME}alu/{pair}/new"] = new + 7.0
        counted = {"harness-new/frame/per_run": 3000, "harness-old/frame/per_run": 2000,
                   "harness-new/frame/own_per_run": 2500}
        with mock.patch.object(results, "stored_count", lambda leg_id, r: 2000 if r == "frame" else None):
            over = results.over_rows("a:0001", recs(time=times, instructions=counted))
        # frame's time is within; its own count is past both of the instructions gate's limits.
        # listed is within its ceiling; the creation call alone is shown, never judged.
        self.assertEqual([(r, kind, round(ratio, 3), round(limit, 3)) for r, kind, ratio, limit in over], [
            ("alu", "host time", 1.085, 1.03), (row, "host time", 1.2, 1.03),
            ("frame", "count", 1.25, 1.03), ("frame", "count", 1.25, 1.0),
            ("sampling", "GPU time", 1.5, 1.03)])

    def test_a_timing_leg_s_line_names_its_base_spread_and_a_leg_void_twice_is_unstable(self):
        leg = {"verdict": "unstable", "suites": {"harness": {"run": 0, "passed": 0, "exit": 0}},
               "records": {}, "problems": [], "parked": [], "node": "n", "jobs": ["j1", "j2"],
               "timed": {"spread": [0.15, "alu"], "states": ["VOID at 14.0% on alu, timed again", "venue unstable"]}}
        line = results.verdict_line("a:0001", leg)
        self.assertTrue(line.startswith("UNSTABLE a:0001"), line)
        self.assertIn("; base spread 15.0% on alu, VOID at 14.0% on alu, timed again, venue unstable", line)
        leg.update(verdict="pass", timed={"spread": None, "states": []})
        line = results.verdict_line("a:0001", leg)
        self.assertTrue(line.startswith("PASS a:0001"), line)
        self.assertTrue(line.endswith("; base spread n/a"), line)
        # A leg that times nothing names no spread.
        del leg["timed"]
        self.assertNotIn("base spread", results.verdict_line("a:0001", leg))

    def test_the_verdict_line_names_the_rule_that_judged_each_row(self):
        cpu = next(iter(results.CLOCK_FOLLOWS_LOAD))
        times = {f"harness-{runtime}/{row}": 100.0 for runtime in ("old", "new")
                 for row in ("frame", "frame_5cb_from_idle", "alu", "burst_160")}
        times["harness-new/authored/first_use_wait_total"] = 900.0
        leg = {"verdict": "pass", "suites": {"harness": {"run": 0, "passed": 0, "exit": 0}},
               "records": {"time": times}, "problems": [], "parked": [], "node": "n", "jobs": ["j1"], "cpu": cpu}
        line = results.verdict_line("1002:15bf", leg)
        self.assertIn(f"; rows judged by instructions per frame, time as a cross-GPU outlier only ({cpu}): "
                      "frame, frame_5cb_from_idle; by time: the other 2", line)
        # Elsewhere, and on a note that names no CPU, every row by its time.
        for other in ("Apple M3 Max", None):
            leg["cpu"] = other
            self.assertIn("; rows judged by time: all 4", results.verdict_line("1002:15bf", leg))
        # A leg that compared no row names no rule.
        leg["records"] = {}
        self.assertNotIn("rows judged", results.verdict_line("1002:15bf", leg))

    def test_a_leg_void_by_the_old_runtime_s_spread_reads_void_on_every_time(self):
        times = {"harness-old/alu": 100.0, "harness-new/alu": 110.0,
                 "harness-retime/alu/1/old": 100.0, "harness-retime/alu/1/new": 110.0,
                 "harness-retime/alu/2/old": 115.0, "harness-retime/alu/2/new": 126.5}
        lines = results.over_lines("a:0001", recs(time=times), None, None, lambda since: None)
        self.assertEqual(len(lines), 1)
        self.assertIn(" VOID ", lines[0])

    def test_a_gated_first_take_is_listed_by_its_own_part_alone(self):
        row, driver = "create_64MiB/first_take", "create_64MiB/first_allocation_total"
        gates = {"a:0001": {row: (driver, "a gap")}}

        def over(part):
            times = {f"harness-old/{row}": 1000.0, f"harness-new/{row}": 3000.0, f"harness-new/{driver}": part}
            for n in range(2, results.FIRST_RUNS + 1):
                times[f"{results.HARNESS_FIRST}{row}/{n}/old"] = 1000.0
                times[f"{results.HARNESS_FIRST}{row}/{n}/new"] = 3000.0
                times[f"{results.HARNESS_FIRST}{driver}/{n}/new"] = part
            with mock.patch.object(results, "OWN_PART_GATES", gates):
                return [(r, kind, round(ratio, 3)) for r, kind, ratio, _ in
                        results.over_rows("a:0001", recs(time=times))]

        self.assertEqual(over(2500.0), [])
        self.assertEqual(over(1500.0), [(row, "own part", 1.5)])

    def test_a_listed_row_reads_its_status_and_an_unlisted_one_new_then_unowned(self):
        entries = [("a:0001", "frame", "open", "core", "abc1234", 1.48, "-"),
                   ("a:0001", "sampling_*", "gap", "a ruling", "-", None, "its ceiling holds"),
                   ("a:0001", "alu", "note", "-", "-", None, "looked at once")]
        times = {"harness-old/frame": 100.0, "harness-new/frame": 150.0,
                 "harness-old/sampling_deep": 100.0, "harness-new/sampling_deep": 150.0,
                 "harness-old/alu": 100.0, "harness-new/alu": 150.0,
                 "harness-old/waves": 100.0, "harness-new/waves": 150.0}
        base = recs(time=dict(times, **{"harness-new/waves": 100.0, "harness-old/frame": 80.0}))
        rounds = {"abc1234": 3}
        with mock.patch.object(results, "STATUS", entries):
            lines = results.over_lines("a:0001@node", recs(time=times), base, None, rounds.get)
            self.assertEqual([line.split(None, 1)[1] for line in lines], [
                "alu                                  host time  1.500 over 1.030  stands   UNOWNED; looked at once",
                "frame                                host time  1.500 over 1.030  VOID     open, core's, 3 rounds, "
                "owned at 1.480, within 1.524",
                "sampling_deep                        host time  1.500 over 1.030  stands   known gap: a ruling; "
                "its ceiling holds",
                "waves                                host time  1.500 over 1.030  stands   NEW"])
            self.assertTrue(lines[0].startswith("a:0001@node "))
            # Without a base, every unlisted row is new and the arm has nothing to say; on another
            # harness, only the creation rows are compared.
            lines = results.over_lines("a:0001", recs(time=times), None, None, rounds.get)
            self.assertEqual([line.split()[-1] for line in lines], ["once", "1.524", "holds", "NEW"])
            # Past its owned ratio by more than OWNED_WORSE, the open record's row says it fails.
            entries[0] = entries[0][:5] + (1.4, "-")
            self.assertTrue(results.over_lines("a:0001", recs(time=times), None, None, rounds.get)[1].endswith(
                "open, core's, 3 rounds, owned at 1.400, WORSE than 1.442: fails its leg"))
            self.assertEqual({line.split("  ")[-2].strip() for line in
                              results.over_lines("a:0001", recs(time=times), base, results.REBASED_TREES,
                                                 rounds.get)}, {"rebased"})
        with mock.patch.object(results, "STATUS", []), \
                mock.patch.object(results, "CARD_VARIANCE", {"a:0001": {"waves": "it varies"}}):
            self.assertEqual(results.row_status("a:0001", "waves", True, rounds.get),
                             "known gap: the card's own variance (CARD_VARIANCE)")

    def test_an_entry_whose_rows_are_all_within_their_limits_is_named_for_removal(self):
        entries = [("a:0001", "frame", "open", "core", "abc1234", 1.48, "-"),
                   ("a:0001", "sampling_*", "gap", "a ruling", "-", None, "-"),
                   ("a:0001", "alu", "open", "core", "abc1234", 1.2, "-"),
                   ("a:0001", "waves", "note", "-", "-", None, "looked at once"),
                   ("a:0001", "untimed", "open", "core", "abc1234", 1.2, "-"),
                   ("b:0002", "frame", "gap", "a ruling", "-", None, "-")]
        within = {"harness-old/frame": 100.0, "harness-new/frame": 101.0,
                  "harness-old/sampling_deep": 100.0, "harness-new/sampling_deep": 102.0,
                  "harness-old/sampling_wide": 100.0, "harness-new/sampling_wide": 150.0,
                  "harness-old/alu": 100.0, "harness-new/alu": 102.0,
                  "harness-old/waves": 100.0, "harness-new/waves": 100.0,
                  "harness-new/untimed": 100.0}
        over = dict(within, **{"harness-new/alu": 110.0})
        with mock.patch.object(results, "STATUS", entries):
            # frame is within; one of sampling_*'s rows is still over; alu is within on one leg of
            # its card and over on the other; a note is never named; untimed and b:0002's frame
            # were timed on no leg of theirs.
            self.assertEqual(results.entries_within({"a:0001": (recs(time=within), None),
                                                     "a:0001@other": (recs(time=over), None)}),
                             [("a:0001", "frame", "open", "core")])
            # sampling_wide back within, the gap is named too.
            back = dict(within, **{"harness-new/sampling_wide": 101.0})
            self.assertEqual(results.entries_within({"a:0001": (recs(time=back), None)}),
                             [("a:0001", "frame", "open", "core"), ("a:0001", "sampling_*", "gap", "a ruling"),
                              ("a:0001", "alu", "open", "core")])


if __name__ == "__main__":
    unittest.main()
