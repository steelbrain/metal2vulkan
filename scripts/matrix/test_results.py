"""The matrix's reading of job output and its rules, on inputs whose answers are known.

    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/matrix

The job output below is nextest's own layout (0.9.140 and 0.9.151 print it alike): status lines,
the summary, then each test's captured output, indented, after a header naming the test.
"""

import subprocess
import unittest
from unittest import mock

import results

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

    def test_a_metric_fails_on_any_increase_and_passes_a_decrease(self):
        base = recs(metric={"a/barriers": 4, "a/passes": 2})
        self.assertEqual(self.problems(recs(metric={"a/barriers": 4, "a/passes": 1}), base), [])
        self.assertEqual(self.problems(recs(metric={"a/barriers": 5, "a/passes": 2}), base),
                         ["metric a/barriers: 4 -> 5"])

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
            # A clock episode in one pair's run decides nothing.
            self.assertEqual(self.budget("1002:1900", pairs(415.7, 662.8, 415.5)), ([], [
                f"{past}; re-timed in 3 pairs, 0.992, 1.582, 0.992, median 0.992, within: judged by the "
                "median"]))
            # A regression in every pair still fails.
            self.assertEqual(self.budget("1002:1900", pairs(470.0, 466.0, 474.0)), ([
                f"{past}; re-timed in 3 pairs, 1.122, 1.112, 1.131, median 1.122"], []))
            # A row with a pair missing was not re-timed: its one time judges it.
            missing = pairs(415.7, 662.8, 415.5)
            del missing["harness-retime/vote_loop/3/new"]
            self.assertEqual(self.budget("1002:1900", missing), ([past], []))
        # A listed row is re-timed past its ceiling's limit, and judged against that limit.
        with mock.patch.object(results, "HARNESS_CEILINGS", {"1002:1900": {"vote_loop": 1.05}}):
            self.assertEqual(self.budget("1002:1900", pairs(440.0, 441.0, 700.0))[0], [])
            self.assertEqual(len(self.budget("1002:1900", pairs(455.0, 456.0, 700.0))[0]), 1)

    def test_a_leg_re_times_the_rows_the_budget_would_fail_on_their_one_time(self):
        ceilings = {"a": {"listed": 2.0, "edge": 2.0}}
        times = {"alu": (100.0, 103.0), "frame": (100.0, 103.5), "listed": (100.0, 230.0),
                 "edge": (100.0, 206.0), "under": (100.0, 90.0)}
        lines = [f"M2V-MATRIX time harness-{runtime}/{row} {pair[i]:.3f}"
                 for row, pair in times.items() for i, runtime in enumerate(("old", "new"))]
        lines += ["M2V-MATRIX time harness-new/authored/first_use_wait_total 900.000",
                  "M2V-MATRIX gauge harness-new/alu/buffers@100 2", "a table row"]
        with mock.patch.object(results, "HARNESS_CEILINGS", ceilings):
            run = subprocess.run(["awk", "-v", f"limits={results.harness_limits('a')}", "-v",
                                  f"budget={results.HARNESS_BUDGET!r}", results.RETIME_SELECT],
                                 input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
            chosen = sorted(line.split()[1] for line in run.stdout.splitlines())
            records = {f"harness-{runtime}/{row}": pair[i]
                       for row, pair in times.items() for i, runtime in enumerate(("old", "new"))}
            failed = sorted(line.split()[2].rstrip(":") for line in self.budget("a", records)[0])
            # Each limit reads back as the double harness_budget compares with.
            self.assertEqual(results.harness_limits("a"), "edge=2.06 listed=2.06")
        self.assertEqual(chosen, ["frame", "listed"])
        self.assertEqual(chosen, failed)

    def test_re_timed_pairs_are_parked_records(self):
        self.assertTrue(f"{results.HARNESS_RETIME}vote_loop/1/new".startswith(results.PARKED_RECORDS))

    def test_every_ceiling_is_over_the_budget(self):
        for leg, rows in results.HARNESS_CEILINGS.items():
            for row, ceiling in rows.items():
                self.assertGreater(ceiling, results.HARNESS_BUDGET, f"{leg} {row}")


if __name__ == "__main__":
    unittest.main()
