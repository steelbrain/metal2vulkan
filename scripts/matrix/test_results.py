"""The matrix's reading of job output and its rules, on inputs whose answers are known.

    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/matrix

The job output below is nextest's own layout (0.9.140 and 0.9.151 print it alike): status lines,
the summary, then each test's captured output, indented, after a header naming the test.
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
