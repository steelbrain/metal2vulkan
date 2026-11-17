"""The matrix's reading of its jobs' output and its comparison rules (docs/VALIDATION.md, "The
matrix"). Pure functions over text and dictionaries: `test_results.py` checks each one.

A job prints, per leg, `M2V-LEG <leg> begin`, `M2V-DEVICE <leg> <field> <value>` lines, then each
suite's output followed by `M2V-SUITE <leg> <suite> exit <code>`. A test records a value as a line
`M2V-MATRIX <kind> <test>/<name> <value>` on its standard output: a count, or for `time`
microseconds, decimals allowed.
"""

import re
import statistics

KINDS = ("metric", "instructions", "gauge", "time")
#: A gauge's key ends in the point it was read at.
GAUGE_POINTS = ("100", "200", "drop")
#: `instructions` may grow by this fraction over the base before it fails.
INSTRUCTIONS_BAND = 0.01
#: A GPU's slowdown over the base fails past this multiple of the median slowdown...
TIME_OUTLIER = 1.25
#: ...of at least this many GPUs.
TIME_MIN_GPUS = 3

#: The old runtime's failures the matrix reports on a leg's line but fails no leg for: the old
#: runtime takes no fixes until the cutover deletes it, and this list goes with it. A test, by its
#: nextest name, with the reason it fails.
PARKED_TESTS = {
    "metal2vulkan-validation runtime_lane::cross_queue_shadows::"
    "a_cpu_write_while_a_command_buffer_runs_reaches_none_of_its_encoders":
        "a GPU-hazard test whose spin can end before the host's write on a loaded host",
    "metal2vulkan-validation runtime_lane::compute_inherited_buffers::"
    "a_host_step_orders_the_commands_by_their_barriers_alone":
        "the old runtime's barrier count moves with its background pipeline builds",
}
#: Gauge keys, by prefix, whose @100-to-@200 flatness is parked for the same reason (their @drop
#: still fails): the old runtime frees idle memory blocks at times of its own.
PARKED_GAUGES = ("harness-old/",)

RECORD = re.compile(r"^\s*M2V-MATRIX (\S+) (\S+) (\S+)\s*$")
LEG = re.compile(r"^M2V-LEG (\S+) begin\s*$")
DEVICE = re.compile(r"^M2V-DEVICE (\S+) (\S+) (.*?)\s*$")
SUITE = re.compile(r"^M2V-SUITE (\S+) (\S+) exit (-?\d+)\s*$")
# nextest: `Summary [   0.003s] 2 tests run: 1 passed, 1 failed, 0 skipped`.
NEXTEST_SUMMARY = re.compile(r"^\s*Summary \[[^\]]*\]\s+(\d+) tests? run: (\d+) passed")
# nextest's status line of a test that did not pass, as the run prints it.
NEXTEST_FAILED = re.compile(
    r"^\s*(FAIL|SIGSEGV|SIGABRT|SIGBUS|SIGILL|SIGKILL|SIGTERM|ABORT|TIMEOUT|LEAK-FAIL)"
    r" \[[^\]]*\]\s+(?:\(\s*\d+/\d+\)\s+)?(\S+ \S+)"
)
# libtest (doctests): `test result: ok. 12 passed; 0 failed; 1 ignored; ...`.
LIBTEST_RESULT = re.compile(r"^test result: \w+\. (\d+) passed; (\d+) failed")
LIBTEST_FAILED = re.compile(r"^test (.+) \.\.\. FAILED\s*$")


def empty_records():
    return {kind: {} for kind in KINDS}


def parse_suite(lines):
    """One suite's output: tests run and passed, the tests that failed, and its records, with
    the problems of the records themselves (an unknown kind, a value that is not a count, a key
    recorded twice, a gauge key without its point)."""
    run = passed = 0
    failed, problems = [], []
    records = empty_records()
    summary = False
    for line in lines:
        m = NEXTEST_SUMMARY.match(line)
        if m and not summary:
            summary = True
            run, passed = int(m.group(1)), int(m.group(2))
            continue
        m = LIBTEST_RESULT.match(line)
        if m:
            run += int(m.group(1)) + int(m.group(2))
            passed += int(m.group(1))
            continue
        m = NEXTEST_FAILED.match(line) or LIBTEST_FAILED.match(line)
        if m:
            name = m.group(m.lastindex)
            if name not in failed:
                failed.append(name)
            continue
        m = RECORD.match(line)
        if m:
            kind, key, value = m.groups()
            if kind not in KINDS:
                problems.append(f"record {key}: unknown kind {kind}")
            elif not (re.fullmatch(r"\d+(\.\d+)?", value) if kind == "time" else value.isdigit()):
                problems.append(f"record {kind} {key}: {value} is not a "
                                + ("duration" if kind == "time" else "count"))
            elif kind == "gauge" and key.rpartition("@")[2] not in GAUGE_POINTS:
                problems.append(f"gauge {key}: the key ends in none of @100, @200, @drop")
            elif key in records[kind]:
                problems.append(f"record {kind} {key}: recorded twice")
            else:
                records[kind][key] = float(value) if kind == "time" else int(value)
    return dict(run=run, passed=passed, failed=failed, records=records, problems=problems)


def parse_job(text):
    """A job's legs: {leg: {"device": {field: value}, "suites": {suite: parsed suite + "exit"}}}.
    Output between a leg's markers belongs to the suite whose `M2V-SUITE` line ends it."""
    legs, leg, buffer = {}, None, []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        m = LEG.match(line)
        if m:
            leg = legs.setdefault(m.group(1), dict(device={}, suites={}))
            buffer = []
            continue
        m = DEVICE.match(line)
        if m and m.group(1) in legs:
            legs[m.group(1)]["device"].setdefault(m.group(2), m.group(3))
            continue
        m = SUITE.match(line)
        if m and m.group(1) in legs:
            suite = parse_suite(buffer)
            suite["exit"] = int(m.group(3))
            legs[m.group(1)]["suites"][m.group(2)] = suite
            buffer = []
            continue
        if leg is not None:
            buffer.append(line)
    return legs


def leg_records(leg):
    """A leg's records over its suites, and the problems of a key two suites both recorded."""
    records, problems = empty_records(), []
    for suite in leg["suites"].values():
        for kind, values in suite["records"].items():
            for key, value in values.items():
                if key in records[kind]:
                    problems.append(f"record {kind} {key}: recorded by two suites")
                records[kind][key] = value
    return records, problems


def leg_parked(leg, records):
    """The leg's parked failures (PARKED_TESTS, PARKED_GAUGES), one line each."""
    out = [f"test {test}" for suite in leg["suites"].values() for test in suite["failed"]
           if test in PARKED_TESTS]
    gauges = records["gauge"]
    for key, value in sorted(gauges.items()):
        name, _, point = key.rpartition("@")
        if (point == "200" and name.startswith(PARKED_GAUGES) and f"{name}@100" in gauges
                and value != gauges[f"{name}@100"]):
            out.append(f"gauge {name}: {gauges[f'{name}@100']} at iteration 100, {value} at 200")
    return out


def leg_problems(leg, records, base):
    """What fails a leg: a suite that failed, a record's own problem, and the rules against the
    base leg's records (BASE None: no base, so only the rules that need none). A suite whose every
    failed test is parked (PARKED_TESTS) has not failed."""
    problems = []
    for name, suite in leg["suites"].items():
        failed = [test for test in suite["failed"] if test not in PARKED_TESTS]
        if failed or (suite["exit"] != 0 and not suite["failed"]):
            shown = ", ".join(failed[:5]) + (" ..." if len(failed) > 5 else "")
            problems.append(f"suite {name} exited {suite['exit']}"
                            + (f", {len(failed)} failed: {shown}" if failed else ""))
        problems += suite["problems"]
    gauges = records["gauge"]
    for key, value in sorted(gauges.items()):
        name, _, point = key.rpartition("@")
        if point == "drop" and value != 0:
            problems.append(f"gauge {name}: {value} live at device drop")
        if (point == "200" and not name.startswith(PARKED_GAUGES) and f"{name}@100" in gauges
                and value != gauges[f"{name}@100"]):
            problems.append(f"gauge {name}: {gauges[f'{name}@100']} at iteration 100, {value} at 200")
    if base is not None:
        for key, value in sorted(records["metric"].items()):
            was = base["metric"].get(key)
            if was is not None and value > was:
                problems.append(f"metric {key}: {was} -> {value}")
        for key, value in sorted(records["instructions"].items()):
            was = base["instructions"].get(key)
            if was is not None and value > was * (1 + INSTRUCTIONS_BAND):
                problems.append(f"instructions {key}: {was} -> {value} (+{(value / was - 1) * 100:.1f}%)")
    return problems


def time_outliers(times, bases):
    """Timing fails only as a cross-GPU outlier. TIMES and BASES: {leg: {key: microseconds}} of
    the GPU legs now and at their bases. For each key that at least TIME_MIN_GPUS legs timed at
    both, a leg whose slowdown (now over base) is more than TIME_OUTLIER times the legs' median
    slowdown fails. Returns {leg: [problems]}."""
    out = {}
    keys = {key for values in times.values() for key in values}
    for key in sorted(keys):
        ratios = {leg: values[key] / bases[leg][key]
                  for leg, values in times.items()
                  if key in values and bases.get(leg, {}).get(key)}
        if len(ratios) < TIME_MIN_GPUS:
            continue
        median = statistics.median(ratios.values())
        for leg, ratio in sorted(ratios.items()):
            if ratio > TIME_OUTLIER * median:
                out.setdefault(leg, []).append(
                    f"time {key}: {ratio:.2f}x its base, against a median of {median:.2f}x "
                    f"over {len(ratios)} GPUs")
    return out


def verdict_line(leg_id, leg):
    """The one line `run` and `show` print for a leg of a note."""
    tests = ", ".join(f"{name} {s['passed']}/{s['run']}" if s["run"] else f"{name} exit {s['exit']}"
                      for name, s in leg.get("suites", {}).items())
    counted = sum(len(values) for values in leg.get("records", {}).values())
    where = " ".join(part for part in (leg.get("device"), leg.get("driver")) if part)
    head = f"{'PASS' if leg['verdict'] == 'pass' else 'FAIL'} {leg_id:10} {where or '?'} on {leg.get('node', '?')}"
    tail = (f"{tests or 'no suite ran'}; {counted} records; base {(leg.get('base') or 'none')[:10]}; "
            f"jobs {' '.join(leg.get('jobs', [])) or '-'}")
    if leg["problems"]:
        tail += " | " + " | ".join(leg["problems"][:4]) + (" ..." if len(leg["problems"]) > 4 else "")
    if leg.get("parked"):
        tail += f" | parked: {len(leg['parked'])} ({leg['parked'][0]}" + (" ..." if len(leg["parked"]) > 1 else "") + ")"
    return f"{head}: {tail}"
