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
    # Each of these failed in one lavapipe run and passed in the runs around it, the old
    # runtime's code unchanged between them.
    "metal2vulkan-runtime tests::view_check_device::the_check_stands_by_while_the_device_s_work_goes_on":
        "failed once on lavapipe (cfa389aa9b) between passing runs of unchanged code",
    "metal2vulkan-runtime tests::queue_share_device::"
    "a_later_queue_s_command_buffer_takes_its_turn_ahead_of_another_queue_s_burst":
        "failed once on lavapipe (c944d63548) between passing runs of unchanged code",
    "metal2vulkan-runtime tests::host_ram_device::"
    "a_chunk_an_object_waited_for_the_thread_s_import_of_is_not_given_back":
        "failed once on lavapipe (73e0af4b66) between passing runs of unchanged code",
    "metal2vulkan-validation runtime_lane::commit_work::"
    "fills_across_rounds_reuse_a_bounded_set_of_command_pools":
        "failed once on lavapipe (cfa389aa9b) between passing runs of unchanged code",
    "metal2vulkan-validation runtime_lane::first_use::new_resources_wait_for_no_gpu_work":
        "failed on lavapipe at c944d63548 and bfe63b5097 between passing runs of unchanged code",
    "metal2vulkan-validation runtime_lane::host_memory::"
    "memory_metal_makes_no_buffer_over_is_refused_when_it_is_made":
        "failed once on lavapipe (7fa6662170) between passing runs of unchanged code",
}
#: Record keys, by prefix, whose gauge @100-to-@200 flatness and cross-GPU time outliers are parked
#: for the same reason (a gauge's @drop still fails): the old runtime frees idle memory blocks at
#: times of its own, and its frame timing on the UHD 630 swings by half between runs of unchanged
#: code (1.58x at a68d520bee, 1.47x at 9af29fbb20). A harness workload's failure on the old
#: runtime (`M2V-FAILED harness-old/<row>`, one burst_2000x5 run on the UHD 630 ended in a GPU
#: recovery at 8b15f6574b) is parked too: harness_budget names its row as not compared.
PARKED_RECORDS = ("harness-old/",)

#: The phrase that opens the subject of a commit allowed to change the old runtime's output: until
#: the cutover, the translator's output through `Device::new` moves only in such a commit. Only the
#: subject's start counts, so a message that merely quotes the phrase exempts nothing.
OLD_PATH_FIX = "old-runtime regression fix"
#: The golden hash table's test (crates/metal2vulkan/src/tests/golden_hashes.rs) compiles through
#: `Device::new`, the old runtime's path. Its failure names the cells that changed under an
#: unchanged key and those under a changed profile fingerprint: either is the old path moving.
GOLDEN_CHANGED = re.compile(r"output changed under an unchanged key: (\d+) cell")
GOLDEN_PROFILE = re.compile(r"(\d+) under a changed profile fingerprint")

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
# The harness command's line for a workload that failed on one runtime.
HARNESS_FAILED = re.compile(r"^\s*M2V-FAILED (\S+)\s*$")


def empty_records():
    return {kind: {} for kind in KINDS}


def parse_suite(lines):
    """One suite's output: tests run and passed, the tests that failed, and its records, with
    the problems of the records themselves (an unknown kind, a value that is not a count, a key
    recorded twice, a gauge key without its point). A key recorded twice keeps its first value and
    is listed, by kind, in `duplicated`."""
    run = passed = 0
    failed, problems = [], []
    records = empty_records()
    duplicated = set()
    golden = dict(changed=0, profile=0)
    summary = False
    for line in lines:
        for kind, pattern in (("changed", GOLDEN_CHANGED), ("profile", GOLDEN_PROFILE)):
            m = pattern.search(line)
            if m:
                golden[kind] = max(golden[kind], int(m.group(1)))
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
        m = NEXTEST_FAILED.match(line) or LIBTEST_FAILED.match(line) or HARNESS_FAILED.match(line)
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
                duplicated.add((kind, key))
            else:
                records[kind][key] = float(value) if kind == "time" else int(value)
    return dict(run=run, passed=passed, failed=failed, records=records, problems=problems,
                golden=golden, duplicated=duplicated)


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


#: Record keys, by prefix, that the harness measures: its counts sum over its own workloads, so,
#: like its times, they compare only with a base the same harness measured.
HARNESS_RECORDS = "harness-"


def base_records(records, same_harness):
    """The base's RECORDS a leg is judged against: all of them under the same harness, else those
    the harness did not measure."""
    if same_harness:
        return records
    return {kind: {key: value for key, value in values.items() if not key.startswith(HARNESS_RECORDS)}
            for kind, values in records.items()}


def leg_parked(leg, records):
    """The leg's parked failures (PARKED_TESTS, PARKED_RECORDS' gauges), one line each."""
    out = [f"test {test}" for suite in leg["suites"].values() for test in suite["failed"]
           if test in PARKED_TESTS]
    gauges = records["gauge"]
    for key, value in sorted(gauges.items()):
        name, _, point = key.rpartition("@")
        if (point == "200" and name.startswith(PARKED_RECORDS) and f"{name}@100" in gauges
                and value != gauges[f"{name}@100"]):
            out.append(f"gauge {name}: {gauges[f'{name}@100']} at iteration 100, {value} at 200")
    return out


def leg_problems(leg, records, base):
    """What fails a leg: a suite that failed, a record's own problem, and the rules against the
    base leg's records (BASE None: no base, so only the rules that need none). A suite whose every
    failed test is parked (PARKED_TESTS, or a harness workload of PARKED_RECORDS) has not failed."""
    problems = []
    for name, suite in leg["suites"].items():
        failed = [test for test in suite["failed"]
                  if test not in PARKED_TESTS and not test.startswith(PARKED_RECORDS)]
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
        if (point == "200" and not name.startswith(PARKED_RECORDS) and f"{name}@100" in gauges
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


def old_path_moved(leg, messages):
    """The old path's output moved: the golden table's test reports a cell changed under an
    unchanged key or under a changed profile fingerprint, and none of MESSAGES (the commits since
    the table's last bless) has a subject that starts with OLD_PATH_FIX. One problem per suite that
    reports it."""
    if any(message.lstrip().lower().startswith(OLD_PATH_FIX) for message in messages):
        return []
    return [f"OLD PATH MOVED: suite {name}: the golden table, compiled through Device::new, has "
            f"{s['golden']['changed']} cell(s) changed under an unchanged key and "
            f"{s['golden']['profile']} under a changed profile fingerprint, and no commit since "
            f"its last bless has a subject starting \"{OLD_PATH_FIX}\""
            for name, s in leg["suites"].items()
            if s.get("golden") and (s["golden"]["changed"] or s["golden"]["profile"])]


def time_outliers(times, bases):
    """Timing fails only as a cross-GPU outlier. TIMES and BASES: {leg: {key: microseconds}} of
    the GPU legs now and at their bases. For each key that at least TIME_MIN_GPUS legs timed at
    both, a leg whose slowdown (now over base) is more than TIME_OUTLIER times the legs' median
    slowdown is an outlier. Returns ({leg: [problems]}, {leg: [parked]}): an outlier of a
    PARKED_RECORDS key is parked, not a problem."""
    out, parked = {}, {}
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
                (parked if key.startswith(PARKED_RECORDS) else out).setdefault(leg, []).append(
                    f"time {key}: {ratio:.2f}x its base, against a median of {median:.2f}x "
                    f"over {len(ratios)} GPUs")
    return out, parked


#: The harness budget: on each GPU, a harness row's time on the new runtime over its time on the
#: old, both timed in the same leg, is at most this. It is a rule of its own, apart from
#: TIME_OUTLIER and PARKED_RECORDS: those judge a time against its base commit's across GPUs and
#: park the old runtime's outliers, while this compares the two runtimes within one leg, the old
#: runtime's time the reference. It ends with the cutover, with the old runtime.
HARNESS_BUDGET = 1.03
HARNESS_NEW = "harness-new/"
HARNESS_OLD = "harness-old/"
#: The harness rows over HARNESS_BUDGET, per GPU leg, each with its ratio as its ceiling: a row
#: past its ceiling fails its leg, so an open row can neither hide nor grow. A fix lowers the
#: ceiling to the ratio the row then reaches, and a row within HARNESS_BUDGET leaves the list.
HARNESS_CEILINGS = {
    "1002:699f": {"burst_160": 1.089, "burst_160_shared": 1.074, "burst_2000": 1.226, "threadgroup_scan": 1.05,
                  "waves": 2.461, "wide_bindings": 1.23},
    "1002:744c": {"simd_reduce": 1.033, "threadgroup_scan": 1.039},
    "10de:1e87": {"burst_160": 1.811, "burst_160_shared": 2.024, "burst_2000": 2.031, "burst_2000x5": 1.724,
                  "frame_5cb": 1.636, "image_filter": 1.527, "wide_bindings": 1.162, "zero_products": 1.031},
    "10de:2204": {"burst_160": 1.259, "burst_160_shared": 1.408, "burst_2000": 1.366, "burst_2000x5": 1.315,
                  "frame_5cb": 1.137, "image_filter": 1.14, "wide_bindings": 1.171},
    "8086:3e9b": {"burst_160": 1.433, "burst_160_shared": 1.258, "burst_2000": 1.108, "burst_2000x5": 1.054,
                  "threadgroup_scan": 1.093, "waves": 1.354, "wide_bindings": 1.208},
}


def harness_budget(leg_id, leg):
    """The harness budget on LEG, the parsed leg LEG_ID: each row timed on both runtimes
    (`harness-new/<row>` and `harness-old/<row>`) against its ceiling in HARNESS_CEILINGS, or
    against HARNESS_BUDGET where none is listed. Returns (problems, parked): a row past its ceiling
    or the budget is a problem, and so is a row whose time was recorded more than once (twice by a
    suite, or by two suites), which has no one time to judge. A listed row passes up to its ceiling
    times HARNESS_BUDGET, the noise allowance parity gets, and is named with its ceiling; one within
    the budget is named as leaving the list. A row whose run failed on the old runtime
    (`M2V-FAILED harness-old/<row>`) is named as not compared and fails nothing here (PARKED_RECORDS);
    one that failed on the new runtime fails its suite (leg_problems)."""
    problems, parked = [], []
    suites = leg["suites"].values()
    for row in sorted({test[len(HARNESS_OLD):] for suite in suites for test in suite["failed"]
                       if test.startswith(HARNESS_OLD)}):
        parked.append(f"harness budget {row}: old failed, not compared")
    times = leg_records(leg)[0]["time"]
    keys = [key for suite in suites for key in suite["records"]["time"]]
    twice = ({key for key in keys if keys.count(key) > 1}
             | {key for suite in suites for kind, key in suite.get("duplicated", ()) if kind == "time"})
    ceilings = HARNESS_CEILINGS.get(leg_id, {})
    for key, new in sorted(times.items()):
        row = key[len(HARNESS_NEW):]
        old = times.get(HARNESS_OLD + row)
        if not key.startswith(HARNESS_NEW) or not old:
            continue
        if {key, HARNESS_OLD + row} & twice:
            problems.append(f"harness budget {row}: its time is recorded more than once, so it has no one "
                            "time to judge")
            continue
        ratio = new / old
        line = f"harness budget {row}: new {new:.1f} us over old {old:.1f} us is {ratio:.3f}"
        if row not in ceilings:
            if ratio > HARNESS_BUDGET:
                problems.append(f"{line}, past {HARNESS_BUDGET}")
        elif ratio > ceilings[row] * HARNESS_BUDGET:
            problems.append(f"{line}, past its ceiling {ceilings[row]} with its allowance, "
                            f"{ceilings[row] * HARNESS_BUDGET:.3f}")
        elif ratio > HARNESS_BUDGET:
            parked.append(f"{line}, its ceiling {ceilings[row]}")
        else:
            parked.append(f"{line}, within {HARNESS_BUDGET}: its ceiling {ceilings[row]} leaves the list")
    return problems, parked


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
