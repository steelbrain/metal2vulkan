"""The matrix's reading of its jobs' output and its comparison rules (docs/VALIDATION.md, "The
matrix"). Pure functions over text and dictionaries: `test_results.py` checks each one.

A job prints, per leg, `M2V-LEG <leg> begin`, `M2V-DEVICE <leg> <field> <value>` lines, then each
suite's output followed by `M2V-SUITE <leg> <suite> exit <code>`. A test records a value as a line
`M2V-MATRIX <kind> <test>/<name> <value>` on its standard output: a count, or for `time`
microseconds, decimals allowed.
"""

import re
import statistics
from pathlib import Path

#: `size` is the size a harness row whose size follows the device ran at (harness_budget).
KINDS = ("metric", "instructions", "gauge", "time", "size", "locks")
#: A gauge's key ends in the point it was read at.
GAUGE_POINTS = ("100", "200", "drop")
#: The new runtime's memory pool's gauges, by their last key component: its `VkDeviceMemory`
#: allocations. The pool frees an idle allocation once the most bytes in use over its window falls
#: past it and the device has been quiet for 20 ms, so allocations an earlier workload left idle
#: may go between a later one's iterations 100 and 200: the harness's long_4cb, whose runs leave
#: 20 ms gaps between submissions, fell from 11 to 8 on the RTX 5080 under a 10 s window, each free
#: at least 20 ms after the last commit (memory.rs's WINDOW_SECS, QUIET_NANOS). Such a gauge fails
#: only if it rises between @100 and @200; at @drop it is 0 like any other.
POOL_GAUGES = ("blocks",)
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
#: recovery at 8b15f6574b) is parked too: harness_budget names its row as not compared. The
#: harness's warm first-use records (harness-new-warm/, the matrix's harness_runs) are parked as
#: well: the cold ones, timed on a fresh driver cache, are the gated ones, and a warm one reads the
#: entries the cold run just wrote, a fraction of a millisecond on Mesa. So are the re-timed pairs
#: of a row over its budget (harness-retime/, RETIME_PAIRS), and the creation rows' first samples
#: from the later processes (harness-first/, FIRST_RUNS): harness_budget judges them. So are the
#: rows' GPU times (harness-gpu-<runtime>/, HARNESS_GPU) and their re-timed pairs'
#: (harness-retime-gpu/): harness_budget judges a row on its GPU time where gpu_judged says so.
PARKED_RECORDS = ("harness-old/", "harness-new-warm/", "harness-retime/", "harness-first/",
                  "harness-gpu-", "harness-retime-gpu/")

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


#: The lock budget's records (`crates/metal2vulkan-rt/src/tests/locks.rs`): the locked
#: instructions per million added commands of each kind, measured on a node whose matrix setup
#: names its CPU's locked-instruction event (`M2V_LOCK_EVENT`). The test holds the bound; the
#: matrix holds that the measurement ran where the node names the event, and carries the values.
LOCK_BUDGET = ("lock_budget/draw", "lock_budget/bound_draw", "lock_budget/dispatch")
#: A node setup's export of the event.
LOCK_EVENT = re.compile(r"\bM2V_LOCK_EVENT=\S")


def names_lock_event(setup):
    """Whether a node's matrix setup (its shell, `matrix.json`'s `setup`) names the
    locked-instruction event."""
    return bool(LOCK_EVENT.search(setup or ""))


def lock_budget_missing(records, named):
    """Each lock budget record a leg's RECORDS lack, where its node names the event (NAMED): the
    budget then measures, and a leg that printed no measurement fails. Elsewhere none: a node that
    names no event measures nothing."""
    if not named:
        return []
    return [f"lock budget {key}: not measured, though the node's setup names M2V_LOCK_EVENT"
            for key in LOCK_BUDGET if key not in records["locks"]]


def card_of(leg_id):
    """The card LEG_ID runs on, by its PCI vendor:device: the leg's name, or an extra leg's
    (extra_leg_id) before its `@`. A card's ceilings, floors and variance rows hold on each of its
    legs."""
    return leg_id.split("@", 1)[0]


def extra_leg_id(card, node):
    """The name of CARD's leg on NODE where the lab holds the card on more than one node and the
    matrix config's `record_nodes` names another as the card's leg of record: `card@node`."""
    return f"{card}@{node}"


def leg_parked(leg, records):
    """The leg's parked failures (PARKED_TESTS and PARKED_RECORDS' gauges), one line each."""
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
                and value != gauges[f"{name}@100"]
                and (name.rpartition("/")[2] not in POOL_GAUGES or value > gauges[f"{name}@100"])):
            problems.append(f"gauge {name}: {gauges[f'{name}@100']} at iteration 100, {value} at 200")
    if base is not None:
        for key, value in sorted(records["metric"].items()):
            was = base["metric"].get(key)
            if was is not None and value > was:
                problems.append(f"metric {key}: {was} -> {value}")
        for key, value in sorted(records["instructions"].items()):
            was = base["instructions"].get(key)
            if (was is not None and not key.startswith(PARKED_RECORDS + ("harness-",))
                    and value > was * (1 + INSTRUCTIONS_BAND)):
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
#: old, both timed in the same leg, is at most this. A row with a GPU time on both runtimes is
#: judged on it (gpu_judged), its host time shown beside it: on the GPUs measured, the old
#: runtime's own spread of a single kernel's GPU time is a few percent where its host time spreads
#: up to 1.3 (gpu-floors.tsv against floors.tsv). The others, and the frame and burst rows
#: (WALL_ROWS), are judged on their time as a program waits on it, timed on the host (the
#: harness's `run::matrix_time`): a run's mean commit-to-complete time where the row times its
#: command buffers', else the host wall time a run adds. It is a rule of its own, apart from
#: TIME_OUTLIER and PARKED_RECORDS: those judge a time against its base commit's across GPUs and
#: park the old runtime's outliers, while this compares the two runtimes within one leg, the old
#: runtime's time the reference. It ends with the cutover, with the old runtime.
HARNESS_BUDGET = 1.03
HARNESS_NEW = "harness-new/"
HARNESS_OLD = "harness-old/"
#: A harness row's GPU time on each runtime, `harness-gpu-<runtime>/<row>`: the p50 over its timed
#: runs of a run's first GPU start to its last GPU end, scaled like its time where the device
#: sizes the row, and absent where the row has none. harness_budget judges a row on it where
#: gpu_judged says so, and shows it beside the row's time otherwise; it is parked for TIME_OUTLIER
#: (PARKED_RECORDS).
HARNESS_GPU = "harness-gpu-"
#: The rows judged on their host time even where they have a GPU time, by the start of their
#: names: a frame's or a burst's command buffers, whose GPU time spans the gaps between them and
#: spreads on the old runtime as widely as the host time on several GPUs (gpu-floors.tsv's
#: measurement: 1.68 on the Iris Xe's frame_5cb, 1.43 on the UHD 630's burst_160_shared). The
#: planner's ruling: these keep the host time's floor and the instructions gate.
WALL_ROWS = ("frame", "burst")
#: WALL_ROWS as the re-time selection's awk pattern.
WALL_RE = "^(" + "|".join(WALL_ROWS) + ")"
#: Rows whose GPU time varies on one card itself, run to run or with its driver's register
#: assignment of an unchanged loop, by GPU leg, with what was seen: harness_budget judges them on
#: their GPU time against their GPU floor like any other, never fails a leg for them, and names
#: them as the card's variance, so no one chases them as the new runtime's.
CARD_VARIANCE = {
    "1002:744c": {
        "sampling": "its GPU p50 on the old runtime ran 16.3-20.3 us over 8 runs (m3-max-45546)",
        "constant_managed": "its GPU p50 on the old runtime is 66.8-69.2 or 74.4 us run to run "
                            "(m3-max-45546)",
    },
    # Informational on the Iris Xe until the harness kernels are authored not to depend on it:
    # with the loop's instructions unchanged, ANV's register assignment moves these rows' GPU time
    # in equal steps, one per two-source instruction whose two vector sources start at GRF numbers
    # equal modulo 16 (m3-max-47683: 0 to 40 dead multiply-adds ahead of the loop). No lane claims
    # a gain or a loss on these rows here without the loop's dump.
    "8086:a7a0": {
        "zero_products": "its GPU p50 ran 4377-5308 us over the same loop code, in steps of 232 us "
                         "(m3-max-47683)",
        "vote_loop": "its GPU p50 ran 824-930 us over the same loop code (m3-max-47683)",
    },
}


def gpu_judged(times, row):
    """Whether harness_budget judges ROW on its GPU time: it has one on both runtimes in TIMES, and
    it is not a WALL_ROWS row."""
    return not row.startswith(WALL_ROWS) and all(
        (times.get(f"{HARNESS_GPU}{runtime}/{row}") or 0) > 0 for runtime in ("old", "new"))


def gpu_beside(times, runtime, row):
    """ROW's GPU time on RUNTIME as its time's lines show it, ` (GPU p50 N us)`, or nothing."""
    gpu = times.get(f"{HARNESS_GPU}{runtime}/{row}")
    return "" if gpu is None else f" (GPU p50 {gpu:.1f} us)"


#: The instructions gate of the frame and burst rows (WALL_ROWS), beside the judgement of their
#: time, on Linux's user-space instructions per timed run, each the p50 over the row's timed runs.
#: The harness records the whole process's (`instructions harness-<runtime>/<row>/per_run`), and
#: where the runtime names the threads it started (the new runtime's `Gauges::threads`), the count
#: of those and the harness's own thread (`.../own_per_run`) and the rest, the threads a driver
#: started for itself (`.../others_per_run`). A driver's threads swing by millions of instructions
#: per run between processes (NVIDIA's on burst_160: 0 to 25 million), the runtime's own do not.
#: The gate is on the runtime's own: EXACT, the new runtime's own count at most HARNESS_BUDGET
#: times its stored count (STORED_INSTRUCTIONS), which ratchets down; and LENIENT, printed on
#: every row, the new runtime's own count at most the old's whole count, the old runtime naming
#: none of its threads, so its driver's share cannot be taken out. The base comparison of
#: leg_problems skips the harness's counts: this gate judges them. A leg that records no
#: instructions (the kernel refused the counter, or Windows) has no gate. Every row records its
#: counts; only these are gated on them.
INSTRUCTIONS_PER_RUN = "/per_run"
#: The new runtime's own threads' count of a row, and the rest of the process's.
OWN_PER_RUN, OTHERS_PER_RUN = "/own_per_run", "/others_per_run"


def read_counts(text):
    """The stored instruction counts' TEXT: `#` comments, a header of `row` and driver families
    (a GPU leg id's vendor, `10de`, or `all` for every leg), then per line a row and its count in
    each, `-` for none. Returns {family: {row: count}}."""
    counts, families = {}, None
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split()
        if families is None:
            families = cells[1:]
            continue
        for family, cell in zip(families, cells[1:]):
            if cell != "-":
                counts.setdefault(family, {})[cells[0]] = int(cell)
    return counts


#: The new runtime's own threads' instructions per timed run (OWN_PER_RUN) that each WALL_ROWS
#: row is held to (`instructions.tsv`, beside this file): one count per row where the legs' counts
#: lie within 3% of each other, else one per driver family. A count only ratchets down: a landing
#: that lowers a row's count by more than 1% stores the lower one, and nothing raises it without a
#: ruling.
STORED_INSTRUCTIONS = read_counts((Path(__file__).resolve().parent / "instructions.tsv").read_text())


def stored_count(leg_id, row):
    """ROW's stored count on LEG_ID: its driver family's, else every leg's, else None."""
    family = leg_id.split(":")[0]
    for column in (family, "all"):
        if row in STORED_INSTRUCTIONS.get(column, {}):
            return STORED_INSTRUCTIONS[column][row]
    return None


def instruction_gate(leg_id, row, records):
    """ROW's instructions gate on LEG_ID from the leg's RECORDS: (problems, parked), both empty
    where ROW is not a WALL_ROWS row or a runtime recorded no whole count. EXACT: the new runtime's
    own count past HARNESS_BUDGET times the stored count is a problem; one more than
    INSTRUCTIONS_BAND under it, or a row with none stored, is named. LENIENT, named on every row:
    the new runtime's own count over the old's whole count is a problem. A new runtime that
    recorded its whole count but not its own is a problem: the gate could not run."""
    counted = records["instructions"]
    new, old = (counted.get(f"harness-{runtime}/{row}{INSTRUCTIONS_PER_RUN}") for runtime in ("new", "old"))
    if not row.startswith(WALL_ROWS) or not new or not old:
        return [], []
    own, others = (counted.get(f"harness-new/{row}{part}") for part in (OWN_PER_RUN, OTHERS_PER_RUN))
    if own is None:
        return [f"harness budget {row}: instructions new {new} per run, but no count of its own threads"], []
    problems, parked = [], []
    stored = stored_count(leg_id, row)
    exact = f"harness budget {row}: EXACT instructions new own {own} per run, " + (
        f"stored {stored}, is {own / stored:.3f}" if stored else "none stored")
    if stored and own > HARNESS_BUDGET * stored:
        problems.append(f"{exact}: past {HARNESS_BUDGET} times the stored count")
    elif stored and own < (1 - INSTRUCTIONS_BAND) * stored:
        parked.append(f"{exact}: more than {INSTRUCTIONS_BAND:.0%} under the stored count, which ratchets "
                      "down to it")
    elif not stored:
        parked.append(exact)
    lenient = (f"harness budget {row}: LENIENT instructions new own {own} per run over old whole {old} is "
               f"{own / old:.3f}; new others {others}, old's driver threads not separable")
    (problems if own > old else parked).append(lenient + (": new own over old whole" if own > old else ""))
    return problems, parked


#: The harness rows over HARNESS_BUDGET that a ruling gave a ceiling, per GPU leg, each a ratio of
#: the times its row is judged on (gpu_judged): a row past its ceiling fails its leg, so an open
#: row can neither hide nor grow. A
#: ceiling is ruled after the work on its row, never taken from a measured ratio; a fix lowers it
#: to the ratio the row then reaches, and a row within HARNESS_BUDGET leaves the list. A row over
#: HARNESS_BUDGET with no ceiling fails its leg.
HARNESS_CEILINGS = {
    # Ruled by the planner on the rows' measured medians, 1.20: a kernel over a grid one row high
    # and one deep pays for running in tiles and for the RX 550's 64-bit address per read (the rt
    # README's known gaps of pipeline states), from a Private table and from a Managed one alike.
    "1002:699f": {"constant_managed": 1.20, "constant_private": 1.20},
    # Ruled by the planner, 1.05, on the rows' medians of 1.040 and 1.032 (e632e01478): a second
    # Shared buffer of 64 MiB on the 780M is rt's memset of the memory the first gave back
    # against old's import of fresh pages, the same zeroing by another hand.
    "1002:15bf": {"create_64MiB/next_take": 1.05},
    "1002:1900": {"create_64MiB/next_take": 1.05},
    # Ruled by the planner on the rows' measured medians, 1.876 and 2.252: old imports fresh
    # anonymous pages for a Shared buffer of 64 MiB, a form chosen by a run-time check; rt
    # allocates and zeroes (on the 2080, a 10.0 ms vkAllocateMemory and a 2.5 ms memset against
    # old's 6.7 ms import; on the 3090, 15.4 to 17.4 ms and 2.6 ms against 6.4 to 8.2 ms).
    "10de:1e87": {"create_64MiB/first_take": 1.88},
    "10de:2204": {"create_64MiB/first_take": 2.26},
    # Ruled by the planner on the row's measured median, 1.877: old carves the buffer from a
    # Private block it already holds, and rt takes a new block. rt's own costs inside the row, its
    # first GPU zero fill's recording and the first GPU use of the new block, are open, and their
    # fixes lower the ceiling.
    "8086:a7a0": {"create_private_16MiB/first_take": 1.88},
    # 8086:3e9b's zero_products has none: the row runs level with old, the new runtime about 1%
    # ahead (m3-max-47846). A grid-edge test that decodes and compares the position ahead of the
    # region's ragged branch once ran 5-6% faster on this card, by the driver compiler's accident
    # on that kernel. Its looped executable had the most instructions of the forms timed (149, and
    # 507 static cycles), yet ran fastest:
    # - the form that branches on the flag first: 153 and 520;
    # - no edge test at all: 117 and 375;
    # - old's: 100 and 336.
    # The driver returns no representation of its code to say why. No form of the test kept both
    # that speed and the sampling rows' gains (m3-max-47851).
}


def read_floors(text):
    """The floors table's TEXT: `#` comments, a header of `row` and GPU leg ids, then per line a
    harness row and its floor on each leg, `-` for none. Returns {leg: {row: floor}}."""
    floors, legs = {}, None
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split()
        if legs is None:
            legs = cells[1:]
            continue
        for leg, cell in zip(legs, cells[1:]):
            if cell != "-":
                floors.setdefault(leg, {})[cells[0]] = float(cell)
    return floors


#: The old runtime's own spread per harness row on each GPU leg, the row's floor: the widest ratio
#: between two of the old runtime's runs of the row in one leg, old over old (`floors.tsv`, beside
#: this file), of its host time; HARNESS_GPU_FLOORS is the same of its GPU time (`gpu-floors.tsv`,
#: from the same runs), the floor of a row judged on it (gpu_judged). A row is past its limit only
#: past HARNESS_BUDGET times the greater of its floor and its ceiling (row_limit): a difference
#: within the old runtime's own spread is not the new runtime's. The tables are measured again
#: when the harness or the old runtime's build changes; a row or leg one does not list has a floor
#: of 1 there.
HARNESS_FLOORS = read_floors((Path(__file__).resolve().parent / "floors.tsv").read_text())
HARNESS_GPU_FLOORS = read_floors((Path(__file__).resolve().parent / "gpu-floors.tsv").read_text())


def row_floor(leg_id, row, gpu=False):
    """ROW's floor on LEG_ID: of its GPU time where GPU, else of its host time; 1 where none is
    listed."""
    return (HARNESS_GPU_FLOORS if gpu else HARNESS_FLOORS).get(card_of(leg_id), {}).get(row, 1.0)


def row_limit(leg_id, row, gpu=False):
    """ROW's limit on LEG_ID, new over old: HARNESS_BUDGET times the greater of its ceiling
    (HARNESS_CEILINGS) and its floor (row_floor, of its GPU time where GPU), each 1 where none is
    listed."""
    return HARNESS_BUDGET * max(HARNESS_CEILINGS.get(card_of(leg_id), {}).get(row, 1.0),
                                row_floor(leg_id, row, gpu))


def past_limit(leg_id, row, gpu=False):
    """How a line names ROW's limit on LEG_ID once past it: the budget, or its ceiling or floor
    (its GPU floor where GPU), whichever sets it, with the allowance."""
    ceiling = HARNESS_CEILINGS.get(card_of(leg_id), {}).get(row, 1.0)
    floor = row_floor(leg_id, row, gpu)
    if max(ceiling, floor) <= 1.0:
        return f"past {HARNESS_BUDGET}"
    name, value = ("ceiling", ceiling) if ceiling >= floor else ("GPU floor" if gpu else "floor", floor)
    return f"past its {name} {value} with its allowance, {row_limit(leg_id, row, gpu):.3f}"


#: The harness's creation rows, by the start of their names. A leg times them apart from the
#: other rows (the matrix's harness_runs): each runtime's run of the others skips them, and they
#: run alone after it, in FIRST_RUNS processes of their own, each on a fresh driver cache and
#: store, so no row before them moves their times.
HARNESS_CREATES = "create_"
#: A creation row's first samples (`<row>/first`, `/first_take`, `/next_take`,
#: `/first_allocation_total`, `/device_first`) are one sample per process, so harness_budget judges
#: each by the median of FIRST_RUNS processes per runtime: the first process's own record and the
#: later processes' `harness-first/<row>/<n>/<runtime>`. A first sample alone is never past its
#: limit.
FIRST_RUNS = 5
HARNESS_FIRST = "harness-first/"
#: A compute creation row's `/first` sample is its creation and one dispatch through to completion,
#: as Metal compiles at the creation and a runtime may compile at the dispatch instead (the old
#: runtime does). A creation row's first sample is a fresh object's on a device that has made one of
#: its kind: the harness makes a warm-up of the row's kind first. The first samples SHOWN_FIRST
#: names are shown with the ratio of their medians, and never judged, each for its reason: the
#: creation call alone (`/first_call`), and the warm-up (`/device_first`), which holds what the
#: device pays once, which a guest that keeps its device pays once in its life.
SHOWN_FIRST = {
    "/first_call": "the creation call alone, which old returns from before it makes its pipeline",
    "/device_first": "the warm-up, the first object of its kind on the device",
}
#: The sed program the first creation process's output goes through: its device's gauges at drop
#: become `harness-<runtime>/creates/device/...`, apart from the other rows' process's, so each
#: key is recorded once.
CREATES_APART = (r"s#^M2V-MATRIX gauge harness-\([a-z]*\)/device/#"
                 r"M2V-MATRIX gauge harness-\1/creates/device/#")
#: The sed program (`sed -n`) each later creation process's output goes through: its first samples
#: alone, as `harness-first/<row>/<n>/<runtime>`, the shell giving `$n` and `$runtime`.
FIRST_KEEP = (r"s#^M2V-MATRIX time harness-$runtime/\(create_[A-Za-z0-9_]*/[a-z_]*\) #"
              r"M2V-MATRIX time harness-first/\1/$n/$runtime #p")


def first_row(row):
    """Whether ROW is a creation row's first sample, judged by the median of FIRST_RUNS."""
    return row.startswith(HARNESS_CREATES) and "/" in row


def first_samples(times, row):
    """ROW's first samples per runtime, (old, new), each FIRST_RUNS long, or None unless every
    process recorded both."""
    samples = []
    for runtime in ("old", "new"):
        values = [times.get(f"harness-{runtime}/{row}")]
        values += [times.get(f"{HARNESS_FIRST}{row}/{n}/{runtime}") for n in range(2, FIRST_RUNS + 1)]
        if any(value is None for value in values):
            return None
        samples.append(values)
    return tuple(samples)


#: A row over its limit in a leg (row_limit) is re-timed once in that leg before the leg ends
#: (the matrix's harness_runs): RETIME_PAIRS pairs of runs of the row alone, old then new, then new
#: then old (A B B A), its times kept as `harness-retime/<row>/<pair>/<runtime>` and its GPU times
#: as `harness-retime-gpu/<row>/<pair>/<runtime>` (RETIME_KEEP). harness_budget judges such a row
#: by the median of its pairs' ratios, new over old, of the times it judges the row on, so a clock
#: episode in one run decides nothing while a regression in every pair still fails. At most
#: RETIME_ROWS rows of a leg, the furthest over first; a row that was not re-timed is judged by its
#: one time.
RETIME_PAIRS = 2
RETIME_ROWS = 8
HARNESS_RETIME = "harness-retime/"
HARNESS_RETIME_GPU = "harness-retime-gpu/"
#: The sed program (`sed -n`) each re-timed run's output goes through: the row's time and GPU time
#: alone, the shell giving `$timed`, `$row` and `$pair`.
RETIME_KEEP = (r"s#^M2V-MATRIX time harness-$timed/$row #M2V-MATRIX time harness-retime/$row/$pair/$timed #p;"
               r"s#^M2V-MATRIX time harness-gpu-$timed/$row #M2V-MATRIX time harness-retime-gpu/$row/$pair/$timed #p")
#: The awk program the leg runs over its runtimes' output to choose the rows it re-times: each row
#: timed on both whose ratio, new over old, is over its limit, as `ratio row`: of its GPU times
#: where it has them on both and does not match WALLRE (gpu_judged), against GPULIMITS', else of
#: its times, against LIMITS' (`row=limit ...`, harness_limits), or BUDGET where a row has no
#: limit listed: harness_budget's comparison, of the same decimals.
RETIME_SELECT = r"""function read(text, into,   n, i, l, p) { n = split(text, l, " "); for (i = 1; i <= n; i++) { split(l[i], p, "="); into[p[1]] = p[2] } }
BEGIN { read(limits, limit); read(gpulimits, glimit) }
$1 == "M2V-MATRIX" && $2 == "time" && $3 ~ /^harness-(old|new)\/[^\/]+$/ {
  split($3, k, "/"); t[k[1] "/" k[2]] = $4; rows[k[2]] = 1
}
$1 == "M2V-MATRIX" && $2 == "time" && $3 ~ /^harness-gpu-(old|new)\/[^\/]+$/ {
  split($3, k, "/"); g[k[1] "/" k[2]] = $4
}
END {
  for (row in rows) {
    old = t["harness-old/" row]; new = t["harness-new/" row]
    if (old == "" || new == "" || old + 0 <= 0) continue
    gold = g["harness-gpu-old/" row] + 0; gnew = g["harness-gpu-new/" row] + 0
    if (row !~ wallre && gold > 0 && gnew > 0) { r = gnew / gold; m = (row in glimit) ? glimit[row] + 0 : budget + 0 }
    else { r = new / old; m = (row in limit) ? limit[row] + 0 : budget + 0 }
    if (r > m) printf "%.6f %s\n", r, row
  }
}"""


def harness_limits(leg_id, gpu=False):
    """The rows of LEG_ID with a ceiling or a floor (of their GPU time where GPU) as
    RETIME_SELECT's LIMITS (GPULIMITS where GPU), `row=limit ...`: each its row_limit, written so
    that it reads back as the same double."""
    floors = HARNESS_GPU_FLOORS if gpu else HARNESS_FLOORS
    rows = set(HARNESS_CEILINGS.get(card_of(leg_id), {})) | set(floors.get(card_of(leg_id), {}))
    return " ".join(f"{row}={row_limit(leg_id, row, gpu)!r}" for row in sorted(rows)
                    if row_limit(leg_id, row, gpu) != HARNESS_BUDGET)


def retimed(times, row, gpu=False):
    """ROW's re-timed pairs' ratios, new over old, of its GPU times where GPU, else of its times, in
    pair order; None unless every pair has both."""
    kept = HARNESS_RETIME_GPU if gpu else HARNESS_RETIME
    ratios = []
    for pair in range(1, RETIME_PAIRS + 1):
        old, new = (times.get(f"{kept}{row}/{pair}/{runtime}") for runtime in ("old", "new"))
        if not old or new is None:
            return None
        ratios.append(new / old)
    return ratios


#: The frozen arm is the noise meter. The old runtime takes regression fixes only, so its time of a
#: harness row moves from one round to the next with the node, the driver and the harness, not
#: with the change under test. Each round compares, per card and row, the old runtime's time that
#: harness_budget judges the row on (frozen_old) with the same time in the previous round: the
#: leg's base, the nearest first-parent ancestor whose note has the leg. Where it moved more than
#: FROZEN_DRIFT either way, the row's comparison is VOID for the round on that card: harness_budget
#: names it and judges nothing from it, and it is re-timed in one process with both runtimes
#: interleaved. harness_rows shows the move on every row the base has.
FROZEN_DRIFT = 0.10


def frozen_old(times, row, gpu):
    """The old runtime's time of ROW in TIMES that harness_budget judges it on: the median of the
    FIRST_RUNS processes' samples for a creation row's first sample (first_row), else its GPU time
    where GPU, else its time; None where TIMES lacks any of it."""
    if first_row(row):
        values = [times.get(f"{HARNESS_OLD}{row}")]
        values += [times.get(f"{HARNESS_FIRST}{row}/{n}/old") for n in range(2, FIRST_RUNS + 1)]
        return None if None in values else statistics.median(values)
    return times.get(f"{HARNESS_GPU}old/{row}" if gpu else f"{HARNESS_OLD}{row}")


def frozen_move(times, base, row):
    """The old runtime's move on ROW from the base's times, BASE, to TIMES, of the time it is
    judged on in TIMES (gpu_judged): (base's, now, now over base's less one), or None where
    either has none."""
    gpu = gpu_judged(times, row)
    was, now = frozen_old(base or {}, row, gpu), frozen_old(times, row, gpu)
    if not was or not now:
        return None
    return was, now, now / was - 1


def frozen_void(times, base):
    """The rows of TIMES whose comparison the frozen arm voids against BASE (FROZEN_DRIFT), each
    `(row, base's, now, move)`, in row order."""
    rows = sorted({key[len(HARNESS_OLD):] for key in times if key.startswith(HARNESS_OLD)})
    moves = ((row, frozen_move(times, base, row)) for row in rows)
    return [(row, *move) for row, move in moves if move and abs(move[2]) > FROZEN_DRIFT]


def harness_budget(leg_id, leg, base=None):
    """The harness budget on LEG, the parsed leg LEG_ID: each row timed on both runtimes
    (`harness-new/<row>` and `harness-old/<row>`) against its limit, row_limit: HARNESS_BUDGET times
    the greater of its ceiling (HARNESS_CEILINGS) and its floor. A row with a GPU time on both
    runtimes that is not a WALL_ROWS row (gpu_judged) is judged on its GPU times against its GPU
    floor (HARNESS_GPU_FLOORS), its times shown beside them; any other on its times against its
    floor (HARNESS_FLOORS). A CARD_VARIANCE row is never a problem: past its limit, or over the
    budget within it, it is named with what its card was seen to do. Returns
    (problems, parked): a row past its limit is a problem, and so is a row whose time was recorded
    more than once (twice by a suite, or by two suites), which has no one time to judge. A row past
    its limit that the leg re-timed (RETIME_PAIRS) is judged by the median of its pairs' ratios
    instead: a problem past the limit, else named with its pairs. A creation row's first sample
    (first_row) is judged by the ratio of the medians of its FIRST_RUNS processes, and is never a
    problem on its one sample: without every process's, it is named as not judged. A listed row
    passes up to its ceiling times HARNESS_BUDGET, the noise allowance parity gets, and is named
    with its ceiling; one within the budget is named as leaving the list. A row whose run failed on the old runtime
    (`M2V-FAILED harness-old/<row>`) is named as not compared and fails nothing here (PARKED_RECORDS);
    one that failed on the new runtime fails its suite (leg_problems). A row whose size follows the
    device (`size` records) is compared only where both runtimes ran it at one size: at two, the
    row is a problem, its times measuring different work. BASE is the times of the leg's base: a
    judged row whose old time moved past FROZEN_DRIFT from the base's is VOID, named and not
    judged (frozen_move)."""
    problems, parked = [], []
    suites = leg["suites"].values()
    for row in sorted({test[len(HARNESS_OLD):] for suite in suites for test in suite["failed"]
                       if test.startswith(HARNESS_OLD)}):
        parked.append(f"harness budget {row}: old failed, not compared")
    records = leg_records(leg)[0]
    times, sizes = records["time"], records["size"]
    keys = [key for suite in suites for key in suite["records"]["time"]]
    twice = ({key for key in keys if keys.count(key) > 1}
             | {key for suite in suites for kind, key in suite.get("duplicated", ()) if kind == "time"})
    ceilings = HARNESS_CEILINGS.get(card_of(leg_id), {})
    for key, new in sorted(times.items()):
        row = key[len(HARNESS_NEW):]
        old = times.get(HARNESS_OLD + row)
        if not key.startswith(HARNESS_NEW) or not old:
            continue
        if {key, HARNESS_OLD + row} & twice:
            problems.append(f"harness budget {row}: its time is recorded more than once, so it has no one "
                            "time to judge")
            continue
        new_size, old_size = sizes.get(key), sizes.get(HARNESS_OLD + row)
        if new_size != old_size:
            problems.append(f"harness budget {row}: new ran at size {new_size}, old at {old_size}, so its "
                            "times measure different work")
            continue
        limit = row_limit(leg_id, row)
        gated, named = instruction_gate(leg_id, row, records)
        problems += gated
        parked += named
        shown = next((why for end, why in SHOWN_FIRST.items() if row.endswith(end)), None)
        if first_row(row) and shown:
            samples = first_samples(times, row)
            if samples is not None:
                old, new = (statistics.median(values) for values in samples)
                parked.append(f"harness budget {row}: median of {FIRST_RUNS} processes, new {new:.1f} us "
                              f"over old {old:.1f} us is {new / old:.3f}: shown, not judged: {shown}")
            continue
        move = frozen_move(times, base, row)
        if move and abs(move[2]) > FROZEN_DRIFT:
            parked.append(f"harness budget {row}: VOID: old moved {move[2]:+.1%} from the base's, "
                          f"{move[0]:.1f} us to {move[1]:.1f} us, so the round judges nothing from it: "
                          "re-time it in one process, both runtimes interleaved")
            continue
        if first_row(row):
            samples = first_samples(times, row)
            if samples is None:
                if new / old > limit:
                    parked.append(f"harness budget {row}: new {new:.1f} us over old {old:.1f} us is "
                                  f"{new / old:.3f}, one sample: a first sample is judged by the median "
                                  f"of {FIRST_RUNS} processes, not judged")
                continue
            old, new = (statistics.median(values) for values in samples)
            line = (f"harness budget {row}: median of {FIRST_RUNS} processes, new {new:.1f} us over old "
                    f"{old:.1f} us is {new / old:.3f}")
            if new / old > limit:
                problems.append(f"{line}, {past_limit(leg_id, row)}")
            elif row in ceilings:
                parked.append(f"{line}, its ceiling {ceilings[row]}")
            continue
        gpu = gpu_judged(times, row)
        if gpu:
            limit = row_limit(leg_id, row, gpu)
            gnew, gold = (times[f"{HARNESS_GPU}{runtime}/{row}"] for runtime in ("new", "old"))
            ratio = gnew / gold
            line = (f"harness budget {row}: GPU p50 new {gnew:.1f} us over old {gold:.1f} us is "
                    f"{ratio:.3f} (time new {new:.1f} us over old {old:.1f} us is {new / old:.3f})")
        else:
            ratio = new / old
            line = (f"harness budget {row}: new {new:.1f} us{gpu_beside(times, 'new', row)} over old "
                    f"{old:.1f} us{gpu_beside(times, 'old', row)} is {ratio:.3f}")
        variance = CARD_VARIANCE.get(card_of(leg_id), {}).get(row)
        if ratio > limit:
            past = f"{line}, {past_limit(leg_id, row, gpu)}"
            ratios = retimed(times, row, gpu)
            if ratios is not None:
                median = statistics.median(ratios)
                past = (f"{past}; re-timed in {RETIME_PAIRS} pairs, "
                        f"{', '.join(f'{r:.3f}' for r in ratios)}, median {median:.3f}")
                if median <= limit:
                    parked.append(f"{past}, within: judged by the median")
                    continue
            if variance:
                parked.append(f"{past}: the card's own variance, not judged ({variance})")
            else:
                problems.append(past)
        elif variance and ratio > HARNESS_BUDGET:
            parked.append(f"{line}, within its limit: the card's own variance ({variance})")
        elif row not in ceilings:
            continue
        elif ratio > HARNESS_BUDGET:
            parked.append(f"{line}, its ceiling {ceilings[row]}")
        else:
            parked.append(f"{line}, within {HARNESS_BUDGET}: its ceiling {ceilings[row]} leaves the list")
    return problems, parked


def harness_rows(records, base=None):
    """Every harness row timed on either runtime in a leg's RECORDS, one line each: its time, its
    GPU time (HARNESS_GPU) and, where recorded, its instructions per run (with its own threads'
    where the runtime names them), on the new runtime and on the old, `-` where it has none, and
    new over old of the times and of the instructions (and of new's own over old's whole). Where
    BASE, the times of the leg's base, has the row, the frozen arm's move: the old runtime's time
    the row is judged on against the base's (frozen_move), VOID past FROZEN_DRIFT."""
    times, counted = records["time"], records["instructions"]
    rows = sorted({key.split("/", 1)[1] for key in times if key.startswith((HARNESS_NEW, HARNESS_OLD))})
    lines = []
    for row in rows:
        cells = []
        for runtime in ("new", "old"):
            wall, gpu = times.get(f"harness-{runtime}/{row}"), times.get(f"{HARNESS_GPU}{runtime}/{row}")
            cell = (f"{runtime} {'-' if wall is None else f'{wall:.1f}':>10} us, "
                    f"GPU p50 {'-' if gpu is None else f'{gpu:.1f}':>10} us")
            instructions = counted.get(f"harness-{runtime}/{row}{INSTRUCTIONS_PER_RUN}")
            if instructions is not None:
                cell += f", {instructions} instructions"
            own = counted.get(f"harness-{runtime}/{row}{OWN_PER_RUN}")
            if own is not None:
                cell += f" ({own} own)"
            cells.append(cell)
        new, old = times.get(HARNESS_NEW + row), times.get(HARNESS_OLD + row)
        ratio = f"{new / old:.3f}" if new is not None and old else "-"
        line = f"{row:<36} {cells[0]} | {cells[1]} | {ratio}"
        new_i, old_i = (counted.get(f"harness-{runtime}/{row}{INSTRUCTIONS_PER_RUN}") for runtime in ("new", "old"))
        if new_i is not None and old_i:
            line += f", instructions {new_i / old_i:.3f}"
            own = counted.get(f"harness-new/{row}{OWN_PER_RUN}")
            if own is not None:
                line += f", own over old {own / old_i:.3f}"
        move = frozen_move(times, base, row)
        if move:
            line += f", old {move[2]:+.1%} on the base" + (" VOID" if abs(move[2]) > FROZEN_DRIFT else "")
        lines.append(line)
    return lines


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
