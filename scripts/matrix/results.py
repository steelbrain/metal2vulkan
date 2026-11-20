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

#: `size` is the size a harness row whose size follows the device ran at, which nothing judges.
KINDS = ("metric", "instructions", "gauge", "time", "size", "locks")
#: A gauge's key ends in the point it was read at.
GAUGE_POINTS = ("100", "200", "drop")
#: The runtime's memory pool's gauges, by their last key component: its `VkDeviceMemory`
#: allocations. The pool frees an idle allocation once the most bytes in use over its window falls
#: past it and the device has been quiet for 20 ms, so allocations an earlier workload left idle
#: may go between a later one's iterations 100 and 200: the harness's long_4cb, whose runs leave
#: 20 ms gaps between submissions, fell from 11 to 8 on the RTX 5080 under a 10 s window, each free
#: at least 20 ms after the last commit (memory.rs's WINDOW_SECS, QUIET_NANOS). Such a gauge fails
#: only if it rises between @100 and @200; at @drop it is 0 like any other.
POOL_GAUGES = ("blocks",)
#: `instructions` may grow by this fraction over the base before it fails.
INSTRUCTIONS_BAND = 0.01
#: Record keys, by prefix, whose `metric` fails where it differs from the base's either way, and
#: where the base has it and the leg does not: the replay player's judged totals of a play on the
#: runtime (`replay-rt/STREAM/RANGE/NAME`, the validation crate's `judged_records`), each shown
#: equal between two plays of one stream at one commit on one card, which a play that did not
#: finish its range does not print.
EQUAL_METRICS = ("replay-",)
#: A GPU's slowdown over the base fails past this multiple of the median slowdown...
TIME_OUTLIER = 1.25
#: ...of at least this many GPUs.
TIME_MIN_GPUS = 3

#: Record keys, by prefix, whose gauge @100-to-@200 flatness and cross-GPU time outliers are parked
#: (a gauge's @drop still fails): the harness's warm first-use records (harness-new-warm/, the
#: matrix's harness_runs), as the cold ones, timed on a fresh driver cache, are the gated ones, and
#: a warm one reads the entries the cold run just wrote, a fraction of a millisecond on Mesa; and
#: the rows' GPU times (harness-gpu-<runtime>/, HARNESS_GPU), which harness_rows shows and nothing
#: judges.
PARKED_RECORDS = ("harness-new-warm/", "harness-gpu-")

#: The phrase that opens the subject of a commit allowed to change the golden table's output: the
#: translator's output through `Device::from_queries` moves only in such a commit. Only the
#: subject's start counts, so a message that merely quotes the phrase exempts nothing.
GOLDEN_FIX = "old-runtime regression fix"
#: The golden hash table's test (crates/metal2vulkan/src/tests/golden_hashes.rs) compiles through
#: `Device::from_queries`. Its failure names the cells that changed under an unchanged key and
#: those under a changed profile fingerprint: either is the table's output moving.
GOLDEN_CHANGED = re.compile(r"output changed under an unchanged key: (\d+) cell")
GOLDEN_PROFILE = re.compile(r"(\d+) under a changed profile fingerprint")

RECORD = re.compile(r"^\s*M2V-MATRIX (\S+) (\S+) (\S+)\s*$")
#: A test's record it did not count on this node (the lock budget's, where the node names no
#: event): `M2V-MATRIX uncounted <test>/<name>`. Its leg's line says `not counted`.
UNCOUNTED = re.compile(r"^\s*M2V-MATRIX uncounted (\S+)\s*$")
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
    uncounted = []
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
        m = UNCOUNTED.match(line)
        if m:
            if m.group(1) not in uncounted:
                uncounted.append(m.group(1))
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
                golden=golden, duplicated=duplicated, uncounted=uncounted)


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


def leg_uncounted(leg):
    """The records a leg's suites said they did not count (UNCOUNTED), sorted, each once."""
    return sorted({key for suite in leg["suites"].values() for key in suite.get("uncounted", [])})


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


#: The lock budget's records (`crates/metal2vulkan-runtime/src/tests/locks.rs`): the locked
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
    (extra_leg_id) before its `@`."""
    return leg_id.split("@", 1)[0]


def extra_leg_id(card, node):
    """The name of CARD's leg on NODE where the lab holds the card on more than one node and the
    matrix config's `record_nodes` names another as the card's leg of record: `card@node`."""
    return f"{card}@{node}"


def leg_parked(leg, records):
    """The leg's parked gauges (PARKED_RECORDS'), one line each."""
    out = []
    gauges = records["gauge"]
    for key, value in sorted(gauges.items()):
        name, _, point = key.rpartition("@")
        if (point == "200" and name.startswith(PARKED_RECORDS) and f"{name}@100" in gauges
                and value != gauges[f"{name}@100"]):
            out.append(f"gauge {name}: {gauges[f'{name}@100']} at iteration 100, {value} at 200")
    return out


def leg_problems(leg, records, base):
    """What fails a leg: a suite that failed, a record's own problem, and the rules against the
    base leg's records (BASE None: no base, so only the rules that need none)."""
    problems = []
    for name, suite in leg["suites"].items():
        failed = suite["failed"]
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
            if was is not None and (value != was if key.startswith(EQUAL_METRICS) else value > was):
                problems.append(f"metric {key}: {was} -> {value}")
        for key, was in sorted(base["metric"].items()):
            if key.startswith(EQUAL_METRICS) and key not in records["metric"]:
                problems.append(f"metric {key}: {was} -> none: the play did not finish its range")
        for key, value in sorted(records["instructions"].items()):
            was = base["instructions"].get(key)
            if (was is not None and not key.startswith(PARKED_RECORDS + ("harness-",))
                    and value > was * (1 + INSTRUCTIONS_BAND)):
                problems.append(f"instructions {key}: {was} -> {value} (+{(value / was - 1) * 100:.1f}%)")
    return problems


def golden_table_moved(leg, messages):
    """The golden table's output moved: its test reports a cell changed under an unchanged key or
    under a changed profile fingerprint, and none of MESSAGES (the commits since the table's last
    bless) has a subject that starts with GOLDEN_FIX. One problem per suite that reports it."""
    if any(message.lstrip().lower().startswith(GOLDEN_FIX) for message in messages):
        return []
    return [f"GOLDEN TABLE MOVED: suite {name}: the golden table, compiled through "
            f"Device::from_queries, has "
            f"{s['golden']['changed']} cell(s) changed under an unchanged key and "
            f"{s['golden']['profile']} under a changed profile fingerprint, and no commit since "
            f"its last bless has a subject starting \"{GOLDEN_FIX}\""
            for name, s in leg["suites"].items()
            if s.get("golden") and (s["golden"]["changed"] or s["golden"]["profile"])]


def outlier_keys(times, bases):
    """The cross-GPU outliers of TIMES against BASES, {leg: {key: microseconds}} of the GPU legs
    now and at their bases: for each key that at least TIME_MIN_GPUS legs timed at both, a leg
    whose slowdown (now over base) is more than TIME_OUTLIER times the legs' median slowdown.
    Returns {leg: {key: (slowdown, median, legs)}}, PARKED_RECORDS keys among them. A row's alone
    runs (HARNESS_ALONE) are samples of its level, judged with its own key, never on their own."""
    out = {}
    keys = {key for values in times.values() for key in values if not key.startswith(HARNESS_ALONE)}
    for key in sorted(keys):
        ratios = {leg: values[key] / bases[leg][key]
                  for leg, values in times.items()
                  if key in values and bases.get(leg, {}).get(key)}
        if len(ratios) < TIME_MIN_GPUS:
            continue
        median = statistics.median(ratios.values())
        for leg, ratio in sorted(ratios.items()):
            if ratio > TIME_OUTLIER * median:
                out.setdefault(leg, {})[key] = (ratio, median, len(ratios))
    return out


def outliers_to_retime(found):
    """The legs of FOUND (outlier_keys) whose harness is timed once more before an outlier fails
    them: those with an outlier of a harness row's time on the runtime (HARNESS_NEW), one
    pass of which never fails a leg on its own. Returns {leg: its outliers in FOUND}, time_outliers's
    FIRST once they are timed again."""
    return {leg: keys for leg, keys in found.items() if any(key.startswith(HARNESS_NEW) for key in keys)}


#: A host can give each process a level of its own: on the RTX 3090 at homelab-um870-1,
#: sampling_deep_4x16x252 took about 37 or about 80 us of wall a run, one level a process, at equal
#: GPU time and CPU instructions (rounds b0c08e51b5, 7cda47f536 and 6b535f55c9; m3-max-51629: 39
#: processes over 13 commits at 36.8 to 40.0 us). Two passes, two processes, both landed high in
#: 6b535f55c9. So a harness row's outlier in both passes is judged by its lowest level: the second
#: pass re-times each such row of the leg (alone_rows) alone in RETIME_PAIRS processes, each with a
#: fresh driver cache and store, its times kept as `harness-alone/<row>/<pair>/<runtime>` (ALONE_KEEP);
#: time_outliers compares the lowest of the runtime's samples (both passes and the alone runs)
#: with the lowest the base recorded of the row (levels). A real regression raises the low level and
#: still fails; a row high in every process fails; a row the lowest brings back is named with every
#: sample, as levels with no verdict, so a high level grown more frequent is still seen.
HARNESS_ALONE = "harness-alone/"
#: The sed program (`sed -n`) each alone run's output goes through: the row's time alone, the shell
#: giving `$timed`, `$row` and `$pair`.
ALONE_KEEP = r"s#^M2V-MATRIX time harness-$timed/$row #M2V-MATRIX time harness-alone/$row/$pair/$timed #p"
#: How many processes a row alone runs in, and at most how many of a leg's rows run alone.
RETIME_PAIRS = 2
RETIME_ROWS = 8


def alone_rows(keys):
    """The rows a leg's second pass re-times alone (HARNESS_ALONE) for KEYS, its first pass's
    outliers (outliers_to_retime's {key: (slowdown, median, legs)}): its harness rows' on the
    runtime (HARNESS_NEW), the furthest out first, at most RETIME_ROWS."""
    rows = sorted(((ratio / median, key[len(HARNESS_NEW):]) for key, (ratio, median, _) in keys.items()
                   if key.startswith(HARNESS_NEW)), reverse=True)
    return [row for _, row in rows[:RETIME_ROWS]]


def levels(values, key):
    """The samples of KEY, a harness row's time on the runtime (HARNESS_NEW), in VALUES, a leg's
    times: its own, then its alone runs' on the runtime (HARNESS_ALONE), each one recorded."""
    row = key[len(HARNESS_NEW):]
    alone = (values.get(f"{HARNESS_ALONE}{row}/{pair}/new") for pair in range(1, RETIME_PAIRS + 1))
    return [value for value in (values.get(key), *alone) if value]


def time_outliers(times, bases, first=None):
    """Timing fails only as a cross-GPU outlier (outlier_keys). Returns ({leg: [problems]}, {leg:
    [parked]}): an outlier of a PARKED_RECORDS key is parked, not a problem. FIRST: the legs whose
    harness was timed once more for a harness row's outlier (outliers_to_retime), each with its
    first pass's outliers; TIMES then hold their second pass, and an outlier of theirs is a
    problem only where their first pass had it too, the two passes agreeing; one the second pass
    alone has is parked. A harness row's outlier in both passes is then judged by its lowest level
    (HARNESS_ALONE): a problem where the lowest of its samples over the lowest its base recorded is
    still past TIME_OUTLIER times the median, else parked with every sample, as levels with no
    verdict."""
    out, parked = {}, {}
    first = first or {}
    now = outlier_keys(times, bases)
    for leg, found in sorted(now.items()):
        for key, (ratio, median, count) in sorted(found.items()):
            line = f"time {key}: {ratio:.2f}x its base, against a median of {median:.2f}x over {count} GPUs"
            if key.startswith(PARKED_RECORDS):
                parked.setdefault(leg, []).append(line)
            elif leg not in first:
                out.setdefault(leg, []).append(line)
            elif key in first[leg]:
                was = first[leg][key]
                agree = f"{line}, and {was[0]:.2f}x against {was[1]:.2f}x in the first pass: both passes agree"
                if not key.startswith(HARNESS_NEW):
                    out.setdefault(leg, []).append(agree)
                    continue
                samples = [was[0] * bases[leg][key], *levels(times[leg], key)]
                based = levels(bases[leg], key)
                low = min(samples) / min(based)
                shown = (f"; its lowest of {len(samples)} processes, {min(samples):.1f} us, over the base's lowest "
                         f"of {len(based)}, {min(based):.1f} us, is {low:.2f}x (samples "
                         f"{', '.join(f'{sample:.1f}' for sample in samples)} us)")
                if low > TIME_OUTLIER * median:
                    out.setdefault(leg, []).append(f"{agree}{shown}: high in every process")
                else:
                    parked.setdefault(leg, []).append(f"{agree}{shown}: levels, no verdict")
            else:
                parked.setdefault(leg, []).append(f"{line} when timed once more, but not in the first pass: "
                                                  "the passes disagree, so it fails nothing")
    for leg, keys in sorted(first.items()):
        for key in sorted(keys):
            if not key.startswith(PARKED_RECORDS) and key not in now.get(leg, {}):
                parked.setdefault(leg, []).append(
                    f"time {key}: an outlier in the first pass, {keys[key][0]:.2f}x its base against a median "
                    f"of {keys[key][1]:.2f}x, but not when timed once more: the passes disagree, so it fails nothing")
    return out, parked


#: A harness row's time on the runtime, `harness-new/<row>` (the harness's `--runtime new`). Its
#: time is judged only as a cross-GPU outlier (time_outliers); the frame and burst rows
#: (WALL_ROWS) are also held to their stored instruction counts (instruction_gate).
HARNESS_NEW = "harness-new/"
#: A harness row's GPU time, `harness-gpu-<runtime>/<row>`: the p50 over its timed runs of a run's
#: first GPU start to its last GPU end, scaled like its time where the device sizes the row, and
#: absent where the row has none. harness_rows shows it beside the row's time; nothing judges it
#: (PARKED_RECORDS).
HARNESS_GPU = "harness-gpu-"
#: The rows the instructions gate holds, by the start of their names: a frame's or a burst's
#: command buffers.
WALL_ROWS = ("frame", "burst")


#: The instructions gate of the frame and burst rows (WALL_ROWS), on Linux's user-space
#: instructions per timed run, each the p50 over the row's timed runs. The harness records the
#: whole process's (`instructions harness-<runtime>/<row>/per_run`), and where the runtime names
#: the threads it started (the runtime's `Gauges::threads`), the count of those and the
#: harness's own thread (`.../own_per_run`) and the rest, the threads a driver started for itself
#: (`.../others_per_run`). A driver's threads swing by millions of instructions per run between
#: processes (NVIDIA's on burst_160: 0 to 25 million), the runtime's own do not. The gate is on the
#: runtime's own: at most HARNESS_BUDGET times its stored count (STORED_INSTRUCTIONS), which
#: ratchets down. The base comparison of leg_problems skips the harness's counts: this gate judges
#: them. A leg that records no instructions (the kernel refused the counter, or Windows) has no
#: gate. Every row records its counts; only these are gated on them.
INSTRUCTIONS_PER_RUN = "/per_run"
#: The runtime's own threads' count of a row, and the rest of the process's.
OWN_PER_RUN, OTHERS_PER_RUN = "/own_per_run", "/others_per_run"
#: The instructions gate's allowance: a row's own count is at most this times its stored count.
HARNESS_BUDGET = 1.03


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


#: The runtime's own threads' instructions per timed run (OWN_PER_RUN) that each WALL_ROWS
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
    where ROW is not a WALL_ROWS row or the runtime recorded no whole count. The runtime's
    own count past HARNESS_BUDGET times the stored count is a problem; one more than
    INSTRUCTIONS_BAND under it, or a row with none stored, is named. A runtime that recorded
    its whole count but not its own is a problem: the gate could not run."""
    counted = records["instructions"]
    new = counted.get(f"harness-new/{row}{INSTRUCTIONS_PER_RUN}")
    if not row.startswith(WALL_ROWS) or not new:
        return [], []
    own = counted.get(f"harness-new/{row}{OWN_PER_RUN}")
    if own is None:
        return [f"harness budget {row}: instructions new {new} per run, but no count of its own threads"], []
    problems, parked = [], []
    stored = stored_count(leg_id, row)
    exact = f"harness budget {row}: instructions new own {own} per run, " + (
        f"stored {stored}, is {own / stored:.3f}" if stored else "none stored")
    if stored and own > HARNESS_BUDGET * stored:
        problems.append(f"{exact}: past {HARNESS_BUDGET} times the stored count")
    elif stored and own < (1 - INSTRUCTIONS_BAND) * stored:
        parked.append(f"{exact}: more than {INSTRUCTIONS_BAND:.0%} under the stored count, which ratchets "
                      "down to it")
    elif not stored:
        parked.append(exact)
    return problems, parked


#: The end of the name of a frames workload's row of its latency from idle, after the workload's
#: name (the harness's `FROM_IDLE`): `frame_from_idle` beside `frame`, whose time is the frames'
#: throughput. A row alone (HARNESS_ALONE) of one runs its workload.
FROM_IDLE = "_from_idle"


#: The harness's creation rows, by the start of their names. A leg times them apart from the
#: other rows (the matrix's harness_runs): the process that times the others skips them, and they
#: run after it in a process of their own on a fresh driver cache and store, so no row before them
#: moves their times.
HARNESS_CREATES = "create_"
#: The sed program the creation process's output goes through: every record of the device's life,
#: of any kind (the runtime's totals, its gauges at drop), becomes
#: `harness-<runtime>/creates/device/...`, apart from the other rows' process's, so each key is
#: recorded once.
CREATES_APART = (r"s#^M2V-MATRIX \([a-z]*\) harness-\([a-z]*\)/device/#"
                 r"M2V-MATRIX \1 harness-\2/creates/device/#")


def harness_budget(leg_id, leg):
    """The harness rows of LEG, the parsed leg LEG_ID, held to the instructions gate
    (instruction_gate). Returns (problems, parked): a gated row's problems and named lines, and a
    row whose time was recorded more than once (twice by a suite, or by two suites), which has no
    one time to show, is a problem. A row's time is judged only as a cross-GPU outlier
    (time_outliers)."""
    problems, parked = [], []
    suites = leg["suites"].values()
    records = leg_records(leg)[0]
    keys = [key for suite in suites for key in suite["records"]["time"]]
    twice = ({key for key in keys if keys.count(key) > 1}
             | {key for suite in suites for kind, key in suite.get("duplicated", ()) if kind == "time"})
    for key in sorted(records["time"]):
        if not key.startswith(HARNESS_NEW):
            continue
        row = key[len(HARNESS_NEW):]
        if key in twice:
            problems.append(f"harness budget {row}: its time is recorded more than once, so it has no one "
                            "time to show")
            continue
        gated, named = instruction_gate(leg_id, row, records)
        problems += gated
        parked += named
    return problems, parked


def harness_rows(records):
    """Every harness row timed in a leg's RECORDS, one line each: its time, its GPU time
    (HARNESS_GPU) and, where recorded, its instructions per run (with its own threads' where the
    runtime names them), `-` where it has none."""
    times, counted = records["time"], records["instructions"]
    lines = []
    for row in sorted(key[len(HARNESS_NEW):] for key in times if key.startswith(HARNESS_NEW)):
        wall, gpu = times.get(HARNESS_NEW + row), times.get(f"{HARNESS_GPU}new/{row}")
        line = (f"{row:<36} {'-' if wall is None else f'{wall:.1f}':>10} us, "
                f"GPU p50 {'-' if gpu is None else f'{gpu:.1f}':>10} us")
        instructions = counted.get(f"harness-new/{row}{INSTRUCTIONS_PER_RUN}")
        if instructions is not None:
            line += f", {instructions} instructions"
        own = counted.get(f"harness-new/{row}{OWN_PER_RUN}")
        if own is not None:
            line += f" ({own} own)"
        lines.append(line)
    return lines


def verdict_line(leg_id, leg):
    """The one line `run` and `show` print for a leg of a note."""
    tests = ", ".join(f"{name} {s['passed']}/{s['run']}" if s["run"] else f"{name} exit {s['exit']}"
                      for name, s in leg.get("suites", {}).items())
    counted = sum(len(values) for values in leg.get("records", {}).values())
    where = " ".join(part for part in (leg.get("device"), leg.get("driver")) if part)
    verdict = {"pass": "PASS", "unstable": "UNSTABLE"}.get(leg["verdict"], "FAIL")
    head = f"{verdict} {leg_id:10} {where or '?'} on {leg.get('node', '?')}"
    tail = (f"{tests or 'no suite ran'}; {counted} records; base {(leg.get('base') or 'none')[:10]}; "
            f"jobs {' '.join(leg.get('jobs', [])) or '-'}")
    if leg["problems"]:
        tail += " | " + " | ".join(leg["problems"][:4]) + (" ..." if len(leg["problems"]) > 4 else "")
    if leg.get("uncounted"):
        tail += " | not counted: " + ", ".join(leg["uncounted"])
    if leg.get("parked"):
        tail += f" | parked: {len(leg['parked'])} ({leg['parked'][0]}" + (" ..." if len(leg["parked"]) > 1 else "") + ")"
    return f"{head}: {tail}"
