#!/usr/bin/env python3
"""The device lane's arithmetic: which tests it is, how they are sharded, and the proof.

`scripts/device-lane/device-lane` calls this; it is not meant to be run by hand, but each
subcommand works alone on an output directory the runner filled.

    lane.py derive --cfg CFG.json --plain PLAIN.json --out DIR [--selection SEL.json]
                   [--max-shard N]
    lane.py prove --out DIR
    lane.py tally --out DIR

derive reads two `cargo nextest list --run-ignored all --message-format json` listings of the
same workspace, one built with `--cfg m2v_gpu` and one without, and writes the lane (the tests
only the cfg build has) and a shard set over it. prove holds that shard set to the lane: every
selected test in exactly one shard, and no shard reaching a test outside the selection. tally
holds the runs that followed to what the shards promised.

A shard's filterset is built here and nowhere else (`filterset`), so the expression prove reasons
about is the expression the runner passes to nextest.
"""

import argparse
import json
import os
import re
import sys

# A test or binary name this tool will put in a filterset unquoted. Module paths in this workspace
# are lowercase Rust identifiers, so a name outside this set means the sharding rule has met
# something it was not written for: refuse rather than emit a filterset whose meaning is unclear.
SAFE = re.compile(r"^[A-Za-z0-9_:.-]+$")


def fail(message):
    sys.exit(f"lane.py: {message}")


# ---------------------------------------------------------------- listings


def read_listing(path):
    """The (binary id, test name) -> ignored map of one `nextest list --message-format json
    --run-ignored all`, holding the tests its filterset takes."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    start = text.find("{")
    if start < 0:
        fail(f"{path}: no JSON object (did the build fail? read the listing's stderr)")
    try:
        document = json.loads(text[start:])
    except json.JSONDecodeError as error:
        fail(f"{path}: {error}")
    suites = document.get("rust-suites") or document.get("rust_suites")
    if suites is None:
        fail(f"{path}: no rust-suites in the listing")
    tests = {}
    for key, suite in suites.items():
        binary = suite.get("binary-id") or suite.get("binary_id") or key
        for name, case in (suite.get("testcases") or {}).items():
            # A listing holds every test of the binary and says of each whether the run's filterset
            # takes it. Only `matches` is in the listing's set. The listings are taken with
            # `--run-ignored all`, so an `#[ignore]`d test is judged by the filterset like any
            # other and `ignored` alone says it will not run.
            match = case.get("filter-match") or case.get("filter_match") or {}
            status = match.get("status", "matches")
            if status == "matches":
                tests[(binary, name)] = bool(case.get("ignored"))
            elif status != "mismatch":
                fail(
                    f"{path}: {binary} {name}: filter-match status {status!r} "
                    "is not one this tool reads"
                )
    if not tests:
        fail(f"{path}: the listing selects no test at all")
    return tests


def write_tests(path, tests):
    with open(path, "w", encoding="utf-8") as handle:
        for binary, name in sorted(tests):
            handle.write(f"{binary}\t{name}\t{'ignored' if tests[(binary, name)] else 'run'}\n")


def read_tests(path):
    tests = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            binary, name, state = line.rstrip("\n").split("\t")
            tests[(binary, name)] = state == "ignored"
    return tests


# ---------------------------------------------------------------- shards


def filterset(binary, terms):
    """The nextest filterset for one shard: its binary, and every test name it holds.

    A shard names its tests one by one. A module prefix would be shorter, but a module holds
    GPU-free tests beside device ones, so a prefix term reaches tests the lane does not hold and
    the run would no longer be the lane.
    """
    for word in [binary] + terms:
        if not SAFE.match(word):
            fail(f"{word!r} is not a name this tool will put in a filterset")
    return f"binary_id(={binary}) and ({' or '.join(f'test(={term})' for term in terms)})"


def matches(binary, terms, tests):
    """Every test of `tests` that `filterset(binary, terms)` selects, by the same rule."""
    held = set(terms)
    return set(key for key in tests if key[0] == binary and key[1] in held)


def module(name):
    """`name`'s area: its first two module components, or fewer if it has fewer. A shard holds
    whole areas where it can, so a failure reads as a place in the tree and not as a test list."""
    return "::".join(name.split("::")[:2])


def shard_binary(names, max_shard):
    """`names` in shards of at most `max_shard`, packed in sorted order and broken at module
    boundaries where a module fits. A module larger than `max_shard` is cut into equal pieces.
    Deterministic in `names` and `max_shard`."""
    groups = {}
    for name in names:
        groups.setdefault(module(name), []).append(name)
    shards, held = [], []
    for key in sorted(groups):
        members = sorted(groups[key])
        while len(members) > max_shard:
            if held:
                shards.append(held)
                held = []
            shards.append(members[:max_shard])
            members = members[max_shard:]
        if held and len(held) + len(members) > max_shard:
            shards.append(held)
            held = []
        held += members
    if held:
        shards.append(held)
    return shards


def shard_name(binary, terms, taken):
    """A short, stable, unique log name for a shard: its binary, and the area it starts in."""
    short = binary.split("-")[-1].split("::")[-1]
    base = f"{short}-{module(terms[0]).replace('::', '.')}"
    name, extra = base, 1
    while name in taken:
        extra += 1
        name = f"{base}~{extra}"
    taken.add(name)
    return name


def derive_shards(selection, max_shard):
    """The shard set over `selection`, as (name, binary, terms) in a stable order."""
    by_binary = {}
    for binary, name in selection:
        by_binary.setdefault(binary, []).append(name)
    shards, taken = [], set()
    for binary in sorted(by_binary):
        for terms in shard_binary(sorted(by_binary[binary]), max_shard):
            shards.append((shard_name(binary, terms, taken), binary, terms))
    return shards


def write_shards(path, shards):
    with open(path, "w", encoding="utf-8") as handle:
        for name, binary, terms in shards:
            handle.write(f"{name}\t{binary}\t{' '.join(terms)}\n")


def read_shards(path):
    shards = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            name, binary, terms = line.rstrip("\n").split("\t")
            shards.append((name, binary, terms.split()))
    return shards


# ---------------------------------------------------------------- census


def census(cfg, plain, lane, selection):
    def per_binary(tests):
        counts = {}
        for binary, _ in tests:
            counts[binary] = counts.get(binary, 0) + 1
        return counts

    lines = [
        f"tests listed with --cfg m2v_gpu   {len(cfg)}",
        f"tests listed without the cfg      {len(plain)}",
        f"the device lane (the delta)       {len(lane)}",
        f"of the lane, #[ignore]d           {sum(1 for key in lane if lane[key])}",
        f"selected for this run             {len(selection)}"
        + ("" if len(selection) == len(lane) else "   (A SLICE, not the lane)"),
        "",
        "the lane by test binary:",
    ]
    counts = per_binary(lane)
    for binary in sorted(counts, key=lambda b: (-counts[b], b)):
        lines.append(f"  {counts[binary]:>6}  {binary}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- subcommands


def do_derive(args):
    cfg = read_listing(args.cfg)
    plain = read_listing(args.plain)
    lane = {key: cfg[key] for key in cfg if key not in plain}
    lost = sorted(key for key in plain if key not in cfg)
    if args.selection:
        listed = read_listing(args.selection)
        selection = {key: lane[key] for key in lane if key in listed}
    else:
        selection = dict(lane)
    os.makedirs(args.out, exist_ok=True)
    write_tests(os.path.join(args.out, "cfg-tests.tsv"), cfg)
    write_tests(os.path.join(args.out, "plain-tests.tsv"), plain)
    write_tests(os.path.join(args.out, "lane.tsv"), lane)
    write_tests(os.path.join(args.out, "selection.tsv"), selection)
    with open(os.path.join(args.out, "lost.tsv"), "w", encoding="utf-8") as handle:
        for binary, name in lost:
            handle.write(f"{binary}\t{name}\n")
    shards = derive_shards(selection, args.max_shard)
    write_shards(os.path.join(args.out, "shards.tsv"), shards)
    text = census(cfg, plain, lane, selection)
    text += f"shards                            {len(shards)}\n"
    with open(os.path.join(args.out, "census.txt"), "w", encoding="utf-8") as handle:
        handle.write(text)
    sys.stdout.write(text)
    return 0


def do_prove(args):
    cfg = read_tests(os.path.join(args.out, "cfg-tests.tsv"))
    selection = read_tests(os.path.join(args.out, "selection.tsv"))
    shards = read_shards(os.path.join(args.out, "shards.tsv"))
    with open(os.path.join(args.out, "lost.tsv"), encoding="utf-8") as handle:
        lost = [line.rstrip("\n") for line in handle if line.strip()]

    faults, seen, overlaps = [], {}, []
    empty = []
    for name, binary, terms in shards:
        hit = matches(binary, terms, cfg)
        if not hit:
            empty.append(name)
        for key in hit:
            if key in seen:
                overlaps.append((key, seen[key], name))
            else:
                seen[key] = name
    strays = sorted(key for key in seen if key not in selection)
    orphans = sorted(key for key in selection if key not in seen)

    if lost:
        faults.append(f"{len(lost)} test(s) the cfg build does not list but the plain build does")
    if empty:
        faults.append(f"{len(empty)} shard(s) select no test: {', '.join(empty)}")
    if overlaps:
        faults.append(f"{len(overlaps)} test(s) in more than one shard")
    if strays:
        faults.append(f"{len(strays)} test(s) a shard reaches that this run did not select")
    if orphans:
        faults.append(f"{len(orphans)} selected test(s) no shard reaches")

    out = [
        "coverage proof",
        f"  selected tests            {len(selection)}",
        f"  shards                    {len(shards)}",
        f"  tests the shards reach    {len(seen)}",
        f"  in two shards             {len(overlaps)}",
        f"  reached but not selected  {len(strays)}",
        f"  selected but unreached    {len(orphans)}",
        f"  listed plain, not by cfg  {len(lost)}",
    ]
    for key in orphans[:20]:
        out.append(f"  UNREACHED  {key[0]} {key[1]}")
    if len(orphans) > 20:
        out.append(f"  UNREACHED  ... and {len(orphans) - 20} more")
    for key in strays[:20]:
        out.append(f"  STRAY      {key[0]} {key[1]}")
    if len(strays) > 20:
        out.append(f"  STRAY      ... and {len(strays) - 20} more")
    for key, first, second in overlaps[:20]:
        out.append(f"  OVERLAP    {key[0]} {key[1]}  in {first} and {second}")
    for line in lost[:20]:
        out.append(f"  LOST       {line}")
    if faults:
        out.append("  PROOF FAILED: " + "; ".join(faults))
    else:
        out.append("  PROOF HOLDS: the shards partition the selection exactly")
    sys.stdout.write("\n".join(out) + "\n")
    return 1 if faults else 0


# nextest's closing line. `tests run` is what it executed; the `skipped` figure beside it counts
# every test of the shard's binary the filterset did not select as well as the `#[ignore]`d ones,
# so only `tests run` says anything about the shard.
SUMMARY = re.compile(r"Summary \[\s*([0-9.]+)s\]\s+(\d+) tests? run:")


def do_tally(args):
    selection = read_tests(os.path.join(args.out, "selection.tsv"))
    shards = read_shards(os.path.join(args.out, "shards.tsv"))
    runnable = sum(1 for key in selection if not selection[key])
    ignored = len(selection) - runnable
    runs = {}
    path = os.path.join(args.out, "runs.tsv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 4:
                    fail(f"{path}: {line.strip()!r} is not a row this tool wrote")
                name, status, build, wall = fields
                runs[name] = (int(status), int(build), int(wall))

    out = [
        "",
        "shard results",
        f"  {'shard':<42}  {'status':<12}{'ran':>5}{'want':>6}{'build':>8}{'run':>9}",
    ]
    ran_total, unaccounted, short, failed = 0, [], [], []
    tested = 0.0
    for name, binary, terms in shards:
        # What this shard's nextest run must have executed: its tests but the `#[ignore]`d ones.
        want = sum(1 for term in terms if not selection.get((binary, term), False))
        log = os.path.join(args.out, f"shard-{name}.log")
        ran = None
        if os.path.exists(log):
            with open(log, encoding="utf-8", errors="replace") as handle:
                for found in SUMMARY.finditer(handle.read()):
                    ran, seconds = int(found.group(2)), float(found.group(1))
        status, build, wall = runs.get(name, (None, 0, 0))
        if ran is None:
            unaccounted.append(name)
            shown = "NO SUMMARY"
        else:
            ran_total += ran
            tested += seconds
            shown = "pass" if status == 0 else f"FAIL exit={status}"
            if status != 0:
                failed.append(name)
            if ran != want:
                short.append(f"{name} ran {ran} of {want}")
        out.append(
            f"  {name:<42}  {shown:<12}"
            f"{'-' if ran is None else ran:>5}{want:>6}{build:>7}s{wall:>8}s"
        )

    faults = []
    if unaccounted:
        named = ", ".join(unaccounted[:4])
        if len(unaccounted) > 4:
            named += f", and {len(unaccounted) - 4} more"
        faults.append(f"{len(unaccounted)} shard(s) wrote no nextest summary: {named}")
    if short:
        faults.append(
            f"{len(short)} shard(s) ran fewer tests than they hold: {'; '.join(short[:4])}"
        )
    if ran_total != runnable:
        faults.append(f"{ran_total} tests ran in all, but the selection holds {runnable} runnable")
    out += [
        "",
        f"  the runs took {sum(run[2] for run in runs.values())}s in all, over {len(runs)} shard(s)",
        f"  nextest's own clock says {tested:.1f}s of tests; building took"
        f" {sum(run[1] for run in runs.values())}s",
        "",
        "coverage of the run",
        f"  selected, runnable        {runnable}",
        f"  selected, #[ignore]d      {ignored}",
        f"  nextest reports run       {ran_total}",
    ]
    if faults:
        out.append("  COVERAGE FAILED: " + "; ".join(faults))
    else:
        out.append("  COVERAGE HOLDS: nextest ran every runnable test of every shard")
    out.append("")
    if not failed and not faults:
        verdict = "VERDICT: every shard passed and the run covered the selection"
    elif not failed:
        verdict = "VERDICT: the run did not cover the selection"
    else:
        verdict = f"VERDICT: {len(failed)} shard(s) failed"
        if faults:
            verdict += ", and the run did not cover the selection"
    out.append(verdict)
    sys.stdout.write("\n".join(out) + "\n")
    return 1 if failed or faults else 0


def do_filterset(args):
    for name, binary, terms in read_shards(os.path.join(args.out, "shards.tsv")):
        if name == args.shard:
            sys.stdout.write(filterset(binary, terms) + "\n")
            return 0
    fail(f"no shard named {args.shard}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    derive = sub.add_parser("derive")
    derive.add_argument("--cfg", required=True)
    derive.add_argument("--plain", required=True)
    derive.add_argument("--selection")
    derive.add_argument("--out", required=True)
    derive.add_argument("--max-shard", type=int, default=150)
    derive.set_defaults(run=do_derive)

    for command, function in (("prove", do_prove), ("tally", do_tally)):
        step = sub.add_parser(command)
        step.add_argument("--out", required=True)
        step.set_defaults(run=function)

    expression = sub.add_parser("filterset")
    expression.add_argument("--out", required=True)
    expression.add_argument("--shard", required=True)
    expression.set_defaults(run=do_filterset)

    args = parser.parse_args()
    sys.exit(args.run(args))


if __name__ == "__main__":
    main()
