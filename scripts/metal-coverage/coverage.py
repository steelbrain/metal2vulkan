#!/usr/bin/env python3
"""Write docs/METAL-COVERAGE.md from coverage.tsv, the Metal API coverage map.

coverage.tsv has one row per public item of the macOS SDK's Metal headers (names only, as
`sdk_items.py` lists them): its header, container, kind, name, whether macOS deprecates it, and
our classification:

- status `I` (implemented): `crate` (runtime or translator) and `counterpart` name the item that
  answers it, `Type::method` or `Type.field` (or a type);
- status `M` (missing): `note` says what it would take, `size` is S, M, L or XL, and `group`
  gathers items built together;
- status `O` (out of scope): `note` says why (ray tracing, macOS-only services, IOSurface
  and presentation, which are the consumer's).

Usage:
  coverage.py            write docs/METAL-COVERAGE.md
  coverage.py --check    fail if docs/METAL-COVERAGE.md is not what the table writes
  coverage.py --sdk      (Mac only) fail if the SDK's names differ from the table's
"""
import subprocess
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TABLE = HERE / "coverage.tsv"
DOC = ROOT / "docs" / "METAL-COVERAGE.md"
SDK = "macOS 27.2 SDK (Xcode 27.2, 27B52)"

STATUS = OrderedDict([("I", "implemented"), ("M", "missing"), ("O", "out of scope")])
SIZES = ["S", "M", "L", "XL"]

# Missing groups in the order we would build them: cheap items plain apps call first, then the
# medium pieces by how often a plain app reaches them, then the large API families. A missing
# call cannot answer wrongly (it does not exist); calls that exist and refuse are in
# REFUSALS.md.
ORDER = [
    "labels", "cmdbuf", "queue", "passdesc", "library", "queries", "barriers", "reflection",
    "function-handles", "functions", "archive", "events", "mesh-descriptor", "sparse",
    "parallel", "icb-depth-stencil", "depthbounds", "min-lod", "specialization",
    "sparsebuf", "counters", "residency", "colormap", "viewpool", "logging", "notifications",
    "compression", "capture", "persistent", "shadervalidation", "stitching", "io", "tensor",
    "mtl4", "ml", "access-counters",
]

GROUP_NAMES = {
    "labels": "labels, debug groups and signposts",
    "cmdbuf": "command buffer options (descriptor, unretained references, `enqueue`, encoder error status)",
    "queue": "command queue descriptor",
    "passdesc": "compute, blit and resource-state pass descriptors",
    "library": "libraries from a file path or URL",
    "queries": "device, state and resource queries",
    "barriers": "`textureBarrier` and queue-stage barriers",
    "reflection": "a pipeline state's reflection after creation",
    "function-handles": "function handles by name, and a handle's type and ID",
    "functions": "a specialized function's name",
    "archive": "more binary archive entries",
    "events": "the shared event listener",
    "mesh-descriptor": "mesh pipeline descriptor fields",
    "sparse": "placement sparse page sizes",
    "parallel": "parallel render encoding",
    "icb-depth-stencil": "depth/stencil states in argument buffers",
    "depthbounds": "depth bounds",
    "min-lod": "a texture's minimum LOD (macOS 27)",
    "specialization": "pipeline specialization (macOS 26)",
    "sparsebuf": "placement sparse buffers",
    "counters": "counter sample buffers",
    "residency": "residency sets",
    "colormap": "color attachment mapping",
    "viewpool": "texture view pools",
    "logging": "shader logging",
    "notifications": "device hot-plug notifications",
    "compression": "lossy texture compression",
    "capture": "GPU capture",
    "persistent": "persistent kernels (macOS 27)",
    "shadervalidation": "shader validation",
    "stitching": "function stitching",
    "io": "fast resource loading (`MTLIOCommandQueue`)",
    "tensor": "tensors",
    "mtl4": "Metal 4's command model",
    "ml": "machine learning encoding",
    "access-counters": "texture access counters (not scheduled)",
}


def rows():
    lines = TABLE.read_text().splitlines()
    head = lines[0].split("\t")
    out = []
    for line in lines[1:]:
        f = line.split("\t")
        if len(f) != len(head):
            sys.exit(f"coverage.tsv: {len(f)} fields, not {len(head)}: {line[:80]}")
        out.append(dict(zip(head, f)))
    return out


def check_rows(table):
    bad = []
    for r in table:
        what = f"{r['container']} {r['name']}"
        if r["status"] not in STATUS:
            bad.append(f"{what}: status {r['status']!r}")
        elif r["status"] == "I" and not (r["crate"] in ("runtime", "translator") and r["counterpart"]):
            bad.append(f"{what}: implemented without a crate and counterpart")
        elif r["status"] == "M" and r["size"] not in SIZES:
            bad.append(f"{what}: missing without a size")
        elif r["status"] in "MO" and not r["note"]:
            bad.append(f"{what}: no note")
        if r["status"] == "M" and r["group"] not in GROUP_NAMES:
            bad.append(f"{what}: group {r['group']!r} has no name")
    if bad:
        sys.exit("coverage.tsv:\n  " + "\n  ".join(bad))


def label(r):
    """An item's name, a member's with its container."""
    if r["kind"] in ("protocol", "class", "category", "function"):
        return r["name"]
    return f"{r['container']}.{r['name']}"


def cell(s):
    return s.replace("|", "\\|")


def render(table):
    out = []
    w = out.append
    total = Counter(r["status"] for r in table)
    crates = Counter(r["crate"] for r in table if r["status"] == "I")
    sizes = Counter(r["size"] for r in table if r["status"] == "M")
    w("# Metal API coverage")
    w("")
    w(
        f"Every public item of the {SDK}'s Metal headers, by name, and what answers it: the "
        "runtime (`metal2vulkan-runtime`, the Metal API on Vulkan) or the translator's plain values "
        "(`metal2vulkan`: pipeline descriptors, reflection, function constants), or nothing yet. "
        "Generated by `scripts/metal-coverage/coverage.py` from `scripts/metal-coverage/coverage.tsv` "
        "(edit the table, never this file); `coverage.py --sdk` checks the table's names against "
        "the SDK on a Mac, and `crates/metal2vulkan-validation/tests/integration/metal_coverage.rs` checks that every counterpart "
        "the table names exists in the sources."
    )
    w("")
    w("- **implemented**: the counterpart answers it. A call can still refuse some inputs, which "
      "[`REFUSALS.md`](REFUSALS.md) lists with the device conditions; the counterpart's Rustdoc says "
      "what differs.")
    w("- **missing**: no counterpart yet; the size is the work to build it (S: a day or less, M: "
      "days, L: a week or more, XL: a new subsystem). A missing call cannot answer wrongly: a "
      "program that needs it does not build.")
    w("- **out of scope**: ray tracing, multi-GPU and macOS-only services, IOSurface, drawables "
      "and source compiles, which the consumer owns, and items macOS does not offer.")
    w("")
    w("## Totals")
    w("")
    w("| Status | Items | Of which deprecated |")
    w("|---|---|---|")
    for s, name in STATUS.items():
        dep = sum(1 for r in table if r["status"] == s and r["deprecated"])
        w(f"| {name} | {total[s]} | {dep} |")
    w(f"| all | {len(table)} | {sum(1 for r in table if r['deprecated'])} |")
    w("")
    w(f"Implemented by the runtime: {crates['runtime']}; by the translator's values: "
      f"{crates['translator']}. Missing by size: " +
      ", ".join(f"{s} {sizes[s]}" for s in SIZES if sizes[s]) + ".")
    w("")
    w("## Missing, in build order")
    w("")
    w("Cheap items a plain app calls first, then medium pieces by how often a plain app reaches "
      "them, then the large API families.")
    w("")
    w("| Group | Size | Items | Examples |")
    w("|---|---|---|---|")
    groups = defaultdict(list)
    for r in table:
        if r["status"] == "M":
            groups[r["group"]].append(r)
    for g in sorted(groups, key=lambda g: (ORDER.index(g) if g in ORDER else len(ORDER), g)):
        items = groups[g]
        size = "/".join(s for s in SIZES if any(r["size"] == s for r in items))
        ex = ", ".join(f"`{label(r)}`" for r in items[:3])
        if len(items) > 3:
            ex += ", ..."
        w(f"| {cell(GROUP_NAMES[g])} | {size} | {len(items)} | {cell(ex)} |")
    w("")
    w("## By header")
    w("")
    by_header = OrderedDict()
    for r in table:
        by_header.setdefault(r["header"], OrderedDict()).setdefault(r["container"], []).append(r)
    for header, containers in by_header.items():
        w(f"### {header}")
        w("")
        w("| Item | Kind | Status | Counterpart or note |")
        w("|---|---|---|---|")
        for container, items in containers.items():
            for r in items:
                name = label(r)
                dep = " (deprecated)" if r["deprecated"] else ""
                if r["status"] == "I":
                    crate = "runtime" if r["crate"] == "runtime" else "translator"
                    what = f"`{r['counterpart']}` ({crate})"
                    if r["note"]:
                        what += f": {r['note']}"
                elif r["status"] == "M":
                    what = f"{r['size']}: {r['note']}"
                else:
                    what = r["note"]
                w(f"| `{cell(name)}`{dep} | {r['kind']} | {STATUS[r['status']]} | {cell(what)} |")
        w("")
    return "\n".join(out)


def sdk_check(table):
    sys.path.insert(0, str(HERE))
    import sdk_items

    sdk = {(it[1], it[2], it[3]) for it in sdk_items.items()}
    have = {(r["container"], r["kind"], r["name"]) for r in table}
    new = sorted(sdk - have)
    gone = sorted(have - sdk)
    for c, k, n in new:
        print(f"in the SDK, not the table: {c} {k} {n}")
    for c, k, n in gone:
        print(f"in the table, not the SDK: {c} {k} {n}")
    if new or gone:
        sys.exit(f"coverage.tsv differs from the SDK: {len(new)} new, {len(gone)} gone")
    print(f"coverage.tsv names every SDK item ({len(sdk)})")


def main():
    table = rows()
    check_rows(table)
    if "--sdk" in sys.argv:
        sdk_check(table)
        return
    text = render(table) + "\n"
    if "--check" in sys.argv:
        if DOC.read_text() != text:
            sys.exit("docs/METAL-COVERAGE.md is not what coverage.tsv writes: run coverage.py")
        print("docs/METAL-COVERAGE.md is current")
        return
    DOC.write_text(text)
    t = Counter(r["status"] for r in table)
    print(f"wrote {DOC.relative_to(ROOT)}: " + ", ".join(f"{STATUS[s]} {t[s]}" for s in STATUS))


if __name__ == "__main__":
    main()
