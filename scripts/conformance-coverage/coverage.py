#!/usr/bin/env python3
"""Write docs/CONFORMANCE-COVERAGE.md: which Metal items the conformance suite verifies.

`docs/METAL-COVERAGE.md` records what we **implement**. This records what the suite **compares
against Metal**, which is a different and narrower claim: an item is only verified where a case's
native Metal leg calls it, the case has a committed recording of Metal's answer, and our side's
answer is compared with it.

Two axes, because one of them alone reads as more than it is. The **items** axis is reachability:
one call of a selector is enough, so a selector that takes an enum counts as exercised whatever
value reached it. The **option space** axis (`option_space.py`) is depth: of each Metal enum's and
option set's values, which ones the suite actually compared, and — because a type whose values are
all compared says nothing about the pairs of them — how many pairs of the option space a case puts
to Metal at once, from the declarations in `crossings.tsv`, and, counted apart, how much
interaction it holds that is no pair of that space, from `other_crossings.tsv`.

Nothing here is a maintained list. Every number is derived, at the moment this runs, from four
things already in the tree:

1. `scripts/metal-coverage/coverage.tsv` — the SDK's items and whether we implement each. This
   does not re-derive that; it joins onto it.
2. `scripts/conformance-coverage/objc2_metal.tsv` — the selector each `objc2-metal` binding sends,
   which type declares it and what it answers. `--api` checks it against the crate's own sources
   on a Mac; it is committed so this runs anywhere.
3. `crates/metal2vulkan-conformance/src/` — the suite's own sources. `metal_leg.py` types each
   receiver in the Metal leg and resolves its calls to SDK items, and the reachability walk below
   says which of the suite's own vocabulary reaches each call.
4. `crates/metal2vulkan-conformance/recordings/` — the committed recordings, which are what makes
   a case's Metal answer a recorded fact rather than an intention.
5. `crates/metal2vulkan/src/metal/generated.rs` and the front end's — the SDK's enumerator names
   and ordinals, which `scripts/regen-metal-enums/` extracts from the Metal headers. They are the
   option-space axis's denominator; no value list is written here.

Usage:
  coverage.py            write docs/CONFORMANCE-COVERAGE.md
  coverage.py --check    fail if docs/CONFORMANCE-COVERAGE.md is not what the sources derive
  coverage.py --api      (Mac only) fail if objc2_metal.tsv is not what objc2-metal's sources say
  coverage.py --selftest run the option-space axis over inputs whose answers are known
"""

import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import metal_leg  # noqa: E402  (the resolver, beside this script)
import option_space  # noqa: E402  (the option-space axis, beside this script)

SUITE = ROOT / "crates/metal2vulkan-conformance"
SRC = SUITE / "src"
DOC = ROOT / "docs/CONFORMANCE-COVERAGE.md"
API = HERE / "objc2_metal.tsv"
TABLE = ROOT / "scripts/metal-coverage/coverage.tsv"

#: The oracle every committed recording is of, as the suite states it.
ORACLE = "apple-m3-max"

STATUS = {"I": "implemented", "M": "missing", "O": "out of scope"}


# --------------------------------------------------------------------------------------------
# The SDK's items


def sdk_rows() -> list[dict[str, str]]:
    """Every row of the Metal API coverage map."""
    with TABLE.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def item_index(rows) -> dict[tuple[str, str], dict[str, str]]:
    """The rows by `(container, name)`."""
    return {(row["container"], row["name"]): row for row in rows}


def self_named(rows) -> dict[str, tuple[str, str]]:
    """Each container whose own implemented row the map holds, keyed by the container.

    A container's own row — `MTLComputeCommandEncoder`'s `MTLComputeCommandEncoder`, and every
    other `class`, `protocol` and `category` row — names the **type**, not a selector of it.
    `objc2-metal` binds no selector by that name, and the resolution below only ever yields
    selectors, so no call site can resolve to one: left to that resolution every such row reads as
    never exercised however much of the type the suite drives.

    So they are credited by what the row actually claims, which is that we implement the type: the
    suite **messages** it. A call site whose receiver is the container or a subtype of it exercises
    the container's own row, gated and attributed to a family exactly as the selector it sends is.
    A class row and a protocol row are one rule here, because a message to the class
    (`MTLTextureDescriptor::alloc()`) types its receiver to the class the same way an instance send
    types its own.
    """
    return {
        row["container"]: (row["container"], row["name"])
        for row in rows
        if row["status"] == "I" and row["container"] and row["name"] == row["container"]
    }


def free_functions(rows) -> dict[str, tuple[str, str]]:
    """The header's free functions the map holds as implemented, keyed by the function's name.

    These no container declares, so `objc2-metal` binds them as Rust functions rather than as
    selectors and the receiver typing never sees one. The leg calls such a function by its own C
    name, which the resolver records as a plain call — an edge of the reachability graph — so the
    walk reads them there instead.
    """
    return {
        row["name"]: (row["container"], row["name"])
        for row in rows
        if row["status"] == "I" and row["kind"] == "function" and not row["container"]
    }


def unbound(rows, index, api: metal_leg.Api) -> list[tuple[str, str]]:
    """The implemented items no case could call, because `objc2-metal` binds no name for them.

    The suite reaches Metal through `objc2-metal` alone. An item the crate binds no selector for
    is not a gap in the **suite**: no case can call what has no binding, so counting it as not
    exercised reads as work that no amount of authoring would close. Reported apart from the gap
    for that reason, and derived rather than listed: every binding the crate declares is resolved
    the same way a call site is, and what no binding lands on is what is left.

    A container's own row and a free function are excluded: neither is reached through a selector,
    and both have their own rule above.
    """
    bindable = set()
    for (ty, rust), selector in api.selector.items():
        found = sdk_item(index, api, ty, selector)
        if found is not None:
            bindable.add(found)
    keys = {(row["container"], row["name"]) for row in rows if row["status"] == "I"}
    other = set(self_named(rows).values()) | set(free_functions(rows).values())
    return sorted(keys - bindable - other)


def sdk_item(index, api: metal_leg.Api, ty: str, selector: str):
    """The coverage map's key for the item a call of `selector` on `ty` exercises, or `None`.

    The map keys an item by the container that **declares** it, so a call on `MTLTexture` of
    `label` is `MTLResource`'s item; and a setter is the property it writes, which the map holds
    once under the property's own name.
    """
    # `objc2` writes an out-parameter error as a trailing `_`, which the header does not.
    selector = selector.removesuffix("_")
    names = [selector]
    if selector.endswith(":") and selector.count(":") == 1 and selector.startswith("set"):
        bare = selector[3:-1]
        names.append(bare[0].lower() + bare[1:])
    # A `BOOL` property's getter is spelled `isFoo`; the header declares the property `foo`.
    if ":" not in selector and re.fullmatch(r"is[A-Z][A-Za-z0-9]*", selector):
        names.append(selector[2].lower() + selector[3:])
    for owner in api.chain(ty):
        for name in names:
            if (owner, name) in index:
                return (owner, name)
    return None


# --------------------------------------------------------------------------------------------
# The suite's vocabulary and what reaches each call


def crate_enums() -> set[str]:
    """Every enum the suite declares."""
    out = set()
    for path in sorted(SRC.rglob("*.rs")):
        text = metal_leg.blanked(path.read_text())
        out.update(re.findall(r"\benum\s+([A-Za-z_][A-Za-z0-9_]*)", text))
    return out


def gate_enums(enums: set[str]) -> set[str]:
    """The enums a `match` arm in the Metal leg may be gated on.

    Only an enum whose variants a case **names** can gate: the attribution reads a case's
    vocabulary off its own text, so an arm on an enum no case names carries no information about
    which case reaches it, and treating it as a gate would drop the call from every family.
    """
    named: set[str] = set()
    for path in sorted((SRC / "cases").glob("*.rs")):
        text = metal_leg.blanked(path.read_text())
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)::[A-Za-z_][A-Za-z0-9_]*", text):
            if match.group(1) in enums:
                named.add(match.group(1))
    return named


def leg_files() -> list[str]:
    """The suite's files that reach Metal, by what they import rather than by name."""
    out = []
    for path in sorted(SRC.rglob("*.rs")):
        if "objc2_metal" in metal_leg.blanked(path.read_text()):
            out.append(str(path.relative_to(SRC)))
    return out


def side_methods() -> list[str]:
    """Every method of the `Side` trait: the suite's whole vocabulary for reaching a device."""
    text = metal_leg.blanked((SRC / "case.rs").read_text())
    at = text.index("trait Side")
    start, end = metal_leg.block(text, at)
    return sorted(set(re.findall(r"\bfn\s+([a-z_][a-z0-9_]*)", text[start:end])))


def variant_constructors(enums: set[str]) -> dict[str, set[str]]:
    """Each `Enum::helper` that builds variants, to the variants it builds.

    A case says `RenderStep::triangle(..)` where the vocabulary's own constructor makes a
    `RenderStep::Draw`, so following these is what keeps a case's step set from reading empty.
    """
    out: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.rs")):
        text = metal_leg.blanked(path.read_text())
        for match in re.finditer(r"\bimpl\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", text):
            owner = match.group(1)
            if owner not in enums:
                continue
            start, end = metal_leg.block(text, match.start())
            for fn in metal_leg.functions(text[start:end], str(path)):
                built = {
                    f"{owner}::{name}"
                    for name in re.findall(
                        rf"(?:Self|{owner})::([A-Za-z_][A-Za-z0-9_]*)",
                        text[start:end][fn.start : fn.end],
                    )
                }
                if built:
                    out.setdefault(f"{owner}::{fn.name}", set()).update(built)
    # A constructor that only calls another constructor resolves through it.
    for _ in range(4):
        for name, built in out.items():
            out[name] = {v for b in built for v in out.get(b, {b})}
    return out


def sdk_keyed(leg: metal_leg.Leg, index, api: metal_leg.Api):
    """Each call's SDK item, and the calls whose item the coverage map does not list.

    `objc2-metal` and the coverage map key an item by different containers: `objc2` by the trait
    nearest the receiver, the map by the header that declares it. And the map holds a property
    once, under its getter's name, while a call may be its setter. So the receiver's own supertype
    chain is walked until the map has the item.
    """
    keyed: dict[metal_leg.Call, tuple[str, str]] = {}
    unlisted: list[tuple[str, str]] = []
    for call in leg.calls:
        if not call.item:
            continue
        found = sdk_item(index, api, call.receiver or call.item[0], call.item[1])
        if found is None:
            unlisted.append(call.item)
        else:
            keyed[id(call)] = found
    return keyed, sorted(set(unlisted))


def messaged_rows(leg: metal_leg.Leg, api: metal_leg.Api, selves) -> dict[int, set[tuple[str, str]]]:
    """For each call site, the containers' own rows its receiver's type exercises.

    A send to a receiver typed `MTLTexture` messages `MTLTexture`, and through it every container
    `MTLTexture` inherits from, so the row of each of those containers is exercised by that one
    call. Keyed by the call so the walk gates and attributes it with the selector it sends.
    """
    out: dict[int, set[tuple[str, str]]] = {}
    for call in leg.calls:
        if not call.receiver:
            continue
        keys = {selves[owner] for owner in api.chain(call.receiver) if owner in selves}
        if keys:
            out[id(call)] = keys
    return out


def gated_items(leg: metal_leg.Leg, entry_points: set[str], keyed, messaged, frees):
    """For each entry point, the items it reaches and the gates each reaching needs.

    One walk of the leg's own call graph. A call inside `Step::GetBytes =>` is reached only by a
    case that encodes that step, and one inside `if let Some(b) = pass.visibility` only by a case
    that states that field, so the gates around a call travel with it; a call under no gate is
    reached by every case that calls the entry point at all. A gate set holds both kinds
    (`metal_leg.GATE_FIELD` tells them apart) and a reaching needs every gate in it.

    Three things a call reaches, and the same gate over all three: the SDK item its selector
    resolves to (`keyed`), the containers' own rows its receiver messages (`messaged`), and, on an
    edge of the graph rather than a call, a header free function the leg calls by name (`frees`).
    """
    reached: dict[str, dict[tuple[str, str], set[frozenset]]] = {}

    def walk(fn: str, stack: tuple[str, ...]) -> dict[tuple[str, str], set[frozenset]]:
        if fn in reached:
            return reached[fn]
        if fn in stack:
            return {}
        here: dict[tuple[str, str], set[frozenset]] = defaultdict(set)
        for call in leg.calls:
            if call.function != fn:
                continue
            arms = frozenset(call.gates)
            if id(call) in keyed:
                here[keyed[id(call)]].add(arms)
            for item in messaged.get(id(call), ()):
                here[item].add(arms)
        for callee in leg.calls_function.get(fn, ()):
            if callee in frees:
                for arms in leg.function_gates.get((fn, callee), {()}):
                    here[frees[callee]].add(frozenset(arms))
            below = walk(callee, stack + (fn,))
            if not below:
                continue
            for arms in leg.function_gates.get((fn, callee), {()}):
                for item, gates in below.items():
                    for gate in gates:
                        here[item].add(frozenset(gate | set(arms)))
        if fn not in stack:
            reached[fn] = dict(here)
        return dict(here)

    return {fn: walk(fn, ()) for fn in sorted(entry_points)}


# --------------------------------------------------------------------------------------------
# The case set


def field_setters() -> dict[str, set[str]]:
    """Each method of the suite's own vocabulary that assigns a field, to the fields it assigns.

    A case states a field either in the struct literal or through the builder the vocabulary
    offers for it — `RenderPass::counting_into(buffer)` is how a case attaches a visibility-result
    buffer — so without this map a case that used the builder would read as stating nothing. The
    same shape `variant_constructors` reads for an enum's helpers, over `self.field = ..` instead.

    Keyed by the method's name alone, like the guards it answers: two structs with a setter of one
    name credit each other's field, which is generous in the direction that cannot refuse a true
    declaration.
    """
    out: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.rs")):
        text = metal_leg.blanked(path.read_text())
        for fn in metal_leg.functions(text, str(path)):
            assigned = {
                match.group(1)
                for match in re.finditer(
                    r"\bself\s*\.\s*([a-z_][a-z0-9_]*)\s*=[^=]", text[fn.start : fn.end]
                )
            }
            if assigned:
                out.setdefault(fn.name, set()).update(assigned)
    return out


def stated_fields(text: str, setters: dict[str, set[str]]) -> set[str]:
    """The vocabulary fields a case module's text states, as a guard in the leg reads them.

    Three spellings, all of them the module's own text: a struct literal's `field: value`, an
    assignment onto a value (`pass.size = cell.b_area`), and a call of a builder that assigns the
    field (`setters`).

    Read generously, and deliberately so. A `let size: u32` reads as stating `size`, and a field a
    module states as `None` or `false` reads the same as one it states a value for — the two
    values of a boolean are nowhere in this text, which is the limit a field axis already carries.
    What the reading is exact about is the only thing the check turns on: a module that never
    mentions the field at all states none of it.
    """
    out = {match.group(1) for match in re.finditer(r"\b([a-z_][a-z0-9_]*)\s*:", text)}
    out |= {
        match.group(1)
        for match in re.finditer(r"\.\s*([a-z_][a-z0-9_]*)\s*=[^=]", text)
    }
    for match in re.finditer(r"\.\s*([a-z_][a-z0-9_]*)\s*\(", text):
        out |= setters.get(match.group(1), set())
    return out


def case_families(
    side: list[str], enums: set[str], constructors, setters: dict[str, set[str]]
) -> dict[str, dict]:
    """Each case module, with the vocabulary its cases use and what it loosens.

    A module's cases share one `run`, so the module is the unit a call site can be attributed to
    without running anything: which `Side` methods its text calls, which variants of the
    vocabulary its text builds, and which of its fields its text states.
    """
    out: dict[str, dict] = {}
    for path in sorted((SRC / "cases").glob("*.rs")):
        source = path.read_text()
        text = metal_leg.blanked(source)
        # An allowance's site is a string, so it is read from the source as written, with the
        # comments dropped so prose about an allowance is not counted as one.
        written = "\n".join(line.split("//")[0] for line in source.splitlines())
        # `rustfmt` breaks `side.mesh_pass(..)` across lines, so the dot is not always beside the
        # receiver.
        calls = {m for m in re.findall(r"\bside\s*\.\s*([a-z_][a-z0-9_]*)\s*\(", text) if m in side}
        variants: set[str] = set()
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)", text):
            name = f"{match.group(1)}::{match.group(2)}"
            if match.group(1) not in enums:
                continue
            variants.update(constructors.get(name, {name}))
        out[path.stem] = {
            "calls": calls,
            "variants": variants,
            # The vocabulary fields the module's text states, which is what a guard in the leg
            # (`metal_leg.guards_of`) is satisfied by: a field written in a struct literal, one
            # assigned onto a value, and one a builder of the vocabulary assigns for it.
            "states": stated_fields(text, setters),
            # What the module loosens, and what it names as the justification. Comparison is
            # exact by default, so a module with no tolerance compares every measured field.
            #
            # A **notice declaration** and a **tolerance** are both `Allowance`s and are not the
            # same thing: a declaration loosens only the notices our side announces, which Metal
            # has none of, so the case can be read at all on a venue that announces one. It
            # loosens no measured value. A tolerance does, and is the number that matters.
            "declarations": _declarations(written),
            "allowances": _tolerances(written),
            "loosenings": sorted(set(re.findall(r"\bLoosening::([A-Za-z]+)", text))),
            "justifications": sorted(
                {
                    {"notice": "Degradation", "accepted": "AcceptedDifference"}.get(name, name)
                    for name in re.findall(r"\bJustification::([A-Za-z]+)", text)
                }
            ),
            # A carried answer record mode must refuse to replace, because its Metal leg is unsafe.
            "carried_only": len(re.findall(r"\bcarried_only\s*\(", text)),
            "cases": [],
            # The cases whose Metal answer record mode took by **running** the native leg. A
            # carried answer, and a call Metal never returned from, run no leg at all, so they
            # exercise nothing however much Metal the module's code names.
            "ran": [],
        }
    return out


def _allowance_paths(text: str) -> list[str]:
    """The site each `Allowance::new` in `text` loosens, as the source writes it."""
    out = []
    for match in re.finditer(r"\bAllowance::new\s*\(", text):
        depth = 1
        i = match.end()
        while i < len(text) and depth:
            if text[i] in "([{":
                depth += 1
            elif text[i] in ")]}":
                depth -= 1
            elif text[i] == "," and depth == 1:
                break
            i += 1
        out.append(text[match.end() : i])
    return out


def _is_declaration(path: str) -> bool:
    """Whether an allowance of this site is a notice declaration rather than a tolerance."""
    return "NOTICES_SITE" in path or "notices" in path


def _declarations(text: str) -> int:
    """How many notice declarations `text` makes.

    `shared_memory_notice` is one of them: it is `notice_allowance` of the one code a venue's
    memory model provokes, which every family that reads an answer back through a `Shared` buffer
    has to declare, so a module that calls it has declared a notice.
    """
    return len(
        re.findall(r"\b(?:notice_allowance|shared_memory_notice)\s*\(", text)
    ) + sum(1 for path in _allowance_paths(text) if _is_declaration(path))


def _tolerances(text: str) -> int:
    """How many measured values `text` loosens."""
    return sum(1 for path in _allowance_paths(text) if not _is_declaration(path))


def _provenance(answer: dict) -> str:
    """Where an answer came from: `metal_run`, or the kind of answer it carries instead."""
    found = answer.get("provenance", "")
    return found if isinstance(found, str) else next(iter(found), "")


def recordings(families: dict[str, dict]) -> list[dict]:
    """Every committed recording, with the case module that declares its case's name.

    A case's name is a literal of its own module, so the recording and the code it came from are
    joined by the name itself rather than by a table saying which is which.
    """
    # The literals of each module, read from the source as written: a case's name is a string, and
    # the blanking the rest of this derivation needs would take exactly the thing being looked up.
    literals = {
        module: set(re.findall(r'"([^"\\\n]*)"', (SRC / "cases" / f"{module}.rs").read_text()))
        for module in families
    }
    out = []
    for path in sorted((SUITE / "recordings").glob("*.json")):
        data = json.loads(path.read_text())
        name = data["case"]
        owners = [module for module, names in literals.items() if name in names]
        if len(owners) != 1:
            raise SystemExit(
                f"{path.name}: the case name {name!r} is a literal of {len(owners)} case modules "
                f"({', '.join(owners) or 'none'}); this derivation needs exactly one"
            )
        answers = data.get("answers", [])
        out.append(
            {
                "case": name,
                "module": owners[0],
                "runs": sum(answer.get("runs", 0) for answer in answers),
                "provenance": sorted({_provenance(answer) for answer in answers}),
                "chips": sorted({answer.get("key", {}).get("chip", "") for answer in answers}),
                "outcomes": sum(len(answer.get("outcomes", [])) for answer in answers),
                "shaders": len(data.get("shaders", [])),
            }
        )
        families[owners[0]]["cases"].append(name)
        if "metal_run" in out[-1]["provenance"]:
            families[owners[0]]["ran"].append(name)
    return out


def known_gaps() -> list[tuple[str, str]]:
    """The ledger's rows, as `(case, what it is waiting on)`."""
    out = []
    for name in ("known-gaps.tsv", "known-refusals.tsv"):
        path = SUITE / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            out.append((fields[0], fields[-1]))
    return out


# --------------------------------------------------------------------------------------------
# The join


def metal_items(
    index: dict[tuple[str, str], dict[str, str]],
    by_item: dict[tuple[str, str], dict[str, set[str]]],
    stated: dict[tuple[str, str], dict[str, set[str]]],
    api: metal_leg.Api,
) -> dict[tuple[str, str], option_space.Item]:
    """Every Metal item a crossing may name as a field axis, from the sources already read.

    The key is the items axis's own -- `(container, item name)` of `scripts/metal-coverage`'s map
    -- so a field axis names an item this document already has a row for rather than a word a
    row's author chose. Each carries the Metal type its call answers, which is what tells a boolean
    or numeric field from an enum-valued one, and the case modules that reach it, in the two
    strengths the sources can say apart:

    * `by_item` -- reached at all, which is the items axis's own join and what that axis counts;
    * `stated` -- reached through a call site whose guards the module's own data can take. The leg
      sets fifty fields of a pass descriptor in one function, each under `if let Some(x) =
      pass.field`, so every case that opens a pass reaches all fifty by the first reading and only
      the ones it states by the second. A crossing's field axis is checked against the second, so
      a row says the module put a value to **that field** and not merely to that Metal type.

    The answered type is `objc2-metal`'s, keyed by the selector it binds for the item: a property
    is bound under its own name, and an Objective-C `BOOL` property under `isName` as well, so
    both spellings are looked for and a property bound under neither answers nothing here. That is
    sound in the one direction this is used: an item whose answered type is unknown is **not**
    refused as an enum under another name, so the check can read short of the truth and never over
    it.
    """
    answered: dict[tuple[str, str], str] = {}
    for (metal, rust), one in api.answers.items():
        selector = api.selector.get((metal, rust), rust)
        for spelling in (selector, rust):
            answered.setdefault((metal, spelling), one)
            if spelling.startswith("is") and len(spelling) > 2:
                tail = spelling[2].lower() + spelling[3:]
                answered.setdefault((metal, tail), one)
    out: dict[tuple[str, str], option_space.Item] = {}
    for key, row in index.items():
        if not key[0] or key[1] == key[0]:
            # A free function's empty container, and a container's own row, which names the type
            # and not a field of it.
            continue
        out[key] = option_space.Item(
            set(stated.get(key, {})),
            answered.get(key, ""),
            set(by_item.get(key, {})),
        )
    return out


def inventory():
    """Everything the document states, derived."""
    api = metal_leg.Api.read(API)
    rows = sdk_rows()
    index = item_index(rows)
    enums = crate_enums()
    gates = gate_enums(enums)
    files = leg_files()
    leg = metal_leg.read_leg(api, SRC, files, gates)
    if leg.mistyped:
        raise SystemExit(
            "a receiver typed to something that does not bind the selector sent to it, which is a "
            "bug in metal_leg.py rather than a Metal call:\n"
            + "\n".join(f"  {f}::{fn} sent {rust} to {ty}" for f, fn, rust, ty in sorted(set(leg.mistyped)))
        )
    keyed, unlisted = sdk_keyed(leg, index, api)
    selves = self_named(rows)
    frees = free_functions(rows)
    messaged = messaged_rows(leg, api, selves)
    side = side_methods()
    constructors = variant_constructors(enums)
    setters = field_setters()
    families = case_families(side, enums, constructors, setters)
    records = recordings(families)

    # The call graph, from each `Side` method the Metal leg implements.
    entries = {fn for fn in leg.calls_function} | {call.function for call in leg.calls}
    gated = gated_items(leg, entries & set(side), keyed, messaged, frees)

    # An item to the families that reach it, and how. And, apart from it, the stronger reading: the
    # families whose own data can take the guards around the call site they reach it through.
    # Only a crossing's field axis is checked against that second reading; the items axis counts
    # reachability, which is what it has always claimed and what its own prose says.
    by_item: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    stated: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for module, family in families.items():
        if not family["ran"]:
            # Nothing of this module's Metal is ever executed.
            continue
        for method in family["calls"]:
            for item, gates in gated.get(method, {}).items():
                for gate in gates:
                    named = {
                        one for one in gate if not one.startswith(metal_leg.GATE_FIELD)
                    }
                    if not named <= family["variants"]:
                        continue
                    by_item[item][module].add(method)
                    fields = {
                        one.removeprefix(metal_leg.GATE_FIELD)
                        for one in gate
                        if one.startswith(metal_leg.GATE_FIELD)
                    }
                    if fields <= family["states"]:
                        stated[item][module].add(method)

    # Calls the leg makes that no family was found to reach: a call site this cannot attribute.
    reached = {item for method in gated.values() for item in method}
    called = (
        set(keyed.values())
        | {item for items in messaged.values() for item in items}
        | {frees[callee] for callees in leg.calls_function.values() for callee in callees if callee in frees}
    )
    return {
        "rows": rows,
        "index": index,
        "api": api,
        "leg": leg,
        "side": side,
        "families": families,
        "records": records,
        "gated": gated,
        "by_item": by_item,
        "stated": stated,
        "called": sorted(called),
        "unattributed": sorted(called - set(by_item)),
        "unreachable": sorted(called - reached),
        "keyed": keyed,
        "selves": selves,
        "frees": frees,
        "messaged": messaged,
        "unbound": unbound(rows, index, api),
        "unlisted": unlisted,
        "gaps": known_gaps(),
        # The second axis. A case whose answer Metal never ran puts no value to Metal, which is
        # the same gate the items axis applies to a call; the axis reads the values a module states
        # and the crossings a case declares, so it is handed the cases and joins them itself.
        #
        # `metal_items` is what a crossing over a Metal **field** is checked against: the items
        # axis's own join of item to the modules reaching it, and what each call answers. One
        # vocabulary, so a field axis names a call this document already reports as exercised.
        "option_space": option_space.inventory(
            ROOT,
            {
                case: module
                for module, family in families.items()
                for case in family["ran"]
            },
            metal_items(index, by_item, stated, api),
        ),
    }


# --------------------------------------------------------------------------------------------
# The document


def _container(name: str) -> str:
    """A container's name as the document prints it; the map leaves a free function's empty."""
    return f"`{name}`" if name else "the header's free functions"


def families_line(modules: dict[str, set[str]]) -> str:
    """The families that reach an item, as the document prints them."""
    return ", ".join(f"`{module}`" for module in sorted(modules))


def names_line(names: list[str]) -> str:
    """Enumerator names as the document prints them."""
    return ", ".join(f"`{name}`" for name in names)


def write_crossings(w, space) -> None:
    """The third axis: the crossings a case declares, in two counts that are never added.

    *The pairs of the option space* is `crossings.tsv`: every axis a Metal enum, so the number is
    comparable with the per-value columns above. *The crossings that are not pairs* is
    `other_crossings.tsv`: a Metal enum crossed with a boolean or numeric Metal field, or one type
    crossed with itself in two slots. Both are declarations and both are floors; they measure
    different things and one headline of the two would measure neither.
    """
    rows = space["rows"]
    held = space["crossings"]
    other = space["other_crossings"]
    pairs = sorted({pair for row in held for pair in row["pairs"]})
    crossed = sorted({axis for row in held for axis in row["axes"]})
    # An enum a row of the second file names is crossed too, so the work list below is read
    # against both. A field axis is no Metal enum and names none.
    crossed_at_all = set(crossed) | {
        axis for row in other for axis in row["axes"] if "." not in axis
    }
    slots = sorted({axis for row in other for axis in row["slots"]})
    fields = sorted({f"{ty}.{name}" for row in other for ty, name in row["fields"]})
    with_slots = sum(1 for row in other if row["slots"])
    with_fields = sum(1 for row in other if row["fields"])
    # The work list: a type the suite has real per-value depth in and crosses with nothing, in
    # either file. One compared value is not depth, so the floor is two.
    thin = sorted(
        (row for row in rows if len(row["compared"]) >= 2 and row["metal"] not in crossed_at_all),
        key=lambda r: (-len(r["compared"]), r["metal"]),
    )

    w("### By crossing, which is depth in the combinations")
    w("")
    w(
        "Every column above is per value, and the pairs are where most of the behaviour an "
        "implementation can get wrong lives: a side can be right about each value of two types "
        "alone and wrong about a cell of the two together. This counts the crossings, which is a "
        "third thing again — not reach, not per-value depth, but how many **pairs of the option "
        "space** a case puts to Metal at once."
    )
    w("")
    w(
        "A crossing is **declared**, in `scripts/conformance-coverage/crossings.tsv`: one row a "
        "crossing, naming the case module, the cell its falsification is read at (one of the "
        "module's recorded cases, or `family` where the cross is read across the family's cases "
        "rather than inside one of them), the Metal types crossed, and what only the cross decides. The count is therefore of "
        "crossings **declared**, never of crossings proved, and what makes the list worth reading "
        "is what the derivation refuses. **This document is not written** if a row names a module "
        "no case of which ran Metal, a cell that is not one of that module's recorded cases, an "
        "axis with no generated value list, fewer than two axes or one axis twice, or an axis of "
        "which the module's own text states **fewer than two values by name** — a module that "
        "cannot vary an axis cannot cross it, so one value would falsify the row."
    )
    w("")
    w("What this axis cannot see, stated rather than implied:")
    w("")
    w(
        "- **Whether the expected answer needs both axes.** No reading of the text can say that: it "
        "is a property of the case's reasoning. The last column says it in words, for a reader to "
        "check against the case, and the refusals above check only that the claim is **stateable** "
        "— that the module varies both axes and the cell ran Metal."
    )
    w(
        "- **A crossing nobody declared.** Every number here is a floor by exactly that. A family "
        "that crosses two types and carries no row counts for nothing, so this axis grows by "
        "declaration and not by discovery."
    )
    w(
        "- **An axis that is not a Metal enum or option set.** A crossing of a type with a boolean "
        "or numeric Metal field — `alphaToCoverageEnabled`, a LOD clamp, an extent, a reset range "
        "— and a type crossed with **itself in two slots**, which the two stencil faces are, are "
        "real depth and are no pair of this option space. They are **not** uncounted: they are "
        "declared in `other_crossings.tsv` and counted in their own table below, apart from these "
        "numbers and never added into them."
    )
    w(
        "- **An axis that is no Metal item at all.** A pixel format's class, the shape of a "
        "shader's output, whether a fragment writes its own sample mask: neither file can name "
        "one, because neither has anything to check such an axis against. That depth is in the "
        "suite and in no count here."
    )
    w(
        "- **A module that takes its value list from the suite's shared vocabulary** states none of "
        "those values itself, so its cross cannot be declared until the module names them. The "
        "per-value columns above hold such values in their own *shared only* column for the same "
        "reason."
    )
    w(
        "- **Which cell varies which axis.** The values are read from the **module's** text while "
        "the claim is about a cell of it, so a module whose cases each fix one axis reads as "
        "varying both. `family` says outright that the cross is read across the cases; a named "
        "cell is the one case the row's reasoning is read at."
    )
    w(
        "- **A triple told apart from three pairs.** A row of three axes counts as its three "
        "pairs, and an interaction no pair of its axes can see is not distinguished from them."
    )
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| pairs of the option space declared | {len(held)} |")
    w(f"| distinct pairs of Metal types crossed | {len(pairs)} |")
    w(f"| types in at least one such pair | {len(crossed)} |")
    w(
        "| types with two or more values compared and in no crossing of **either** table | "
        f"{len(thin)} |"
    )
    w("")
    if not held:
        w(
            "None are declared, so the pairs of this option space are measured at nothing and "
            "every per-value column above stands alone."
        )
        w("")
        return
    w("| case module | cell | crossed | what only the cross decides |")
    w("|---|---|---|---|")
    for row in sorted(held, key=lambda r: (r["module"], r["cell"], r["axes"])):
        axes = " x ".join(f"`{axis}`" for axis in row["axes"])
        cell = "its cases" if row["cell"] == option_space.FAMILY else f"`{row['cell']}`"
        w(f"| `{row['module']}` | {cell} | {axes} | {row['why']} |")
    w("")
    write_other_crossings(w, other, slots, fields, with_slots, with_fields)
    w("#### Types with per-value depth and no crossing")
    w("")
    w(
        "The work list of this axis, and the shape of the answer to *is the depth sufficient?*: "
        "each of these is a type the suite compared two or more values of and crossed with "
        "nothing in **either** table. The number is the values compared. A row is a question and "
        "not a verdict — the type may be crossed by a family that declared no row, and a type "
        "whose values cannot interact with anything has nothing to answer — but a type deep on "
        "the per-value axis and absent here is where a cell of a pair is least likely to have "
        "been looked at."
    )
    w("")
    for row in thin:
        w(f"- **`{row['metal']}`**, {len(row['compared'])} of {len(row['values'])} compared")
    w("")


def write_other_crossings(w, other, slots, fields, with_slots, with_fields) -> None:
    """The crossings that are not pairs of the option space, counted apart from the pairs."""
    w("#### Crossings that are not pairs of the option space")
    w("")
    w(
        "The table above counts pairs of two Metal **enums**, because that is what the per-value "
        "columns measure the depth of. A crossing of an enum with a Metal field that is a "
        "**boolean** or a **number**, and a crossing of one type with **itself in two or more "
        "slots**, is depth of the same kind and is no pair of that space. These are declared in "
        "`scripts/conformance-coverage/other_crossings.tsv`, in rows of the same four fields, and "
        "counted here. **The two counts are never added together**: one says how much of the "
        "option space's own grid a case puts to Metal, the other how much interaction the suite "
        "holds that the option space cannot express, and a sum of them would say neither."
    )
    w("")
    w(
        "An axis there is a Metal enum with a generated value list, which **may** be named more "
        "than once — that repetition is the type-in-two-slots crossing and the count of names is "
        "the count of slots — or one Metal item of one Metal type, spelled as the items axis keys "
        "it (`MTLRenderPipelineDescriptor.alphaToCoverageEnabled`, "
        "`MTLArgumentEncoder.setArgumentBuffer:offset:`)."
    )
    w("")
    w(
        "**This document is not written** if such a row names a module no case of which ran Metal, "
        "a cell that is not one of that module's recorded cases, fewer than two axes, nothing that "
        "the cross decides, or an **enum** axis of which the module's own text states fewer than "
        "two values by name. Three refusals are this table's own:"
    )
    w("")
    w(
        "- **a row every axis of which is a Metal enum named once.** That is a pair of the option "
        "space: it belongs in the table above, under the check that the module states two values "
        "of each axis, and a row parked here would be the same claim under a weaker rule;"
    )
    w(
        "- **a field axis `objc2-metal` binds no such item for, or one this module's own data "
        "never reaches.** The leg must hold a call of that field at a site whose guards this "
        "module's text satisfies. Reachability is the items axis's own join, so a field axis names "
        "a call this document already reports as exercised rather than a word a row's author "
        "chose; the guards are what make the row a claim about that **field** and not about any "
        "field of its Metal type. One shared encoder sets a pass descriptor's fields in one "
        "function, each under `if let Some(x) = pass.field`, so by reachability alone every case "
        "that opens a pass reaches them all — which would let a case attaching no visibility "
        "buffer declare a crossing on `visibilityResultBuffer`;"
    )
    w(
        "- **a field axis whose call answers a Metal type with a generated value list.** Such a "
        "field is an enum under another name — `MTLTextureDescriptor.storageMode` is one — and "
        "naming it as a field would dodge the two-values check the enum spelling carries."
    )
    w("")
    w(
        "What **cannot** be checked for a field axis, and is therefore what only a reader can "
        "check: that the module put **two values** of the field to Metal. A boolean's two values "
        "and a number's range are nowhere in the text this reads and in no recording's "
        "transcript, so a field axis carries the field being Metal's and this module stating it, "
        "and nothing about variation — and a field a module states as `None` or `false` reads "
        "here exactly like one it states a value for. For the same reason a **bare boolean** "
        "guard in the leg (`if descriptor.read_only {`) is no guard here: its body is reached on "
        "one of the field's two values, and which one a case states is what this cannot see. An "
        "enum axis carries the two-values check here exactly as it does above, which is why a "
        "row's enum axes are the stronger half of it. The last column, as above, is for a reader "
        "against the case."
    )
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| crossings declared that are not pairs of the option space | {len(other)} |")
    w(f"| of those, naming a type in two or more slots | {with_slots} |")
    w(f"| of those, naming a Metal field that is a boolean or a number | {with_fields} |")
    w(f"| distinct Metal fields crossed | {len(fields)} |")
    w(f"| distinct Metal types crossed with themselves | {len(slots)} |")
    w("")
    w(
        "The two *of those* rows are not a partition and do not sum to the first: a row may name "
        "both, as the sampler row crossing its two filter slots with a LOD clamp does."
    )
    w("")
    if not other:
        w(
            "None are declared, so how much interaction the suite holds outside the option "
            "space's own pairs is measured at nothing."
        )
        w("")
        return
    w("| case module | cell | crossed | what only the cross decides |")
    w("|---|---|---|---|")
    for row in sorted(other, key=lambda r: (r["module"], r["cell"], r["axes"])):
        axes = " x ".join(f"`{axis}`" for axis in row["axes"])
        cell = "its cases" if row["cell"] == option_space.FAMILY else f"`{row['cell']}`"
        w(f"| `{row['module']}` | {cell} | {axes} | {row['why']} |")
    w("")


def write_option_space(w, space) -> None:
    """The second axis: of each Metal enum's and option set's values, which the suite compared."""
    rows = space["rows"]
    values = sum(len(row["values"]) for row in rows)
    compared = sum(len(row["compared"]) for row in rows)
    shared = sum(len(row["shared"]) for row in rows)
    missing = sum(len(row["missing"]) for row in rows)
    settled = sum(len(row["settled"]) for row in rows)
    reachable = values - settled

    w("## By option space, which is depth rather than reach")
    w("")
    w(
        "The axis above counts a Metal **item** as exercised where one call of it ran. A selector "
        "that takes an enum therefore counts whatever value reached it, so that axis says nothing "
        "about how much of an enum the suite ever put to Metal. This one does, over the values "
        "themselves."
    )
    w("")
    w(
        "The denominator is generated, never written here: "
        "`crates/metal2vulkan/src/metal/generated.rs` and the front end's hold every enumerator "
        "name and ordinal `scripts/regen-metal-enums/` extracts from the SDK's Metal headers, and "
        "`scripts/conformance-coverage/metal_enumerators.tsv` holds the same for the Metal types "
        "this tree declares by hand instead, which have no generated Rust. A Metal type neither "
        "covers has **no denominator here** and is listed apart, below."
    )
    w("")
    w("A value is **compared** where either of two things holds, which are different claims:")
    w("")
    w(
        "- **answered** — the enumerator's name is in what a site of a committed recording "
        "**observed**, in an answer Metal **ran**. Metal named the value back at a site the "
        "comparison reads, so the value is in the compared text itself."
    )
    site = sum(len(row["at_a_site"]) for row in rows)
    w(
        f"- **at a site** — the enumerator's name is a committed recording's site **name**, and "
        f"in nothing that site observed: {site} of them. The comparison pairs sites by name, so "
        "the value is in the compared text and the case did put it to Metal — but the name is the "
        "**case's** label for the site, so it is no evidence that Metal said the value. It is its "
        "own column for that reason: read together with *answered* it reads as Metal having "
        "named back values that a case spelled itself, which for `MTLPixelFormat` is most of "
        "them."
    )
    w(
        "- **in effect** — a case module's own text names the value, and that module has at least "
        "one case whose recording is a Metal run. Metal's recorded answer was taken with the "
        "value in effect, so what is compared is Metal's behaviour **under** it, even where no "
        "site reads it back. A blend factor is this and not the first: what the recording holds is "
        "the texels Metal blended with it."
    )
    w("")
    w("What this axis cannot see, measured rather than asserted:")
    w("")
    w(
        "- **A value Metal answers as a bare ordinal.** A transcript holds a `uint`, and nothing "
        "in it says which enum the number belongs to, so only a site that records the enumerator's "
        "**name** is read. The *answered* column is a floor by every site that records an ordinal "
        "instead."
    )
    w(
        "- **A spelling of a value this derivation does not read.** Both numerators read names, "
        "and a name written in a shape the reading does not cover is counted nowhere and looks "
        "exactly like a value no case reaches. Four shapes are read: a variant beside its type, "
        "`objc2`'s own spelling, a renaming `use`, and a braced `use` whose variants the module "
        "then writes bare. A fifth shape would be invisible the same way, so every count here is "
        "a floor by that unknown — and the two shapes already found were each caught by someone "
        "checking a count against the code, not by this derivation, so the unknown is not small "
        "on the evidence."
    )
    w(
        "- **A value a case states as a raw ordinal.** A case may hand the Metal leg an enum's "
        "ordinal as a number — a sweep over `0..n` rather than over named variants — and no "
        "reading of the text can say which value a computed number is. So *in effect*, and the "
        "*stated* column of the hand-modelled table below, are floors by every sweep written that "
        "way: a type they report short of its values may in fact be swept exhaustively, and the "
        "case is what says so. Read the case before taking a row of the work list as work."
    )
    w(
        "- **Which case put a value in effect.** The unit is the case module, as it is for the "
        "items axis: a value named anywhere in a module whose cases include a Metal run counts for "
        "that module, even where one of its cases names it and the rest do not."
    )
    w(
        f"- **A value only the suite's shared sources name** — {shared} of them. Some case puts it "
        "to Metal through the render or texture vocabulary, but which case is not a property of "
        "the text, so it is neither compared nor missing and has its own column."
    )
    w(
        "- **Combinations.** Every column here is per value. A type whose values are all compared "
        "says nothing about the pairs and triples of them, which is where most of the behaviour "
        "an implementation can get wrong lives. *By crossing* below counts what a case states "
        "together, in two numbers kept apart: the pairs of this option space, and the crossings "
        "that are no pair of it — an enum with a boolean or numeric field, a type with itself in "
        "two slots. Each is a floor by every crossing nobody declared."
    )
    w(
        "- **Metal's API validation is excluded.** Its complaints name enumerators, and the suite "
        "records that leg and never compares it, so reading it would count Apple's prose as a "
        "comparison. So are the cost and timing tables: a site name there is what was measured, "
        "not an answer that was compared."
    )
    w(
        "- **A `#[cfg(test)]` item is excluded.** One of the suite's unit tests names four blend "
        "factors to assert that no case uses them; counting those would have this report the "
        "opposite of what the code says."
    )
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| Metal enums and option sets with a generated value list | {len(rows)} |")
    w(f"| their values | {values} |")
    w(f"| of those, unreachable: no case can put them to Metal | {settled} |")
    w(f"| so reachable by some case | {reachable} |")
    w(f"| values compared | {compared} |")
    w(f"| share of the reachable compared | {100.0 * compared / reachable:.0f}% |")
    w(f"| values only the suite's shared sources name | {shared} |")
    w(f"| values no source names and no recording holds | {missing} |")
    # "every REACHABLE value", because `MTLPixelFormat` has 266 compared and 35 that no case can
    # put to Metal, and calling that "every value compared" would be false of the type.
    whole = sum(1 for row in rows if not row["missing"] and not row["shared"])
    w(f"| types every reachable value of which is compared | {whole} |")
    w(f"| types no value of which is compared | {sum(1 for r in rows if not r['compared'])} |")
    w("")
    largest = sorted(rows, key=lambda r: -len(r["values"]))[:4]
    w(
        "The share is weakly informative on its own: "
        + ", ".join(f"`{row['metal']}`" for row in largest)
        + f" hold {sum(len(row['values']) for row in largest)} of the {values} values between "
        "them, so an aggregate over types this uneven is mostly theirs. The per-type table is the "
        "one to read."
    )
    w("")
    w(
        "### Depth by type"
    )
    w("")
    w(
        "Deepest gap first, which is the order the work list reads in. *answered*, *at a site* "
        "and *in effect* overlap; *compared* is their union."
    )
    w("")
    w(
        "| Metal type | values | compared | answered | at a site | in effect | shared only | "
        "unreachable | missing |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    for row in sorted(rows, key=lambda r: (-len(r["missing"]), -len(r["shared"]), r["metal"])):
        w(
            f"| `{row['metal']}` | {len(row['values'])} | {len(row['compared'])} | "
            f"{len(row['answered'])} | {len(row['at_a_site'])} | {len(row['in_effect'])} | "
            f"{len(row['shared'])} | {len(row['settled'])} | {len(row['missing'])} |"
        )
    w("")
    if settled:
        w("### The values no case can put to Metal")
        w("")
        w(
            "Counted apart from the gap above, because a gap is work and these are not. Each is "
            "declared in `scripts/conformance-coverage/unreachable_values.tsv` with the "
            "measurement that settled it and the case module that holds the finding, and this "
            "document is **not** written if a value listed there turns out to be one some case "
            "compared, or one its type does not have. Nothing is listed for being obscure, out of "
            "scope, or merely unbuilt: a Metal value this suite has simply not reached yet is "
            "`missing`, which is the column to read as work."
        )
        w("")
        by_reason: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for (metal, value), (why, finding) in sorted(space["unreachable"].items()):
            by_reason[(metal, why, finding)].append(value)
        for (metal, why, finding), named in sorted(by_reason.items()):
            w(f"- **`{metal}`**, {len(named)} of {len(next(r for r in rows if r['metal'] == metal)['values'])}: {why}.")
            w(f"  Held by `{finding}`: " + ", ".join(f"`{value}`" for value in named))
        w("")
    write_crossings(w, space)
    unreadable = [row for row in rows if row["named_no_value"]]
    if unreadable:
        w("### Types a case is about, whose values this axis cannot read")
        w("")
        w(
            "The raw-ordinal limit above as a work list rather than a sentence. Each row is a type "
            "whose name a recorded case module uses and **none of whose values that module states "
            "by name**: a sweep written `for bits in 0..8` reads exactly this way, and so does a "
            "module that only mentions the type in passing. So a row is a question for the case, "
            "never a verdict on it -- the module may put every value of the type to Metal, and "
            "this axis would still count none of them."
        )
        w("")
        w(
            "It names the type and the module, and no value: which value a computed number is is "
            "not a property of the text, and a rule that answered it would be inventing the "
            "answer. Closing a row means the case stating its values by name, or a transcript site "
            "recording the enumerator beside the ordinal it already stores."
        )
        w("")
        w("| Metal type | values | compared | modules naming it that state no value |")
        w("|---|---|---|---|")
        for row in sorted(
            unreadable, key=lambda r: (len(r["compared"]), -len(r["values"]), r["metal"])
        ):
            modules = ", ".join(f"`{m}`" for m in row["named_no_value"])
            w(
                f"| `{row['metal']}` | {len(row['values'])} | {len(row['compared'])} | "
                f"{modules} |"
            )
        w("")

    w("### The values not compared")
    w("")
    w(
        "The work list. A name here is a value of Metal's that no committed recording holds and no "
        "case module states **by name**: it is not a claim that Metal accepts the value on every "
        "device, and not a claim that no case reaches it, because a case that sweeps a type by "
        "ordinal states none of its values here. Each row is a question for the case, not a "
        "verdict on it."
    )
    w("")
    for row in sorted(rows, key=lambda r: (-len(r["missing"]), r["metal"])):
        if not row["missing"]:
            continue
        w(f"- **`{row['metal']}`**, {len(row['missing'])} of {len(row['values'])}: "
          + names_line(row["missing"]))
    w("")
    only = [row for row in rows if row["shared"]]
    if only:
        w(
            "And the values only the suite's shared sources name, which some case reaches and this "
            "cannot attribute:"
        )
        w("")
        for row in sorted(only, key=lambda r: r["metal"]):
            w(f"- **`{row['metal']}`**: " + names_line(row["shared"]))
        w("")
    w("### Metal types with no generated value list")
    w("")
    w(
        "These the tree models by hand: a Rust item whose own doc comment opens by naming a Metal "
        "type, and whose shape is a closed enum or a newtype over one integer. The tree declares "
        "them itself because the compile-and-plan layer never reads them, and that stays — but a "
        "declaration measured against itself is no measurement, so "
        "`scripts/regen-metal-enums/`'s `TABLE_ONLY` list generates "
        "`scripts/conformance-coverage/metal_enumerators.tsv`, the SDK's own values for them, and "
        "every type it covers is in the per-type table above with a real denominator. What is left "
        "here is what the table does not cover."
    )
    w("")
    w(
        "For a row here **this axis has no denominator**: *declared* is our own sources' variant "
        "count rather than the SDK's — a newtype option set declares none — and there is no "
        "*answered* column, because a recording names a value by its Metal enumerator and this has "
        "no list of those names to match against."
    )
    w("")
    w(
        f"{len(space['by_hand'])} are left, and the list is a floor: a type the tree models "
        "without saying so on the item lands in neither table, and so is nowhere in this axis at "
        "all."
    )
    w("")
    w("| Metal type | the tree's type | declared | stated by a recorded case | shared only |")
    w("|---|---|---|---|---|")
    for row in space["by_hand"]:
        w(
            f"| `{row['metal']}` | `{row['rust']}` | {len(row['declared']) or '— a bit set'} | "
            f"{len(row['stated'])} | {len(row['shared'])} |"
        )
    w("")


def write(data) -> str:
    """The document's text."""
    rows, index, api, leg = data["rows"], data["index"], data["api"], data["leg"]
    families, records, by_item = data["families"], data["records"], data["by_item"]
    # One row per `(container, name)`, not one per row of the map: it holds
    # `MTLRasterizationRateLayerDescriptor` twice, once as a class and once as a category on it,
    # and every number here is over keys. Counting the rows instead states a denominator the
    # numerator cannot reach, and `item_index` resolves a call to the later row anyway.
    implemented = list(
        {
            (row["container"], row["name"]): row for row in rows if row["status"] == "I"
        }.values()
    )
    keys = {(row["container"], row["name"]) for row in implemented}
    unreachable = set(data["unbound"])
    exercised = {key for key in by_item if key in keys}
    cases_of = defaultdict(list)
    for record in records:
        cases_of[record["module"]].append(record)

    out: list[str] = []
    w = out.append
    w("# Metal conformance coverage")
    w("")
    w(
        "Generated by `scripts/conformance-coverage/coverage.py` from the suite's own sources and "
        "its committed recordings. Do not edit this file; change the sources or the suite and run "
        "the script."
    )
    w("")
    w(
        "This is not [METAL-COVERAGE.md](METAL-COVERAGE.md), which says what we **implement**. "
        "This says what the conformance suite **compares against Metal**. An item counts as "
        "*exercised* here only where a case's native Metal leg calls it; a case's Metal answer is "
        "a recorded fact only where the recording is committed. The two claims are different "
        "sizes, and only the second is conformance."
    )
    w("")
    w(
        "It says it along **two axes**, because either one alone reads as more than it is. The "
        "**items** axis is reach: one call of a selector is enough, so a selector that takes an "
        "enum counts as exercised whatever value reached it. The **option space** axis is depth: "
        "of each Metal enum's and option set's values, which ones the suite compared. A high "
        "number on the first and a low one on the second is what a suite that touches everything "
        "shallowly looks like."
    )
    w("")
    w(
        "Both are **per value**, which is the limit the suite's own blind spot sits in: a side can "
        "be right about every value of two types alone and wrong about a cell of the two together. "
        "*By crossing* inside the option-space axis counts what a case puts to Metal at once, "
        "from declarations the generator checks: the pairs of the option space, and — counted "
        "apart and never added in — the crossings that are no pair of it. They are the only "
        "numbers here about interactions, and each is a floor by every crossing nobody declared."
    )
    w("")

    w("## What this measures, and what it cannot")
    w("")
    w(
        "The suite reaches Metal through `objc2-metal`, whose Rust name for a binding is the "
        "selector with each `:` written `_`. So a call site names its selector outright, and the "
        "derivation resolves the **receiver's type** by following what made it — a class's "
        "constructor, a getter whose answer `objc2_metal.tsv` records, a field of the side's own "
        "struct, a parameter's declared type — and then walks that type's supertypes to the "
        "container the SDK declares the item on. A message to a class, written "
        "`MTLType::selector(..)`, names its own receiver and needs no typing."
    )
    w("")
    w("Its limits, measured rather than asserted:")
    w("")
    sites = len(leg.calls)
    typed = len(data["keyed"])
    w(
        f"- **{typed} of {sites} call sites** in the Metal leg resolve to an item of the SDK. "
        f"{len(leg.by_uniqueness)} of those resolve because only one type in `objc2-metal` binds "
        "the name at all, so no typing was needed."
    )
    w(
        f"- **{len(leg.foreign)} call sites send a selector `objc2-metal` binds on no Metal type "
        "at all**, so they are an item of no Metal container and there is no count for them to "
        "cost: " + ", ".join(f"`{rust}`" for rust in sorted({r for _, _, r, _ in leg.foreign}))
        + ", which are `NSObject`'s."
    )
    # The sites that DO cost something, stated in items rather than selector names. Listing the
    # selector names alone -- which this did until 2026-10-24 -- hides exactly what a reader wants:
    # eight items of one family read as uncalled while the leg demonstrably called them, and the
    # document said only that some selectors were ambiguous.
    #
    # The careful part is the bound. An untyped site sent its selector to ONE receiver, so one of
    # the items binding that name is exercised -- but where another call site already credits one
    # of them, that may be the one, and the site proves nothing. What proves something is a
    # selector NONE of whose candidates anything else credits: then whichever type received it,
    # an item that reads here as uncalled was called. Candidates of different selectors are
    # disjoint, so each such selector contributes at least one.
    credited = set(data["called"])
    candidates: dict[str, set[tuple[str, str]]] = {}
    for _, _, rust, _ in leg.unresolved:
        if rust not in candidates:
            candidates[rust] = {
                item
                for ty in api.receivers(rust)
                if (item := api.binds(ty, rust)) and item in keys
            }
    blind = {rust: items for rust, items in candidates.items() if items and not items & credited}
    hidden = sorted({item for items in blind.values() for item in items})
    head = (
        f"- **{len(leg.unresolved)} call sites resolve to no item although some Metal type binds "
        "the selector**: an instance send whose receiver this could not type, or a message to a "
        "class that does not bind the name sent to it. Each sent its selector to exactly one "
        "receiver, so each exercised one item; which one is what the typing could not say."
    )
    if blind:
        w(
            head + f" {len(blind)} of the selectors they send are bound only by items **nothing "
            "else credits**, so whichever type received those calls, an item that reads below as "
            f"not exercised was exercised: at least {len(blind)} of the {len(hidden)} candidates "
            "listed here, and at most all of them."
        )
        for item in hidden:
            w(f"  - `{item[0]}`'s `{item[1]}`")
        w(
            "  So every per-item count below is a **floor**, by at least these. To close one, "
            "spell the receiver's type at the call site — a `let` with a type, rather than a "
            "pattern binding or an inferred one — and this re-derives with the item credited to "
            "the case that calls it."
        )
    else:
        w(
            head + " **Every one of their selectors is bound by at least one item some other call "
            "site credits**, so none of these sites proves an item is missing from the count. That "
            "is not a proof that the count is complete: such a site may have sent its selector to "
            "a type other than the credited one, and this cannot tell. To rule it out, spell the "
            "receiver's type at the call site — a `let` with a type, rather than a pattern binding "
            "or an inferred one — and this re-derives with the item credited to the case that "
            "calls it."
        )
    w(
        "- **0 call sites typed to a receiver that does not bind the selector sent to it.** The "
        "generator fails rather than writing this document if any does, which is what keeps the "
        "typing from quietly counting the wrong item."
    )
    selves, frees = data["selves"], data["frees"]
    w(
        f"- **{len(selves)} of the map's implemented rows are a container's own**: the `class`, "
        "`protocol` and `category` rows, whose name is the container's and not a selector's. No "
        "call site can resolve to one, because `objc2-metal` binds no selector by that name, so "
        "these are credited by the suite **messaging** the type instead — a call whose receiver is "
        "the container or a subtype of it — under the same gate and the same family attribution as "
        "the selector it sends."
    )
    w(
        f"- **{len(frees)} are the header's free functions**, which no container declares. "
        "`objc2-metal` binds these as Rust functions, so the receiver typing never sees one; the "
        "leg calls such a function by its own C name and the walk reads it as an edge of the call "
        "graph."
    )
    unreach = data["unbound"]
    w(
        f"- **{len(unreach)} implemented items `objc2-metal` binds no name for at all**, so no "
        "case can call them whatever it is authored to do. They are counted apart from the gap "
        "below rather than folded into it, because a gap is work and these are not:"
    )
    for container, name in unreach:
        w(f"  - {_container(container)}'s `{name}`")
    w(
        "- The unit a call is attributed to is the **case module**, not the case: a module's cases "
        "share one `run`, so which of them encodes which step is not a property of the text. "
        "Where a module holds one case the two are the same."
    )
    w(
        "- **No verdict is recorded here.** A verdict is what a replay on one venue measured, and "
        "a number like that in a generated document is a journal entry that rots. What is durable, "
        "and is here, is whether a case compares **exactly**: the allowances a module declares, "
        "and the rows of the suite's ledger."
    )
    w("")

    w("## The headline")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| implemented Metal items | {len(implemented)} |")
    w(f"| of those, unreachable: `objc2-metal` binds no name for them | {len(unreachable)} |")
    w(f"| so reachable by some case | {len(keys) - len(unreachable)} |")
    w(f"| of the reachable, exercised by the suite's Metal leg | {len(exercised)} |")
    w(f"| of the reachable, not exercised | {len(keys) - len(unreachable) - len(exercised)} |")
    share = 100.0 * len(exercised) / (len(keys) - len(unreachable))
    w(f"| share of the reachable exercised | {share:.0f}% |")
    loose = {
        key
        for key in exercised
        if any(families[module]["allowances"] for module in by_item[key])
    }
    w(f"| of those exercised, no case reaching it loosens anything | {len(exercised) - len(loose)} |")
    w(f"| of those exercised, a case reaching it loosens some site | {len(loose)} |")
    w(f"| case modules | {len(families)} |")
    w(f"| committed recordings | {len(records)} |")
    w(
        f"| distinct SDK items the Metal leg reaches, of any status | {len(data['called'])} |"
    )
    w(
        f"| of those, reached only where this cannot name the family | "
        f"{len(data['unattributed'])} |"
    )
    w(f"| ledger rows (a case whose gap is recorded) | {len(data['gaps'])} |")
    space = data["option_space"]
    w(
        "| declared crossings of the option space (pairs put to Metal at once) | "
        f"{len(space['crossings'])} |"
    )
    w(
        "| declared crossings that are no pair of it (a boolean or numeric field, or one type in "
        f"two slots) | {len(space['other_crossings'])} |"
    )
    w("")
    w(
        "A tolerance names a **site of a transcript**, not an item, so the last two rows are read "
        "as written: a case that loosens one site of its own transcript may compare every other "
        "field of it exactly, and this derivation cannot tell which of a case's fields came from "
        "which Metal call. Where the distinction matters, read the case."
    )
    w("")
    w(
        "*Exercised* and *recorded* are one number here, not two, and deliberately: a case is "
        "credited with exercising an item only through its cases whose answer record mode took by "
        "**running** Metal, and such a case has a committed recording by construction. A case that "
        "carries an answer, or whose call Metal never returned from, runs no native leg and "
        "exercises nothing, however much Metal its code names."
    )
    w("")

    write_option_space(w, data["option_space"])

    w("## By container")
    w("")
    w(
        "Every container with an implemented item, by how much of it the suite reaches. *no "
        "binding* is the container's share of the unreachable items above, held out of "
        "*unexercised* so each ratio reads as what authoring could close."
    )
    w("")
    w("| container | implemented | no binding | exercised | unexercised | families |")
    w("|---|---|---|---|---|---|")
    by_container: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in implemented:
        by_container[row["container"]].append(row)
    def container_left(container: str) -> int:
        return sum(
            1
            for row in by_container[container]
            if (row["container"], row["name"]) not in exercised
            and (row["container"], row["name"]) not in unreachable
        )
    order = sorted(by_container, key=lambda c: (-container_left(c), c))
    for container in order:
        rows_here = by_container[container]
        hit = {key for key in exercised if key[0] == container}
        blocked = sum(1 for key in unreachable if key[0] == container)
        modules = sorted({m for key in hit for m in by_item[key]})
        w(
            f"| {_container(container)} | {len(rows_here)} | {blocked} | {len(hit)} | "
            f"{container_left(container)} | "
            + (", ".join(f"`{m}`" for m in modules[:4]) + ("…" if len(modules) > 4 else ""))
            + " |"
        )
    w("")

    w("## By header, which is the family of behaviour")
    w("")
    w(
        "A container is one Objective-C type; the SDK header it sits in is the family of Metal "
        "behaviour. This is the table to pick a batch from: a header with nothing exercised is a "
        "family the suite says nothing about, and the rest are ordered by how much of each is left."
    )
    w("")
    w("| header | implemented | exercised | unexercised |")
    w("|---|---|---|---|")
    by_header: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in implemented:
        by_header[row["header"]].append(row)
    def left(header: str) -> int:
        return sum(
            1
            for row in by_header[header]
            if (row["container"], row["name"]) not in exercised
            and (row["container"], row["name"]) not in unreachable
        )
    for header in sorted(by_header, key=lambda h: (-left(h), h)):
        hit = len(by_header[header]) - left(header)
        w(f"| `{header}` | {len(by_header[header])} | {hit} | {left(header)} |")
    w("")
    empty = [h for h in sorted(by_header) if left(h) == len(by_header[h])]
    w(
        f"{len(empty)} of {len(by_header)} headers have no exercised item at all, "
        f"{sum(len(by_header[h]) for h in empty)} implemented items between them: "
        + ", ".join(f"`{h}`" for h in empty)
        + "."
    )
    w("")

    w("## Containers with no case at all")
    w("")
    w(
        "An implemented container no case's Metal leg touches. These are the families of Metal "
        "behaviour the suite says nothing about."
    )
    w("")
    w("| container | implemented items | header |")
    w("|---|---|---|")
    untouched = [c for c in sorted(by_container) if not any(k[0] == c for k in exercised)]
    for container in untouched:
        header = by_container[container][0]["header"]
        w(f"| {_container(container)} | {len(by_container[container])} | `{header}` |")
    w("")
    w(f"{len(untouched)} containers, {sum(len(by_container[c]) for c in untouched)} implemented items.")
    w("")

    w("## The suite's shape")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| case modules | {len(families)} |")
    w(f"| committed recordings | {len(records)} |")
    runs = sorted(record["runs"] for record in records)
    w(f"| recorded runs behind a case, median | {runs[len(runs) // 2]} |")
    w(f"| recorded runs, least and most | {runs[0]} and {runs[-1]} |")
    w(f"| cases whose recording holds more than one outcome | {sum(1 for r in records if r['outcomes'] > 1)} |")
    w(f"| cases running a committed shader fixture | {sum(1 for r in records if r['shaders'])} |")
    provenance = Counter(kind for record in records for kind in record["provenance"])
    for kind, count in provenance.most_common():
        w(f"| cases whose Metal answer is `{kind}` | {count} |")
    w(f"| modules that loosen a measured value | {sum(1 for f in families.values() if f['allowances'])} |")
    w(f"| tolerances in all | {sum(f['allowances'] for f in families.values())} |")
    w(f"| modules declaring a notice (loosening no value) | {sum(1 for f in families.values() if f['declarations'])} |")
    loosenings = Counter(k for f in families.values() for k in f["loosenings"])
    w(f"| loosenings used | {', '.join(f'`{k}`' for k, _ in loosenings.most_common()) or 'none'} |")
    justs = Counter(k for f in families.values() for k in f["justifications"])
    w(f"| justifications named | {', '.join(f'`{k}`' for k, _ in justs.most_common()) or 'none'} |")
    w(f"| modules forbidding a re-record | {sum(1 for f in families.values() if f['carried_only'])} |")
    w(f"| `Side` methods (the suite's whole device vocabulary) | {len(data['side'])} |")
    w(f"| `Side` methods the Metal leg reaches an SDK item through | {sum(1 for m in data['gated'] if data['gated'][m])} |")
    per_case = sorted(
        len({key for key in by_item if module in by_item[key]})
        for record in records
        for module in [record["module"]]
    )
    w(f"| SDK items a case's Metal leg reaches, median | {per_case[len(per_case) // 2]} |")
    w(f"| the same, least and most | {per_case[0]} and {per_case[-1]} |")
    w(f"| Metal call sites in the leg | {len(leg.calls)} |")
    w(f"| distinct selectors the leg sends | {len({c.rust for c in leg.calls})} |")
    w(f"| of those, resolved to an item of the coverage map | {len({c.rust for c in leg.calls if id(c) in data['keyed']})} |")
    chips = Counter(chip for record in records for chip in record["chips"])
    w(f"| oracles the recordings name | {', '.join(f'`{c}` ×{n}' for c, n in chips.most_common())} |")
    w("")

    w("## By family")
    w("")
    w(
        "Each case module: the cases it holds, the `Side` methods it calls, and how many "
        "implemented items its Metal leg reaches through them."
    )
    w("")
    w("| family | cases | Metal ran | compared | items reached |")
    w("|---|---|---|---|---|")
    for module in sorted(families):
        family = families[module]
        here = {key for key in exercised if module in by_item[key]}
        exact = (
            "every field"
            if not family["allowances"]
            else f"{family['allowances']} tolerance(s): "
            + ", ".join(f"`{k}`" for k in family["justifications"])
        )
        w(
            f"| `{module}` | {len(family['cases'])} | {len(family['ran'])} | {exact} | "
            f"{len(here)} |"
        )
    w("")

    w("## Every implemented item")
    w("")
    w(
        "One row per implemented item of the SDK, in the coverage map's order. *exercised* is "
        "whether a case's native Metal leg calls it; *families* are the case modules that reach "
        "it; the last column is how many of their cases record an answer Metal actually ran."
    )
    w("")
    for container in sorted(by_container):
        w(f"### {_container(container)}")
        w("")
        w("| item | exercised | families | cases Metal ran |")
        w("|---|---|---|---|")
        for row in by_container[container]:
            key = (container, row["name"])
            modules = by_item.get(key, {})
            count = sum(len(families[module]["ran"]) for module in modules)
            w(
                f"| `{row['name']}` | {'yes' if key in exercised else 'no'} | "
                f"{families_line(modules) or '—'} | {count or '—'} |"
            )
        w("")

    w("## Exercised, but not ours")
    w("")
    w(
        "An item the suite's Metal leg calls whose coverage-map status is not implemented. Read "
        "each one before concluding anything: a call the leg makes to **set its own Metal up** is "
        "not a behaviour under test, and our side need not have it. A call the case's answer "
        "depends on is the other thing, and would be a case that can only be exact by accident."
    )
    w("")
    w("| item | status | note | families |")
    w("|---|---|---|---|")
    for key in sorted(by_item):
        row = index.get(key)
        if row is None or row["status"] == "I":
            continue
        w(
            f"| `{key[0]}` `{key[1]}` | {STATUS[row['status']]} | {row['note'] or '—'} | "
            f"{families_line(by_item[key])} |"
        )
    w("")

    w("## Called, attributed to no family")
    w("")
    w(
        "An item the Metal leg calls that no case module was found to reach: a `Side` method no "
        "case calls, a step of the vocabulary no case encodes, or a call the leg makes outside "
        "every `Side` method — the side's own constructor, which runs for every case and which "
        "this walk has no family to attribute, because a family is read off which `Side` methods "
        "a case's text calls. The first two are suite code that no case runs; the third is "
        "measured and named here rather than counted as exercised."
    )
    w("")
    if data["unattributed"]:
        w("| item | status |")
        w("|---|---|")
        for key in data["unattributed"]:
            row = index.get(key)
            w(
                f"| {_container(key[0])} `{key[1]}` | "
                f"{STATUS[row['status']] if row else 'not in the map'} |"
            )
    else:
        w("None: every item the leg calls is reached by a case module.")
    w("")

    if data["unlisted"]:
        w("## Called, not in the coverage map")
        w("")
        w(
            "The Metal leg sends these selectors to a receiver this typed, and "
            "`scripts/metal-coverage/coverage.tsv` holds no row for the item. Either the selector "
            "is `NSObject`'s, which `objc2-metal` generates on every class it binds and no Metal "
            "header declares, or the map is short of the SDK, or the name is spelled differently "
            "there; the last two are worth a look."
        )
        w("")
        w("| receiver | selector |")
        w("|---|---|")
        for ty, selector in data["unlisted"]:
            w(f"| `{ty}` | `{selector}` |")
        w("")

    w("## The ledger")
    w("")
    w(
        "A case whose difference from Metal is recorded rather than compared away. The ledger "
        "fails a run in both directions, so a row that no longer holds is a report of its own."
    )
    w("")
    if data["gaps"]:
        w("| case | waiting on |")
        w("|---|---|")
        for case, waiting in data["gaps"]:
            w(f"| `{case}` | {waiting[:160]}{'…' if len(waiting) > 160 else ''} |")
    else:
        w("No rows.")
    w("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------
# Entry points


def check_api() -> int:
    """Hold `objc2_metal.tsv` to what `objc2-metal`'s own sources say."""
    found = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--manifest-path", str(SUITE / "Cargo.toml")],
        capture_output=True,
        text=True,
        check=True,
    )
    meta = json.loads(found.stdout)
    roots = [
        Path(package["manifest_path"]).parent / "src"
        for package in meta["packages"]
        if package["name"] == "objc2-metal"
    ]
    if not roots:
        print("objc2-metal is not in this workspace's dependency graph")
        return 1
    import objc2_items

    types, calls = objc2_items.rows(roots[0])
    want = "# type\tkind\tsupertypes\n" + "\n".join(types) + "\n# type\tselector\trust\tanswers\n"
    want += "\n".join(calls) + "\n"
    if API.read_text() != want:
        print(
            f"{API.relative_to(ROOT)} is not what objc2-metal's sources say: run\n"
            f"  python3 -I scripts/conformance-coverage/objc2_items.py {roots[0]} > "
            f"{API.relative_to(ROOT)}"
        )
        return 1
    print(f"{API.relative_to(ROOT)}: {len(calls)} bindings, as objc2-metal declares them")
    return 0


def selftest() -> int:
    """Run the derivations over inputs whose answers are known.

    Two modules: the option-space axis, and the Metal leg's call graph, which is what attributes a
    call to a case module. Both are derivations over text, and both fail by quietly matching
    nothing, so each builds its own little tree where the answer is known by construction.
    """
    import unittest

    import test_metal_leg
    import test_option_space

    loader = unittest.TestLoader()
    loaded = unittest.TestSuite(
        loader.loadTestsFromModule(module) for module in (test_option_space, test_metal_leg)
    )
    found = unittest.TextTestRunner(verbosity=2).run(loaded)
    return 0 if found.wasSuccessful() else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    if "--api" in argv:
        return check_api()
    text = write(inventory())
    if "--check" in argv:
        if not DOC.exists() or DOC.read_text() != text:
            print(
                f"{DOC.relative_to(ROOT)} is not what the suite's sources and recordings derive: "
                "run\n  python3 -I scripts/conformance-coverage/coverage.py"
            )
            return 1
        print(f"{DOC.relative_to(ROOT)}: as derived")
        return 0
    DOC.write_text(text)
    print(f"wrote {DOC.relative_to(ROOT)} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
