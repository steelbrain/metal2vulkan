"""The suite's depth in each Metal enum and option set: which of a type's values it compared.

`coverage.py`'s other axis is **reachability**: an item counts as exercised where a case's Metal
leg calls it, so one call is enough and a selector that takes an enum says nothing about how many
of that enum's values ever reached Metal. This is the second axis, over the values themselves.

The denominator is generated, never written here: `crates/metal2vulkan/src/metal/generated.rs` and
`crates/metal2vulkan-air/src/metal/generated.rs` hold every enumerator name and ordinal of the
Metal types `scripts/regen-metal-enums/` extracts from the SDK headers. A Metal type the tree
models by hand instead has **no denominator here** and is reported as such rather than guessed at;
the fix is appending its name to that script's lists and regenerating.

The numerator is two derived sets, which are different claims and are kept apart:

- **answered** — the enumerator's name appears in a committed recording's transcript of an answer
  Metal **ran**. Metal named the value back to us at a site the comparison reads, so the value is
  in the compared text itself.
- **in effect** — a case module's own text names the value, and that module has at least one case
  whose recording is a Metal run. Metal's recorded answer was taken with the value in effect, so
  the comparison is of Metal's behaviour **under** that value even where no site reads it back.

`compared` is their union. A value that only the suite's **shared** sources name (a default of the
render vocabulary, say) is counted in neither: some case puts it to Metal, but which case is not a
property of the text, so it is reported in its own column and is neither compared nor missing.

A case states most values by a Metal type's own variant, but not all of them: the suite has enums
of its own for what a case asks a resource for, and a storage mode is written `Storage::Private`
and never `MTLStorageModePrivate`. `vocabulary` reads those spellings off the vocabulary enum's own
declaration -- each variant opens its doc with the Metal enumerator it is -- so a case naming one
counts as a case naming the value. Without it `shared_creator_formats`' exhaustive sweep of all
four storage modes over 24 recorded cells counted for nothing and `MTLStorageMode` read `0 of 4`.

# What the floor is made of

Every column here is a floor, and the honest thing is to say what of rather than only that it is
one. Three things, and as far as has been measured no fourth:

1. **A value a case states as a raw ordinal**, which the next section is about. Measured where it
   can be: the document lists each type a recorded module names and states no value of by name,
   which is how a sweep over `0..n` reads, so the floor has a work list and not just a warning.
2. **A spelling of a value that `_stated_in` does not read.** A name written in a shape the
   reading does not cover is counted nowhere and looks exactly like a value no case reaches.
   `_stated_in` lists every shape read, `_resolved` is the one rule the type-keyed ones go
   through, and `vocabulary` is the one rule the suite's own spellings go through, so a shape on
   neither list is the floor. Three have been found, each by someone checking a printed count
   against the code rather than by this file, which is the evidence for taking the unknown
   seriously rather than calling it small.
3. **A value a case reaches by naming a shared item rather than a value.** `vocabulary` reads the
   suite's own **enums**, because a variant of one is a value's name under another spelling. A
   shared `const` or `fn` is not: `heap_facts` and `resource_facts` sweep `MTLPurgeableState`
   through `heap_query::purgeable_pairs()` and `MTLCPUCacheMode` through its `CACHE_MODES`, and
   nothing in those modules' text names a state or a cache mode. Following a name to the item it
   imports would be a resolution of the suite's text and not a spelling of a value, so it is not
   read here and those values stay in the *shared only* column, which is where the document says
   a value some case reaches and no module can be credited with belongs.

What is **not** part of the floor, because it is measured: the generated denominator (every type
`regen-metal-enums` covers, by `ENUMS`, `OPTIONS` or `TABLE_ONLY`), the `#[cfg(test)]` exclusion,
Metal's validation leg, and which case put a value in effect. A type the tree models by hand and
says so nowhere on the item is outside both tables and so outside the axis, which the document
states where it lists them.

# The crossings, which are the pairs rather than the values

Every column above is per value, and a type whose values are all compared says nothing about the
pairs of them. `crossings.tsv` is the declaration of the pairs: one row a crossing, naming the case
module, the cell the falsification is read at, the Metal types crossed, and what only the cross
decides. It is a **declaration**, so what makes it worth reading is not the list but what the
derivation refuses:

- a module no case of which ran Metal, and a cell that is not one of that module's recorded cases;
- an axis that is not a Metal enum or option set with a generated value list;
- fewer than two axes, or one axis twice;
- an axis of which the module's own text states fewer than two values by name, which is the one
  check that bears on the claim itself: a module that cannot vary the axis cannot hold a cross over
  it, so a single-value falsification would satisfy the row.

What the derivation cannot check is whether the expected answer **needs** both axes. That is a
property of the case's reasoning, not of its text, so the row states it in words for a reader and
the count is of crossings declared, never of crossings proved. The document says so where it
prints the number.

# The crossings that are not pairs of the option space

A crossing of a Metal enum with a **boolean** field (`alphaToCoverageEnabled`), with a **number**
(an extent, a level count, an offset, a sample count), or of one type with **itself in two slots**
(the two stencil faces, a sampler's three address modes, two members of an argument buffer) is real
depth and is not a pair of the option space. `crossings.tsv` refuses every one of those, which left
the shape of them stated as a limit and measured at nothing: a reader could not tell whether one
such crossing or forty fell outside the count.

`other_crossings.tsv` is where they are declared, counted **apart** from the pairs and never added
into one number -- the same separation `unreachable_values.tsv` keeps between a measured
impossibility and a gap. Its rows take the same four fields, and the axes are of two kinds:

- a Metal enum or option set with a generated value list, which **may** be named more than once:
  that repetition is the type-in-two-slots crossing, and the count of slots is the count of names;
- `MTLType.item`, one Metal item of one Metal type, spelled as the items axis keys it --
  `MTLRenderPipelineDescriptor.alphaToCoverageEnabled`,
  `MTLArgumentEncoder.setArgumentBuffer:offset:`.

What this second file refuses, beyond the shape checks the first one makes:

- a row every axis of which is a Metal enum named once, which is a pair of the option space and
  belongs in `crossings.tsv` under its checks, not here under weaker ones;
- an `MTLType.item` axis `objc2-metal` binds no such item for, and one **this module's Metal leg
  does not reach** -- the same join the items axis makes, so a field axis names a call the suite
  is already measured to make;
- an `MTLType.item` axis whose call **answers a Metal type with a generated value list**. Such a
  field is an enum under another name, and declaring it here would dodge the pairs file's check
  that the module states two of its values. `MTLTextureDescriptor.storageMode` is one.

What it cannot check, and what no reading of the sources could: that the module puts **two values**
of a boolean or numeric field to Metal. The enum axes carry that check here as they do in the pairs
file -- the module's own text must state two values by name -- and a field axis carries only that
the field is Metal's and that this module's Metal leg sets it. So a field axis is a weaker
declaration than an enum axis, which is why the two counts stay apart and why the document says
which part only a reader can check.

# Why a value stated as a raw ordinal is not resolved, and must not be

Both numerators read **names**. A case may instead hand the Metal leg an enum's ordinal as a
number — a sweep over `0..n`, or a literal field — and such a value appears nowhere by name, so
every column reads short of what the case does. That is stated as a limit in the document rather
than worked around, because the obvious way to close it is unsound:

the leg types some of these fields itself (`MTLPixelFormat(request.format as usize)` says that
`format` holds a pixel format's ordinal), which invites reading every `format: <number>` in the
suite as a pixel format. It is not one. `recording.rs` declares `format: u32` for the **recording
format's version**, so `format: 1` there is version 1 and not `MTLPixelFormatA8Unorm`, and a rule
keyed on the field's name would quietly credit the suite with a value nothing ever set. One
vocabulary's word for a thing is not another's.

So the floor stays a floor and says so. Closing it properly means a case stating its values by
name, or a transcript site recording the enumerator beside the ordinal it already stores — not a
cleverer reading of the text here.
"""

import csv
import json
import re
import sys
from pathlib import Path

# The sibling resolver, found beside this file rather than through the importer's sys.path, so
# this module imports the same under `python3 -I` (which drops the script's own directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metal_leg  # noqa: E402  (after the path above)

#: The generated files holding the SDK's enumerator names and ordinals.
GENERATED = (
    "crates/metal2vulkan/src/metal/generated.rs",
    "crates/metal2vulkan-air/src/metal/generated.rs",
)

#: The published crates' sources, where a Metal type the tree models by hand is declared.
BY_HAND = (
    "crates/metal2vulkan-runtime/src",
    "crates/metal2vulkan/src",
    "crates/metal2vulkan-air/src",
)

#: A recording answer whose transcript is Metal's own, rather than one the case carries.
RAN = "metal_run"


#: The generated table of SDK enumerators for the Metal types this tree models by hand.
ENUMERATORS = "scripts/conformance-coverage/metal_enumerators.tsv"


class Type:
    """One Metal enum or option set, with the SDK's values as the generated table holds them."""

    def __init__(
        self,
        metal: str,
        rust: str,
        kind: str,
        values: list[tuple[str, str]],
        aliases: dict[str, str] | None = None,
    ):
        #: The Metal type name, `MTLBlendFactor`.
        self.metal = metal
        #: The Rust name the tree gives it, `BlendFactor`.
        self.rust = rust
        #: `enum` for an `NS_ENUM`, `flags` for an `NS_OPTIONS`.
        self.kind = kind
        #: Each value as `(Rust name, Metal enumerator name)`, in declaration order.
        self.values = values
        #: A deprecated enumerator to the canonical enumerator of the same value. A name, not a
        #: value, so it is in no denominator -- but a recording that names a value by it is naming
        #: that value, and a numerator reading names has to know so.
        self.aliases = aliases or {}

    @property
    def names(self) -> list[str]:
        """The Metal enumerator names, in declaration order."""
        return [metal for _, metal in self.values]

    def value_named(self, name: str) -> str | None:
        """The canonical enumerator `name` names, following an alias, or `None`."""
        if name in self.aliases:
            return self.aliases[name]
        return name if name in self.names else None


def read_types(root: Path) -> dict[str, Type]:
    """Every Metal type the generated tables hold, by Metal name.

    `MTLDataType` is in both files (the front end's, which `metal` re-exports); the first reading
    wins, and the two are the same table.
    """
    out: dict[str, Type] = {}
    for name in GENERATED:
        path = root / name
        text = path.read_text()
        found = 0
        for block in re.finditer(r"metal_(enum|flags)!\s*\{(.*?)\n\}\n", text, re.S):
            kind, body = block.group(1), block.group(2)
            head = re.match(r".*?\b(\w+)\s*=\s*\"(MTL\w+)\"\s*\{", body, re.S)
            if head is None:
                raise SystemExit(f"{name}: a {kind} block with no `Rust = \"MTLName\" {{` head")
            declared, _, _ = body[head.end() :].partition("aliases {")
            values = re.findall(
                r"(\w+)\s*=\s*(?:0x[0-9a-fA-F]+|\d+),\s*\"(MTL\w+)\"\s*;", declared
            )
            if not values:
                raise SystemExit(f"{name}: {head.group(2)} parsed to no values")
            found += 1
            out.setdefault(head.group(2), Type(head.group(2), head.group(1), kind, values))
        if not found:
            # The parse silently matching nothing is the one way this axis can report a denominator
            # of zero and look like a closed option space, so it is a failure, not a count.
            raise SystemExit(f"{name}: no metal_enum!/metal_flags! block parsed")
    return out


#: The settled-unreachable declaration, relative to the tree's root.
UNREACHABLE = "scripts/conformance-coverage/unreachable_values.tsv"


def read_unreachable(root: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """`(type, value)` to `(why, finding)` for every value NO case can put to Metal.

    This axis's counterpart to the items axis's "`objc2-metal` binds no name for them": a value
    behind an assertion that ends the process, or with no API that answers it, is not work, and
    counting it as a gap puts 51 permanent rows on a work list. The share below is then taken over
    what a case can reach, the way the items axis takes it.

    The declaration is `unreachable_values.tsv`, beside this script rather than in the suite, for
    one reason: a `#[cfg(test)]` item is deliberately invisible to this axis (a unit test naming a
    value must not credit it), so a list in the suite's tests could not be read, and a list in its
    product text would be a case naming the value.

    The caller checks it rather than trusting it -- see `inventory`, which fails on a row whose
    value some case did compare and on a row naming a value its type does not have.
    """
    path = root / UNREACHABLE
    out: dict[tuple[str, str], tuple[str, str]] = {}
    if not path.exists():
        # A tree that declares none, which is every synthetic tree the self-test builds. Absence is
        # "nothing is settled", never "everything is", so the share it feeds can only read LOW.
        return out
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise SystemExit(f"{path.name}: {line!r} is not `type\tvalue\twhy\tfinding`")
        metal, value, why, finding = fields
        if (metal, value) in out:
            raise SystemExit(f"{path.name}: {value} of {metal} is listed twice")
        if not why.strip() or not finding.strip():
            raise SystemExit(f"{path.name}: {value} of {metal} carries no reason or no finding")
        out[(metal, value)] = (why, finding)
    return out


#: The declared crossings of the option space -- every axis a Metal enum -- relative to the
#: tree's root.
CROSSINGS = "scripts/conformance-coverage/crossings.tsv"

#: The declared crossings that are **not** pairs of the option space: one axis a boolean or
#: numeric Metal field, or one type in two slots. Counted apart from the pairs, never added in.
OTHER = "scripts/conformance-coverage/other_crossings.tsv"

#: What a row's `cell` field says where the cross is read **across** the family's cases -- one case
#: a cell of it, or one axis fixed per case and the others swept inside each -- rather than inside
#: one case that holds the whole grid.
FAMILY = "family"


class Item:
    """One Metal item a field axis may name: who reaches it, and what its call answers.

    Built by `coverage.py` from the two sources it already reads -- `objc2_metal.tsv` for what
    `objc2-metal` binds and what each call answers, and the items axis's own join for which case
    modules' Metal legs reach the item. A field axis is checked against this and nothing else, so
    it names a call the suite is already measured to make rather than a word a row's author chose.

    Reaching is held in two strengths, because the weaker one is not per **field**. The suite's
    Metal leg builds a descriptor in one function, setting each of its fields under a guard on the
    suite's own vocabulary (`if let Some(buffer) = pass.visibility`), so every case that opens a
    pass reaches every one of those call sites by reachability alone -- which would let a case that
    attaches no visibility buffer declare a crossing on `visibilityResultBuffer`. `modules` is the
    stronger reading, and the one a field axis is checked against: the module's own data states
    the field the guard tests.
    """

    def __init__(self, modules: set[str], answers: str, reached: set[str] | None = None):
        #: The case modules whose Metal leg reaches this item through a call site **their own data
        #: can take**: every guard around it is on a field the module's text states.
        self.modules = frozenset(modules)
        #: The Metal type the call answers, or `""`. A type with a generated value list here means
        #: the field is an enum under another name and belongs in `crossings.tsv`.
        self.answers = answers
        #: The case modules whose Metal leg reaches this item at all, guards aside, which is what
        #: the items axis counts. A module here and not in `modules` reaches the call only where
        #: another case's data would have taken it, which is what the refusal says.
        self.reached = frozenset(modules if reached is None else reached)


def _read_rows(
    path: Path, repeats: bool
) -> list[tuple[str, str, tuple[str, ...], str]]:
    """Every row of a crossings file as `(module, cell, axes, what only the cross decides)`.

    The shape of the row and nothing about the suite: the caller checks it against the sources, the
    way `read_unreachable`'s caller checks that list. Refused here is only what the row itself gets
    wrong -- a field missing, a crossing of one axis, a row declared twice, or a row that says
    nothing about what the cross decides.

    A crossing of **one axis** is a sweep, which the per-value columns above already count, so a row
    with fewer than two axes is a mistake and not a smaller claim, in either file.

    `repeats` is the one difference between the two files. An axis named twice in `crossings.tsv`
    is a type crossed with itself in two slots, which is not a pair of the option space and is
    refused there; in `other_crossings.tsv` that repetition **is** the claim, and the count of
    names is the count of slots.
    """
    out: list[tuple[str, str, tuple[str, ...], str]] = []
    if not path.exists():
        # A tree that declares none, which is every synthetic tree the self-test builds unless it
        # asks for one. Absence is "none declared", which the counts read as nothing crossed.
        return out
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise SystemExit(
                f"{path.name}: {line!r} is not `module\tcell\taxes\twhat only the cross decides`"
            )
        module, cell, axes, why = (field.strip() for field in fields)
        held = tuple(axis.strip() for axis in axes.split("+") if axis.strip())
        if len(held) < 2:
            raise SystemExit(
                f"{path.name}: {module} {cell} crosses {axes!r}, which is one axis -- a sweep of "
                "one type is what the per-value columns already count"
            )
        if not repeats and len(set(held)) != len(held):
            raise SystemExit(
                f"{path.name}: {module} {cell} names an axis twice in {axes!r}; a type crossed "
                f"with itself in two slots is no pair of the option space and is declared in "
                f"{Path(OTHER).name}"
            )
        if not why:
            raise SystemExit(
                f"{path.name}: {module} {cell} says nothing about what only the cross decides"
            )
        key = (module, cell, held)
        if key in seen:
            raise SystemExit(f"{path.name}: {module} {cell} {axes} is declared twice")
        seen.add(key)
        out.append((module, cell, held, why))
    return out


def read_crossings(root: Path) -> list[tuple[str, str, tuple[str, ...], str]]:
    """Every declared pair of the option space, by its row's shape alone.

    Every axis is a Metal enum, each named once: `crossings`, below, is what checks that against
    the generated value lists and the module's own text.
    """
    return _read_rows(root / CROSSINGS, repeats=False)


def read_other_crossings(root: Path) -> list[tuple[str, str, tuple[str, ...], str]]:
    """Every declared crossing that is not a pair of the option space, by its row's shape alone.

    An axis may repeat here, which is the type-in-two-slots crossing; `other_crossings`, below, is
    what checks the axes against the Metal items and the generated value lists.
    """
    return _read_rows(root / OTHER, repeats=True)


def _holds_the_cell(
    modules_ran: dict[str, set[str]], where: str, module: str, cell: str
) -> set[str]:
    """The module's recorded cases, with the row's cell checked to be one of them.

    Refused: a module no case of which ran Metal, a cell that is not one of **that** module's
    recorded cases -- which is what catches a row left behind by a renamed or deleted case -- and
    `family` where the module has fewer than two recorded cases, since a cross read across the
    family's cases needs at least two of them.
    """
    held = modules_ran.get(module)
    if not held:
        raise SystemExit(
            f"{where} declares {module} {cell}, and no case of {module} has a recording "
            "of an answer Metal ran -- a module that ran no Metal put no pair to it"
        )
    if cell == FAMILY:
        if len(held) < 2:
            raise SystemExit(
                f"{where} says {module}'s cross is read across its own cases, and it "
                f"has {len(held)} recorded case -- a cross across cases needs at least two"
            )
    elif cell not in held:
        raise SystemExit(
            f"{where} declares the cell {cell} of {module}, which is not a case of "
            f"{module} with a recording of an answer Metal ran -- a renamed or deleted case, "
            "or the wrong module"
        )
    return held


def _varies_the_axis(
    types: dict[str, Type],
    effect: dict[tuple[str, str], set[str]],
    where: str,
    module: str,
    cell: str,
    axis: str,
) -> None:
    """That `module`'s own text states two values of the Metal enum `axis` by name.

    The one check that bears on the claim, and it is the same check in both files. A module that
    states one value of an axis, or none, cannot vary it, so nothing in the text tells the claimed
    cross from a sweep of the other axis at a fixed cell -- a single-value falsification would
    satisfy the row. It is a check of what the text can state, never of what the case's expected
    answer needs.
    """
    stated = [value for value in types[axis].names if module in effect.get((axis, value), ())]
    if len(stated) < 2:
        raise SystemExit(
            f"{where} says {module} {cell} crosses {axis}, and {module}'s own text "
            f"states {len(stated)} of its values by name -- a module that cannot vary an "
            "axis cannot cross it, so one value would falsify the row"
        )


def _pairs(axes: tuple[str, ...]) -> list[tuple[str, str]]:
    """Each unordered pair of a row's axes, so a three-axis grid holds three.

    A pair is sorted, so two rows crossing the same two types in either order are one pair and not
    two, and a type crossed with itself contributes none: `(A, A)` is no pair of two types.
    """
    return sorted(
        {
            tuple(sorted((one, other)))
            for i, one in enumerate(axes)
            for other in axes[i + 1 :]
            if one != other
        }
    )


def crossings(
    root: Path,
    types: dict[str, Type],
    effect: dict[tuple[str, str], set[str]],
    ran_cases: dict[str, str],
) -> list[dict]:
    """The declared pairs of the option space, each checked against the sources it names.

    `ran_cases` is every case whose committed recording holds an answer Metal **ran**, to the
    module that declares its name -- the same join the items axis makes, and the same gate: a case
    that ran no Metal put no pair to Metal however much its module's text names.

    Four things are refused, and each is a way a row can be false of the tree rather than merely
    optimistic:

    - a module no case of which ran Metal, or a cell that is not one of **that** module's recorded
      cases, and `family` where the module has fewer than two recorded cases
      (`_holds_the_cell`);
    - an axis with no generated value list, which is a typo or a Metal type outside this axis;
    - an axis of which the module's own text states fewer than **two** values by name
      (`_varies_the_axis`).

    What is **not** refused, because no reading of the text can see it: whether the expected answer
    is a function of both axes. A row's last field says that in words, for a reader to check against
    the case.
    """
    where = Path(CROSSINGS).name
    modules_ran: dict[str, set[str]] = {}
    for case, module in ran_cases.items():
        modules_ran.setdefault(module, set()).add(case)
    out: list[dict] = []
    for module, cell, axes, why in read_crossings(root):
        _holds_the_cell(modules_ran, where, module, cell)
        for axis in axes:
            if axis not in types:
                raise SystemExit(
                    f"{where} crosses {axis}, which has no generated value list here"
                )
            _varies_the_axis(types, effect, where, module, cell, axis)
        out.append(
            {
                "module": module,
                "cell": cell,
                "axes": list(axes),
                "why": why,
                "pairs": _pairs(axes),
            }
        )
    return out


def other_crossings(
    root: Path,
    types: dict[str, Type],
    effect: dict[tuple[str, str], set[str]],
    ran_cases: dict[str, str],
    items: dict[tuple[str, str], Item],
) -> list[dict]:
    """The declared crossings that are not pairs of the option space, each checked the same way.

    These are counted apart from the pairs and never added into them: a crossing of a type with a
    boolean or numeric field, or of a type with itself in two or more slots, is real depth and is
    **not** a pair of the option space, which is what the columns above and `crossings.tsv` measure.

    `items` is every Metal item a field axis may name, keyed as the items axis keys it, holding the
    case modules whose Metal leg reaches it and the Metal type its call answers.

    What is refused here that `crossings` does not refuse:

    - a row every axis of which is a Metal enum named **once**. That is a pair of the option space:
      it belongs in `crossings.tsv`, under the check that the module states two values of each
      axis, and a row parked here would be the same claim under a weaker rule;
    - a field axis naming an item `objc2-metal` binds nothing for, and one **this module's own
      data never reaches**: the leg must hold a call of that field at a site whose guards this
      module's text satisfies. The reachability half is the items axis's own join, so a field axis
      names a call the suite is measured to make rather than a word chosen here; the guard half is
      what makes it a claim about that **field** and not about any field of its Metal type, since
      one shared encoder sets fifty fields of a descriptor and every case reaches all fifty;
    - a field axis whose call answers a Metal type with a generated value list. Such a field is an
      enum under another name, and naming it as a field would dodge the two-values check that the
      enum spelling carries.

    Everything `crossings` refuses is refused here too, by the same helpers: the cell must be one
    of the module's recorded cases, and an enum axis -- repeated or not -- must be one the module's
    own text states two values of.

    What **cannot** be checked, for a field axis, is that the module put two values of the field to
    Metal. The guard check reads that the module states the field, not which values it states: a
    boolean's two values and a number's range are nowhere in the text this reads, and nothing in a
    recording names them, so that part of a field row is what only a reader can check. The document
    says so where it prints the count.
    """
    where = Path(OTHER).name
    modules_ran: dict[str, set[str]] = {}
    for case, module in ran_cases.items():
        modules_ran.setdefault(module, set()).add(case)
    out: list[dict] = []
    for module, cell, axes, why in read_other_crossings(root):
        if len(set(axes)) == len(axes) and all(axis in types for axis in axes):
            raise SystemExit(
                f"{where} declares {module} {cell} over {'+'.join(axes)}, every axis of which is "
                f"a Metal enum named once -- that is a pair of the option space and is declared "
                f"in {Path(CROSSINGS).name}, under its own checks"
            )
        _holds_the_cell(modules_ran, where, module, cell)
        fields: list[tuple[str, str]] = []
        for axis in axes:
            if axis in types:
                _varies_the_axis(types, effect, where, module, cell, axis)
                continue
            if "." not in axis:
                raise SystemExit(
                    f"{where} crosses {axis}, which is neither a Metal type with a generated "
                    "value list nor a `MTLType.item` naming one Metal item"
                )
            metal, name = axis.split(".", 1)
            held = items.get((metal, name))
            if held is None:
                raise SystemExit(
                    f"{where} crosses {axis}, and `objc2-metal` binds no {name} on {metal} -- a "
                    "typo, a renamed selector, or the wrong Metal type"
                )
            if held.answers in types:
                raise SystemExit(
                    f"{where} crosses the field {axis}, whose call answers {held.answers}, which "
                    f"has a generated value list -- cross {held.answers} itself, so the module is "
                    "held to stating two of its values"
                )
            if module not in held.modules:
                if module in held.reached:
                    raise SystemExit(
                        f"{where} says {module} crosses {axis}, and every call of it {module}'s "
                        f"Metal leg reaches sits behind a guard on a field of the suite's own "
                        f"vocabulary that {module}'s text never states -- the leg sets {name} only "
                        f"where a case asks for it, and this one does not, so no case of "
                        f"{module} ever put a value to it"
                    )
                raise SystemExit(
                    f"{where} says {module} crosses {axis}, and {module}'s Metal leg reaches no "
                    f"call of it -- the items axis attributes {name} to "
                    f"{', '.join(sorted(held.modules)) or 'no module'}"
                )
            fields.append((metal, name))
        out.append(
            {
                "module": module,
                "cell": cell,
                "axes": list(axes),
                "why": why,
                # The pairs of **distinct** Metal enums this row holds, which is how a row that
                # crosses an enum with a field still reports the enums it names: a field axis is
                # no Metal type and is in no pair, so a row naming one enum and one field holds
                # none. These are reported and never counted into the option space's own pairs.
                "pairs": _pairs(tuple(axis for axis in axes if axis in types)),
                # The Metal fields the row names, and how many slots of one enum it crosses.
                "fields": fields,
                "slots": sorted(
                    {axis for axis in axes if axis in types and axes.count(axis) > 1}
                ),
            }
        )
    return out


def read_table(root: Path, rust_names: dict[str, str]) -> dict[str, Type]:
    """The SDK's values for the Metal types this tree models by hand, from the generated table.

    These have no generated Rust -- the runtime declares them itself, and `render_table`'s comment
    says why that stays -- so without this table the axis had no SDK list for them and measured a
    hand-written variant list against itself. It could not report a value the SDK declares and the
    tree does not, and it reported an option-set newtype, which declares no variants at all, as a
    type with no values: `MTLPipelineOption` read `0 of 0` while its four defined values were being
    swept.

    `rust_names` is the tree's own name for each, which `by_hand` reads off the declaring item, so
    a case naming `StoreAction::Unknown` resolves the same way one naming a generated type's
    variant does.
    """
    path = root / ENUMERATORS
    values: dict[str, list[tuple[str, str]]] = {}
    aliases: dict[str, dict[str, str]] = {}
    kinds: dict[str, str] = {}
    canonical: dict[tuple[str, str], str] = {}
    with path.open() as handle:
        rows = list(
            csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
        )
    if not rows:
        # A table that parsed to nothing would hand every type in it a denominator of zero and
        # read as a closed option space, which is the one way this axis fails silently.
        raise SystemExit(f"{ENUMERATORS}: parsed to no rows")
    for row in rows:
        metal, role = row["type"], row["role"]
        kinds[metal] = row["kind"]
        if role == "value":
            values.setdefault(metal, []).append((row["rust"], row["enumerator"]))
            canonical[(metal, row["rust"])] = row["enumerator"]
        elif role == "alias":
            aliases.setdefault(metal, {})[row["enumerator"]] = canonical[(metal, row["rust"])]
        else:
            raise SystemExit(f"{ENUMERATORS}: {metal} has a row whose role is {role!r}")
    out: dict[str, Type] = {}
    for metal, held in sorted(values.items()):
        out[metal] = Type(
            metal, rust_names.get(metal, metal[3:]), kinds[metal], held, aliases.get(metal)
        )
    return out


def without_tests(text: str) -> str:
    """`blanked(text)` with every `#[cfg(test)]` item blanked too.

    A unit test naming a value is not the suite putting it to Metal, and one of them names four
    blend factors to **assert that no case uses them**. Counting those would have the axis report
    the opposite of what the code says.
    """
    out = list(metal_leg.blanked(text))
    for attr in re.finditer(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]", "".join(out)):
        i = attr.end()
        n = len(out)
        # Skip the attributes between this one and the item it sits on.
        while True:
            while i < n and out[i].isspace():
                i += 1
            if i < n and out[i] == "#":
                depth = 0
                while i < n:
                    if out[i] in "([{":
                        depth += 1
                    elif out[i] in ")]}":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                    i += 1
                continue
            break
        # The item ends at its body's closing brace, or at the `;` of one with no body. A `;` or a
        # brace inside brackets (`[u8; 4]`) is the type's, not the item's.
        depth = 0
        end = None
        while i < n:
            char = out[i]
            if char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            elif depth == 0 and char == "{":
                _, end = metal_leg.block("".join(out), i)
                break
            elif depth == 0 and char == ";":
                end = i + 1
                break
            i += 1
        if end is None:
            raise SystemExit("a #[cfg(test)] item with neither a body nor a `;`")
        for at in range(attr.start(), end):
            if not out[at].isspace():
                out[at] = " "
    return "".join(out)


def _resolved(held: Type, tail: str, metal_spelling: bool) -> set[tuple[str, str]]:
    """The values of `held` that the variant name `tail` names, by every spelling of one value.

    One reading of the rule, so every place that resolves a name uses the same one: a variant name
    beside its type, a variant name brought in bare by a `use`, and the `objc2` spelling are the
    same question asked of different text.

    Three ways a name reaches a value, and no fourth:

    - the Rust name the generated table gives it, which is what `BlendFactor::SourceAlpha` names;
    - the type's name and the variant's spelled back together, where that concatenation is an
      enumerator the table holds. `objc2` names a variant with its type's prefix dropped, so
      `MTLStorageMode::Private` is `MTLStorageModePrivate` — and the same rule reads our own
      spelling of a type we model unlike the SDK, `BlitOption::None` against `MTLBlitOptionNone`,
      because we make a closed enum of what the SDK makes an `NS_OPTIONS`. Never by assuming the
      rule held: the value credited is the one whose enumerator is exactly the concatenation, and
      nothing is credited where the table holds no such name.
    - a hand-modelled type with no table has no SDK enumerator list at all, so our own variant
      name is the only name there is.
    """
    out: set[tuple[str, str]] = set()
    for rust, value in held.values:
        if rust == tail:
            out.add((held.metal, value))
    found = held.value_named(held.metal + tail)
    if found is not None:
        out.add((held.metal, found))
    elif metal_spelling and held.kind == "by hand" and tail in held.names:
        out.add((held.metal, tail))
    return out


#: The suite's shared sources, where its own vocabulary is declared: everything under the crate's
#: `src` but the case modules, the test trees and the binaries.
SHARED = "crates/metal2vulkan-conformance/src"


def shared_sources(root: Path) -> list[tuple[str, str, str]]:
    """Each shared source as `(path as the crate sees it, its text, `without_tests` of it)`.

    One reading for the two things that walk these files -- the vocabulary below and the *shared*
    column -- so the two cannot disagree about which file is shared, and neither re-reads 2.6 MB
    of source the other already has. The raw text is kept because a vocabulary enum's
    **declaration** is in its doc comments, which `without_tests` blanks: what a case states is
    read off code, what the vocabulary *is* is read off the declaration.
    """
    src = root / SHARED
    out: list[tuple[str, str, str]] = []
    for path in sorted(src.rglob("*.rs")):
        held = path.relative_to(src)
        if held.parts[0] in ("cases", "tests", "bin") or held.name == "tests.rs":
            continue
        text = path.read_text()
        out.append((str(held), text, without_tests(text)))
    return out


#: One enum of the suite's own vocabulary: a `pub` or `pub(crate)` enum of a shared source. A case
#: names a value of it by its variant, so this is the item the spelling below is read off.
VOCABULARY = re.compile(r"\bpub(?:\s*\([^)]*\))?\s+enum\s+([A-Za-z_]\w*)\s*\{")

#: What a variant's **first** doc line declares it to be: an inline code span that **opens** the
#: line, holding one Metal enumerator or several of one type joined by `|`. Opening the line is the
#: `_opens_the_doc` discipline in the small -- a name part-way through a sentence is prose about a
#: value ("unlike `MTLStorageModePrivate`, this one is ours") and declares nothing.
DECLARES = re.compile(r"^`(MTL\w+(?:\s*\|\s*MTL\w+)*)`")

#: An attribute, so a variant's own name is the first identifier after it rather than one of
#: `#[cfg(feature = "x")]`'s.
ATTRIBUTE = re.compile(r"#\s*\[")


def _chunks(source: str, blank: str) -> list[tuple[str, str]]:
    """An enum body's depth-0 comma-separated chunks, each as `(its text, its blanked text)`.

    A variant with a payload holds commas of its own (`Rect { width: u32, height: u32 }`), and a
    field of it is spelled exactly like a variant, so splitting on every comma would read a field
    as a variant and report a complete vocabulary as one whose variants are not all declared. The
    depth counter is what keeps a payload inside its variant.
    """
    out: list[tuple[str, str]] = []
    depth = 0
    start = 0
    for i, char in enumerate(blank):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            out.append((source[start:i], blank[start:i]))
            start = i + 1
    if source[start:].strip():
        out.append((source[start:], blank[start:]))
    return out


def _variant(source: str, blank: str) -> tuple[str, str] | None:
    """One chunk read as `(the variant's name, what its first doc line says)`, or `None`.

    The name is the first identifier of the blanked chunk once its attributes are skipped, so
    neither a comment nor a `#[cfg(...)]`'s own words can be it. The doc line is the first
    non-empty one of the chunk's `///` block, read off the **raw** text, which is where a
    declaration lives.
    """
    i = 0
    n = len(blank)
    while True:
        while i < n and blank[i].isspace():
            i += 1
        if ATTRIBUTE.match(blank, i) is None:
            break
        depth = 0
        while i < n:
            if blank[i] in "([{":
                depth += 1
            elif blank[i] in ")]}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
    name = re.match(r"[A-Za-z_]\w*", blank[i:])
    if name is None:
        return None
    said = ""
    for line in source.splitlines():
        line = line.strip()
        if not line.startswith("///"):
            continue
        held = line[3:].strip()
        if held:
            said = held
            break
    return name.group(0), said


def _declares(said: str, of_name: dict[str, tuple[str, str]]) -> set[tuple[str, str]] | None:
    """The values a variant's first doc line declares it to be, or `None` for no declaration.

    `None` and the empty set are different answers. `None` is "this line declares nothing", which
    is every variant whose doc is prose; the empty set says the line **is** a declaration and
    names a value of no generated type, which its caller refuses rather than reading as absent.
    """
    found = DECLARES.match(said)
    if found is None:
        return None
    out: set[tuple[str, str]] = set()
    for part in found.group(1).split("|"):
        held = of_name.get(part.strip())
        if held is None:
            return set()
        out.add(held)
    return out


def vocabulary(
    sources: list[tuple[str, str, str]], types: dict[str, Type]
) -> dict[str, dict[str, frozenset[tuple[str, str]]]]:
    """The suite's own enums that spell a Metal enum, by enum name and variant.

    This is the fifth spelling `_stated_in` reads, and the one a reader cannot see in a case: a
    case states a storage mode as `Storage::Private`, never as `MTLStorageModePrivate`, so without
    this mapping the exhaustive sweep in `shared_creator_formats` counted for nothing and
    `MTLStorageMode` read `0 of 4` while all four were being put to Metal in 24 cells.

    It is derived from the vocabulary enum's **own declaration** rather than kept as a table of
    string pairs: a variant whose first doc line opens with the Metal enumerator it spells declares
    the value, and `Storage`, `Plane`, `TextureRole`, `StageAsked` and `Dispatching` each say so on
    each of their variants. `DECLARES` reads one enumerator or several of one type joined by `|`,
    because the suite's vocabulary collapses an option set's combinations into variants of their
    own (`TextureRole::ReadWrite`, `StageAsked::VertexAndFragment`), and a combination is as much a
    declaration as a single bit.

    Four things are refused, each a way the mapping can be **wrong** rather than absent:

    - a variant of a spelling enum whose first doc line declares nothing. That is the one that
      makes the mapping fail where it is incomplete: a vocabulary that gains a variant nobody gave
      a Metal spelling would otherwise go on reading as a closed sweep of the values it still
      covers, which is the defect this whole function exists to fix, one variant smaller;
    - a Metal-shaped declaration naming a value no generated type holds -- a typo, or an
      enumerator the SDK renamed;
    - two Metal types across one enum's variants, which is no spelling of either;
    - two variants declaring the very same values, which is one spelling written twice.

    An enum **no** variant of which declares anything is not a spelling and is passed over in
    silence: that is most of the suite's shared enums, and so is one that spells a Metal type with
    no generated value list (`MTLSparseTextureMappingMode`), since this axis has no denominator for
    it and says so rather than guessing.
    """
    of_name = {metal: (held.metal, metal) for held in types.values() for metal in held.names}
    for held in types.values():
        for name, canonical in held.aliases.items():
            of_name.setdefault(name, (held.metal, canonical))
    out: dict[str, dict[str, frozenset[tuple[str, str]]]] = {}
    for where, source, blank in sources:
        for head in VOCABULARY.finditer(blank):
            # `block` holds the braces; the body is what they hold, so a chunk's depth starts at 0
            # and a depth-0 comma is a variant's.
            start, end = metal_leg.block(blank, head.end() - 1)
            body = slice(start + 1, end - 1)
            held = [_variant(*chunk) for chunk in _chunks(source[body], blank[body])]
            read = {
                found[0]: _declares(found[1], of_name) for found in held if found is not None
            }
            declared = {one: values for one, values in read.items() if values}
            if not declared:
                continue
            name = head.group(1)
            failed = sorted(one for one, values in read.items() if values == set())
            if failed:
                raise SystemExit(
                    f"{where}: {name} spells a Metal enum, and its {', '.join(failed)} declares "
                    "an enumerator no generated value list holds -- a typo, an enumerator the SDK "
                    "renamed, or a second Metal type in one spelling"
                )
            silent = sorted(one for one, values in read.items() if values is None)
            if silent:
                raise SystemExit(
                    f"{where}: {name} spells a Metal enum, and its {', '.join(silent)} says which "
                    "value on no line of its own -- open that variant's doc with the Metal "
                    "enumerator it is, as its siblings do, so a case naming it is counted. The "
                    "option-space axis reads this declaration and nothing else"
                )
            metal = {one for values in declared.values() for one, _ in values}
            if len(metal) != 1:
                raise SystemExit(
                    f"{where}: {name}'s variants declare values of {', '.join(sorted(metal))} -- "
                    "one enum of the suite's vocabulary spells one Metal type"
                )
            seen: dict[frozenset[tuple[str, str]], str] = {}
            for one, values in sorted(declared.items()):
                key = frozenset(values)
                if key in seen:
                    raise SystemExit(
                        f"{where}: {name}'s {seen[key]} and {one} declare the same value of "
                        f"{metal.pop()} -- one spelling written twice"
                    )
                seen[key] = one
            if name in out:
                raise SystemExit(
                    f"{where}: a second {name} spelling a Metal enum; the suite's vocabulary has "
                    "one declaration a name, since a case names the value by the name alone"
                )
            out[name] = {one: frozenset(values) for one, values in declared.items()}
    return out


def _stated_in(
    text: str,
    by_rust: dict[str, Type],
    by_metal: dict[str, Type],
    by_vocab: dict[str, dict[str, frozenset[tuple[str, str]]]],
):
    """The `(Metal type, value)` pairs `text` names, in every spelling the suite uses.

    The suite names a Metal value four ways, and all four have to be read:

    - through the tree's own type, `BlendFactor::SourceAlpha` — and `rotated_targets.rs` writes
      `use BlendFactor as F;` and then every factor as `F::SourceAlpha`, so a derivation reading
      only `BlendFactor::` spellings reports a third of that module's blend factors as never
      exercised;
    - through `objc2-metal`'s own enum, `MTLStorageMode::Private`, which the Metal leg writes
      directly;
    - through a **braced `use`**, `use SamplerBorderColor::{OpaqueBlack, OpaqueWhite,
      TransparentBlack};`, after which the module writes the variants bare and names
      `SamplerBorderColor::` nowhere. `border_swizzles` sweeps all three border colours in nine
      samplers written that way, and a reading of `Head::Tail` alone finds none of them.
    - through the suite's **own vocabulary**, `Storage::Private`, which `vocabulary` reads off the
      variant's own declaration of the Metal enumerator it spells. A case states a storage mode no
      other way, so without this the exhaustive four-mode sweep of `shared_creator_formats`
      counted for nothing at all.

    A `use` is taken as the module naming the value, and the bare name is not looked for
    afterwards. That is sound here and nowhere else: an import nothing uses is a warning, warnings
    are errors in this tree, so a variant in a `use` is one the text goes on to use. Looking for
    the bare name instead would be the unsound reading — `border_swizzles` imports
    `TextureSwizzle::Alpha as A`, and crediting a type's value for every `A` in a module is how a
    derivation invents coverage.
    """
    alias: dict[str, str] = {}
    for found in re.finditer(r"\b([A-Za-z_]\w*)\s+as\s+([A-Za-z_]\w*)\b", text):
        if found.group(1) in by_rust or found.group(1) in by_vocab:
            alias[found.group(2)] = found.group(1)
    out: set[tuple[str, str]] = set()
    for found in re.finditer(r"\b([A-Za-z_]\w*)::([A-Za-z_]\w*)\b", text):
        head, tail = found.group(1), found.group(2)
        held = by_rust.get(alias.get(head, head))
        if held is not None:
            out |= _resolved(held, tail, False)
        held = by_metal.get(head)
        if held is not None:
            out |= _resolved(held, tail, True)
        spelled = by_vocab.get(alias.get(head, head))
        if spelled is not None:
            out |= spelled.get(tail, frozenset())
    # The braced form, which the loop above cannot see: after `::` comes `{` and not a name. The
    # path may be qualified (`metal2vulkan::metal::TextureSwizzle::{..}`), so the type is its last
    # segment, and each item may be renamed, in which case what names the value is the name
    # imported and not the local one.
    for found in re.finditer(r"\buse\s+([A-Za-z_][\w:]*?)\s*::\s*\{([^{}]*)\}", text):
        head = found.group(1).split("::")[-1].strip()
        imports = [item.split(" as ")[0].strip() for item in found.group(2).split(",")]
        for held, metal_spelling in (
            (by_rust.get(alias.get(head, head)), False),
            (by_metal.get(head), True),
        ):
            if held is None:
                continue
            for imported in imports:
                if imported:
                    out |= _resolved(held, imported, metal_spelling)
        spelled = by_vocab.get(alias.get(head, head))
        if spelled is not None:
            for imported in imports:
                out |= spelled.get(imported, frozenset())
    return out


def _types_named_in(text: str, held: dict[str, Type]) -> set[str]:
    """The Metal types `text` names **as a type**, by either spelling, as whole identifiers.

    A type's own name, never a value's: this says what a module is about, not what it put to
    Metal. It is what turns the raw-ordinal limit from a sentence into a work list -- a module that
    names `MTLPipelineOption` and states none of its values by name is a case this axis cannot read
    the depth of, and saying which module that is costs no guess about which value it reached.
    """
    found: set[str] = set()
    for one in held.values():
        for spelling in (one.metal, one.rust):
            if re.search(rf"\b{re.escape(spelling)}\b", text):
                found.add(one.metal)
                break
    return found


def stated(
    root: Path,
    types: dict[str, Type],
    ran: set[str],
    by_vocab: dict[str, dict[str, frozenset[tuple[str, str]]]],
    sources: list[tuple[str, str, str]],
):
    """The values each case module names, and the ones only the suite's shared sources name.

    `ran` is the case modules with at least one committed recording of an answer Metal ran; a
    module not in it puts nothing to Metal however much its text names.

    `by_vocab` is the suite's own spellings, and `sources` the shared sources they were read off,
    so the *shared* column below reads exactly the files the vocabulary came from.
    """
    by_rust = {held.rust: held for held in types.values()}
    by_metal = {held.metal: held for held in types.values()}
    src = root / SHARED
    in_effect: dict[tuple[str, str], set[str]] = {}
    # Each module to the types its text is about, so a type a module names and states no value of
    # can be reported as one whose depth is unreadable here rather than as one nothing reaches.
    about: dict[str, set[str]] = {}
    for path in sorted((src / "cases").glob("*.rs")):
        if path.stem not in ran:
            continue
        text = without_tests(path.read_text())
        stated_here = _stated_in(text, by_rust, by_metal, by_vocab)
        for pair in stated_here:
            in_effect.setdefault(pair, set()).add(path.stem)
        said = {metal for metal, _ in stated_here}
        for metal in _types_named_in(text, types) - said:
            about.setdefault(metal, set()).add(path.stem)
    shared: set[tuple[str, str]] = set()
    for _, _, text in sources:
        shared |= _stated_in(text, by_rust, by_metal, by_vocab)
    return in_effect, shared, about


def _tokens(value, key: str | None, out: list[str]) -> None:
    """Every identifier-shaped token of a transcript value, less the base64 of its bytes."""
    if isinstance(value, str):
        if key != "bytes":
            out.extend(re.findall(r"[A-Za-z0-9_]+", value))
    elif isinstance(value, dict):
        for name, held in value.items():
            _tokens(held, name, out)
    elif isinstance(value, list):
        for held in value:
            _tokens(held, key, out)


def _site_and_observed(node, named: list[str], observed: list[str]) -> None:
    """A transcript's tokens, split by where they came from: a site's **name**, or what it observed.

    The split is the whole point. A node's name is the **case's** label for a site, chosen by
    whoever wrote the case; what the node observed is Metal's answer. Read into one bag they credit
    a value to Metal because a case spelled the enumerator in its own site name, which across the
    committed recordings is most of `MTLPixelFormat`: the suite labels the site it asks about with
    the format's name and the site answers a number.
    """
    if not isinstance(node, dict):
        return
    name = node.get("name")
    if isinstance(name, str):
        named.extend(re.findall(r"[A-Za-z0-9_]+", name))
    _tokens(node.get("observed"), None, observed)
    for child in node.get("children") or []:
        _site_and_observed(child, named, observed)


def answered(
    root: Path, types: dict[str, Type]
) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], set[str]]]:
    """The values a committed recording's compared transcript holds, by the cases holding them.

    Two dicts, because they are two claims of different strength and were one number:

    - **answered**: the value is in what a site **observed**, which is Metal's own answer. This is
      the strong claim the document makes for the column -- Metal named the value back at a site
      the comparison reads.
    - **at a site**: the value is in a site's **name**. The comparison does read those names, and
      pairs sites by them, so the value is in the compared text and the case did put it to Metal.
      But it is the case's label and not Metal's answer, so it is no evidence that Metal **said**
      the value, and folding it into the first overstated what the recording holds.

    Both are inside *compared*, so no total moves; what moves is which column carries them.

    Only `answers[].outcomes[].transcript` is read. An answer's `validated` leg is Metal's API
    validation, which the suite records and **never compares**, and its complaints name
    enumerators; reading it would count Apple's prose as a comparison. The cost and timing tables
    are not read either: a site name there is what was measured, not an answer that was compared.
    """
    of_name = {metal: (held.metal, metal) for held in types.values() for metal in held.names}
    # A deprecated spelling names the value it is an alias of, so a recording holding
    # `MTLPipelineOptionBindingInfo` is Metal naming bit 1 back at us, whatever the headers call
    # that bit first.
    for held in types.values():
        for name, canonical in held.aliases.items():
            of_name.setdefault(name, (held.metal, canonical))
    out: dict[tuple[str, str], set[str]] = {}
    at_a_site: dict[tuple[str, str], set[str]] = {}
    for path in sorted((root / "crates/metal2vulkan-conformance/recordings").glob("*.json")):
        data = json.loads(path.read_text())
        named: list[str] = []
        observed: list[str] = []
        for answer in data.get("answers", []):
            if answer.get("provenance") != RAN:
                continue
            for outcome in answer.get("outcomes", []):
                _site_and_observed(
                    (outcome.get("transcript") or {}).get("root"), named, observed
                )
        for token in set(observed):
            if token in of_name:
                out.setdefault(of_name[token], set()).add(data["case"])
        for token in set(named):
            if token in of_name and of_name[token] not in out:
                at_a_site.setdefault(of_name[token], set()).add(data["case"])
    # A value Metal observed somewhere is answered, whatever else spells it, so the two columns
    # never hold the same value twice.
    for key in list(at_a_site):
        if key in out:
            del at_a_site[key]
    return out, at_a_site


#: A Rust item the tree declares to be one Metal enum or option set: a doc comment naming the
#: Metal type, then the item it documents, which is a closed enum (an `NS_ENUM`) or a newtype over
#: one integer (an `NS_OPTIONS`). A multi-field struct is a Metal *structure*, and a class or
#: descriptor is neither, so the shape of the item is what tells the three apart.
MODELS = re.compile(
    r"///\s*`(MTL\w+)`[^\n]*\n(?:[ \t]*(?:///[^\n]*|#\[[^\n]*\])\n)*"
    r"[ \t]*pub\s+(?:(enum)\s+([A-Za-z_]\w*)\s*\{"
    r"|(struct)\s+([A-Za-z_]\w*)\s*\(\s*u(?:8|16|32|64|size)\s*\)\s*;)"
)


def _opens_the_doc(source: str, at: int) -> bool:
    """Whether the doc line at `at` is the first of its item's doc block.

    The generated enums write the Metal name on the first line, and so do the hand-written ones. A
    name mentioned part-way through a paragraph is prose about a Metal type, not a declaration of
    being one: `CommandDescriptor`'s doc says it models `MTLIndirectCommandBufferDescriptor` "as
    far as the record format needs it", which is a description and not that enum.
    """
    before = source.rfind("\n", 0, at)
    if before < 0:
        return True
    line = source[source.rfind("\n", 0, before) + 1 : before]
    return not line.lstrip().startswith("///")


def by_hand(root: Path, types: dict[str, Type]) -> list[tuple[str, str, list[str]]]:
    """The Metal enums and option sets the tree models by hand, with our own variant names.

    Found by shape: a doc comment naming a Metal type on an item that is a closed Rust enum or a
    newtype over one integer. These have **no generated value list**, so this axis has no
    denominator for them, and the names returned are our own sources', not the SDK's. An option-set
    newtype declares no variants and returns an empty list.

    The list is itself a floor: a type the tree models without saying so in a doc comment on the
    item is in neither table and nowhere in this axis.
    """
    out: dict[str, tuple[str, list[str]]] = {}
    for where in BY_HAND:
        for path in sorted((root / where).rglob("*.rs")):
            if "tests" in path.parts or path.name in ("generated.rs", "tests.rs"):
                continue
            source = path.read_text()
            # `blanked` keeps every offset, so the brace walk below reads against the source. It
            # is a character loop over the whole file, so it runs only where a match needs it:
            # almost no file in these crates declares one of these types.
            text: str | None = None
            for found in MODELS.finditer(source):
                metal = found.group(1)
                if metal in types or not _opens_the_doc(source, found.start()):
                    continue
                if found.group(2):
                    if text is None:
                        text = metal_leg.blanked(source)
                    start, end = metal_leg.block(text, found.end() - 1)
                    names = re.findall(r"^\s*([A-Z]\w*)\s*(?:,|\{|\(|$)", text[start:end], re.M)
                    out[metal] = (found.group(3), names)
                else:
                    out[metal] = (found.group(5), [])
    return [(metal, rust, names) for metal, (rust, names) in sorted(out.items())]


def inventory(
    root: Path,
    ran_cases: dict[str, str],
    items: dict[tuple[str, str], Item] | None = None,
) -> dict:
    """The axis, derived: every type with its three sets of values, and the declared crossings.

    `ran_cases` is every case whose committed recording holds an answer Metal **ran**, to the case
    module that declares its name. The module-level gate the per-value columns apply is that same
    fact read coarsely -- a module with at least one such case -- so there is one input and not two
    that could disagree.

    `items` is every Metal item a field axis of `other_crossings.tsv` may name. A tree that passes
    none declares no field axis that can be checked, so any it declares is refused rather than
    taken on trust.
    """
    ran = set(ran_cases.values())
    types = read_types(root)
    # The tree's own name for each type it models by hand, read off the declaring item, so the
    # generated table of their SDK values can be keyed the way a case spells them.
    named = {metal: rust for metal, rust, _ in by_hand(root, types)}
    # A hand-modelled type with a generated table of the SDK's values is no longer hand-modelled
    # as far as this axis is concerned: it has a denominator, so it joins the per-type table and
    # leaves the one below it.
    types.update(read_table(root, named))
    hand = [row for row in by_hand(root, types) if row[0] not in types]
    # A hand-modelled type the table does not cover is carried through the same resolution, keyed
    # by its own Rust variant names: this axis has no SDK enumerator list for it, so a variant is
    # all there is to name.
    held = dict(types)
    for metal, rust, names in hand:
        held[metal] = Type(metal, rust, "by hand", [(name, name) for name in names])
    # The suite's own spellings, read off the shared sources once and handed to the reading, so a
    # case naming `Storage::Private` is read as naming `MTLStorageModePrivate`. Resolved against
    # `types` rather than `held`: a hand-modelled type with no generated table has no SDK
    # enumerator for a variant's doc to declare, so there is nothing for this to key on.
    sources = shared_sources(root)
    vocab = vocabulary(sources, types)
    effect, shared, about = stated(root, held, ran, vocab, sources)
    read_back, at_a_site = answered(root, types)
    unreachable = read_unreachable(root)
    for metal, value in unreachable:
        if metal not in types:
            raise SystemExit(
                f"unreachable_values.tsv names {metal}, which has no generated value list here"
            )
        if value not in types[metal].names:
            raise SystemExit(
                f"unreachable_values.tsv names {value}, which is not a value of {metal} -- an "
                "enumerator renamed in the SDK, or a typo"
            )
    rows = []
    for metal, held in sorted(types.items()):
        compared, only_shared, missing, settled = [], [], [], []
        for name in held.names:
            key = (metal, name)
            if key in read_back or key in at_a_site or key in effect:
                if key in unreachable:
                    # The list claims no case can reach this value, and a case compared it. One of
                    # the two is wrong and neither may stand: a wrong row here is how a share gets
                    # inflated, which is the whole thing this column has to be trusted not to do.
                    raise SystemExit(
                        f"unreachable_values.tsv calls {name} of {metal} unreachable, but a case "
                        "compared it -- delete the row, or correct what the case claims"
                    )
                compared.append(name)
            elif key in unreachable:
                settled.append(name)
            elif key in shared:
                only_shared.append(name)
            else:
                missing.append(name)
        rows.append(
            {
                "metal": metal,
                "rust": held.rust,
                "kind": held.kind,
                "values": held.names,
                "compared": compared,
                "answered": [n for n in held.names if (metal, n) in read_back],
                # In a site's **name** rather than in what it observed: the case's label for the
                # site, which the comparison does pair by, so the value is in the compared text --
                # but it is not Metal saying the value, which is what *answered* claims.
                "at_a_site": [n for n in held.names if (metal, n) in at_a_site],
                "in_effect": [n for n in held.names if (metal, n) in effect],
                "shared": only_shared,
                "missing": missing,
                # Values no case can reach, settled by a measurement rather than unexercised. Kept
                # out of `missing` so a work list is work.
                "settled": settled,
                "modules": sorted({m for n in held.names for m in effect.get((metal, n), ())}),
                # Modules whose text is about this type and states no value of it by name. A
                # sweep written over ordinals reads this way, and so does a module that only
                # mentions the type, so the row is a question for the case and not a verdict.
                "named_no_value": sorted(about.get(metal, ())),
            }
        )
    hand_rows = [
        {
            "metal": metal,
            "rust": rust,
            "declared": names,
            "stated": [name for name in names if (metal, name) in effect],
            "shared": [
                name
                for name in names
                if (metal, name) not in effect and (metal, name) in shared
            ],
        }
        for metal, rust, names in hand
    ]
    return {
        "rows": rows,
        "by_hand": hand_rows,
        "answered": read_back,
        "at_a_site": at_a_site,
        "in_effect": effect,
        "unreachable": unreachable,
        # The suite's own enums that spell a Metal enum, as `{spelling: {variant: [values]}}`. The
        # document prints how many Metal types are reached this way, because a spelling nobody
        # declared is the one floor this fix replaced a defect with.
        "vocabulary": {
            name: {one: sorted(values) for one, values in sorted(held.items())}
            for name, held in sorted(vocab.items())
        },
        # The pairs axis. Checked against the sources each row names, and a floor by every crossing
        # the suite holds and nobody declared.
        "crossings": crossings(root, types, effect, ran_cases),
        # The crossings that are no pair of the option space: an axis that is a boolean or numeric
        # Metal field, or one type in two slots. A separate count, never added into the one above.
        "other_crossings": other_crossings(
            root, types, effect, ran_cases, items or {}
        ),
    }


if __name__ == "__main__":
    sys.exit(
        f"{Path(__file__).name} is a module of scripts/conformance-coverage, not a command: it "
        "derives one part of the document and has no output of its own. The generator is\n"
        "  python3 -I scripts/conformance-coverage/coverage.py"
    )
