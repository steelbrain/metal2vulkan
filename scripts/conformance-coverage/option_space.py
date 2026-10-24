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

# What the floor is made of

Every column here is a floor, and the honest thing is to say what of rather than only that it is
one. Two things, and as far as has been measured no third:

1. **A value a case states as a raw ordinal**, which the next section is about. Measured where it
   can be: the document lists each type a recorded module names and states no value of by name,
   which is how a sweep over `0..n` reads, so the floor has a work list and not just a warning.
2. **A spelling of a value that `_stated_in` does not read.** A name written in a shape the
   reading does not cover is counted nowhere and looks exactly like a value no case reaches.
   `_stated_in` lists every shape read and `_resolved` is the one rule they all go through, so a
   shape not on that list is the floor. Two have been found, both by someone checking a printed
   count against the code rather than by this file, which is the evidence for taking the unknown
   seriously rather than calling it small.

What is **not** part of the floor, because it is measured: the generated denominator (every type
`regen-metal-enums` covers, by `ENUMS`, `OPTIONS` or `TABLE_ONLY`), the `#[cfg(test)]` exclusion,
Metal's validation leg, and which case put a value in effect. A type the tree models by hand and
says so nowhere on the item is outside both tables and so outside the axis, which the document
states where it lists them.

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
from pathlib import Path

import metal_leg

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


def _stated_in(text: str, by_rust: dict[str, Type], by_metal: dict[str, Type]):
    """The `(Metal type, value)` pairs `text` names, in every spelling the suite uses.

    The suite names a Metal value three ways, and all three have to be read:

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

    A `use` is taken as the module naming the value, and the bare name is not looked for
    afterwards. That is sound here and nowhere else: an import nothing uses is a warning, warnings
    are errors in this tree, so a variant in a `use` is one the text goes on to use. Looking for
    the bare name instead would be the unsound reading — `border_swizzles` imports
    `TextureSwizzle::Alpha as A`, and crediting a type's value for every `A` in a module is how a
    derivation invents coverage.
    """
    alias: dict[str, str] = {}
    for found in re.finditer(r"\b([A-Za-z_]\w*)\s+as\s+([A-Za-z_]\w*)\b", text):
        if found.group(1) in by_rust:
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
    # The braced form, which the loop above cannot see: after `::` comes `{` and not a name. The
    # path may be qualified (`metal2vulkan::metal::TextureSwizzle::{..}`), so the type is its last
    # segment, and each item may be renamed, in which case what names the value is the name
    # imported and not the local one.
    for found in re.finditer(r"\buse\s+([A-Za-z_][\w:]*?)\s*::\s*\{([^{}]*)\}", text):
        head = found.group(1).split("::")[-1].strip()
        for held, metal_spelling in (
            (by_rust.get(alias.get(head, head)), False),
            (by_metal.get(head), True),
        ):
            if held is None:
                continue
            for item in found.group(2).split(","):
                imported = item.split(" as ")[0].strip()
                if imported:
                    out |= _resolved(held, imported, metal_spelling)
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


def stated(root: Path, types: dict[str, Type], ran: set[str]):
    """The values each case module names, and the ones only the suite's shared sources name.

    `ran` is the case modules with at least one committed recording of an answer Metal ran; a
    module not in it puts nothing to Metal however much its text names.
    """
    by_rust = {held.rust: held for held in types.values()}
    by_metal = {held.metal: held for held in types.values()}
    src = root / "crates/metal2vulkan-conformance/src"
    in_effect: dict[tuple[str, str], set[str]] = {}
    # Each module to the types its text is about, so a type a module names and states no value of
    # can be reported as one whose depth is unreadable here rather than as one nothing reaches.
    about: dict[str, set[str]] = {}
    for path in sorted((src / "cases").glob("*.rs")):
        if path.stem not in ran:
            continue
        text = without_tests(path.read_text())
        stated_here = _stated_in(text, by_rust, by_metal)
        for pair in stated_here:
            in_effect.setdefault(pair, set()).add(path.stem)
        said = {metal for metal, _ in stated_here}
        for metal in _types_named_in(text, types) - said:
            about.setdefault(metal, set()).add(path.stem)
    shared: set[tuple[str, str]] = set()
    for path in sorted(src.rglob("*.rs")):
        held = path.relative_to(src)
        if held.parts[0] in ("cases", "tests", "bin") or held.name == "tests.rs":
            continue
        shared |= _stated_in(without_tests(path.read_text()), by_rust, by_metal)
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


def inventory(root: Path, ran: set[str]) -> dict:
    """The axis, derived: every type with its three sets of values."""
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
    effect, shared, about = stated(root, held, ran)
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
    }
