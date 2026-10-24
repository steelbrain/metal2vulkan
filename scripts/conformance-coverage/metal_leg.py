#!/usr/bin/env python3
"""Resolve the Metal calls the conformance suite's native leg makes, to items of the SDK.

The suite's Metal leg reaches Metal through `objc2-metal`, whose Rust name for a binding is its
selector with each `:` turned into `_`. So a call site names its selector outright, and the one
thing a reader cannot see from the name is **which type** receives it: twenty types bind
`setLabel:`, four bind `setUsage:`, and the SDK item a call exercises is a property of the receiver.

This types the receiver. Within a function it follows what each local was made from — a class
method of a named Metal class, a getter whose answer `objc2_metal.tsv` records, a field of the
side's own struct, a parameter's declared type — and resolves the call against the type it lands
on, walking that type's supertypes to the one that declares the selector (`MTLTexture`'s `label`
is `MTLResource`'s item).

The resolution checks itself: a receiver typed `T` sent a selector that `objc2-metal` does not bind
on `T` or any of its supertypes is a **typing error**, not a Metal call, and
[`Leg.mistyped`][Leg] carries it so the caller can fail rather than quietly count the wrong item.
The honest limit is the other direction, and it is kept in two piles rather than one, because they
cost different things. A site whose selector `objc2-metal` binds on no Metal type at all
(`alloc`, `class`) is in [`Leg.foreign`][Leg]: no item exists, so nothing is hidden. A site whose
selector some Metal type DOES bind, but whose receiver this could not type, is in
[`Leg.unresolved`][Leg], and that one hides items: the call is real, one of the types binding the
name received it, and the caller can enumerate which. Pooled together they read as one harmless
limit, which is how eight items of one family read as uncalled while the leg demonstrably called
them. Neither pile is dropped.

A message to a class — `MTLRenderPassDescriptor::renderPassDescriptor()` — needs no typing at all:
the path names its receiver. It is resolved against that class the same way an instance send is
resolved against the type its receiver typed to.

A call site also carries where it sits: the function that holds it, and the **gates** around it, so
a call made only for one step of the case vocabulary is attributed to that step and not to every
case that encodes any step at all. Two kinds of gate are read, and a call carries every one that
encloses it:

* a `match` arm's variant, as `Enum::Variant` ([`arms_of`]), which a case satisfies by naming that
  variant;
* a guard on one field of the suite's **own** vocabulary, as `field:name` ([`guards_of`]), which a
  case satisfies by stating that field. One shared encoder sets fifty fields of a descriptor, each
  behind `if let Some(x) = pass.field`, so without this a case that states one of them reads as
  reaching all fifty.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

#: Rust types that wrap the Metal type a value really is.
UNWRAP = re.compile(r"\b(?:Option|Retained|ProtocolObject|Vec|Box|NonNull|dyn|mut|Self)\b")

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Calls that answer what they were given: an `Option` or `Result` adapter around a Metal object,
#: which the suite's leg uses to turn a `nil` into an error. Following them is what lets
#: `device.newCommandQueue().ok_or_else(..)?` type as an `MTLCommandQueue`.
THROUGH = {
    "ok_or",
    "ok_or_else",
    "unwrap",
    "expect",
    "clone",
    "as_ref",
    "as_deref",
    "to_owned",
    "retain",
    "map_err",
    "unwrap_or_else",
    "borrow",
    # A pool lookup and an `Option`'s contents: `self.libraries.get(i)`, `self.encoder.take()`.
    "get",
    "take",
    "first",
    "last",
    "pop",
    "as_mut",
}


def blanked(text: str) -> str:
    """`text` with comments and string literals replaced by spaces of the same length.

    Keeping the length means every offset still points where it did, so a brace walk over the
    result can be read against the original text.
    """
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        two = text[i : i + 2]
        if two == "//":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif two == "/*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and text[i - 1 : i + 1] != "*/":
                out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                i += 1
        elif text[i] == '"':
            out[i] = " "
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    out[i] = " "
                    i += 1
                if i < n:
                    out[i] = " "
                    i += 1
            # The quote that closes it, so the next one found opens a string rather than closing
            # the one just left.
            if i < n:
                out[i] = " "
                i += 1
        elif text[i] == "'" and text[i + 1 : i + 2] and text[i + 2 : i + 3] == "'":
            # A character literal. A lifetime (`'a`) is one quote and is left alone.
            out[i] = out[i + 1] = out[i + 2] = " "
            i += 3
        else:
            i += 1
    return "".join(out)


def block(text: str, open_at: int) -> tuple[int, int]:
    """The half-open span of the brace block that opens at or after `open_at`."""
    start = text.find("{", open_at)
    if start < 0:
        return (len(text), len(text))
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return (start, i + 1)
    return (start, len(text))


def metal_type(decl: str) -> str:
    """The Metal class or protocol a type declaration is, or `""`.

    `&ProtocolObject<dyn MTLDevice>`, `Retained<MTLTextureDescriptor>` and
    `Vec<Retained<ProtocolObject<dyn MTLBuffer>>>` each name one Metal type; anything naming none,
    or more than one, is not a receiver this follows.
    """
    bare = UNWRAP.sub(" ", decl)
    names = [w for w in IDENT.findall(bare) if w.startswith("MTL")]
    return names[0] if len(names) == 1 and len(set(names)) == 1 else ""


@dataclass
class Api:
    """What `objc2-metal` binds: which type answers which selector, and with what."""

    #: `(type, rust name)` to the selector it sends.
    selector: dict[tuple[str, str], str]
    #: `(type, rust name)` to the Metal type the call answers, where it answers one.
    answers: dict[tuple[str, str], str]
    #: A type to the types it inherits from, nearest first.
    supertypes: dict[str, list[str]]
    #: Every Objective-C type `objc2-metal` binds, to whether it is a `class` or a `protocol`.
    #: A name outside this is not a Metal type at all, however it is spelled: a Rust struct the
    #: crate also defines (`MTLSize`), a bitflags set (`MTLBlitOption`) or an enum.
    kinds: dict[str, str]

    @classmethod
    def read(cls, path: Path) -> "Api":
        selector: dict[tuple[str, str], str] = {}
        answers: dict[tuple[str, str], str] = {}
        supertypes: dict[str, list[str]] = {}
        kinds: dict[str, str] = {}
        section = ""
        for line in path.read_text().splitlines():
            if line.startswith("#"):
                section = line
                continue
            fields = line.split("\t")
            if "supertypes" in section:
                supertypes[fields[0]] = [f for f in fields[2].split(" ") if f]
                kinds[fields[0]] = fields[1]
            else:
                ty, sel, rust, answered = fields
                selector[(ty, rust)] = sel
                if answered:
                    answers[(ty, rust)] = answered
        return cls(selector, answers, supertypes, kinds)

    def declares(self, ty: str) -> bool:
        """Whether `objc2-metal` binds `ty` as an Objective-C class or protocol."""
        return ty in self.kinds

    def chain(self, ty: str) -> list[str]:
        """`ty` and every type it inherits from, nearest first."""
        out = [ty]
        i = 0
        while i < len(out):
            for parent in self.supertypes.get(out[i], []):
                if parent not in out:
                    out.append(parent)
            i += 1
        return out

    def binds(self, ty: str, rust: str) -> tuple[str, str] | None:
        """The `(declaring type, selector)` a receiver of type `ty` sends `rust` to, or `None`."""
        for owner in self.chain(ty):
            if (owner, rust) in self.selector:
                return (owner, self.selector[(owner, rust)])
        return None

    def answered(self, ty: str, rust: str) -> str:
        """The Metal type a call of `rust` on `ty` answers, or `""`."""
        for owner in self.chain(ty):
            if (owner, rust) in self.selector:
                return self.answers.get((owner, rust), "")
        return ""

    def receivers(self, rust: str) -> set[str]:
        """Every type that binds the Rust name `rust`."""
        return {ty for ty, name in self.selector if name == rust}


@dataclass
class Call:
    """One Metal call the leg's source makes."""

    #: The file it is in, relative to the crate.
    file: str
    #: The function that holds it.
    function: str
    #: The gates around it, outermost first: a `match` arm's variant as `Enum::Variant`, and a
    #: guard on one field of the suite's own vocabulary as `field:name`.
    gates: tuple[str, ...]
    #: The Rust name, which is the selector with `:` as `_`.
    rust: str
    #: The receiver's type, or `""` where it could not be typed.
    receiver: str
    #: The SDK item, as `(declaring type, selector)`, where the receiver typed.
    item: tuple[str, str] | None


@dataclass
class Leg:
    """Every Metal call the leg makes, and what the typing could not reach."""

    calls: list[Call] = field(default_factory=list)
    #: Call sites this attributes to no item although some Metal type binds the selector, as
    #: `(file, function, rust name, receiver text)`. The call is real and one of those types
    #: received it; which one is what the typing could not say, so these are the sites that hide
    #: items, and the candidates are enumerable from `receivers`.
    unresolved: list[tuple[str, str, str, str]] = field(default_factory=list)
    #: Call sites whose selector `objc2-metal` binds on no Metal type at all, as
    #: `(file, function, rust name, receiver text)`: `alloc` and `class`, which are `NSObject`'s.
    #: There is no item to hide, so these are counted apart from `unresolved`.
    foreign: list[tuple[str, str, str, str]] = field(default_factory=list)
    #: Call sites whose receiver typed to something that does not bind the selector: a bug in this
    #: typing, never a Metal call.
    mistyped: list[tuple[str, str, str, str]] = field(default_factory=list)
    #: Call sites whose receiver did not type but whose Rust name only one type in `objc2-metal`
    #: binds, so the receiver is that type whatever made it.
    by_uniqueness: list[tuple[str, str, str]] = field(default_factory=list)
    #: Which functions call which, by name, for the reachability closure.
    calls_function: dict[str, set[str]] = field(default_factory=dict)
    #: The gates each function's calls to other functions sit under, the same two kinds as
    #: [`Call.gates`][Call]: what a case must name, and what it must state, to reach through here.
    function_gates: dict[tuple[str, str], set[tuple[str, ...]]] = field(default_factory=dict)


@dataclass
class Function:
    """A function of the source, with the span of its body."""

    name: str
    #: The `impl` header it is in, or `""` at module level.
    impl: str
    file: str
    signature: str
    start: int
    end: int


def body_brace(text: str, after: int) -> int:
    """Where the body of the `fn` whose name ends at `after` opens, or `-1` for a declaration.

    The signature is skipped by **bracket depth**, not by looking for the first `{` and treating a
    `;` before it as the end of a declaration. A parameter of an array type carries a `;` inside
    its brackets — `imageblock: [u32; 3]` — and reading that `;` as the end of a declaration drops
    the whole function, with every Metal call it makes. Measured when this was written: it dropped
    fourteen functions of the Metal side, `texture_2d`, `read_texture`, `replace_region`,
    `dispatch`, `dispatch_threads` and `tile_pass` among them, so their calls were attributed to no
    item at all and were missing from the unresolved count too.

    `->` is skipped as a unit, because its `>` is not a closing bracket.
    """
    depth = 0
    i = after
    while i < len(text):
        two = text[i : i + 2]
        if two == "->":
            i += 2
            continue
        char = text[i]
        if char in "([<":
            depth += 1
        elif char in ")]>":
            depth = max(0, depth - 1)
        elif depth == 0 and char == "{":
            return i
        elif depth == 0 and char == ";":
            return -1
        i += 1
    return -1


def functions(text: str, file: str) -> list[Function]:
    """Every function the (blanked) source declares, with its body's span."""
    out: list[Function] = []
    impls: list[tuple[str, int]] = []
    for match in re.finditer(r"\bimpl\b[^{;]*\{", text):
        header = match.group(0)[:-1].strip()
        span = block(text, match.start())
        impls.append((header, span[1]))
        impls[-1] = (header, span[1])
    for match in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", text):
        open_at = body_brace(text, match.end())
        if open_at < 0:
            # A declaration with no body (a trait's required method).
            continue
        start, end = block(text, match.end())
        here = [h for h, stop in impls if h and match.start() < stop and start < stop]
        out.append(
            Function(
                name=match.group(1),
                impl=min(here, key=len) if here else "",
                file=file,
                signature=text[match.end() : start],
                start=start,
                end=end,
            )
        )
    return out


def arms_of(text: str, start: int, end: int, enums: set[str]) -> list[tuple[int, int, str]]:
    """Every `match` arm in `text[start:end]`, with the variants of `enums` its pattern names.

    Read from the `match` keyword, not from the variant names: a call inside an arm is only gated
    by the arm it is in, and `Failure::named(..)` before an unrelated `=>` is not an arm at all.
    An arm matching several variants (`A | B`) is attributed to each, and a nested `match` is
    found by this same walk, so a call carries every arm around it.
    """
    out: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\bmatch\b", text[start:end]):
        at = start + match.start()
        open_at, close = block(text, at)
        if close > end:
            continue
        i = open_at + 1
        while i < close - 1:
            arrow = _at_depth(text, i, close - 1, "=>")
            if arrow < 0:
                break
            pattern = text[i:arrow]
            body_end = _arm_end(text, arrow + 2, close - 1)
            for name in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)", pattern):
                if name.group(1) in enums:
                    out.append((arrow, body_end, f"{name.group(1)}::{name.group(2)}"))
            i = body_end + 1
    return out


#: How a gate on a field of the suite's own vocabulary is spelled, so a reader of a gate set can
#: tell it from a `match` arm's `Enum::Variant` and strip it where only the variants are wanted.
GATE_FIELD = "field:"

#: The `if` conditions whose body is reached only where the field is **stated**, read from the
#: condition text between `if` and the block it opens. An `Option` unwrapped, one asked whether it
#: is there, and a collection asked whether it is not empty: in each the field names what the
#: case had to state for the call inside to happen at all.
#:
#: A bare boolean field (`if descriptor.read_only {`) is **not** one of these. Its body is reached
#: on one of the two values the field can hold, and which value a case states is nowhere in the
#: text this reads -- the same limit that keeps a field axis from carrying a two-values check --
#: so reading it as a gate would refuse a case that states the field and sets it false.
STATED = (
    # `if let Some(x) = pass.visibility`, `if let Some(depth) = &pass.depth`, and the same with an
    # adapter on the end: `if let Some(label) = self.pass.label.as_ref()`.
    re.compile(
        r"^let\s+(?:Some|Ok)\s*\(.*\)\s*=\s*&?\s*(?:mut\s+)?[A-Za-z_][\w.]*?\.(\w+)\s*"
        r"(?:\.\s*(?:as_ref|as_deref|as_mut|clone|to_owned)\s*\(\s*\))?$"
    ),
    re.compile(r"^&?\s*[A-Za-z_][\w.]*?\.(\w+)\s*\.\s*is_some\s*\(\s*\)$"),
    re.compile(r"^!\s*&?\s*[A-Za-z_][\w.]*?\.(\w+)\s*\.\s*is_empty\s*\(\s*\)$"),
)


def _condition_end(text: str, start: int, end: int) -> int:
    """Where the `if` condition that starts at `start` ends: the `{` it opens, or `-1`.

    Only round and square brackets nest here: a `{` at bracket depth zero is the body's, and one
    inside a call (`if let Some(x) = make(Thing { a: 1 })`) is not.
    """
    depth = 0
    for i in range(start, end):
        c = text[i]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "{" and depth == 0:
            return i
        elif c == ";" and depth == 0:
            # A `let .. else` or a statement boundary: this `if` opens no block here.
            return -1
    return -1


def guards_of(text: str, start: int, end: int) -> list[tuple[int, int, str]]:
    """Every guard in `text[start:end]` on a field of the suite's own vocabulary, with its span.

    The gate is the **field's name**, as `field:name`, and the span is the guarded block alone: an
    `else` branch opens its own block, which this one does not cover, so a call Metal makes only
    where the field is absent carries no gate for it.

    This is what makes an item's attribution per **field** rather than per shared encoder. The
    Metal leg builds one descriptor in one function, setting each field under its own guard, so
    every case that calls that encoder reaches every call site in it; the guard says which of them
    the case's own data could ever have taken.

    The field is read by name and not by the struct it belongs to, because the typing here follows
    Metal's types and not the suite's. Two vocabulary structs with a field of one name are read as
    one gate, which is generous in the safe direction: it can let a declaration through, never
    refuse a true one.
    """
    out: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\bif\s", text[start:end]):
        at = start + match.end()
        brace = _condition_end(text, at, end)
        if brace < 0:
            continue
        condition = " ".join(text[at:brace].split())
        for shape in STATED:
            found = shape.match(condition)
            if found:
                out.append((brace, block(text, brace)[1], GATE_FIELD + found.group(1)))
                break
    return out


def _at_depth(text: str, start: int, end: int, want: str) -> int:
    """Where `want` next appears in `text[start:end]` at bracket depth zero."""
    depth = 0
    i = start
    while i < end:
        c = text[i]
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
        elif depth == 0 and text[i : i + len(want)] == want:
            return i
        i += 1
    return -1


def _arm_end(text: str, start: int, end: int) -> int:
    """Where the arm body that starts at `start` ends: its block, or the comma after it."""
    i = start
    while i < end and text[i].isspace():
        i += 1
    if i < end and text[i] == "{":
        return block(text, i)[1]
    comma = _at_depth(text, i, end, ",")
    return comma if comma >= 0 else end


def receiver_of(text: str, at: int) -> str:
    """The text of the receiver expression whose `.` is at `at`.

    Whitespace is part of an expression because `rustfmt` breaks a call chain across lines, so a
    receiver is very often the line above the `.`; a newline is only an end when what precedes it
    is not an expression at all.
    """
    i = at
    depth = 0
    while i > 0:
        c = text[i - 1]
        if c in ")]":
            depth += 1
        elif c in "([":
            if depth == 0:
                break
            depth -= 1
        elif c in "{}":
            break
        elif c.isspace():
            # Whitespace continues a chain only where an expression precedes it.
            before = text[:i].rstrip()
            if depth == 0 and not (before and (before[-1].isalnum() or before[-1] in "_)]?")):
                break
        elif depth == 0 and not (c.isalnum() or c in "_.:&*?"):
            break
        i -= 1
    return " ".join(text[i:at].split())


class Typing:
    """What each name in one function's body is.

    A name is typed by the `let` that binds it **nearest before** the use, so a body that reuses
    one name for two objects — `let attachment = ...depthAttachment()` after
    `let attachment = ...colorAttachments()...` — types each use as what it was last made from
    rather than as whichever binding the file happens to end with.
    """

    def __init__(self, api: Api, fields: dict[str, str], locals_: dict[str, str] | None = None):
        self.api = api
        self.fields = fields
        #: The Metal type each function of the leg itself answers, where it answers one. A pool
        #: getter (`self.held(handle)`) is how most of this source reaches a Metal object.
        self.locals = locals_ or {}
        #: Each name's bindings, as `(offset, declared type, initializer)`, in source order.
        self.bindings: dict[str, list[tuple[int, str, str]]] = {}
        #: Parameters, which hold for the whole body.
        self.names: dict[str, str] = {}
        #: Where the expression being typed sits, so a binding after it does not answer it.
        self.at = 0

    def closure_params(self, body: str, offset: int) -> None:
        """Take the declared types of every closure parameter in `body`.

        A closure is how this source hands one encoder's work to two call sites, so its parameters
        type as many receivers as a function's do.
        """
        for match in re.finditer(r"\|([^|]*:[^|]*)\|", body):
            for part in _split(match.group(1)):
                if ":" not in part:
                    continue
                name, decl = part.split(":", 1)
                name = name.strip().removeprefix("mut ").strip()
                ty = metal_type(decl)
                if IDENT.fullmatch(name) and ty:
                    self.bind(offset + match.start(), name, decl, "")

    def param_types(self, signature: str) -> None:
        """Take the declared types of a signature's parameters."""
        depth = 0
        start = signature.find("(")
        parts: list[str] = []
        piece = ""
        for c in signature[start + 1 :]:
            if c in "(<[":
                depth += 1
            elif c in ")>]":
                if depth == 0:
                    break
                depth -= 1
            if c == "," and depth == 0:
                parts.append(piece)
                piece = ""
            else:
                piece += c
        parts.append(piece)
        for part in parts:
            if ":" not in part:
                continue
            name, decl = part.split(":", 1)
            name = name.strip().removeprefix("mut ").strip()
            if IDENT.fullmatch(name):
                ty = metal_type(decl)
                if ty:
                    self.names[name] = ty

    def expression(self, expr: str) -> str:
        """The Metal type `expr` is, or `""`."""
        expr = _unsafe_as_parens(" ".join(expr.split()))
        expr = expr.strip("&*").strip().rstrip("?;").strip()
        if not expr:
            return ""
        if expr.endswith(")"):
            open_at = _matching(expr)
            if open_at >= 0:
                head = expr[:open_at].strip()
                if "::" in head:
                    ty, _, called = head.rpartition("::")
                    ty = ty.strip().strip("&*<>").split("::")[-1].strip()
                    return self.api.answered(ty, called.strip()) if ty.startswith("MTL") else ""
                if "." in head:
                    recv, _, called = head.rpartition(".")
                    called = called.strip()
                    if called in THROUGH:
                        # An `Option` or `Result` adapter answers what it was given.
                        return self.expression(recv)
                    if recv.strip() == "self":
                        return self.locals.get(called, "")
                    owner = self.expression(recv)
                    return self.api.answered(owner, called) if owner else ""
                if not head:
                    # A parenthesised expression, or the block of an `unsafe` the next rule turned
                    # into one.
                    return self.expression(expr[open_at + 1 : -1])
                if head.strip() in self.locals:
                    return self.locals[head.strip()]
                return ""
        if expr.endswith("]"):
            open_at = expr.rfind("[")
            if open_at > 0:
                return self.expression(expr[:open_at])
        if expr.startswith("MTL") and "::" not in expr and IDENT.fullmatch(expr):
            return expr
        steps = [piece.strip() for piece in expr.split(".")]
        head = steps[0].split("[")[0].strip()
        owner = self.fields if head == "self" else None
        if owner is not None:
            steps = steps[1:]
            if not steps:
                return ""
            ty = self.fields.get(steps[0].split("[")[0], "")
            steps = steps[1:]
        else:
            ty = self.name(head)
            steps = steps[1:]
        for step in steps:
            if not ty:
                return ""
            ty = self.api.answered(ty, step.split("[")[0].strip())
        return ty

    def bind(self, at: int, name: str, decl: str, expr: str) -> None:
        """Take `let name: decl = expr`, bound at offset `at`."""
        self.bindings.setdefault(name, []).append((at, decl, expr))

    def name(self, name: str) -> str:
        """The Metal type `name` holds where it is being used, or `""`."""
        before = [b for b in self.bindings.get(name, []) if b[0] < self.at]
        if before:
            at, decl, expr = before[-1]
            ty = metal_type(decl) if decl else ""
            if ty:
                return ty
            here, self.at = self.at, at
            try:
                return self.expression(expr)
            finally:
                self.at = here
        return self.names.get(name, "")

    def typed(self, expr: str, at: int) -> str:
        """The Metal type the expression `expr`, which sits at offset `at`, is."""
        self.at = at
        return self.expression(expr)


def _unsafe_as_parens(expr: str) -> str:
    """`expr` with each `unsafe { .. }` block written as `( .. )`.

    An `unsafe` block is an expression here — `unsafe { device.newX(..) }.ok_or_else(..)?` — so
    reading its braces as parentheses lets one rule type the block and the chain after it.
    """
    out = expr
    while True:
        at = out.find("unsafe")
        if at < 0:
            return out
        rest = out[at + len("unsafe") :]
        if not rest.lstrip().startswith("{"):
            return out
        open_at = at + len("unsafe") + rest.index("{")
        depth = 0
        close = len(out) - 1
        for i in range(open_at, len(out)):
            if out[i] == "{":
                depth += 1
            elif out[i] == "}":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        out = out[:at] + "(" + out[open_at + 1 : close] + ")" + out[close + 1 :]


def _split(text: str) -> list[str]:
    """`text` split on the commas at depth zero."""
    out: list[str] = []
    piece = ""
    depth = 0
    for c in text:
        if c in "(<[":
            depth += 1
        elif c in ")>]":
            depth -= 1
        if c == "," and depth == 0:
            out.append(piece)
            piece = ""
        else:
            piece += c
    out.append(piece)
    return out


def _matching(expr: str) -> int:
    """Where the `(` that closes at `expr`'s last character opens."""
    depth = 0
    for i in range(len(expr) - 1, -1, -1):
        if expr[i] in ")]}":
            depth += 1
        elif expr[i] in "([{":
            depth -= 1
            if depth == 0:
                return i
    return -1


def struct_fields(text: str, name: str) -> dict[str, str]:
    """The Metal type of each field of struct `name`, where it has one."""
    match = re.search(rf"\bstruct\s+{name}\b[^{{;]*\{{", text)
    if not match:
        return {}
    start, end = block(text, match.start())
    out: dict[str, str] = {}
    depth = 0
    piece = ""
    for c in text[start + 1 : end - 1]:
        if c in "(<[":
            depth += 1
        elif c in ")>]":
            depth -= 1
        if c == "," and depth == 0:
            if ":" in piece:
                field_name, decl = piece.split(":", 1)
                ty = metal_type(decl)
                if ty:
                    out[field_name.strip().removeprefix("pub ").strip()] = ty
            piece = ""
        else:
            piece += c
    if ":" in piece:
        field_name, decl = piece.split(":", 1)
        ty = metal_type(decl)
        if ty:
            out[field_name.strip().removeprefix("pub ").strip()] = ty
    return out


def read_leg(api: Api, src: Path, files: list[str], enums: set[str]) -> Leg:
    """Every Metal call the leg's `files` make, typed."""
    leg = Leg()
    blank = {name: blanked((src / name).read_text()) for name in files}
    fields: dict[str, str] = {}
    answers: dict[str, str] = {}
    for name, text in blank.items():
        for struct in re.findall(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)", text):
            fields.update(struct_fields(text, struct))
        for fn in functions(text, name):
            arrow = fn.signature.rfind("->")
            if arrow >= 0:
                ty = metal_type(fn.signature[arrow + 2 :])
                if ty:
                    answers[fn.name] = ty
    for name, text in blank.items():
        for fn in functions(text, name):
            gates = arms_of(text, fn.start, fn.end, enums) + guards_of(text, fn.start, fn.end)
            typing = Typing(api, fields, answers)
            typing.param_types(fn.signature)
            typing.closure_params(text[fn.start : fn.end], fn.start)
            # Every `let` of the body, so a use of a name is typed by the one nearest before it.
            # A binding, including the `let Some(x) = ..` and `let Ok(x) = ..` forms this source
            # uses to unwrap a pool slot or an `Option` field.
            for match in re.finditer(
                r"\blet\s+(?:mut\s+)?(?:(?:Some|Ok)\(\s*(?:mut\s+)?)?"
                r"([A-Za-z_][A-Za-z0-9_]*)\)?\s*(?::([^=;]*))?=",
                text[fn.start : fn.end],
            ):
                at = fn.start + match.end()
                stop = _statement_end(text, at, fn.end)
                typing.bind(fn.start + match.start(), match.group(1), match.group(2) or "", text[at:stop])
            # `r#type`, which is how `objc2` spells a binding whose selector is a Rust keyword.
            # Without the `r#` such a call site matches nothing at all — the identifier after the
            # dot is `r`, which is not followed by `(` — so every `MTLBinding`'s and
            # `MTLArgument`'s `type` the leg reads was invisible here.
            for match in re.finditer(
                r"\.\s*(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", text[fn.start : fn.end]
            ):
                rust = match.group(1)
                at = fn.start + match.start()
                here = tuple(v for lo, hi, v in gates if lo <= at < hi)
                owners = api.receivers(rust)
                recv = receiver_of(text, at)
                if recv == "self" or not owners:
                    # `self` is the leg's own side, never a Metal object, so a name it shares with
                    # a selector (`buffer`, `texture`) is this source's own function.
                    # Not a Metal call at all. A call to another function of the leg is an edge of
                    # the reachability graph instead.
                    leg.calls_function.setdefault(fn.name, set()).add(rust)
                    leg.function_gates.setdefault((fn.name, rust), set()).add(here)
                    continue
                ty = typing.typed(recv, at)
                if not ty and len(owners) == 1:
                    # Only one type binds the name, so no typing is needed to know the receiver.
                    ty = next(iter(owners))
                    leg.by_uniqueness.append((name, fn.name, rust))
                if not ty:
                    leg.unresolved.append((name, fn.name, rust, recv))
                    leg.calls.append(Call(name, fn.name, here, rust, "", None))
                    continue
                item = api.binds(ty, rust)
                if item is None:
                    leg.mistyped.append((name, fn.name, rust, ty))
                    leg.calls.append(Call(name, fn.name, here, rust, ty, None))
                    continue
                leg.calls.append(Call(name, fn.name, here, rust, ty, item))
            # `MTLType::selector(..)`, which is how `objc2` spells a message to the **class**: a
            # descriptor's own constructor (`MTLRenderPassDescriptor::renderPassDescriptor()`) and
            # the `init` an `alloc()` is handed. The receiver is the class the path names, so
            # nothing has to be typed; the path is resolved against what `objc2-metal` binds on
            # that class, the same way an instance send is.
            #
            # `\s*` after each `::` because a long constructor's name does not fit beside its
            # class: `MTLRasterizationRateMapDescriptor::` ends one line and
            # `rasterizationRateMapDescriptorWithScreenSize_layer(..)` begins the next, and with
            # the whitespace unmatched that call site was in no table at all -- not a call, not
            # unresolved, nowhere.
            for match in re.finditer(
                r"\b((?:[A-Za-z_][A-Za-z0-9_]*::\s*)+)(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                text[fn.start : fn.end],
            ):
                rust = match.group(2)
                at = fn.start + match.start()
                ty = [part for part in match.group(1).split("::") if part.strip()][-1].strip()
                if not api.declares(ty):
                    # Not an Objective-C type of `objc2-metal` at all, so this sends no message:
                    # a Rust type's associated function (`Vec::new`), an enum's variant, or a
                    # bitflags constructor whose name only looks like Metal's
                    # (`MTLBlitOption::empty`). The loop below reads it as the leg's own call.
                    continue
                here = tuple(v for lo, hi, v in gates if lo <= at < hi)
                item = api.binds(ty, rust)
                if item is None:
                    # The class named here does not bind this selector. Not `mistyped`: nothing was
                    # typed wrongly, the path says its own receiver. Which pile it belongs in is
                    # whether ANY Metal type binds the name: `alloc` and `class` are `NSObject`'s
                    # and so are an item of no Metal container, while a name some other Metal type
                    # binds means a real item received this call and the table cannot say which.
                    pile = leg.unresolved if api.receivers(rust) else leg.foreign
                    pile.append((name, fn.name, rust, ty))
                    leg.calls.append(Call(name, fn.name, here, rust, "", None))
                    continue
                leg.calls.append(Call(name, fn.name, here, rust, ty, item))
            # A call to another function of the leg: an edge of the reachability graph. A path
            # qualifier is kept where it is a module (`crate::metal_work::run`) or `Self`, and
            # dropped where it is a type, because `MTLTextureDescriptor::new` is the class message
            # the loop above resolves and not this source's own `new`.
            #
            # A method send is excluded by the `.` before its name, and by that alone: `rustfmt`
            # breaks a chain **before** the dot, so a continuation line reads `.label(` and the
            # dot is beside the name wherever the call sits. Measured over the whole leg: 0 call
            # sites have whitespace between the dot and the name.
            for match in re.finditer(
                r"\b((?:[A-Za-z_][A-Za-z0-9_]*::)*)([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                text[fn.start : fn.end],
            ):
                rust = match.group(2)
                at = fn.start + match.start()
                if text[at - 1 : at] == "." or rust in ("fn", "if", "match", "while", "for"):
                    continue
                qualifier = [part for part in match.group(1).split("::") if part]
                if qualifier and qualifier[-1][:1].isupper() and qualifier[-1] != "Self":
                    continue
                here = tuple(v for lo, hi, v in gates if lo <= at < hi)
                leg.calls_function.setdefault(fn.name, set()).add(rust)
                leg.function_gates.setdefault((fn.name, rust), set()).add(here)
    return leg


def _statement_end(text: str, at: int, end: int) -> int:
    """Where the statement that starts at `at` ends: its `;` at depth zero."""
    depth = 0
    for i in range(at, end):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return i
            depth -= 1
        elif c == ";" and depth == 0:
            return i
    return end
