#!/usr/bin/env python3
"""The signature-level public surface of the three published crates, against its committed table.

    signatures.py --check     # derive it and compare with the table; the diff, and a bad status
    signatures.py --bless     # write the table, printing what moved
    signatures.py --print     # the derived records, one a line
    signatures.py --selftest  # the derivation over little trees whose answer is known

`crates/metal2vulkan/tests/api_signatures.tsv` is the table, and
`crates/metal2vulkan/tests/api_signatures.rs` says what it pins, what it cannot see, and what
blessing it asserts. Every verb takes `--root DIR` (the repository, the default being this
script's own two directories up) and `--table FILE`.

It reads source text only: no build, no toolchain, no rustdoc, whose JSON surface is unstable and
changes between nightlies. So it is the same five seconds on any checkout, under any
feature set: a `#[cfg]`-gated item is read as present, with its gate kept in the signature.

The surface is the set of names a consumer may write -- the module roots the crates export, their
`pub use` leaves and the items a declaring macro states there -- and for each name every signature
the sources give it anywhere in the crate: the declaration, its public fields or variants, the
`pub fn`, `pub const` and `pub type` of its inherent impls wherever they live, the header of each
of its trait impls, and a public trait's own items. The sources are `cargo fmt`-formatted, so a
declaration's indentation names its nesting level.

A record is `kind<TAB>key<TAB>signature`. A constant's declaration and its value are two records
(`const` and `const-value`, `assoc-const` and `assoc-const-value`), so that a value that moves and
a signature that moves are two lines to bless rather than one; the signature records carry no
value at all.
"""
import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------- masking

TOKEN = re.compile(r'//|/\*|r#*"|b?"|\'')


def strip(text):
    """(code, mask): `text` with comments blanked, and with literals blanked as well.

    Both keep every offset and newline of `text`. Slicing `code` gives a signature with no comment
    in it; matching braces on `mask` never trips over one inside a literal.
    """
    code = list(text)
    lit = []          # the literal ranges, blanked in the mask but kept in the code
    n = len(text)
    i = 0
    while True:
        mo = TOKEN.search(text, i)
        if mo is None:
            break
        i = mo.start()
        kind = mo.group(0)
        if kind == "//":
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                code[k] = " "
            i = j
        elif kind == "/*":
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            for k in range(i, j):
                if code[k] != "\n":
                    code[k] = " "
            i = j
        elif kind.startswith("r"):
            close = '"' + kind[1:]
            j = text.find(close, mo.end())
            j = n if j < 0 else j + len(close)
            lit.append((i, j))
            i = j
        elif kind.endswith('"'):
            j = mo.end()
            while j < n:
                if text[j] == "\\":
                    j += 2
                elif text[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            lit.append((i, j))
            i = j
        else:  # a quote: a char literal, or a lifetime
            if i + 1 < n and text[i + 1] == "\\":
                j = text.find("'", i + 2)
                j = n if j < 0 else j + 1
                lit.append((i, j))
                i = j
            elif i + 2 < n and text[i + 2] == "'":
                lit.append((i, i + 3))
                i += 3
            else:
                i += 1
    code = "".join(code)
    m = list(code)
    for a, b in lit:
        for k in range(a, b):
            if m[k] != "\n":
                m[k] = " "
    return code, "".join(m)


def match_pair(m, i, open_c, close_c):
    """The offset just past the `close_c` closing the `open_c` at `i`."""
    depth = 0
    while i < len(m):
        if m[i] == open_c:
            depth += 1
        elif m[i] == close_c:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(m)


# ---------------------------------------------------------------- declarations

VIS = re.compile(r"^pub(\s*\([^)]*\))?\s+")
KEEP_ATTRS = re.compile(r"^#\[(cfg|cfg_attr|non_exhaustive|derive|repr|must_use|deprecated|"
                        r"track_caller|doc\(hidden)")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MACROCALL = re.compile(r"^([a-z_][a-z0-9_]*)\s*!\s*[{(\[]")
MODIFIERS = ("default ", "unsafe ", "async ", 'extern "C" ', 'extern "system" ', "const ")
KINDS = ("fn ", "struct ", "enum ", "trait ", "type ", "const ", "static ", "mod ", "union ")


class Item:
    __slots__ = ("vis", "kind", "name", "header", "decl", "attrs", "body")

    def __init__(self, vis, kind, name, header, decl, attrs, body):
        self.vis, self.kind, self.name = vis, kind, name
        self.header, self.decl = header, decl
        self.attrs, self.body = attrs, body

    def hidden(self):
        return any(a.startswith("#[doc(hidden") for a in self.attrs)

    def test_only(self):
        return any(a.startswith("#[cfg(test") or a.startswith("#[cfg(all(test") for a in self.attrs)


def classify(decl):
    rest = decl
    changed = True
    while changed:
        changed = False
        for mo in MODIFIERS:
            if rest.startswith(mo):
                if mo == "const " and not rest.startswith("const fn "):
                    continue
                rest, changed = rest[len(mo):], True
                break
    for kw in KINDS:
        if rest.startswith(kw):
            m = IDENT.search(rest[len(kw):])
            return kw.strip(), (m.group(0) if m else "?")
    if rest.startswith("impl") and not rest[4:5].isalnum() and rest[4:5] != "_":
        return "impl", ""
    if rest.startswith("use "):
        return "use", ""
    if rest.startswith("macro_rules!"):
        m = IDENT.search(rest[len("macro_rules!"):])
        return "macro", (m.group(0) if m else "?")
    mo = MACROCALL.match(rest)
    if mo:
        return "macrocall", mo.group(1)
    return None, None


def items(code, m, start, end, indent):
    """Every declaration at exactly `indent` columns between `start` and `end`."""
    out, pending = [], []
    pad = " " * indent
    i = start
    while i < end:
        eol = code.find("\n", i)
        if eol < 0 or eol > end:
            eol = end
        line = code[i:eol]
        if not line.strip():
            # A blank line separates an item from the attributes above it. `code` has every
            # comment blanked, and the sources are `cargo fmt`-formatted with no trailing
            # whitespace, so a line of spaces was a comment and only an empty one is blank: a
            # doc comment written between an attribute and its item keeps the attribute.
            if not line:
                pending = []
            i = eol + 1
            continue
        if not line.startswith(pad) or line[indent:indent + 1] == " ":
            i = eol + 1
            continue
        at_indent = line[indent:]
        if at_indent.startswith("#["):
            j = match_pair(m, i + indent + 1, "[", "]")
            attr = " ".join(code[i + indent:j].split())
            if KEEP_ATTRS.match(attr):
                pending.append(attr)
            # Step over the newline the attribute's own line ends with, so the zero-length
            # remainder after `]` does not read as the blank line that drops `pending`.
            nl = code.find("\n", j)
            i = nl + 1 if nl >= 0 and not code[j:nl].strip() else j
            continue
        vm = VIS.match(at_indent)
        vis = vm.group(0).strip() if vm else ""
        at = i + indent + (vm.end() if vm else 0)
        kind, name = classify(code[at:at + 400])
        if kind is None:
            pending = []
            i = eol + 1
            continue
        j, stop = at, None
        if kind == "macrocall":
            while j < end and m[j] not in "{([":
                j += 1
            opener = m[j] if j < end else "{"
            close = {"{": "}", "(": ")", "[": "]"}[opener]
            e = match_pair(m, j, opener, close)
            body = (j + 1, e - 1)
            out.append(Item(vis, kind, name, code[i + indent:j], code[at:j], list(pending), body))
            pending = []
            i = e
            continue
        while j < end:
            ch = m[j]
            if ch == "{" and kind != "use":
                stop = "{"
                break
            if ch == ";":
                stop = ";"
                break
            if ch == "{" and kind == "use":
                j = match_pair(m, j, "{", "}")
                continue
            if ch == "(":
                j = match_pair(m, j, "(", ")")
                continue
            if ch == "[":
                # `[T; N]` in a type, and a tuple struct's or a call's own group: the `;` and `{`
                # inside them do not end the declaration.
                j = match_pair(m, j, "[", "]")
                continue
            j += 1
        header = code[i + indent:j]
        body = None
        if stop == "{":
            e = match_pair(m, j, "{", "}")
            body, j = (j + 1, e - 1), e
        else:
            j += 1
        out.append(Item(vis, kind, name, header, code[at:j].rstrip("{;"), list(pending), body))
        pending = []
        i = j
    return out


def norm(s):
    return " ".join(s.split()).rstrip("{").strip()


# One spelling for a declaration `cargo fmt` wrapped and one it left on a line. Wrapping puts a
# space inside the brackets and a trailing comma before the closing one, so a signature that
# crossed the line width would otherwise read as a change in every one of its arguments. A one
# element tuple keeps its comma (`(T,)` is not `(T)`): only `, ` before a bracket goes.
CANON = ((", )", ")"), (", >", ">"), (", ]", "]"), (", }", " }"),
         ("( ", "("), (" )", ")"), ("[ ", "["), (" ]", "]"), ("< ", "<"), (" >", ">"))


def canon(text):
    """`text` with its bracket spacing and wrap commas in one form.

    Only for a declaration. A constant's value keeps the sources' spelling, since `<<`, `>` and
    `(` carry meaning there that this would mangle.
    """
    while True:
        before = text
        for old, new in CANON:
            text = text.replace(old, new)
        if text == before:
            return text


def sig(it, head=None):
    text = canon(norm(head if head is not None else it.header))
    keep = [a for a in it.attrs if not a.startswith("#[doc(hidden")]
    return (" ".join(sorted(keep)) + " " + text).strip() if keep else text


def split_value(text):
    """(declaration, value) of a `const` or `static`, at the `=` that separates them.

    The `=` of an attribute (`#[cfg(feature = "ash")]`) or of a type sits inside brackets, so the
    separator is the first one at bracket depth zero that is not part of a two character operator.
    """
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "=" and depth == 0:
            if text[i - 1:i] in ("=", "!", "<", ">") or text[i + 1:i + 2] == "=":
                continue
            return text[:i].rstrip(), text[i + 1:].strip()
    return text, None


# ---------------------------------------------------------------- the crate tree

SKIP = re.compile(r"(^|/)tests?(/|\.rs$)|_tests\.rs$|(^|/)test_")


class Crate:
    def __init__(self, src, name):
        self.src, self.name = src, name
        self.cache = {}
        self.sources = [p for p in sorted(src.rglob("*.rs"))
                        if not SKIP.search(str(p.relative_to(src)))]

    def read(self, path):
        got = self.cache.get(path)
        if got is None:
            text = path.read_text(encoding="utf-8", errors="replace")
            code, m = strip(text)
            got = (code, m, items(code, m, 0, len(code), 0))
            self.cache[path] = got
        return got

    def module_file(self, segs):
        if not segs:
            return self.src / "lib.rs"
        base = self.src.joinpath(*segs)
        for cand in (base.with_suffix(".rs"), base / "mod.rs"):
            if cand.is_file():
                return cand
        return None


def use_leaves(prefix, tree, out):
    tree = tree.strip()
    if tree.startswith("{") and tree.endswith("}"):
        depth, part = 0, ""
        for ch in tree[1:-1]:
            if ch in "{(":
                depth += 1
            elif ch in "})":
                depth -= 1
            if ch == "," and depth == 0:
                use_leaves(prefix, part, out)
                part = ""
            else:
                part += ch
        if part.strip():
            use_leaves(prefix, part, out)
        return
    if "::{" in tree:
        head, rest = tree.split("::{", 1)
        use_leaves(prefix + [s for s in head.split("::") if s], "{" + rest, out)
        return
    segs = [s for s in tree.split("::") if s]
    if not segs:
        return
    last = segs[-1]
    if " as " in last:
        real, alias = [s.strip() for s in last.split(" as ")]
        out.append((prefix + segs[:-1] + [real], alias))
    else:
        out.append((prefix + segs, last))


# ---------------------------------------------------------------- declaring macros

PARAM_DECL = re.compile(r"pub\s+(enum|struct|fn|const)\s+\$(\w+)")
FIXED_DECL = re.compile(r"pub\s+(enum|struct|fn|const)\s+([A-Z]\w*)")


def declaring_macros(crates):
    """macro name -> the public item it declares: ("param", None) or ("fixed", Name)."""
    out = {}
    for crate in crates:
        for path in crate.sources:
            code, m, decls = crate.read(path)
            for it in decls:
                if it.kind != "macro" or it.body is None:
                    continue
                body = code[it.body[0]:it.body[1]]
                if PARAM_DECL.search(body):
                    out[it.name] = ("param", None)
                else:
                    mo = FIXED_DECL.search(body)
                    if mo:
                        out.setdefault(it.name, ("fixed", mo.group(2)))
    return out


# A declaring call states its item at the call's own member level; what sits deeper are the
# item's values, which `collect` records as that item's members.
CALL_NAME = re.compile(r"^ {4}([A-Z]\w*)\s*(\([^)]*\))?\s*(=|\{|,|$)", re.M)


def call_declares(it, code, macros):
    """The public names a declaring macro call states."""
    what = macros.get(it.name)
    if what is None or it.body is None:
        return []
    if what[0] == "fixed":
        return [what[1]]
    body = code[it.body[0]:it.body[1]]
    return [mo.group(1) for mo in CALL_NAME.finditer(body)]


# ---------------------------------------------------------------- the surface names

def root_names(crate, segs, macros, seen=None):
    """name -> (kind, item) at the root of module `segs`."""
    seen = set() if seen is None else seen
    if tuple(segs) in seen:
        return {}
    seen.add(tuple(segs))
    path = crate.module_file(segs)
    if path is None:
        return {}
    code, m, decls = crate.read(path)
    names = {}
    for it in decls:
        if it.test_only():
            continue
        if it.kind == "macrocall":
            for nm in call_declares(it, code, macros):
                names[nm] = ("macro:" + it.name, it, nm)
            continue
        if it.vis != "pub":
            continue
        if it.kind == "use":
            leaves = []
            use_leaves([], norm(it.decl)[len("use "):], leaves)
            for p, nm in leaves:
                if nm == "*":
                    target = resolve_mod(crate, segs, p[:-1])
                    if target is not None:
                        names.update(root_names(crate, target, macros, seen))
                elif not it.hidden():
                    names[nm] = ("use", it, p[-1] if p else nm)
        elif it.kind == "mod":
            if not it.hidden():
                names[it.name] = ("mod", it, it.name)
        elif it.kind in ("fn", "struct", "enum", "trait", "type", "const", "static", "union"):
            if not it.hidden():
                names[it.name] = (it.kind, it, it.name)
    return names


def resolve_mod(crate, here, segs):
    if not segs:
        return list(here)
    first = segs[0]
    if first == "crate":
        segs = segs[1:]
    elif first == "self":
        segs = list(here) + segs[1:]
    elif first == "super":
        segs = list(here[:-1]) + segs[1:]
    elif first == crate.name.replace("-", "_"):
        segs = segs[1:]
    else:
        cand = list(here) + segs
        if crate.module_file(cand) is not None:
            return cand
    return segs if crate.module_file(segs) is not None else None


TABLE_MOD = re.compile(r'^\s*"([^"]*)" =>')
TABLE_ITEM = re.compile(r"^\s+(struct|enum|fn|const|static|trait|type|mod|hidden)\s+(\w+)")


def table_modules(tree):
    """The surface module paths the committed table names, for `metal2vulkan`.

    `crates/metal2vulkan/tests/api_surface.rs` is the crate's own statement of which module roots
    are the consumer surface, so the tool takes the set from there rather than guessing it from
    `pub mod`: a public submodule inside a root is not surface.
    """
    path = tree / "crates" / "metal2vulkan" / "tests" / "api_surface.rs"
    if not path.is_file():
        return None
    mods, here = {}, None
    inside = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("surface! {"):
            inside = True
            continue
        if not inside:
            continue
        if line == "}":
            break
        mo = TABLE_MOD.match(line)
        if mo:
            here = mo.group(1)
            mods.setdefault(here, {})
            continue
        mo = TABLE_ITEM.match(line)
        if mo and here is not None:
            mods[here][mo.group(2)] = mo.group(1)
    return mods


def definition_file(crate, segs, name, macros, depth=0):
    """The file that declares `name` as module `segs` names it, following `pub use`."""
    if depth > 16:
        return None
    path = crate.module_file(segs)
    if path is None:
        return None
    code, m, decls = crate.read(path)
    globs = []
    for it in decls:
        if it.test_only():
            continue
        if it.kind in DECL_KINDS and it.name == name:
            return path
        if it.kind == "macrocall" and name in call_declares(it, code, macros):
            return path
        if it.kind == "use" and it.vis == "pub":
            leaves = []
            use_leaves([], norm(it.decl)[len("use "):], leaves)
            for pth, nm in leaves:
                if nm == name:
                    target = resolve_mod(crate, segs, pth[:-1])
                    if target is None:
                        return None
                    return definition_file(crate, target, pth[-1], macros, depth + 1)
                if nm == "*":
                    globs.append(pth[:-1])
    for pth in globs:
        target = resolve_mod(crate, segs, pth)
        if target is None:
            continue
        got = definition_file(crate, target, name, macros, depth + 1)
        if got is not None:
            return got
    return None


def declared_in(crate):
    """name -> the files of the crate that declare it, at any visibility."""
    out = {}
    for path in crate.sources:
        _, _, decls = crate.read(path)
        for it in decls:
            if it.kind in DECL_KINDS and not it.test_only():
                out.setdefault(it.name, set()).add(path)
    return out


def surface_names(crate, macros, roots=None):
    """name -> the path a consumer writes it at, for one crate.

    `roots` names the surface modules (the committed table's, for the translator); without it the
    crate root and its own `pub mod` roots are the surface, which is the shape of the runtime's
    and the front end's.
    """
    out = {}
    if roots is None:
        roots = [[]]
        for nm, (kind, it, defname) in sorted(root_names(crate, [], macros).items()):
            if kind == "mod":
                roots.append([nm])
    real, home = {}, {}
    for segs in roots:
        for nm, (kind, it, defname) in sorted(root_names(crate, segs, macros).items()):
            if kind == "mod":
                continue
            if nm not in out:
                out[nm] = "::".join(segs + [nm]) if segs else nm
                real[defname] = nm
                home[nm] = definition_file(crate, segs, defname, macros)
    return out, real, home


# ---------------------------------------------------------------- the signatures

DECL_KINDS = ("struct", "enum", "trait", "type", "const", "static", "union", "fn")
SELF_TY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The constants whose value this repository moves on purpose, several times a day: AGENTS.md
# requires the matching bump whenever behaviour, an encoding or a run changes. Their declarations
# are pinned like any other; their values are not recorded at all, because a table that holds them
# is blessed by reflex within a day, and a check blessed by reflex is worse than no check. Nothing
# else is excused: every other public constant's value is a record. `--check` holds each name here
# to a constant the surface still has, so the list cannot rot into a stale excuse.
VOLATILE_VALUES = (
    "metal2vulkan::cache::ARTIFACT_FORMAT_VERSION",
    "metal2vulkan::cache::RUNTIME_VERSION",
    "metal2vulkan::cache::SEMANTICS_VERSION",
)


def impl_self(header):
    """(trait, base type) of an impl header."""
    h = norm(header)[len("impl"):].strip()
    if h.startswith("<"):
        h = h[match_pair(h, 0, "<", ">"):].strip()
    if " where " in h:
        h = h.split(" where ")[0]
    parts = re.split(r"(?<![\w:])for(?![\w:])", h, maxsplit=1)
    tr, ty = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (None, h)
    ty = re.sub(r"^&\s*('[A-Za-z_]\w*\s*)?(mut\s+)?", "", ty.strip())
    mo = SELF_TY.search(ty)
    return tr, (mo.group(0) if mo else None)


def body_parts(code, m, body):
    """A braced body's comma-separated members, comments and attributes gone."""
    out, depth, part = [], 0, ""
    for k in range(body[0], body[1]):
        ch = m[k]
        if ch in "{([<":
            depth += 1
        elif ch in "})]>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(part)
            part = ""
        else:
            part += code[k]
    out.append(part)
    cleaned = []
    for p in out:
        p = canon(re.sub(r"#\[[^]]*\]", "", " ".join(p.split())).strip())
        if p:
            cleaned.append(p)
    return cleaned


def collect(crate, names, real, home, extra=()):
    """Every signature the sources state for a surface name, keyed by its consumer path.

    The crate's own sources answer first; another crate's (the front end's, whose items this one
    re-exports) answer only for a name the crate itself never declares, so a private type of the
    same name elsewhere never joins a surface name's record.
    """
    recs, defs = {}, {}
    live = dict(names)
    elsewhere = declared_in(crate)

    def add(kind, k, signature):
        """One record. A constant's value becomes a second record under `<kind>-value`.

        The value never reaches the declaration's record, so a constant the repository bumps on
        purpose moves one line, in the value section, and can never carry a changed type or a
        changed name past a reader of the diff.
        """
        if kind in ("const", "static", "assoc-const"):
            decl, value = split_value(signature)
            recs.setdefault((kind, k), set()).add(decl)
            if value is not None and k not in VOLATILE_VALUES:
                recs.setdefault((kind + "-value", k), set()).add(norm(value))
            return
        recs.setdefault((kind, k), set()).add(signature)

    def surface(defined):
        """The surface name a definition answers for, through a `pub use ... as` if need be."""
        if defined in live:
            return defined
        nm = real.get(defined)
        return nm if nm in live else None

    def key(name, suffix=""):
        return crate.name + "::" + names[name] + suffix

    def mine(name, path, own):
        """Whether a declaration or impl in `path` is the surface name's, not a namesake's.

        A name the crate declares in one place alone takes its impls from anywhere, which is where
        they live: `RenderPipelineDescriptor`'s sit in `reflection/`. A name the crate declares in
        several places takes them only from the module that defines the surface one.
        """
        if not own:
            return True
        at = home.get(name)
        if at is None or len(elsewhere.get(name, ())) < 2:
            return True
        return path == at or at.parent in path.parents

    macros = crate.macros
    for tree in [crate] + list(extra):
        if tree is not crate:
            live = {n: p for n, p in live.items() if n not in defs}
            if not live:
                break
        for path in tree.sources:
            code, m, decls = tree.read(path)
            rel = str(path.relative_to(tree.src))
            for it in decls:
                if it.test_only() or (it.kind == "mod" and it.name in ("tests", "test")):
                    continue
                nm = surface(it.name) if it.kind in DECL_KINDS else None
                if nm is not None and home.get(nm) is not None and path != home[nm]:
                    nm = None          # a namesake elsewhere in the crate, not the surface's
                if nm is not None and it.vis == "pub":
                    defs.setdefault(nm, []).append(rel)
                    it = it
                    add(it.kind, key(nm), sig(it))
                    nm2 = nm
                    if it.kind == "struct" and it.body:
                        for p in body_parts(code, m, it.body):
                            if p.startswith("pub "):
                                nm = IDENT.search(p[4:])
                                if nm:
                                    add("field", key(nm2, "." + nm.group(0)), p)
                    elif it.kind == "enum" and it.body:
                        for p in body_parts(code, m, it.body):
                            nm = IDENT.match(p)
                            if nm:
                                add("variant", key(nm2, "::" + nm.group(0)), p)
                    elif it.kind == "trait" and it.body:
                        for t in items(code, m, it.body[0], it.body[1], 4):
                            if t.kind in ("fn", "type", "const"):
                                add("trait-item", key(nm2, "::" + t.name), sig(t))
                elif it.kind == "impl":
                    tr, base = impl_self(it.header)
                    base = surface(base) if base else None
                    if base is None or not mine(base, path, tree is crate):
                        continue
                    if tr is not None:
                        add("trait-impl", key(base, " impl " + norm(tr)), sig(it))
                    elif it.body:
                        for f in items(code, m, it.body[0], it.body[1], 4):
                            if f.vis != "pub" or f.test_only():
                                continue
                            if f.kind == "fn":
                                add("method", key(base, "::" + f.name), sig(f))
                            elif f.kind == "const":
                                add("assoc-const", key(base, "::" + f.name), sig(f))
                            elif f.kind == "type":
                                add("assoc-type", key(base, "::" + f.name), sig(f))
                elif it.kind == "macro" and it.body:
                    body = code[it.body[0]:it.body[1]]
                    if it.name in macros:
                        add("macro-template", crate.name + "::" + it.name + "!", norm(body))
                elif it.kind == "macrocall" and it.body:
                    for nm in call_declares(it, code, macros):
                        nm = surface(nm)
                        if nm is None:
                            continue
                        add("macro-decl", key(nm), it.name + "! " + nm)
                        at = it.body[0]
                        while at < it.body[1]:
                            eol = code.find("\n", at)
                            if eol < 0 or eol > it.body[1]:
                                eol = it.body[1]
                            raw, bare = code[at:eol], m[at:eol]
                            if bare.strip() and (len(raw) - len(raw.lstrip())
                                                 == len(bare) - len(bare.lstrip())):
                                line = norm(raw)
                                mo = IDENT.match(line)
                                if mo and line not in ("}", "{"):
                                    add("macro-member", key(nm, "::" + mo.group(0)), line)
                            at = eol + 1
    return recs, defs


# ---------------------------------------------------------------- driver

CRATES = ["metal2vulkan", "metal2vulkan-runtime", "metal2vulkan-air"]

# A floor on each crate's record count. The table is the pin, so a parse that went blind shows as
# thousands of removals and fails on its own; the floor is what makes a *blessed* collapse loud as
# well, so a table of four items cannot pass quietly. It is a long way under the real counts and
# is raised only when a crate's surface grows past it.
FLOORS = {"metal2vulkan": 6000, "metal2vulkan-runtime": 2500, "metal2vulkan-air": 1800}

HEADER = """\
# The signature-level public surface of metal2vulkan, metal2vulkan-runtime and metal2vulkan-air:
# one record a line, "kind<TAB>key<TAB>signature", derived from the sources by
# scripts/api-signatures/signatures.py and pinned by crates/metal2vulkan/tests/api_signatures.rs,
# which says what it covers, what it cannot see, and what blessing a change here asserts.
#
# Write it with `python3 scripts/api-signatures/signatures.py --bless`, which prints what moved.
# A removed or changed line is a breaking change of a published crate.
"""


def build(tree, want):
    """crate name -> (its surface names, its records, where each name is defined).

    `metal2vulkan`'s surface modules come from its own committed table
    (`crates/metal2vulkan/tests/api_surface.rs`), which is the crate's statement of which module
    roots a consumer may write; a public submodule inside a root is not surface. The other two
    crates' roots are their `pub mod`s, which is the shape they have.
    """
    crates = {}
    for name in CRATES:
        src = tree / "crates" / name / "src"
        if src.is_dir():
            crates[name] = Crate(src, name)
    macros = declaring_macros(crates.values())
    for c in crates.values():
        c.macros = macros
    air = crates.get("metal2vulkan-air")
    table = table_modules(tree)
    out = {}
    for name in want:
        crate = crates.get(name)
        if crate is None:
            continue
        roots = None
        if name == "metal2vulkan" and table:
            roots = [[s for s in k.split("::") if s] for k in sorted(table)]
        names, real, home = surface_names(crate, macros, roots)
        # A crate's surface may re-export another's item (the runtime re-exports the translator's,
        # both re-export the front end's), so those trees answer for a name this crate never
        # declares itself.
        extra = [c for c in (crates.get("metal2vulkan"), air)
                 if c is not None and c is not crate]
        out[name] = (names,) + collect(crate, names, real, home, extra)
    return out


def records(root, want=None):
    """{(kind, key): {signature}}, the per-crate record counts, and the per-crate surface names."""
    built = build(root, want or CRATES)
    recs, counts, surface = {}, {}, {}
    for name, (names, got, _defs) in built.items():
        counts[name] = len(got)
        surface[name] = set(names)
        for k, sigs in got.items():
            recs.setdefault(k, set()).update(sigs)
    return recs, counts, surface


def lines_of(recs):
    """The records as the table's lines, in one order whatever the sources' order."""
    return [f"{kind}\t{key}\t{s}"
            for (kind, key), sigs in sorted(recs.items()) for s in sorted(sigs)]


def load_table(path):
    """The table back into {(kind, key): {signature}}."""
    recs = {}
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        kind, key, signature = line.split("\t", 2)
        recs.setdefault((kind, key), set()).add(signature)
    return recs


def is_value(kind):
    """Whether a record is a constant's value rather than a signature.

    The two are reported apart, so a constant's value that moved can never sit in the same list as
    a signature that moved and be blessed along with it unread.
    """
    return kind.endswith("-value")


def report(old, new, out):
    """The three lists, signatures and constant values apart. True when nothing moved."""
    added = sorted(k for k in new if k not in old)
    removed = sorted(k for k in old if k not in new)
    changed = sorted(k for k in new if k in old and new[k] != old[k])

    def show(title, keys, both):
        if not keys:
            return
        out.append("")
        out.append(f"## {title} ({len(keys)})")
        for kind, key in keys:
            out.append("")
            out.append(f"{key}  [{kind}]")
            if both:
                for s in sorted(old[(kind, key)]):
                    out.append(f"  - {s}")
                for s in sorted(new[(kind, key)]):
                    out.append(f"  + {s}")
            else:
                src = new if (kind, key) in new else old
                for s in sorted(src[(kind, key)]):
                    out.append(f"    {s}")

    lists = (("added", added, False), ("removed", removed, False), ("changed", changed, True))
    for section, values in (("signature", False), ("constant value", True)):
        part = [(word, [k for k in keys if is_value(k[0]) == values], both)
                for word, keys, both in lists]
        if not any(keys for _, keys, _ in part):
            continue
        out.append("")
        out.append(f"# {section} changes")
        for word, keys, both in part:
            show(f"{section} {word}", keys, both)
    return not (added or removed or changed)


def check(root, table_path):
    """The table is what the sources derive. The report, and whether it holds."""
    out = []
    recs, counts, surface = records(root)
    ok = True

    have = {key for (_kind, key) in recs}
    for name in VOLATILE_VALUES:
        if name not in have:
            ok = False
            out.append(f"the volatile-value list names {name}, which the surface no longer has: "
                       f"remove it from VOLATILE_VALUES in scripts/api-signatures/signatures.py")

    for name, floor in FLOORS.items():
        got = counts.get(name)
        if got is None:
            ok = False
            out.append(f"{name}: no records at all, so the derivation did not run")
        elif got < floor:
            ok = False
            out.append(f"{name}: {got} records, under the floor of {floor}: either the parse went "
                       f"blind, or the crate's surface really is that much smaller and the floor "
                       f"in this script needs its new value")
        else:
            out.append(f"{name}: {got} records")

    # The module roots `metal2vulkan` exports are the crate's own committed statement of its
    # surface, so the parse that finds them is checked against it on the real tree: a parse that
    # silently stopped seeing a root's items would otherwise derive a smaller surface and call it
    # the truth.
    table = table_modules(root)
    named = {n for m in (table or {}).values() for n, k in m.items() if k != "mod"}
    if not named:
        ok = False
        out.append("crates/metal2vulkan/tests/api_surface.rs names no module-root item, so the "
                   "cross-check of this parse against the crate's own table did not run")
    else:
        parsed = surface.get("metal2vulkan", set())
        out.append(f"metal2vulkan: {len(named)} names in tests/api_surface.rs, {len(parsed)} parsed")
        for n in sorted(named - parsed):
            ok = False
            out.append(f"  in tests/api_surface.rs, not parsed: {n}")
        for n in sorted(parsed - named):
            ok = False
            out.append(f"  parsed, not in tests/api_surface.rs: {n}")

    pinned = load_table(table_path)
    if pinned is None:
        return False, f"{table_path} does not exist; write it with --bless"
    if not report(pinned, recs, out):
        ok = False
        out.append("")
        out.append(f"the public surface moved: {table_path.name} is not what the sources derive.")
        out.append("A line here is a published crate's API. When the move is intended, write the")
        out.append("table with `python3 scripts/api-signatures/signatures.py --bless` and say in")
        out.append("the commit message what broke for a consumer; when it is not, fix the source.")
    return ok, "\n".join(out)


def bless(root, table_path):
    """Write the table, and print every record that moved and both sides of every change.

    What it prints is the assertion the writer makes: each of those moves is intended, and a
    removed or changed line is a breaking change of a published crate that the commit means.
    """
    out = []
    recs, counts, _surface = records(root)
    pinned = load_table(table_path)
    if pinned is not None:
        report(pinned, recs, out)
    for name in sorted(counts):
        out.append(f"# {name}: {counts[name]} records")
    table_path.write_text(HEADER + "".join(f"{line}\n" for line in lines_of(recs)), encoding="utf-8")
    out.append(f"wrote {table_path} ({sum(len(v) for v in recs.values())} records)")
    return True, "\n".join(out)


# ---------------------------------------------------------------- the self test

# Each case is (name, {relative path: source}, [records that must be there],
# [records that must not be]). The trees are the crate layout the derivation reads, so a case
# states its answer by construction: what a reader of the sources can see a consumer may write,
# and what a reader can see it may not.
CASES = [
    (
        "a method's signature, its return type and its loss of `pub`",
        {"crates/metal2vulkan-runtime/src/lib.rs": """\
pub mod event;
""",
         "crates/metal2vulkan-runtime/src/event.rs": """\
/// An event.
pub struct SharedEvent(u32);

/// A device.
pub struct Device(u32);

impl SharedEvent {
    /// The device.
    pub fn device(&self) -> Option<&Device> {
        None
    }

    /// Not public, so not surface.
    fn signalled(&self) -> bool {
        false
    }
}
"""},
        ["method\tmetal2vulkan-runtime::event::SharedEvent::device\t"
         "pub fn device(&self) -> Option<&Device>",
         "struct\tmetal2vulkan-runtime::event::SharedEvent\tpub struct SharedEvent(u32)"],
        ["method\tmetal2vulkan-runtime::event::SharedEvent::device\tpub fn device(&self) -> &Device",
         "method\tmetal2vulkan-runtime::event::SharedEvent::signalled\tpub fn signalled(&self) -> bool"],
    ),
    (
        "a unit variant that became a struct variant, and a field",
        {"crates/metal2vulkan-runtime/src/lib.rs": """\
/// How a vertex is fetched.
pub enum VertexFetch {
    /// Fixed function.
    FixedFunction { narrow_float: bool },
    /// A shader.
    Shader,
}

/// What a device says.
pub struct DeviceProperties {
    /// The render node.
    pub drm_render: Option<(i64, i64)>,
    seen: bool,
}
"""},
        ["variant\tmetal2vulkan-runtime::VertexFetch::FixedFunction\t"
         "FixedFunction { narrow_float: bool }",
         "variant\tmetal2vulkan-runtime::VertexFetch::Shader\tShader",
         "field\tmetal2vulkan-runtime::DeviceProperties.drm_render\t"
         "pub drm_render: Option<(i64, i64)>"],
        ["variant\tmetal2vulkan-runtime::VertexFetch::FixedFunction\tFixedFunction",
         "field\tmetal2vulkan-runtime::DeviceProperties.seen\tseen: bool"],
    ),
    (
        "a wrapped signature reads as the one the formatter left on a line",
        {"crates/metal2vulkan-runtime/src/lib.rs": """\
/// A plan.
pub struct Plan(u32);

/// A budget.
pub struct Budget(u32);

/// Entries.
pub struct StageEntries(u32);

impl Plan {
    /// Wrapped by the formatter, because it is long.
    pub fn vertex_returns_value(
        &self,
        budget: &Budget,
        entries: &StageEntries,
    ) -> Result<bool, ()> {
        Ok(true)
    }
}
"""},
        ["method\tmetal2vulkan-runtime::Plan::vertex_returns_value\t"
         "pub fn vertex_returns_value(&self, budget: &Budget, entries: &StageEntries) "
         "-> Result<bool, ()>"],
        ["method\tmetal2vulkan-runtime::Plan::vertex_returns_value\t"
         "pub fn vertex_returns_value( &self, budget: &Budget, entries: &StageEntries, ) "
         "-> Result<bool, ()>"],
    ),
    (
        "a constant's declaration and its value are two records",
        {"crates/metal2vulkan-runtime/src/lib.rs": """\
/// How many bytes an argument handle takes.
pub const ARGUMENT_HANDLE_BYTES: u64 = 8;

/// A table.
pub struct Table(u32);

impl Table {
    /// The least number of entries.
    pub const MIN_ENTRIES: u64 = 16;
}
"""},
        ["const\tmetal2vulkan-runtime::ARGUMENT_HANDLE_BYTES\tpub const ARGUMENT_HANDLE_BYTES: u64",
         "const-value\tmetal2vulkan-runtime::ARGUMENT_HANDLE_BYTES\t8",
         "assoc-const\tmetal2vulkan-runtime::Table::MIN_ENTRIES\tpub const MIN_ENTRIES: u64",
         "assoc-const-value\tmetal2vulkan-runtime::Table::MIN_ENTRIES\t16"],
        ["const\tmetal2vulkan-runtime::ARGUMENT_HANDLE_BYTES\t"
         "pub const ARGUMENT_HANDLE_BYTES: u64 = 8",
         "const-value\tmetal2vulkan-runtime::ARGUMENT_HANDLE_BYTES\t16"],
    ),
    (
        "what is not surface: crate-private, hidden, and test-only",
        {"crates/metal2vulkan-runtime/src/lib.rs": """\
pub mod gap;
mod inner;
""",
         "crates/metal2vulkan-runtime/src/gap.rs": """\
/// A gap.
pub enum RuntimeGap {
    /// Options.
    FunctionOptions(u32),
}

/// Crate-private: no cfg makes it reachable.
pub(crate) struct Inside(u32);

/// Public, but not API.
#[doc(hidden)]
pub struct Plumbing(u32);

#[cfg(test)]
mod tests {
    /// Only under test.
    pub struct Probe(u32);
}
""",
         "crates/metal2vulkan-runtime/src/inner.rs": """\
/// Reachable from nowhere a consumer may write.
pub struct Hidden(u32);
"""},
        ["variant\tmetal2vulkan-runtime::gap::RuntimeGap::FunctionOptions\tFunctionOptions(u32)"],
        ["struct\tmetal2vulkan-runtime::gap::Inside\tpub(crate) struct Inside(u32)",
         "struct\tmetal2vulkan-runtime::gap::Plumbing\tpub struct Plumbing(u32)",
         "struct\tmetal2vulkan-runtime::gap::Probe\tpub struct Probe(u32)",
         "struct\tmetal2vulkan-runtime::inner::Hidden\tpub struct Hidden(u32)"],
    ),
    (
        "a trait's own items, and a trait impl's header",
        {"crates/metal2vulkan-runtime/src/lib.rs": """\
use std::fmt;

/// A thing with a length.
pub trait Measured {
    /// The unit.
    type Unit;

    /// How long.
    fn length(&self) -> u64;
}

/// A buffer.
pub struct Buffer(u32);

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Buffer")
    }
}
"""},
        ["trait-item\tmetal2vulkan-runtime::Measured::length\tfn length(&self) -> u64",
         "trait-item\tmetal2vulkan-runtime::Measured::Unit\ttype Unit",
         "trait-impl\tmetal2vulkan-runtime::Buffer impl fmt::Debug\timpl fmt::Debug for Buffer"],
        ["method\tmetal2vulkan-runtime::Buffer::fmt\t"
         "fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result"],
    ),
    (
        "the attributes that are the contract reach the signature",
        {"crates/metal2vulkan-runtime/src/lib.rs": """\
/// A gap a consumer must read completely.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum RuntimeGap {
    /// One.
    One,
}

/// Behind a feature.
#[cfg(feature = "ash")]
pub struct Interop(u32);

/// An attribute with a doc comment under it.
#[non_exhaustive]
/// Written between the attribute and the item.
pub struct Awkward {
    /// A field.
    pub one: u32,
}
"""},
        ["enum\tmetal2vulkan-runtime::RuntimeGap\t"
         "#[derive(Clone, Copy, Debug)] #[non_exhaustive] pub enum RuntimeGap",
         "struct\tmetal2vulkan-runtime::Interop\t#[cfg(feature = \"ash\")] pub struct Interop(u32)",
         "struct\tmetal2vulkan-runtime::Awkward\t#[non_exhaustive] pub struct Awkward"],
        ["enum\tmetal2vulkan-runtime::RuntimeGap\tpub enum RuntimeGap",
         "struct\tmetal2vulkan-runtime::Interop\tpub struct Interop(u32)",
         "struct\tmetal2vulkan-runtime::Awkward\tpub struct Awkward"],
    ),
]


def selftest():
    """The derivation over trees whose answer is known, both directions on every case."""
    out, ok = [], True
    work = Path(tempfile.mkdtemp(prefix="api-signatures-selftest."))
    try:
        for n, (name, files, want, unwanted) in enumerate(CASES):
            tree = work / f"case{n}"
            for rel, text in files.items():
                path = tree / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            recs, _counts, _surface = records(tree, ["metal2vulkan-runtime"])
            got = set(lines_of(recs))
            missing = [r for r in want if r not in got]
            present = [r for r in unwanted if r in got]
            if missing or present:
                ok = False
                out.append(f"case {n} ({name}):")
                for r in missing:
                    out.append(f"  not derived, and it must be: {r}")
                for r in present:
                    out.append(f"  derived, and it must not be: {r}")
                out.append("  what the case derived:")
                for r in sorted(got):
                    out.append(f"    {r}")
            else:
                out.append(f"case {n}: {len(want)} records found, {len(unwanted)} refused ({name})")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return ok, "\n".join(out)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=here.parent.parent)
    ap.add_argument("--table", type=Path, default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bless", action="store_true")
    ap.add_argument("--print", dest="show", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    table = args.table or root / "crates/metal2vulkan/tests/api_signatures.tsv"

    if args.selftest:
        ok, text = selftest()
        print(text)
        sys.exit(0 if ok else 1)
    if args.check or args.bless:
        ok, text = (check if args.check else bless)(root, table)
        print(text)
        sys.exit(0 if ok else 1)
    if not args.show:
        ap.error("one of --check, --bless, --print, --selftest")
    recs, _counts, _surface = records(root)
    for line in lines_of(recs):
        print(line)


if __name__ == "__main__":
    main()
