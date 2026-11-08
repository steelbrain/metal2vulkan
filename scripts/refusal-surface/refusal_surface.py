#!/usr/bin/env python3
"""Generate docs/REFUSALS.md: every refusal of something Metal accepts, from the code.

The refusal surface is read from the construction sites of the typed refusals, outside tests:

- the translator's and the AIR front end's `Unsupported` (a `Feature` and its `Needs`): the
  constructors `Unsupported::not_implemented`, `not_representable`, `needs`, `needs_any_of`,
  `driver_defect`, `runtime_driver_defect`, `metal_gpu_family`, `device_capability`, and
  `Unsupported { .. }`
  literals, in the `src` of crates/metal2vulkan-air, crates/metal2vulkan and
  crates/metal2vulkan-runtime;
- the runtime's `RuntimeGap` (`Error::NotImplemented`) and `AppleOnly` (`Error::AppleOnly`);
- the binder's refusals of valid bind states at a device limit (`BindError`);
- the translation caps (`Limit`, `Error::LimitExceeded`);
- the rasterization-rate maps the compile layer does not compile (`UnpinnedCompile`, the
  runtime's `Error::RateMap`);
- the translation deadline (`BudgetExceeded`, `Error::Budget`; the runtime's
  `Error::Compile(Budget)`);
- what a call did wrong (the runtime's `Problem`, carried by `Error::Metal`), and the problem enums
  it forwards to, found from the forwarding variants' payloads: each of these is a claim that Metal
  itself refuses an input, so each carries a declared basis.

Each refusal is one row per variant, with what it blocks (the variant's doc comment), the device
condition (from the constructor: every device, or the capabilities, driver defects or families
its sites name), its sites, its basis, and a reach mark, which is the one hand-kept column
(`reach.tsv` beside this script).

The basis says where the belief that this input must be refused comes from. It is derived from the
construction site wherever the site states it, and declared in the refusal's own doc comment, as a
last line `Basis: <value> (the evidence)`, wherever the claim is about Metal itself, which no site
can state. `--check` requires one of every `Problem`, and of every variant of a problem enum a
`Problem` forwards to with `Basis: delegated(ThatEnum)`: a refusal that asserts Metal stops an input
cannot be added with nothing behind it, and one whose basis is not settled says
`Basis: unknown (why)` rather than nothing.

    scripts/refusal-surface/refusal_surface.py                 # write docs/REFUSALS.md
    scripts/refusal-surface/refusal_surface.py --check         # fail when a variant, its
                                                               # condition, basis or reach mark is
                                                               # stale (the site lists may drift)
    scripts/refusal-surface/refusal_surface.py --table         # print the plain table only
    scripts/refusal-surface/refusal_surface.py --missing-basis # list refusals whose basis is
                                                               # absent or malformed
    scripts/refusal-surface/refusal_surface.py --selftest      # the basis marker's own cases
    scripts/refusal-surface/refusal_surface.py --delegations    # the problem enums a `Problem`
                                                               # forwards to, as the scan finds
                                                               # them

It reads source text only (no build), so it runs anywhere.
"""
import argparse
import re
import subprocess
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "REFUSALS.md"
REACH = Path(__file__).resolve().parent / "reach.tsv"

CRATES = ROOT / "crates"
TREES = [
    ("air", CRATES / "metal2vulkan-air" / "src"),
    ("translator", CRATES / "metal2vulkan" / "src"),
    ("runtime", CRATES / "metal2vulkan-runtime" / "src"),
]

CONSTRUCTORS = {
    "not_implemented": "NotImplemented",
    "not_representable": "NotRepresentable",
    "needs": "AnyOf",
    "needs_any_of": "AnyOf",
    "driver_defect": "DriverDefect",
    "runtime_driver_defect": "RuntimeDriverDefect",
    "metal_gpu_family": "MetalGpuFamily",
    "device_capability": "DeviceCapability",
}

# Binder refusals of a valid bind state: a device limit or an unestablished reading. The other
# `BindError` variants are Metal's own validation (misuse), or carry an `Unsupported` already
# counted under its feature.
BIND_REFUSALS = [
    "DescriptorRangeExceeded",
    "TexelBufferTooLarge",
    "HeapTooLarge",
    "TessellationFactorRange",
    "PatchIndexRange",
    "ControlPointIndexRange",
    "MissingStageInRegion",
    "UniformViewAliased",
    "MissingDeviceAddress",
]

REACH_MARKS = OrderedDict(
    [
        ("system", "a guest's CoreAnimation, CoreImage or VideoToolbox plausibly reaches it"),
        ("app", "a typical Metal app plausibly reaches it from public API"),
        ("rare", "public API, but specialized: few apps reach it"),
        ("none", "no public API path: malformed or non-Metal input, Apple-private calls, or a cap no valid program of ordinary size meets"),
    ]
)

# The basis of a refusal: where the belief that this input must be refused comes from. One
# vocabulary for the whole surface.
#
# Where the construction site already says it, it is **derived** from the site and nothing is kept
# by hand: a capability a device lacks, a piece not built, a cap of this crate's. Where the claim
# is about Metal itself, no site can say it, so it is **declared** in the refusal's own doc comment
# as a last line, `Basis: <value> (the evidence)`, and `--check` requires one of every variant of
# `Problem` (section `invalid`), which is nothing but such claims.
BASIS = OrderedDict(
    [
        (
            "metal-measured",
            "Metal itself refuses the input, and a run of ours on a named device or release says so",
        ),
        (
            "metal-documented",
            "Metal itself refuses it, and Apple's own words say so: a quoted assertion, `NSError` "
            "or validation message, a header, or Apple's documentation",
        ),
        (
            "metal-takes",
            "measured: plain Metal **takes** this input, so the refusal reaches past Metal. An open "
            "defect, recorded here rather than left unsaid",
        ),
        (
            "device",
            "the mechanism is absent: from this Vulkan device, from its driver, or from every "
            "Vulkan device",
        ),
        (
            "ours",
            "this crate's own gap, cap or deadline, or the misuse of a runtime extension Metal has "
            "no counterpart for",
        ),
        (
            "unknown",
            "not established: the refusal stands, and why Metal is believed to refuse this has not "
            "been settled",
        ),
    ]
)

# A basis that claims something about Metal's own behaviour carries its evidence with it.
CITED_BASIS = ("metal-measured", "metal-documented", "metal-takes")
# A measurement names the device or the release it ran on.
NAMES_A_DEVICE = re.compile(r"\bM\d\b|\bApple M|\bmacOS\b|\b2[0-9]\.[0-9]")
# An aggregate variant carries no basis of its own: it forwards to another problem enum, whose
# variants hold the claims. `delegated(ThatEnum)` says so, and is held to the variant's payload.
DELEGATED = "delegated"

# A payload that is one problem enum: `Problem`'s forwarding variants carry nothing else, and the
# enum they name is read here too, so the claims inside it are held to a basis as the parent's are.
ONE_PROBLEM = re.compile(r"[A-Z][A-Za-z0-9_]*Problem")


def split_basis(doc):
    """(prose, marker): the doc without its trailing `Basis:` line, and that line's text.

    The marker is the last thing in the doc comment, so one `rfind` finds it wherever the prose
    above it goes; a `Basis:` left in the prose makes the tail unparsable, which `--check` reports
    rather than passing over.
    """
    i = doc.rfind("Basis:")
    if i < 0:
        return doc, None
    return doc[:i].rstrip(), doc[i + len("Basis:") :].strip()


BASIS_MARKER = re.compile(r"([a-z][a-z-]*)\s*(?:\((.*)\))?\.?")


def parse_basis(marker, payload=""):
    """(value, citation, complaint) of one `Basis:` marker; `value` is None where it is malformed.

    `payload` is the variant's payload type, which a `delegated` marker must name.
    """
    m = BASIS_MARKER.fullmatch(marker.strip())
    if not m:
        return None, "", "is not `Basis: <value>` or `Basis: <value> (the evidence)`"
    value, cite = m.group(1), (m.group(2) or "").strip()
    if value == DELEGATED:
        if not cite:
            return None, "", "`delegated` names no enum: write `Basis: delegated(TheProblem)`"
        if cite != payload.strip():
            return None, "", f"`delegated({cite})` is not this variant's payload (`{payload.strip() or 'none'}`)"
        return DELEGATED, cite, ""
    if value not in BASIS:
        return None, "", f"`{value}` is no basis: " + ", ".join(BASIS)
    if value in CITED_BASIS and not cite:
        return None, "", f"`{value}` claims something about Metal and cites no evidence"
    if value in ("metal-measured", "metal-takes") and not NAMES_A_DEVICE.search(cite):
        return None, "", f"`{value}` cites a run that names no device or release: `{cite}`"
    return value, cite, ""


# How each section's basis follows from the construction site, where it does. `None` is a section
# whose basis no site states: each variant's doc comment declares it.
def derived_basis(section, kinds):
    if section == "feature":
        if not kinds:
            return ""
        # A site that refuses on every device because the lowering is not built settles the basis
        # whatever the other sites need of a device.
        return "ours" if "NotImplemented" in kinds else "device"
    if section == "gap":
        return "ours"
    if section == "bind":
        return "device"
    if section in ("limit", "budget"):
        return "ours"
    return ""


def lex(text):
    """Blank comments, strings and char literals (keeping newlines and offsets)."""
    out = list(text)
    i, n = 0, len(text)

    def blank(a, b):
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
            continue
        if c == "r" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.match(r'r(#*)"', text[i : i + 10])
            if m:
                close = '"' + m.group(1)
                j = text.find(close, i + len(m.group(0)))
                j = n if j < 0 else j + len(close)
                blank(i + 1, j - 1)
                i = j
                continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            blank(i + 1, j)
            i = j + 1
            continue
        if c == "'":
            m = re.match(r"'(\\.[^']*|[^'\\])'", text[i : i + 12])
            if m:
                blank(i + 1, i + len(m.group(0)) - 1)
                i += len(m.group(0))
                continue
        i += 1
    return "".join(out)


def close_of(text, start, open_ch, close_ch):
    depth = 0
    for k in range(start, len(text)):
        if text[k] == open_ch:
            depth += 1
        elif text[k] == close_ch:
            depth -= 1
            if depth == 0:
                return k
    return len(text) - 1


def blank_test_items(text):
    """Blank `#[cfg(test)]` items (modules, functions, impls) so their sites do not count."""
    out = list(text)
    for m in re.finditer(r"#\[cfg\(test\)\]", text):
        k = text.find("{", m.end())
        semi = text.find(";", m.end())
        if k < 0 or (0 <= semi < k):
            continue
        e = close_of(text, k, "{", "}")
        for p in range(m.start(), e + 1):
            if out[p] != "\n":
                out[p] = " "
    return "".join(out)


def source_files():
    for tree, base in TREES:
        for p in sorted(base.rglob("*.rs")):
            rel = p.relative_to(ROOT).as_posix()
            if "/tests/" in rel or rel.endswith("/tests.rs") or rel.endswith("_tests.rs"):
                continue
            yield tree, rel, blank_test_items(lex(p.read_text()))


def fn_spans(text):
    """(start, end, name) of every function body in text."""
    spans = []
    for m in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^{;]*?>)?\s*\(", text):
        k = close_of(text, m.end() - 1, "(", ")")
        b = text.find("{", k)
        semi = text.find(";", k)
        if b < 0 or (0 <= semi < b):
            continue
        name = m.group(1)
        # A public `Device` method's body on the generation that serves is `<name>_now`, beside
        # the method that forwards to it (`device/line.rs`): its sites are the method's.
        if name.endswith("_now") and re.search(r"self\.current\(\)\s*\." + name + r"\(", text):
            name = name[: -len("_now")]
        spans.append((b, close_of(text, b, "{", "}"), name))
    return spans


def enclosing(spans, pos):
    best = None
    for b, e, name in spans:
        if b <= pos <= e and (best is None or b > best[0]):
            best = (b, e, name)
    return best


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def split_args(s):
    args, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


CAP_TOKENS = r"(VkFeature|VkExtension|VkProperty|VkLimit|ApiVersion|DriverWorkaround|RuntimeDriverDefect)::([A-Za-z0-9_]+)"


def conditions(kind, needs_text):
    if kind == "NotImplemented":
        return {"every device (not implemented)"}
    if kind == "NotRepresentable":
        return {"every device (no Vulkan mechanism)"}
    if kind == "DeviceCapability":
        # A Vulkan capability the compile layer's vocabulary does not name, which only the runtime
        # needs: the device's own gap, so the condition is the devices without it and never "every
        # device". The site names it as a string, so there is no capability token to read.
        return {"devices without the capability the site names"}
    toks = re.findall(CAP_TOKENS, needs_text)
    if kind == "AnyOf":
        caps = sorted({f"{a}::{b}" for a, b in toks if a.startswith("Vk") or a == "ApiVersion"})
        if "Capability::Format" in needs_text or "FormatUsage::" in needs_text:
            caps.append("a format's usage")
        if not caps:
            expr = " ".join(needs_text.split())
            expr = re.sub(r"^vec!\[\s*(.*?)\s*\]$", r"\1", expr)
            if "addressed_needs" in expr:
                return {"devices without buffer device addresses (`addressed_needs`)"}
            for cap, what in (("Limit", "a device limit"), ("Feature", "a Vulkan feature"), ("Property", "a Vulkan property"), ("Extension", "a Vulkan extension"), ("Format", "a format's usage")):
                if expr.startswith(f"Capability::{cap}"):
                    return {f"devices short of {what} the site names"}
            if expr and len(expr) <= 70:
                return {f"devices without a capability the site computes (`{expr}`)"}
            return {"devices without a capability the site computes"}
        return {"devices without " + c for c in caps}
    if kind == "DriverDefect":
        ws = sorted({b for a, b in toks if a == "DriverWorkaround"})
        return {"drivers with workaround " + w for w in ws} or {"drivers with a defect"}
    if kind == "RuntimeDriverDefect":
        ws = sorted({b for a, b in toks if a == "RuntimeDriverDefect"})
        return {"drivers with defect " + w for w in ws} or {"drivers with a defect"}
    if kind == "MetalGpuFamily":
        return {"devices that do not claim the Metal GPU family"}
    return {"varies (" + kind + ")"}


def is_pattern(text, start, end):
    """Whether the variant path at [start, end) is a match pattern rather than a value."""
    k = end
    while k < len(text) and text[k] in " \t":
        k += 1
    if k < len(text) and text[k] in "({":
        k = close_of(text, k, text[k], ")" if text[k] == "(" else "}") + 1
    rest = text[k : k + 3].lstrip()
    if rest.startswith("=>") or (rest.startswith("|") and not rest.startswith("||")):
        return True
    before = text[max(0, start - 30) : start].rstrip()
    # A closure's parameters (`let refuse = |a, b| Variant { .. }`) end in `|` too: the value
    # after them is built, not matched.
    closure = re.search(r"(?:=|\(|,|\bmove)\s*\|[^|]*\|$", text[max(0, start - 80) : start].rstrip())
    if before.endswith("|") and not before.endswith("||") and not closure:
        return True
    if re.search(r"\b(let|matches!\([^,]*,)\s*$", text[max(0, start - 60) : start]):
        return True
    if before.endswith("let") or before.endswith("Some(") and "let" in text[max(0, start - 40) : start]:
        return True
    return False


def scan(sub_problems=()):
    """Rows keyed by (section, variant): conditions, sites.

    `sub_problems` are the problem enums `Problem` forwards to: their variants are construction
    sites of a refusal too, and are counted as `Problem`'s own are.
    """
    rows = defaultdict(lambda: {"conds": set(), "kinds": set(), "sites": [], "indirect": []})
    feature_refs = defaultdict(list)
    for tree, rel, text in source_files():
        spans = fn_spans(text)

        def site(pos):
            """A construction site: its file, its line, and the function that builds the refusal.

            The line is carried only so two matches of one construction count as one site; `cite`
            does not print it.
            """
            enc = enclosing(spans, pos)
            return (rel, line_of(text, pos), enc[2] if enc else "")

        # Unsupported constructors.
        for m in re.finditer(r"\bUnsupported::(" + "|".join(CONSTRUCTORS) + r")\s*\(", text):
            k = m.end() - 1
            e = close_of(text, k, "(", ")")
            args = split_args(text[k + 1 : e])
            kind = CONSTRUCTORS[m.group(1)]
            feat = args[0] if args else ""
            fm = re.match(r"(?:\w+::)*Feature::([A-Za-z0-9_]+)", feat)
            name = fm.group(1) if fm else None
            needs_text = " ".join(args[1:])
            if name is None:
                key = ("feature", "(computed)")
            else:
                key = ("feature", name)
            rows[key]["conds"] |= conditions(kind, needs_text)
            rows[key]["kinds"].add(kind)
            rows[key]["sites"].append(site(m.start()))
        # Unsupported { feature: .., needs: .. } literals.
        for m in re.finditer(r"\bUnsupported\s*\{", text):
            pre = text[max(0, m.start() - 12) : m.start()]
            if "struct" in pre or "enum" in pre:
                continue
            e = close_of(text, m.end() - 1, "{", "}")
            body = text[m.end() : e]
            fm = re.search(r"\bfeature\s*:\s*(?:\w+::)*Feature::([A-Za-z0-9_]+)", body)
            nm = re.search(r"\bneeds\s*:\s*(?:\w+::)*Needs::([A-Za-z0-9_]+)", body)
            if not re.search(r"\bfeature\s*:", body):
                continue  # a pattern (`Unsupported { feature, .. }`)
            if is_pattern(text, m.start(), e + 1):
                continue
            name = fm.group(1) if fm else "(computed)"
            kind = nm.group(1) if nm else "computed"
            needs_text = body[nm.start() :] if nm else ""
            rows[("feature", name)]["conds"] |= conditions(kind, needs_text)
            rows[("feature", name)]["kinds"].add(kind)
            rows[("feature", name)]["sites"].append(site(m.start()))
        # Indirect feature references (a feature handed to a helper that builds the refusal).
        for m in re.finditer(r"\bFeature::([A-Za-z0-9_]+)", text):
            if is_pattern(text, m.start(), m.end()):
                continue
            feature_refs[m.group(1)].append(site(m.start()))
        if tree == "runtime":
            # `Problem`: what a call did wrong. `Problem::` is a suffix of every other problem
            # enum's name (`CopyProblem::`), so the match requires a boundary the word `\b` does
            # not give between two letters.
            for m in re.finditer(r"(?<![A-Za-z0-9_])Problem::([A-Z][A-Za-z0-9_]*)", text):
                if rel.endswith("metal2vulkan-runtime/src/error.rs") or is_pattern(text, m.start(), m.end()):
                    continue
                rows[("invalid", m.group(1))]["sites"].append(site(m.start()))
            gap_alias = ["RuntimeGap"]
            am = re.search(r"\bRuntimeGap\s+as\s+(\w+)", text)
            if am:
                gap_alias.append(am.group(1))
            for alias in gap_alias:
                for m in re.finditer(r"\b" + alias + r"::([A-Z][A-Za-z0-9_]*)", text):
                    if rel.endswith("metal2vulkan-runtime/src/error.rs"):
                        continue
                    if is_pattern(text, m.start(), m.end()):
                        continue
                    section = "gap_family" if "/families/" in rel else "gap"
                    rows[(section, m.group(1))]["sites"].append(site(m.start()))
            # The problem enums `Problem` forwards to. The enum's own module declares it and
            # also builds it, so the file cannot be skipped the way `error.rs` is: `is_pattern`
            # keeps the match arms of its `Display` out.
            for enum in sub_problems:
                for m in re.finditer(r"(?<![A-Za-z0-9_])" + enum + r"::([A-Z][A-Za-z0-9_]*)", text):
                    if is_pattern(text, m.start(), m.end()):
                        continue
                    rows[("subproblem", f"{enum}::{m.group(1)}")]["sites"].append(site(m.start()))
            for m in re.finditer(r"\bAppleOnly::([A-Z][A-Za-z0-9_]*)", text):
                if rel.endswith("apple_only.rs") and re.match(r"\s*[,{(]", text[m.end() : m.end() + 2]) is None:
                    pass
                if is_pattern(text, m.start(), m.end()):
                    continue
                rows[("apple", m.group(1))]["sites"].append(site(m.start()))
        for m in re.finditer(r"\bBindError::([A-Z][A-Za-z0-9_]*)", text):
            if m.group(1) not in BIND_REFUSALS or is_pattern(text, m.start(), m.end()):
                continue
            rows[("bind", m.group(1))]["sites"].append(site(m.start()))
        for m in re.finditer(r"\bLimit::([A-Z][A-Za-z0-9_]*)", text):
            if is_pattern(text, m.start(), m.end()):
                continue
            if rel.endswith("metal2vulkan-air/src/error.rs"):
                continue
            rows[("limit", m.group(1))]["sites"].append(site(m.start()))
        for enum, section in (("UnpinnedCompile", "ratemap"), ("BudgetExceeded", "budget")):
            for m in re.finditer(r"\b" + enum + r"::([A-Z][A-Za-z0-9_]*)", text):
                if is_pattern(text, m.start(), m.end()):
                    continue
                rows[(section, m.group(1))]["sites"].append(site(m.start()))
    return rows, feature_refs


def problem_arms(path):
    """(variant, doc, payload) of `pub enum Problem`, in order.

    Every one of these is a claim that Metal itself refuses an input, so every one carries a
    declared basis. The payload is read too: an aggregate variant's `delegated` marker must name
    it, so the marker cannot be used where a variant does not forward.
    """
    return enum_arms(path, "Problem")


def enum_arms(path, enum):
    """(variant, doc, payload) of `pub enum <enum>` in path, in order.

    Struct variants are stepped over by brace depth, so a field's own doc comment is never read as
    the next variant's. This is the reader for `Problem` and for every problem enum it forwards to:
    those are the refusals whose basis is declared, and a doc comment read wrongly would lose one.
    """
    lines = path.read_text().split("\n")
    out, doc, inside, depth = [], [], False, 0
    for line in lines:
        if not inside:
            if re.match(r"pub enum " + enum + r" \{", line):
                inside = True
            continue
        if depth == 0 and line.startswith("}"):
            break
        if depth:
            depth += line.count("{") - line.count("}")
            continue
        s = line.strip()
        if s.startswith("///"):
            doc.append(s[3:].strip())
            continue
        m = re.match(r"^    ([A-Z][A-Za-z0-9_]*)\s*(\(|\{|,)", line)
        if m:
            payload = ""
            if m.group(2) == "(" and ")" in line:
                payload = line[line.index("(") + 1 : line.rindex(")")]
            elif m.group(2) == "{":
                depth = 1
            out.append((m.group(1), " ".join(doc), payload))
            doc = []
        elif s and not s.startswith("//") and not s.startswith("#"):
            doc = []
    return out


def enum_variants(path, enum):
    """(variant, doc) of `pub enum <enum>` in path, in order."""
    lines = path.read_text().split("\n")
    out, doc, inside = [], [], False
    for line in lines:
        if not inside:
            if re.match(r"pub enum " + enum + r"\b", line):
                inside = True
            continue
        if line.startswith("}"):
            break
        s = line.strip()
        if s.startswith("///"):
            doc.append(s[3:].strip())
            continue
        m = re.match(r"^    ([A-Z][A-Za-z0-9_]*)\b", line)
        if m:
            out.append((m.group(1), " ".join(doc)))
        if s and not s.startswith("//") and not s.startswith("#"):
            doc = []
    return out


def runtime_enum_files():
    """Each `pub enum` the runtime declares outside tests, by name: the file that declares it.

    Read from the sources, so a problem enum that moves to another module is still found.
    """
    where = {}
    for tree, rel, text in source_files():
        if tree != "runtime":
            continue
        for m in re.finditer(r"^pub enum ([A-Za-z0-9_]+) \{", text, re.M):
            where.setdefault(m.group(1), ROOT / rel)
    return where


def forwarded_problems(path):
    """Every problem enum `Problem` forwards to, and every one those forward to, in their order.

    Structural, and no list of names is kept here: a variant whose whole payload is one enum the
    runtime declares is a forwarding variant, and the enum it names is read as `Problem` is. A
    sub-problem added later is read with it, and one that forwards again is followed.

    Each is (enum, the file that declares it, the forwarding variant, that variant's basis, whether
    this family's variants must declare a basis of their own). The forwarding variant's own basis
    decides the last: `delegated(ThatEnum)` says the claims live in `ThatEnum`, so each of its
    variants declares its own; any other value is a claim about the whole family (a runtime
    extension Metal has no counterpart for), which its variants derive from it.
    """
    where = runtime_enum_files()
    out, seen, queue = [], {"Problem"}, [("Problem", path)]
    while queue:
        owner, file = queue.pop(0)
        for name, doc, payload in enum_arms(file, owner):
            payload = payload.strip()
            if not ONE_PROBLEM.fullmatch(payload) or payload in seen:
                continue
            seen.add(payload)
            marker = split_basis(doc)[1]
            value = parse_basis(marker, payload)[0] if marker is not None else None
            target = where.get(payload)
            out.append((payload, target, f"{owner}::{name}", value or "", value == DELEGATED))
            if target is not None:
                queue.append((payload, target))
    return out


# A citation of the design record, a work package or an open item, which a reader of this
# file cannot follow: dropped from the summaries.
REF = r"(?:refining |DESIGN )?D\d+(?:\.\d+)?(?: (?:items?|part) \d+(?:(?:, | and )\d+)*)?|WP-[A-Z0-9]+[a-z]?|U\d+"


def strip_refs(text):
    text = re.sub(rf"\s*\((?:{REF})(?:[,;] (?:{REF}))*\)", "", text)
    text = re.sub(rf"[,;] (?:{REF})(?=[,;)])", "", text)
    return re.sub(rf"\((?:{REF})[,;] ", "(", text)


def summary(doc):
    doc = strip_refs(doc.strip())
    if not doc:
        return ""
    m = re.match(r"(.+?[.:])(\s|$)", doc)
    first = m.group(1) if m else doc
    first = first.rstrip(":").rstrip(".")
    if len(first) > 150:
        first = first[:147].rsplit(" ", 1)[0] + "..."
    return first.replace("|", "\\|")


def load_reach():
    reach = {}
    if REACH.exists():
        for line in REACH.read_text().split("\n"):
            if not line.strip() or line.startswith("#"):
                continue
            f = line.split("\t")
            while len(f) < 4:
                f.append("")
            reach[(f[0], f[1])] = (f[2], f[3])
    return reach


def cite(sites):
    """The sites of one refusal as `file (function)`, in file order, deduplicated.

    No line number: it is right for one state of the tree only, while the function is right in
    every tree that has it and is what a reader searches for. Where one function builds more than
    one site the count says so, so dropping the line loses no site.
    """
    counted = Counter((rel, fn) for rel, _, fn in set(sites))
    out = []
    for (rel, fn), n in sorted(counted.items()):
        inside = ", ".join(x for x in (fn, f"{n} sites" if n > 1 else "") if x)
        out.append(f"{rel} ({inside})" if inside else rel)
    return out


def sites_cell(sites):
    """How many construction sites, then the first two as `file (function)`."""
    if not sites:
        return "0"
    first = cite(sites)
    shown = ", ".join(f"`{s}`" for s in first[:2])
    more = f" (+{len(first) - 2})" if len(first) > 2 else ""
    return f"{len(set(sites))}: {shown}{more}"


def head_sha():
    """HEAD's short sha, for the `--table` print. A tracked file carries no such stamp: it would
    differ on every regeneration without a refusal having changed."""
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build():
    error_rs = CRATES / "metal2vulkan-runtime" / "src" / "error.rs"
    subs = forwarded_problems(error_rs)
    rows, feature_refs = scan([enum for enum, _, _, _, _ in subs])
    reach = load_reach()
    fdocs = OrderedDict(enum_variants(CRATES / "metal2vulkan" / "src" / "error" / "mod.rs", "Feature"))
    for n, d in enum_variants(CRATES / "metal2vulkan-air" / "src" / "error.rs", "Feature"):
        fdocs.setdefault(n, d)
    gdocs = OrderedDict(enum_variants(CRATES / "metal2vulkan-runtime" / "src" / "error.rs", "RuntimeGap"))
    adocs = OrderedDict(enum_variants(CRATES / "metal2vulkan-runtime" / "src" / "apple_only.rs", "AppleOnly"))
    apple_gaps = apple_only_gaps(CRATES / "metal2vulkan-runtime" / "src" / "apple_only.rs")
    bdocs = OrderedDict(enum_variants(CRATES / "metal2vulkan" / "src" / "error" / "mod.rs", "BindError"))
    ldocs = OrderedDict(enum_variants(CRATES / "metal2vulkan-air" / "src" / "error.rs", "Limit"))
    rdocs = OrderedDict(enum_variants(CRATES / "metal2vulkan" / "src" / "bind" / "rate_map.rs", "UnpinnedCompile"))
    tdocs = OrderedDict(enum_variants(CRATES / "metal2vulkan-air" / "src" / "error.rs", "BudgetExceeded"))
    for n, refs in feature_refs.items():
        if n in fdocs and not rows[("feature", n)]["sites"]:
            rows[("feature", n)]["indirect"] = refs
    pdocs = problem_arms(error_rs)
    payloads = {n: pay for n, _, pay in pdocs}
    tables = OrderedDict()
    missing_reach = []
    basis_faults = []
    # The problem enums `Problem` forwards to, each family in the order its forwarding variant
    # comes: one row for each of their variants, keyed `Enum::Variant`.
    sdocs = OrderedDict()
    sub_family = {}
    for enum, file, parent, value, required in subs:
        sub_family[enum] = (parent, value, required)
        if file is None:
            basis_faults.append(("subproblem", enum, f"forwards from `{parent}` to an enum no source declares"))
            continue
        arms = enum_arms(file, enum)
        if required and not arms:
            basis_faults.append(("subproblem", enum, f"forwards from `{parent}` to an enum with no variants to read"))
        for n, d, pay in arms:
            sdocs[f"{enum}::{n}"] = d
            payloads[f"{enum}::{n}"] = pay
    sections = [
        ("feature", fdocs, "Translator and front-end refusals (`Unsupported`: a `Feature` and its `Needs`)"),
        ("gap", gdocs, "Runtime gaps (`Error::NotImplemented(RuntimeGap)`)"),
        ("apple", adocs, "Apple-only calls (`Error::AppleOnly`)"),
        ("bind", OrderedDict((n, bdocs.get(n, "")) for n in BIND_REFUSALS), "Binder refusals at a device limit (`BindError`)"),
        ("limit", ldocs, "Translation caps (`Error::LimitExceeded(Limit)`)"),
        (
            "ratemap",
            rdocs,
            "Rate maps the compile layer does not compile (`Error::RateMap(UnpinnedCompile)`)",
        ),
        ("budget", tdocs, "The translation deadline (`Error::Budget`; the runtime's `Error::Compile(Budget)`)"),
        (
            "invalid",
            OrderedDict((n, d) for n, d, _ in pdocs),
            "What a call did wrong (`Error::Metal(MetalError)`: a `Problem`)",
        ),
        (
            "subproblem",
            sdocs,
            "What a call did wrong, in detail (the problem enums a `Problem` forwards to)",
        ),
    ]
    sub_section = next(i for i, (s, _, _) in enumerate(sections, 1) if s == "subproblem")
    for section, docs, title in sections:
        out = []
        for name, doc in docs.items():
            r = rows.get((section, name))
            fam = rows.get(("gap_family", name)) if section == "gap" else None
            sites = (r["sites"] if r else [])
            indirect = (r["indirect"] if r else [])
            conds = sorted(r["conds"]) if r else []
            if section == "gap":
                if name in ("FeatureEnable", "ExtensionEnable"):
                    conds = ["devices without the feature or extension it names"]
                else:
                    conds = ["every device (runtime piece not built)"]
            elif section == "apple":
                gap = apple_gaps.get(name, "?")
                conds = [f"every device (Apple-private, {APPLE_GAP_WORDS.get(gap, gap)})"]
            elif section in ("bind", "limit"):
                conds = ["at the device's limit" if section == "bind" else "every device (a deterministic cap)"]
            elif section == "ratemap":
                conds = ["every device (this crate's bound: Apple's compile unmeasured or crashing there)"]
            elif section == "budget":
                conds = ["every device (the attempt's deadline, 20 seconds by default)"]
            elif section in ("invalid", "subproblem"):
                conds = ["every device (a public call's own check of its arguments)"]
            if not sites and not indirect and not (fam and fam["sites"]):
                state = "no site"
            elif not sites and indirect:
                state = "indirect"
            else:
                state = "direct"
            notsched = " (not scheduled)" if "not scheduled" in doc.lower() else ""
            if section in ("invalid", "subproblem"):
                # Reach is not marked here: every one of these is an argument of a public call,
                # so a reach mark would say `app` for all of them.
                rm = ("-", "")
            else:
                rm = reach.get((section, name))
                if rm is None:
                    missing_reach.append((section, name))
                    rm = ("?", "")
            prose, marker = split_basis(doc)
            if marker is None:
                basis, cite = derived_basis(section, (r["kinds"] if r else set())), ""
                if section == "invalid":
                    basis_faults.append((section, name, "carries no `Basis:` line"))
                elif section == "subproblem":
                    parent, value, required = sub_family[name.split("::")[0]]
                    if required:
                        basis_faults.append((section, name, "carries no `Basis:` line"))
                    else:
                        # The forwarding variant's own basis is a claim about the whole family, so
                        # this one derives it rather than repeating it.
                        basis, cite = value, f"from `{parent}`"
            else:
                basis, cite, complaint = parse_basis(marker, payloads.get(name, ""))
                if basis is None:
                    basis_faults.append((section, name, complaint))
                    basis = ""
                doc = prose
            if section == "invalid" and basis == DELEGATED:
                cite = f"{cite}, section {sub_section}"
            if sites:
                sc = sites_cell(sites)
            elif indirect:
                sc = "via a helper: " + sites_cell(indirect)
            elif fam and fam["sites"]:
                sc = "family audit only: " + sites_cell(fam["sites"])
            else:
                sc = "0"
            out.append(
                {
                    "name": name,
                    "what": summary(doc) + notsched,
                    "cond": "; ".join(conds) if conds else "-",
                    "sites": sc,
                    "state": state,
                    "reach": rm[0],
                    "why": rm[1].replace("|", "\\|"),
                    "basis": basis or "-",
                    "cite": cite.replace("|", "\\|"),
                    "size": "",
                    "all": sorted(set(sites or indirect or (fam["sites"] if fam else []))),
                }
            )
        tables[section] = (title, out)
    return tables, missing_reach, basis_faults


def render(tables):
    lines = []
    lines.append("# The refusal surface")
    lines.append("")
    lines.append(
        "Every refusal of something Metal accepts, read from the code: the construction sites of the "
        "typed refusals outside tests (`scripts/refusal-surface/refusal_surface.py`, which writes this "
        "file; edit its `reach.tsv` and the refusals' own doc comments, never the tables). Sections 8 "
        "and 9 are the other half: what a call did wrong, each row a claim that Metal itself refuses "
        "an input, with the basis of that claim; section 9 holds the problem enums a `Problem` "
        "forwards to, where most of those claims live. Malformed input of no Metal call at all is "
        "misuse and is not listed. [`METAL-COVERAGE.md`](METAL-COVERAGE.md) lists the Metal API the "
        "runtime has no counterpart for at all."
    )
    lines.append("")
    lines.append("Columns:")
    lines.append("")
    lines.append(
        "- **What it blocks**: the variant's doc comment, first sentence, and \"not scheduled\" "
        "where the doc says so."
    )
    lines.append(
        "- **Where**: the device condition, from the constructor at its sites: `NotImplemented` and "
        "`NotRepresentable` refuse on every device; `needs` names the Vulkan feature, extension, "
        "property or limit a device lacks; a driver defect or deviation names the driver. A site that "
        "computes its capability says so."
    )
    lines.append(
        "- **Sites**: how many construction sites, then the first two as `file (function)`, with "
        "the count where one function builds more than one. \"via a helper\" counts the places "
        "that hand the feature to a function that builds the refusal; \"family audit only\" is a gap "
        "no call refuses, only `supportsFamily:`'s audit (the API is missing: see the coverage map); "
        "0 is a variant nothing builds today."
    )
    lines.append(
        "- **Basis**: where the belief that this input must be refused comes from. Derived from the "
        "construction site where the site says it (`device`, `ours`), declared as a last `Basis:` "
        "line in the refusal's own doc comment where the claim is about Metal itself, and `-` where "
        "neither says it yet:"
    )
    for k, v in BASIS.items():
        lines.append(f"  - `{k}`: {v};")
    lines.append(
        "  - `delegated`: the variant forwards to another problem enum, whose own variants carry the "
        "claims. Those enums are read too: each is a family of section 9, named in the row, and a "
        "variant of one whose basis is absent or malformed fails the check as a `Problem`'s does. "
        "A family whose forwarding variant states a basis of its own instead (a runtime extension "
        "Metal has no counterpart for) carries that value in section 9, cited to the variant it "
        "comes from."
    )
    lines.append("- **Reach** (hand-kept, `reach.tsv`):")
    for k, v in REACH_MARKS.items():
        lines.append(f"  - `{k}`: {v};")
    lines.append("  - `?`: not assessed yet.")
    lines.append(
        "  Metal's own refusals (sections 8 and 9) carry no reach mark: every one of them is an "
        "argument of a public call."
    )
    lines.append("")
    # totals
    lines.append("## Totals")
    lines.append("")
    lines.append("| Kind | Variants | With a site | Reach system | Reach app | Reach rare | Reach none | Unassessed |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for section, (title, rows) in tables.items():
        c = defaultdict(int)
        for r in rows:
            c[r["reach"]] += 1
        live = sum(1 for r in rows if r["state"] != "no site")
        lines.append(
            f"| {title.split(' (')[0]} | {len(rows)} | {live} | {c['system']} | {c['app']} | {c['rare']} | {c['none']} | {c['?']} |"
        )
    lines.append("")
    lines.append("## Totals by basis")
    lines.append("")
    cols = list(BASIS) + ["delegated", "-"]
    lines.append("| Kind | " + " | ".join(f"`{k}`" for k in cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for section, (title, rows) in tables.items():
        c = defaultdict(int)
        for r in rows:
            c[r["basis"]] += 1
        lines.append(f"| {title.split(' (')[0]} | " + " | ".join(str(c[k]) for k in cols) + " |")
    lines.append("")
    for i, (section, (title, rows)) in enumerate(tables.items(), 1):
        lines.append(f"## {i}. {title}")
        lines.append("")
        lines.append("| Variant | What it blocks | Where | Basis | Sites | Reach |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            reach = f"`{r['reach']}`" + (f": {r['why']}" if r["why"] else "")
            basis = f"`{r['basis']}`" + (f": {r['cite']}" if r["cite"] else "")
            lines.append(
                f"| `{r['name']}` | {r['what']} | {r['cond']} | {basis} | {r['sites']} | {reach} |"
            )
        lines.append("")
    lines.append("## Appendix: every site")
    lines.append("")
    lines.append("Each refusal's sites, with the function that builds it.")
    lines.append("")
    for section, (title, rows) in tables.items():
        for r in rows:
            if not r["all"]:
                continue
            lines.append(f"- `{section}::{r['name']}`: " + "; ".join(f"`{x}`" for x in cite(r["all"])))
    lines.append("")
    return "\n".join(lines) + "\n"


def plain_table(tables):
    """One Markdown table of every refusal, most reachable first, with short device conditions."""
    order = {k: i for i, k in enumerate(REACH_MARKS)}

    def short(c):
        parts = []
        for p in c.split("; "):
            p = p.replace("devices without one of ", "lacking one of ").replace("devices without ", "lacking ")
            p = p.replace("every device (not implemented)", "all devices: not built")
            p = p.replace("every device (no Vulkan mechanism)", "all devices: no Vulkan form")
            p = p.replace("every device (runtime piece not built)", "all devices: not built")
            p = re.sub(r"every device \(Apple-private, ([^)]*)\)", r"all devices (Apple-private): \1", p)
            p = re.sub(r"Vk(Feature|Extension|Limit|Property)::", "", p)
            parts.append(p)
        # Every device's refusal (not built, no Vulkan form) first, so a long list of device
        # conditions never cuts it off.
        parts.sort(key=lambda p: not p.startswith("all devices"))
        s = "; ".join(parts)
        return s if len(s) < 160 else s[:157] + "..."

    rows = [(order.get(x["reach"], len(order)), sec, x) for sec, (_, rs) in tables.items() for x in rs]
    rows.sort(key=lambda t: (t[0], t[1], t[2]["name"]))
    out = [
        f"# The refusal surface at `{head_sha()}`",
        "",
        "Reach: " + "; ".join(f"{k} = {v}" for k, v in REACH_MARKS.items()) + ".",
        "",
        "Basis: " + "; ".join(f"{k} = {v}" for k, v in BASIS.items()) + ".",
        "",
        "| Reach | Basis | Kind | Refusal | Blocks | Where | Sites |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, sec, x in rows:
        n = x["sites"].split(" ")[0].rstrip(":")
        if x["sites"].startswith("via"):
            n = "via a helper"
        if x["sites"].startswith("family"):
            n = "family audit only"
        what = x["what"] if len(x["what"]) < 110 else x["what"][:107] + "..."
        out.append(f"| {x['reach']} | {x['basis']} | {sec} | {x['name']} | {what} | {short(x['cond'])} | {n} |")
    return "\n".join(out) + "\n"


# Why each Apple-only refusal is refused, from `AppleOnly::gap`.
APPLE_GAP_WORDS = {
    "Unmeasured": "Metal's answer not measured",
}


def apple_only_gaps(path):
    """Each `AppleOnly` variant's `AppleOnlyGap`, read from `AppleOnly::gap`'s match arms."""
    text = path.read_text()
    start = text.index("pub const fn gap(self) -> AppleOnlyGap")
    body = text[start:text.index("\n    }\n", start)]
    gaps = {}
    for arm in re.finditer(r"((?:\|?\s*Self::\w+\s*)+)=>\s*AppleOnlyGap::(\w+)", body):
        for name in re.findall(r"Self::(\w+)", arm.group(1)):
            gaps[name] = arm.group(2)
    return gaps


def selftest():
    """The basis marker's parse, over markers written here, both ways.

    The column is worth no more than this parse: a marker shape it stops recognising takes a claim
    out of the table, and a smaller table then reads like the truth. So each case asserts both that
    a good marker is read as what it says and that a bad one is named, and `--check` cannot pass a
    tree whose `Problem` carries no marker at all.
    """
    good = [
        ("metal-measured (an Apple M3 Max, macOS 27.0.1: the call asserts)", "", "metal-measured"),
        ("metal-documented (Metal asserts \"commit an already committed command buffer\")", "", "metal-documented"),
        ("metal-takes (an M3 Max: plain Metal returns and the pass completes)", "", "metal-takes"),
        ("device", "", "device"),
        ("ours", "", "ours"),
        ("unknown (Metal's answer here is unmeasured)", "", "unknown"),
        ("unknown", "", "unknown"),
        ("delegated(TextureProblem)", "TextureProblem", "delegated"),
        ("metal-measured (an M4).", "", "metal-measured"),
    ]
    bad = [
        ("", "", "an empty marker"),
        ("measured on an M4", "", "a value outside the vocabulary"),
        ("metal-meausred (an M4)", "", "a misspelled value"),
        ("metal-measured", "", "a measurement with no evidence"),
        ("metal-takes", "", "a plain-Metal claim with no evidence"),
        ("metal-documented", "", "a documented refusal with no citation"),
        ("metal-measured (the call asserts)", "", "a measurement that names no device or release"),
        ("metal-takes (plain Metal returns)", "", "a plain-Metal claim that names no device"),
        ("delegated", "TextureProblem", "`delegated` naming no enum"),
        ("delegated(ViewProblem)", "TextureProblem", "`delegated` naming another enum"),
        ("delegated(TextureProblem)", "", "`delegated` on a variant that forwards nothing"),
        ("Device", "", "a value of the wrong case"),
    ]
    bad_news = []
    for marker, payload, want in good:
        value, _, complaint = parse_basis(marker, payload)
        if value != want:
            bad_news.append(f"`Basis: {marker}` read as {value!r} ({complaint}), not {want!r}")
    for marker, payload, what in bad:
        value, _, complaint = parse_basis(marker, payload)
        if value is not None:
            bad_news.append(f"`Basis: {marker}` ({what}) was taken as {value!r}")
        elif not complaint:
            bad_news.append(f"`Basis: {marker}` ({what}) was refused without saying why")
    # The marker is the doc comment's last line, and the prose above it is left alone.
    prose, marker = split_basis("A thing Metal refuses. Basis: ours")
    if (prose, marker) != ("A thing Metal refuses.", "ours"):
        bad_news.append(f"split_basis read {(prose, marker)!r}")
    if split_basis("No marker here.") != ("No marker here.", None):
        bad_news.append("split_basis invented a marker")
    for line in bad_news:
        print("selftest:", line, file=sys.stderr)
    if bad_news:
        print(f"{len(bad_news)} basis cases failed", file=sys.stderr)
        return 1
    print(f"basis parse: {len(good)} markers read, {len(bad)} refused", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail when docs/REFUSALS.md is stale")
    ap.add_argument("--table", action="store_true", help="print one plain Markdown table, most reachable first")
    ap.add_argument("--missing-reach", action="store_true", help="list variants without a reach mark")
    ap.add_argument("--missing-basis", action="store_true", help="list refusals whose basis is absent or malformed")
    ap.add_argument("--selftest", action="store_true", help="run the basis marker's own cases")
    ap.add_argument(
        "--delegations",
        action="store_true",
        help="list the problem enums a `Problem` forwards to, as the scan finds them",
    )
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.delegations:
        error_rs = CRATES / "metal2vulkan-runtime" / "src" / "error.rs"
        for enum, file, parent, value, required in forwarded_problems(error_rs):
            n = len(enum_arms(file, enum)) if file is not None else 0
            where = file.relative_to(ROOT).as_posix() if file is not None else "?"
            print(f"{enum}\t{'required' if required else 'derived'}\t{where}\t{n}\t{parent}\t{value or '-'}")
        return 0
    tables, missing, faults = build()
    if a.missing_reach:
        for s, n in missing:
            print(f"{s}\t{n}")
        return 0
    if a.missing_basis:
        for s, n, why in faults:
            print(f"{s}\t{n}\t{why}")
        return 0
    if a.table:
        sys.stdout.write(plain_table(tables))
        return 0
    text = render(tables)
    if a.check:
        # Structural: the variants, what each blocks, where, and its reach. Sites and line
        # numbers drift with every edit and are not checked.
        def rows(t):
            out = []
            for line in t.split("\n"):
                f = line.split(" | ")
                if line.startswith("| `") and len(f) == 6:
                    out.append((f[0], f[1], f[2], f[3], f[5]))
            return out

        old = OUT.read_text() if OUT.exists() else ""
        if rows(old) != rows(text):
            stale = sorted(set(rows(old)) ^ set(rows(text)))
            for r in stale[:10]:
                print("changed:", " | ".join(r)[:200], file=sys.stderr)
            print("docs/REFUSALS.md is stale: rerun scripts/refusal-surface/refusal_surface.py", file=sys.stderr)
            return 1
        if missing:
            print(f"{len(missing)} variants without a reach mark (--missing-reach)", file=sys.stderr)
            return 1
        if faults:
            for s, n, why in faults[:20]:
                print(f"{s}::{n}: its `Basis:` {why}", file=sys.stderr)
            print(
                f"{len(faults)} refusals whose basis is absent or malformed (--missing-basis). Every "
                "variant of `Problem` carries a last doc-comment line `Basis: <value> (the "
                "evidence)`; a refusal whose basis is not settled says `Basis: unknown (why)` rather "
                "than nothing.",
                file=sys.stderr,
            )
            return 1
        return 0
    OUT.write_text(text)
    print(
        f"wrote {OUT.relative_to(ROOT)}; {len(missing)} variants without a reach mark, "
        f"{len(faults)} without a basis",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
