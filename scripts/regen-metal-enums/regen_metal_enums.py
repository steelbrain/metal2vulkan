#!/usr/bin/env python3
"""Generate the Metal enum and option-set values from the macOS SDK headers.

`crates/metal2vulkan/src/metal/generated.rs` holds the translator's;
`crates/metal2vulkan-air/src/metal/generated.rs` holds the ones the
AIR front end reads (`MTLDataType`, the type of a function constant), which the translator's
`metal` module re-exports.

Only enumerator names and their numeric values are extracted (the values are the Metal API's wire
contract, which guest command streams carry as raw ordinals); no header text is copied. Values
must be integer literals, shifts, ors, or references to earlier enumerators of the same type;
anything else fails generation instead of being guessed.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "crates" / "metal2vulkan" / "src" / "metal" / "generated.rs"
AIR_OUTPUT = ROOT / "crates" / "metal2vulkan-air" / "src" / "metal" / "generated.rs"
TABLE_OUTPUT = ROOT / "scripts" / "conformance-coverage" / "metal_enumerators.tsv"

# NS_ENUM types, emitted as closed Rust enums (`metal_enum!`).
ENUMS = [
    "MTLArgumentBuffersTier",
    "MTLAttributeFormat",
    "MTLBindingAccess",
    "MTLBindingType",
    "MTLBlendFactor",
    "MTLBlendOperation",
    "MTLCompareFunction",
    "MTLCullMode",
    "MTLDepthClipMode",
    "MTLDispatchType",
    "MTLFunctionType",
    "MTLIndexType",
    "MTLMutability",
    "MTLPatchType",
    "MTLPixelFormat",
    "MTLPrimitiveTopologyClass",
    "MTLPrimitiveType",
    "MTLReadWriteTextureTier",
    "MTLSamplerAddressMode",
    "MTLSamplerBorderColor",
    "MTLSamplerMinMagFilter",
    "MTLSamplerMipFilter",
    "MTLSamplerReductionMode",
    "MTLStencilOperation",
    "MTLStepFunction",
    "MTLTessellationControlPointIndexType",
    "MTLTessellationFactorFormat",
    "MTLTessellationFactorStepFunction",
    "MTLTessellationPartitionMode",
    "MTLTextureSwizzle",
    "MTLTextureType",
    "MTLTriangleFillMode",
    "MTLVertexFormat",
    "MTLVertexStepFunction",
    "MTLWinding",
]

# The NS_ENUM types a consumer maps completely, emitted without `#[non_exhaustive]`:
# a value added to one is a compile error in every consumer, the runtime included, rather than a
# wildcard arm. Every other enum is `#[non_exhaustive]`.
EXHAUSTIVE = {
    "MTLCullMode",
    "MTLIndexType",
    "MTLPrimitiveType",
    "MTLTextureType",
    "MTLWinding",
}

# Enumerators the SDK headers do not declare but Metal makes on Apple GPUs, appended after a type's
# header enumerators: each name is the one Metal prints in an object's description for the raw
# value, where it prints one (an Apple M3 Max on macOS 26.7 and an Apple M4 on macOS 27.0.1).
# Programs that page Core Video surfaces create textures in the pixel formats.
PRIVATE = {
    "MTLPixelFormat": [
        ("MTLPixelFormatARGB8Unorm", 84),
        ("MTLPixelFormatRGB10A8_2P_XR10", 550),
        ("MTLPixelFormatRGB10A8_2P_XR10_sRGB", 551),
        # The names a texture descriptor's description prints for these raw values, on an Apple
        # M4 under macOS 27.0.1: the two-plane YCbCr formats (plain) and the three after them (plain
        # and validated). Metal makes a texture of the two-plane ones only over an IOSurface plane,
        # and none of the others from a descriptor alone.
        ("MTLPixelFormatYCBCR8_420_2P", 500),
        ("MTLPixelFormatYCBCR8_422_2P", 502),
        ("MTLPixelFormatYCBCR8_444_2P", 503),
        ("MTLPixelFormatYCBCR10_420_2P", 505),
        ("MTLPixelFormatYCBCR10_422_2P", 506),
        ("MTLPixelFormatYCBCR10_444_2P", 507),
        ("MTLPixelFormatYCBCR10_420_2P_PACKED", 508),
        ("MTLPixelFormatYCBCR10_422_2P_PACKED", 509),
        ("MTLPixelFormatYCBCR10_444_2P_PACKED", 510),
        ("MTLPixelFormatYCBCR8_420_2P_sRGB", 520),
        ("MTLPixelFormatYCBCR8_422_2P_sRGB", 522),
        ("MTLPixelFormatYCBCR8_444_2P_sRGB", 523),
        ("MTLPixelFormatYCBCR10_420_2P_sRGB", 525),
        ("MTLPixelFormatYCBCR10_422_2P_sRGB", 526),
        ("MTLPixelFormatYCBCR10_444_2P_sRGB", 527),
        ("MTLPixelFormatYCBCR10_420_2P_PACKED_sRGB", 528),
        ("MTLPixelFormatYCBCR10_422_2P_PACKED_sRGB", 529),
        ("MTLPixelFormatYCBCR10_444_2P_PACKED_sRGB", 530),
        ("MTLPixelFormatRGB8_420_2P", 540),
        ("MTLPixelFormatRGB8_422_2P", 541),
        ("MTLPixelFormatRGB8_444_2P", 542),
        ("MTLPixelFormatRGB10_420_2P", 543),
        ("MTLPixelFormatRGB10_422_2P", 544),
        ("MTLPixelFormatRGB10_444_2P", 545),
        ("MTLPixelFormatRGB10_420_2P_PACKED", 546),
        ("MTLPixelFormatRGB10_422_2P_PACKED", 547),
        ("MTLPixelFormatRGB10_444_2P_PACKED", 548),
        ("MTLPixelFormatYCBCR10_420_2P_PQ", 564),
        ("MTLPixelFormatYCBCR10_422_2P_PQ", 565),
        ("MTLPixelFormatYCBCR10_444_2P_PQ", 566),
        ("MTLPixelFormatYCBCR10_420_2P_PACKED_PQ", 567),
        ("MTLPixelFormatYCBCR10_422_2P_PACKED_PQ", 568),
        ("MTLPixelFormatYCBCR10_444_2P_PACKED_PQ", 569),
        ("MTLPixelFormatYCBCR12_420_2P", 570),
        ("MTLPixelFormatYCBCR12_422_2P", 571),
        ("MTLPixelFormatYCBCR12_444_2P", 572),
        ("MTLPixelFormatYCBCR12_420_2P_PQ", 573),
        ("MTLPixelFormatYCBCR12_422_2P_PQ", 574),
        ("MTLPixelFormatYCBCR12_444_2P_PQ", 575),
        ("MTLPixelFormatYCBCR12_420_2P_PACKED", 580),
        ("MTLPixelFormatYCBCR12_422_2P_PACKED", 581),
        ("MTLPixelFormatYCBCR12_444_2P_PACKED", 582),
        ("MTLPixelFormatYCBCR12_420_2P_PACKED_PQ", 583),
        ("MTLPixelFormatYCBCR12_422_2P_PACKED_PQ", 584),
        ("MTLPixelFormatYCBCR12_444_2P_PACKED_PQ", 585),
        ("MTLPixelFormatYCBCR10_420_2P_XR", 591),
        ("MTLPixelFormatYCBCR10_422_2P_XR", 592),
        ("MTLPixelFormatYCBCR10_444_2P_XR", 593),
        ("MTLPixelFormatYCBCR10_420_2P_PACKED_XR", 594),
        ("MTLPixelFormatYCBCR10_422_2P_PACKED_XR", 595),
        ("MTLPixelFormatYCBCR10_444_2P_PACKED_XR", 596),
        ("MTLPixelFormatYCBCR12_420_2P_XR", 597),
        ("MTLPixelFormatYCBCR12_422_2P_XR", 598),
        ("MTLPixelFormatYCBCR12_444_2P_XR", 599),
        ("MTLPixelFormatYCBCR12_420_2P_PACKED_XR", 600),
        ("MTLPixelFormatYCBCR12_422_2P_PACKED_XR", 601),
        ("MTLPixelFormatYCBCR12_444_2P_PACKED_XR", 602),
        ("MTLPixelFormatYCBCR10_420_2P_HLG", 608),
        ("MTLPixelFormatYCBCR10_422_2P_HLG", 609),
        ("MTLPixelFormatYCBCR10_444_2P_HLG", 610),
        ("MTLPixelFormatYCBCR10_420_2P_PACKED_HLG", 611),
        ("MTLPixelFormatYCBCR10_422_2P_PACKED_HLG", 612),
        ("MTLPixelFormatYCBCR10_444_2P_PACKED_HLG", 613),
        ("MTLPixelFormatYCBCR12_420_2P_HLG", 615),
        ("MTLPixelFormatYCBCR12_422_2P_HLG", 616),
        ("MTLPixelFormatYCBCR12_444_2P_HLG", 617),
        ("MTLPixelFormatYCBCR12_420_2P_PACKED_HLG", 618),
        ("MTLPixelFormatYCBCR12_422_2P_PACKED_HLG", 619),
        ("MTLPixelFormatYCBCR12_444_2P_PACKED_HLG", 620),
        ("MTLPixelFormatRGBA16Float_XR", 556),
        ("MTLPixelFormatAnyDepth", 651),
        ("MTLPixelFormatAnyStencil", 652),
        # ASTC's 4x2 and 8x4 footprints, which the headers do not declare: the names a texture's
        # description prints for these raw values, on an Apple M4 under macOS 27.0.1 (a census of
        # every raw value Metal makes a texture of). Metal makes a 2D texture of each from a
        # descriptor alone.
        ("MTLPixelFormatASTC_4x2_sRGB", 185),
        ("MTLPixelFormatASTC_8x4_sRGB", 191),
        ("MTLPixelFormatASTC_4x2_LDR", 203),
        ("MTLPixelFormatASTC_8x4_LDR", 209),
        ("MTLPixelFormatASTC_4x2_HDR", 221),
        ("MTLPixelFormatASTC_8x4_HDR", 227),
        # Every other raw value a texture descriptor's description names, on an Apple M4 under
        # macOS 27.0.1, that neither the headers nor the entries above declare (every value from 0
        # to 1100): the descriptor's name for each, which `MTLPixelFormatGetInfoForDevice` gives too
        # for all but `RGB16Bfloat` and the 12-bit `Uint` formats from 646 on. An Apple M3 Max under
        # macOS 26.7 names each of them the same but `RGB16Bfloat`, and describes none from 630 on.
        ("MTLPixelFormatR16Bfloat", 26),
        ("MTLPixelFormatBGRA4Unorm", 44),
        ("MTLPixelFormatRG16Bfloat", 66),
        ("MTLPixelFormatARGB8Unorm_sRGB", 85),
        ("MTLPixelFormatRGB16Bfloat", 100),
        ("MTLPixelFormatRGBA16Bfloat", 116),
        ("MTLPixelFormatRGBX8Unorm", 300),
        ("MTLPixelFormatRGBX8Unorm_sRGB", 301),
        ("MTLPixelFormatRGBX8Snorm", 302),
        ("MTLPixelFormatRGBX8Uint", 303),
        ("MTLPixelFormatRGBX8Sint", 304),
        ("MTLPixelFormatRGBX16Uint", 313),
        ("MTLPixelFormatRGBX16Sint", 314),
        ("MTLPixelFormatRGBX16Float", 315),
        ("MTLPixelFormatRGBX32Uint", 323),
        ("MTLPixelFormatRGBX32Sint", 324),
        ("MTLPixelFormatRGBX32Float", 325),
        ("MTLPixelFormatYCBCR8_422_1P", 501),
        ("MTLPixelFormatYCBCR10_444_1P", 504),
        ("MTLPixelFormatYCBCR8_422_1P_sRGB", 521),
        ("MTLPixelFormatYCBCR10_444_1P_sRGB", 524),
        ("MTLPixelFormatYCBCRA8_444_1P", 560),
        ("MTLPixelFormatYCBCR8_422_1P_REV", 562),
        ("MTLPixelFormatYCBCR10_444_1P_PQ", 563),
        ("MTLPixelFormatR10Unorm_X6", 576),
        ("MTLPixelFormatR10Unorm_X6_sRGB", 577),
        ("MTLPixelFormatRG10Unorm_X12", 578),
        ("MTLPixelFormatRG10Unorm_X12_sRGB", 579),
        ("MTLPixelFormatRGB10A2Unorm_sRGB", 586),
        ("MTLPixelFormatRGB10A2Unorm_PQ", 587),
        ("MTLPixelFormatR10Unorm_PACKED", 588),
        ("MTLPixelFormatRG10Unorm_PACKED", 589),
        ("MTLPixelFormatYCBCR10_444_1P_XR", 590),
        ("MTLPixelFormatR12Unorm_X4", 603),
        ("MTLPixelFormatR12Unorm_X4_PQ", 604),
        ("MTLPixelFormatRG12Unorm_X8", 605),
        ("MTLPixelFormatR10Unorm_X6_PQ", 606),
        ("MTLPixelFormatYCBCR10_444_1P_HLG", 607),
        ("MTLPixelFormatR10Unorm_X6_HLG", 621),
        ("MTLPixelFormatR12Unorm_X4_HLG", 622),
        ("MTLPixelFormatRGB10A2Unorm_HLG", 623),
        ("MTLPixelFormatRGB10A8_2P", 624),
        ("MTLPixelFormatRGB10A8_2P_sRGB", 625),
        ("MTLPixelFormatRGB10A8_2P_PQ", 626),
        ("MTLPixelFormatRGB10A8_2P_HLG", 627),
        ("MTLPixelFormatBGR10A2Unorm_sRGB", 628),
        ("MTLPixelFormatBGR10A2Unorm_PQ", 629),
        ("MTLPixelFormatBGRA10Uint_PACKED", 630),
        ("MTLPixelFormatBGRA10_XR_PACKED", 631),
        ("MTLPixelFormatBGRA10_XR_PACKED_sRGB", 632),
        ("MTLPixelFormatBGRA10Unorm_PACKED", 633),
        ("MTLPixelFormatBGRA10Unorm_PACKED_sRGB", 634),
        ("MTLPixelFormatBGRA10Unorm_PACKED_HLG", 635),
        ("MTLPixelFormatBGRA10Unorm_PACKED_PQ", 636),
        ("MTLPixelFormatBGRA10Uint", 637),
        ("MTLPixelFormatBGRA10Unorm", 638),
        ("MTLPixelFormatBGRA10Unorm_sRGB", 639),
        ("MTLPixelFormatBGRA10Unorm_HLG", 640),
        ("MTLPixelFormatBGRA10Unorm_PQ", 641),
        ("MTLPixelFormatR10Uint_X6", 642),
        ("MTLPixelFormatR10Uint_PACKED", 643),
        ("MTLPixelFormatRG10Uint_X12", 644),
        ("MTLPixelFormatRG10Uint_PACKED", 645),
        ("MTLPixelFormatR12Uint_X4", 646),
        ("MTLPixelFormatR12Uint_PACKED", 647),
        ("MTLPixelFormatRG12Uint_X8", 648),
        ("MTLPixelFormatRG12Uint_PACKED", 649),
        ("MTLPixelFormatRGB10A8Uint_2P", 650),
    ],
    # Metal's own name for the raw value 2 of `minFilter` and `magFilter`, which a sampler
    # descriptor's description prints (an Apple M4 on macOS 27.0.1): Metal makes
    # a sampler state of it, and samples through it with its bicubic filter. The
    # descriptor keeps two bits; 3 stops the process in `newSamplerStateWithDescriptor:`.
    "MTLSamplerMinMagFilter": [("MTLSamplerMinMagFilterCubic", 2)],
    # The `functionType` Metal answers for a Core Image kernel library's extern function (`TYPE` 5,
    # from `newFunctionWithName:` or `newExternFunctionWithName:`), whose description prints
    # "Unknown" (an Apple M3 Max on macOS 27.0.1): the name is the one Metal's
    # `newExternFunctionWithName:` and `externFunctionNames` give the kind.
    "MTLFunctionType": [("MTLFunctionTypeExtern", 4)],
    # Every texture creator takes these bits and the texture's usage keeps them, its views' and
    # shared opens' too; the names are the ones its description prints (an Apple
    # M4 on macOS 27.0.1, plain and validated).
    "MTLTextureUsage": [
        ("MTLTextureUsageNoConcurrentAccess", 0x4000),
        ("MTLTextureUsageVideo", 0x8000),
        ("MTLTextureUsageBlockWritesOnly", 0x10000),
    ],
}

# Bits of an option set the SDK headers do not declare that Metal takes and drops: every texture
# creator takes usage bit 0x8 and makes the texture its other bits make, whose usage answers
# without it, while the heap size query of a descriptor of it alone answers as for a usage.
# `TryFrom<u32>` keeps them; `without_dropped` is what an object made of the value answers.
DROPPED = {
    "MTLTextureUsage": [0x8],
}

# NS_ENUM types the AIR front end reads, emitted into its own `metal` module (`metal_enum!`).
AIR_ENUMS = [
    "MTLDataType",
]

# Metal enums and option sets the tree models **by hand** rather than from these headers, because
# the compile-and-plan layer never reads them: the runtime declares `LoadAction`, `StoreAction`,
# `PipelineOption` and the rest in its own sources and says so there.
#
# No Rust is generated for them, and none should be: that decision belongs to whoever made it.
# What is generated is `TABLE_OUTPUT`, a table of their SDK enumerators and ordinals, which is the
# option-space axis's denominator for them (`scripts/conformance-coverage/option_space.py`).
# Without it that axis measures a hand-written variant list against itself, which cannot report a
# value the SDK has and the tree does not, and reports an option-set newtype -- which declares no
# variants at all -- as a type with no values.
#
# `MTLResourceOptions` is **not** here: the SDK writes its values in terms of `MTLCPUCacheMode`'s
# and `MTLStorageMode`'s enumerators shifted by `#define`d amounts, and `evaluate` takes integer
# literals, shifts, ors and earlier enumerators of the same type and fails on anything else rather
# than guessing it. So that one type keeps no denominator, with a reason, instead of one this
# script would have to widen its contract to produce.
TABLE_ONLY = [
    "MTLBlitOption",
    "MTLCPUCacheMode",
    "MTLCommandBufferStatus",
    "MTLHazardTrackingMode",
    "MTLHeapType",
    "MTLLoadAction",
    "MTLMultisampleDepthResolveFilter",
    "MTLMultisampleStencilResolveFilter",
    "MTLPipelineOption",
    "MTLPurgeableState",
    "MTLSparsePageSize",
    "MTLSparseTextureRegionAlignmentMode",
    "MTLStorageMode",
    "MTLStoreAction",
    "MTLStoreActionOptions",
    "MTLVisibilityResultMode",
]

# NS_OPTIONS types, emitted as bit-set newtypes (`metal_flags!`).
OPTIONS = [
    "MTLBarrierScope",
    "MTLColorWriteMask",
    "MTLFunctionOptions",
    "MTLIndirectCommandType",
    "MTLRenderStages",
    "MTLResourceUsage",
    "MTLTextureUsage",
]

DECL = re.compile(
    r"typedef\s+NS_(ENUM|OPTIONS)\s*\(\s*(\w+)\s*,\s*(MTL\w+)\s*\)\s*\{(.*?)\}", re.S)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def strip_macros(text):
    """Removes `ALL_CAPS(...)` annotations (availability, Swift names), balancing parentheses."""
    out = []
    i = 0
    pattern = re.compile(r"\b[A-Z][A-Z0-9_]*\s*\(")
    while True:
        m = pattern.search(text, i)
        if not m:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:m.start()])
        depth = 0
        j = m.end() - 1
        while j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            raise ValueError("unbalanced annotation parentheses")
        i = j + 1


def evaluate(expr, known):
    expr = expr.strip()
    tokens = re.findall(r"0[xX][0-9a-fA-F]+[uUlL]*|[0-9]+[uUlL]*|[A-Za-z_]\w*|<<|[|()]|\S", expr)
    parts = []
    for tok in tokens:
        if re.fullmatch(r"0[xX][0-9a-fA-F]+[uUlL]*", tok):
            parts.append(str(int(tok.rstrip("uUlL"), 16)))
        elif re.fullmatch(r"[0-9]+[uUlL]*", tok):
            parts.append(str(int(tok.rstrip("uUlL"))))
        elif re.fullmatch(r"[A-Za-z_]\w*", tok):
            if tok not in known:
                raise ValueError(f"unknown identifier {tok} in {expr!r}")
            parts.append(str(known[tok]))
        elif tok in ("<<", "|", "(", ")"):
            parts.append(tok)
        else:
            raise ValueError(f"unsupported token {tok!r} in {expr!r}")
    if not parts:
        raise ValueError("empty value expression")
    return int(eval(" ".join(parts), {"__builtins__": {}}, {}))


def parse_body(name, body):
    body = strip_macros(strip_comments(body))
    entries = []
    known = {}
    previous = -1
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        m = re.fullmatch(r"(\w+)\s*(?:=\s*(.+))?", item, re.S)
        if not m:
            raise ValueError(f"unparsed enumerator in {name}: {item!r}")
        ident, expr = m.group(1), m.group(2)
        value = evaluate(expr, known) if expr is not None else previous + 1
        known[ident] = value
        entries.append((ident, value))
        previous = value
    if not entries:
        raise ValueError(f"{name} has no enumerators")
    return entries


def collect(headers):
    found = {}
    for path in sorted(headers.glob("*.h")):
        text = strip_comments(path.read_text())
        for m in DECL.finditer(text):
            kind, _, name, body = m.groups()
            if name in found:
                raise ValueError(f"{name} declared twice")
            found[name] = (kind, body)
    return found


def variant_name(ident, prefixes):
    for prefix in prefixes:
        if ident.startswith(prefix) and len(ident) > len(prefix):
            rest = ident[len(prefix):]
            break
    else:
        raise ValueError(f"{ident} does not start with {prefixes[0]}")
    words = [w for w in rest.split("_") if w]
    rest = "".join(w[0].upper() + w[1:] for w in words)
    if rest[0].isdigit():
        rest = re.findall(r"[A-Z][a-z]*", prefixes[0])[-1] + rest
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", rest):
        raise ValueError(f"unexpected variant name {rest} from {ident}")
    return rest


def const_name(ident, prefixes):
    camel = variant_name(ident, prefixes)
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", camel)
    return "_".join(w.upper() for w in words)


def rows(name, entries, flags):
    """Canonical (rust, enumerator, value) rows and (alias enumerator, rust) pairs.

    The canonical enumerator of a value is the first one spelled with the type's own prefix; other
    enumerators with the same value (deprecated spellings) become aliases.
    """
    prefixes = [name, name.rstrip("s")]
    canonical = {}
    for ident, value in entries:
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError(f"{ident} = {value} does not fit a u32 ordinal")
        if value not in canonical and any(ident.startswith(p) for p in prefixes):
            canonical[value] = ident
    out = []
    names = {}
    for ident, value in entries:
        if value not in canonical:
            raise ValueError(f"{ident} has no enumerator spelled with the {name} prefix")
        if canonical[value] != ident:
            continue
        rust = const_name(ident, prefixes) if flags else variant_name(ident, prefixes)
        if rust in names:
            raise ValueError(f"colliding name {rust} in {name}")
        names[ident] = rust
        out.append((rust, ident, value))
    aliases = [(ident, names[canonical[value]]) for ident, value in entries
               if canonical[value] != ident]
    return out, aliases


def render_table(found, sdk_version, names):
    """The SDK's enumerators for `names`, as a tab-separated table.

    One row per name: the Metal type, `enum` or `flags`, whether the row is the `value` or an
    `alias` spelling of one, the Rust name this generator derives, the SDK enumerator and the
    ordinal. An alias shares its value row's ordinal and Rust name.

    A table rather than Rust because nothing in the tree should declare these types twice: the
    runtime already declares them by hand, and the only thing missing was a denominator to measure
    that declaration against.
    """
    out = [
        f"# Generated from the macOS {sdk_version} SDK Metal.framework headers (names and values).",
        "# Regenerate with scripts/regen-metal-enums/regen_metal_enums.py; do not edit.",
        "# The SDK's values for the Metal enums and option sets this tree models by hand, which is",
        "# the conformance suite's option-space denominator for them.",
        "type\tkind\trole\trust\tenumerator\tvalue",
    ]
    for name in names:
        if name not in found:
            raise ValueError(f"{name} not found in the SDK headers")
        kind, body = found[name]
        flags = kind == "OPTIONS"
        parsed = parse_body(name, body)
        known = {value for _, value in parsed}
        for ident, value in PRIVATE.get(name, []):
            if value in known:
                raise ValueError(f"{ident} = {value} is declared by the headers now")
            parsed.append((ident, value))
        entries, aliases = rows(name, parsed, flags)
        of_rust = {rust: value for rust, _, value in entries}
        for rust, ident, value in entries:
            out.append(f"{name}\t{'flags' if flags else 'enum'}\tvalue\t{rust}\t{ident}\t{value}")
        # A deprecated spelling of a value already listed. It is a name, not a value, so it adds
        # no row to any denominator -- but a recording or a case may name a value by it, and a
        # numerator that reads names has to know it means the same value. `MTLPipelineOption`'s
        # bit 1 is exactly this: the headers declare `MTLPipelineOptionArgumentInfo` first and
        # `MTLPipelineOptionBindingInfo` after it, so the spelling the tree and Apple's
        # documentation both use is the alias.
        for ident, rust in aliases:
            out.append(
                f"{name}\t{'flags' if flags else 'enum'}\talias\t{rust}\t{ident}"
                f"\t{of_rust[rust]}"
            )
    return "\n".join(out) + "\n"


def render(found, sdk_version, enums, options):
    out = [
        f"// Generated from the macOS {sdk_version} SDK Metal.framework headers (names and values).",
        "// Regenerate with scripts/regen-metal-enums/regen_metal_enums.py; do not edit.",
        "",
    ]
    for name, flags in [(n, False) for n in enums] + [(n, True) for n in options]:
        if name not in found:
            raise ValueError(f"{name} not found in the SDK headers")
        kind, body = found[name]
        if (kind == "OPTIONS") != flags:
            raise ValueError(f"{name} is NS_{kind}, expected the other kind")
        parsed = parse_body(name, body)
        known = {value for _, value in parsed}
        for ident, value in PRIVATE.get(name, []):
            if value in known:
                raise ValueError(f"{ident} = {value} is declared by the headers now")
            parsed.append((ident, value))
        entries, aliases = rows(name, parsed, flags)
        macro = "metal_flags" if flags else "metal_enum"
        out.append(f"{macro}! {{")
        out.append(f"    /// `{name}`.")
        if not flags and name not in EXHAUSTIVE:
            out.append("    #[non_exhaustive]")
        out.append(f'    {name[3:]} = "{name}" {{')
        for rust, ident, value in entries:
            shown = f"0x{value:x}" if flags else str(value)
            out.append(f'        {rust} = {shown}, "{ident}";')
        out.append("    }")
        out.append("    aliases {")
        for ident, rust in aliases:
            out.append(f'        "{ident}" => {rust};')
        out.append("    }")
        if flags:
            out.append("    dropped {")
            for value in DROPPED.get(name, []):
                if value in known or any(value & v for _, _, v in entries):
                    raise ValueError(f"{name} bit 0x{value:x} is named and dropped")
                out.append(f"        0x{value:x};")
            out.append("    }")
        out.append("}")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def sdk():
    path = subprocess.run(["xcrun", "--sdk", "macosx", "--show-sdk-path"], check=True,
                          capture_output=True, text=True).stdout.strip()
    version = subprocess.run(["xcrun", "--sdk", "macosx", "--show-sdk-version"], check=True,
                             capture_output=True, text=True).stdout.strip()
    return Path(path) / "System/Library/Frameworks/Metal.framework/Headers", version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", type=Path, help="Metal.framework/Headers directory")
    parser.add_argument("--sdk-version", help="SDK version recorded in the output header")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--air-output", type=Path, default=AIR_OUTPUT)
    parser.add_argument("--table-output", type=Path, default=TABLE_OUTPUT,
                        help="the SDK values of the types this tree models by hand")
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed output differs from a fresh generation")
    args = parser.parse_args()
    if args.headers:
        if not args.sdk_version:
            parser.error("--headers requires --sdk-version")
        headers, version = args.headers, args.sdk_version
    else:
        headers, version = sdk()
    found = collect(headers)
    outputs = [
        (args.output, render(found, version, ENUMS, OPTIONS)),
        (args.air_output, render(found, version, AIR_ENUMS, [])),
        (args.table_output, render_table(found, version, TABLE_ONLY)),
    ]
    if args.check:
        stale = [path for path, text in outputs if path.read_text() != text]
        for path in stale:
            print(f"{path} is stale; rerun without --check", file=sys.stderr)
        if stale:
            sys.exit(1)
        return
    for path, text in outputs:
        path.write_text(text)


if __name__ == "__main__":
    main()
