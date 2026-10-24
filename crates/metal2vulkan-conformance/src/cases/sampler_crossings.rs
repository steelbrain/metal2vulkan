//! Pairs and triples of a sampler descriptor's fields, where each member alone is already
//! compared.
//!
//! The option-space axis reads `MTLSamplerAddressMode`, `MTLSamplerBorderColor`,
//! `MTLSamplerMinMagFilter`, `MTLSamplerMipFilter` and `MTLCompareFunction` at every one of their
//! values, and [`super::sampler_filtering`], [`super::nearest_positions`],
//! [`super::border_swizzles`] and [`super::pixel_samplers`] are where that happened. What none of
//! them put to Metal is a **crossing**: each of those four sweeps holds every field but one at a
//! value chosen to keep the one it sweeps legible, and three of those held values are exactly the
//! ones that interact.
//!
//! - `sampler_filtering`'s address sweep pins `s` and `t` to `Repeat` and moves `r` alone, so no
//!   row of it ever had **three different modes at once**; nor has any other case — `border_swizzles`
//!   and `nearest_positions` set `s` and `t` together and leave `r` at the default.
//! - Its border sweep likewise reads a border on the `r` axis with `s` and `t` repeating and
//!   **inside** the texture, so no row asks what a border-reading axis does while another axis is
//!   the one out of range.
//! - Its mip sweep pins **both** filters to nearest, and says so in as many words: "two nearest
//!   filters answer it identically whichever Metal picks". That is what makes the sweep readable
//!   and it is also what drops the question of **which** of the two filters a clamped level picks.
//! - Its comparison sweep samples at texel **centres**, where a linear footprint is one texel, so
//!   its eight `MTLCompareFunction` rows read the same under either filter and the one crossing a
//!   depth sampler has — filtering the comparison **results** rather than comparing a filtered
//!   depth — cannot show. The tree's `msf_compare_filter` fixture was written for that question
//!   and no case of this suite reads it.
//!
//! Four families here, each a crossing and each with a falsification only the crossing can see.
//!
//! # `axes` — three address modes at once
//!
//! `msf_runtime_samplers`' `msf_slot3d_linear_repeat` samples a 4 by 4 by 4 `R32Float` texture
//! whose texel `(x, y, z)` holds `1 + x + 4 y + 16 z`, through its own constexpr sampler (linear,
//! repeating on all three axes) and through the runtime sampler at `[[sampler(0)]]`. The runtime
//! sampler filters **nearest** on both filters, so its answer **names the texel it read** rather
//! than a blend of two.
//!
//! The lanes are the whole 5 by 5 by 5 grid of `AXIS_COORDS` — one coordinate inside and four
//! outside, two of them negative — so every combination of "this axis is inside, that one is a
//! period out, the third is far out the other way" is a lane of its own. That is the point: an
//! implementation that decides *whether* to wrap from the coordinate as a whole, or that applies
//! one axis's mode to all three, reads the right texel on every lane whose axes agree and the
//! wrong one as soon as they do not.
//!
//! `TRIPLES`'s nine rows are three with all three modes equal — the controls, which no collapse
//! onto one axis can move — and six with them unequal: the three rotations of
//! `(Repeat, MirrorRepeat, ClampToEdge)`, so a collapse onto `s` and a collapse onto `t` give
//! different wrong answers, and three rotations of `(MirrorClampToEdge, Repeat, ClampToEdge)`,
//! which crosses the one mode MSL cannot spell in a constexpr sampler against the two it can.
//!
//! # `axis-border` — a border on one axis while another is out of range
//!
//! The same 3D kernel over an `RGBA8Unorm` texture, because `R32Float` cannot part the three
//! border colours (Metal fills a one-channel sample as `(r, 0, 0, 1)`, so transparent and opaque
//! black both read 0 in the only channel there is — `sampler_filtering` measured that and the
//! format is chosen for the same reason here). One case per `MTLSamplerBorderColor`, and
//! `BORDER_TRIPLES`'s six rows put a border-reading mode on `s` alone, on `t` alone, on `r`
//! alone, on all three (the control), and twice crossed with `ClampToZero` on another axis, which
//! on this oracle reads the border colour too.
//!
//! The question only the crossing asks: on a lane whose `s` is inside and whose `t` is a period
//! out, does a sampler whose **`s`** mode is `ClampToBorderColor` read the border? Metal's answer
//! is recorded, not assumed.
//!
//! # `mip-filter-pairs` — which of the two filters a clamped level picks
//!
//! `msf_slot2d_linear_edge` over the 4 by 4 `R32Float` chain of three levels whose texels are
//! distinct numbers, sampled at a quarter of a texel past each texel's lower corner — **not** at a
//! centre, which is where a nearest and a linear filter inside one level read the same texel and
//! where `sampler_filtering`'s mip sweep deliberately sits.
//!
//! One case per `MTLSamplerMipFilter`; `FILTER_PAIRS`'s four `(min, mag)` pairs crossed with
//! `PAIR_CLAMPS`'s five `lodMinClamp` values are the twenty rows. The kernel samples at an
//! explicit `level(0.0)` and Metal raises that level to `lodMinClamp`, so the row's clamp is the
//! level — and **which filter reads it** is the crossing. Vulkan states the rule in its own terms
//! (the magnifying filter at a level of zero or less, the minifying one above it), Metal's rule for
//! a `NotMipmapped` sampler that ignores the clamp outright is a third thing again, and the two
//! rows whose filters agree cannot tell any of them apart.
//!
//! # `compare-filters` — filtering the comparison, not the depth
//!
//! `rrp_mirror_clamp`'s `rrp_mc_compare` takes five `sample_compare`s of a 4 by 4
//! `Depth32Float` texture through the runtime sampler, at texel offsets 0, -3, 2, -7 and 5 on `s`.
//! The sixteen rows are every `MTLCompareFunction` crossed with a nearest and with a linear min and
//! mag filter.
//!
//! The lanes sample at two coordinates with four texels in the footprint — the shared corner of
//! texels `(1, 1)`, `(2, 1)`, `(1, 2)` and `(2, 2)`, where a linear filter weights them a quarter
//! each, and a point three quarters of the way towards that corner, where the four weights are
//! 9/16, 3/16, 3/16 and 1/16 — against `compare_references`'s nine references: below all four
//! depths, exactly on each of them, between each adjacent pair, and above all four.
//!
//! A depth sampler compares **per texel** and filters the ones and zeros that come out. So a
//! linear row's answer at the equal-weight coordinate is a quarter, a half or three quarters as the
//! reference passes each texel's depth, and at the unequal-weight one a sixteenth, three
//! sixteenths and the sums of those — while a nearest row's answer is 0 or 1 and nothing else. An
//! implementation that filtered the depth and then compared would answer 0 or 1 on **every** row,
//! and its nearest rows would be right.
//!
//! # Nothing here can fault a GPU
//!
//! One compute pass a row, read-only textures of at most 4 by 4 by 4, two buffers a row, one
//! thread a lane, no loop, no fragment stage. The `axes` and `axis-border` families' coordinates
//! reach outside their texture on purpose, which is what an address mode is for; every other
//! family's are inside.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side,
    TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
    SamplerMipFilter, TextureType, TextureUsage,
};

/// The fixture whose slot kernels sample through the runtime sampler at `[[sampler(0)]]` beside a
/// constexpr sampler of their own.
const SLOTS: &str = "msf_runtime_samplers";

/// Its 2D kernel, whose constexpr sampler is `filter::linear, address::clamp_to_edge` and has no
/// mip filter, so its column is pinned to level 0.
const SLOT_2D: &str = "msf_slot2d_linear_edge";

/// Its 3D kernel, whose constexpr sampler repeats on all three axes.
const SLOT_3D: &str = "msf_slot3d_linear_repeat";

/// The fixture whose `rrp_mc_compare` takes five depth comparisons through the runtime sampler.
const COMPARES: Fixture = Fixture {
    name: "rrp_mirror_clamp",
    entry: "rrp_mc_compare",
};

/// Every texture's side: a power of two, so the 2D chain's levels are whole texels of the one
/// above and the 3D texture's periods are exact.
const SIDE: u32 = 4;

/// The 3D texture's depth, which the `r` axis addresses.
const DEPTH: u32 = 4;

/// The 2D chain's levels: 4 by 4, 2 by 2 and 1 by 1.
const LEVELS: u32 = 3;

/// The `float4` results one lane of a slot kernel writes: the constexpr sampler's and the runtime
/// sampler's.
const SLOT_RESULTS: u32 = 2;

/// The bytes one lane of a slot kernel writes.
const SLOT_LANE: u32 = SLOT_RESULTS * 16;

/// The `float4` results one lane of `rrp_mc_compare` writes: the comparison with no offset, then
/// with the offsets -3, 2, -7 and 5 on `s`.
const COMPARE_RESULTS: u32 = 5;

/// The bytes one lane of `rrp_mc_compare` writes.
const COMPARE_LANE: u32 = COMPARE_RESULTS * 16;

/// The byte every output buffer holds before a row runs, so a word no thread wrote is a recorded
/// value rather than whatever the allocator left.
const FILL: u8 = 0xab;

/// The threads a group of an axis family's dispatch: 125 lanes are 25 groups of five.
///
/// Nothing a sampler does depends on the group size, and each family's is a **divisor of its lane
/// count** for one reason: a dispatch whose last group is short is one the runtime splits in two,
/// which announces `metal2vulkan-degrade-split-dispatch` — a notice about the dispatch and not
/// about the sampler. Dividing exactly leaves every case here declaring no notice at all, so a
/// notice from any row reads as divergent at once.
const AXIS_GROUP: u32 = 5;

/// The threads a group of the mip-filter family's dispatch: 16 lanes are four groups of four.
const MIP_GROUP: u32 = 4;

/// The threads a group of the comparison family's dispatch: 18 lanes are three groups of six.
const COMPARE_GROUP: u32 = 6;

// ---------------------------------------------------------------------------------------------
// The 3D texture the axis families sample

/// The 3D texture's texel `(x, y, z)`: `1 + x + SIDE * y + SIDE * SIDE * z`, every value a small
/// integer and exact in `float`, so a nearest sample returns the texel's own number bit for bit.
fn texel_3d(x: u32, y: u32, z: u32) -> f32 {
    (1 + x + SIDE * y + SIDE * SIDE * z) as f32
}

/// The 3D texture's texels, `x` fastest then `y` then `z`.
fn bytes_3d() -> Vec<u8> {
    slices()
        .flat_map(|(x, y, z)| texel_3d(x, y, z).to_le_bytes())
        .collect()
}

/// The four-channel 3D texture the border family samples: red from `x`, green from `y`, blue from
/// `z` and a fixed alpha, so a sample names the texel it read in three channels and no channel
/// takes a border colour's own value.
fn bytes_3d_rgba() -> Vec<u8> {
    slices()
        .flat_map(|(x, y, z)| {
            [
                (16 + 16 * x) as u8,
                (16 + 16 * y) as u8,
                (16 + 16 * z) as u8,
                96,
            ]
        })
        .collect()
}

/// Every texel of the 3D texture, `x` fastest then `y` then `z`.
fn slices() -> impl Iterator<Item = (u32, u32, u32)> {
    (0..DEPTH).flat_map(|z| (0..SIDE).flat_map(move |y| (0..SIDE).map(move |x| (x, y, z))))
}

/// The 3D texture of `format`, loaded with `uploads`.
fn texture_3d<'a>(format: PixelFormat, uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
    TextureRequest {
        format: format.as_raw(),
        texture_type: TextureType::Type3D,
        size: [SIDE, SIDE, DEPTH],
        levels: 1,
        slices: 1,
        usage: TextureUsage::SHADER_READ,
        private: false,
        gpu_optimized: true,
        swizzle: metal2vulkan::metal::TextureSwizzleChannels::DEFAULT,
        over: None,
        uploads,
    }
}

/// The 3D texture's one upload, whose pitches are `channels` bytes a texel.
fn upload_3d(bytes: &[u8], channels: u32) -> Upload<'_> {
    Upload {
        level: 0,
        slice: 0,
        origin: [0, 0, 0],
        size: [SIDE, SIDE, DEPTH],
        bytes,
        row_pitch: SIDE * channels,
        rows: SIDE * DEPTH,
        image_pitch: SIDE * SIDE * channels,
    }
}

// ---------------------------------------------------------------------------------------------
// The axis families' coordinates

/// The value each axis of the axis families' grid takes, one inside the texture and four outside.
///
/// `0.375` is `1.5 / 4`, the centre of texel 1, so an axis at it reads texel 1 under every address
/// mode there is and is the lane's "inside" value. `1.25` and `-0.3` are one period out either
/// way, which is where `Repeat` and `MirrorRepeat` part (0.25 against 0.75, and 0.7 against 0.3).
/// `2.6` and `-1.7` are further out, and they are **not** symmetric on purpose: a device that
/// folded an axis the wrong way round would read the same texel at `-x` and at `x`, and a
/// symmetric pair could not tell.
const AXIS_COORDS: [f32; 5] = [0.375, 1.25, -0.3, 2.6, -1.7];

/// The lanes of an axis family: every `(s, t, r)` of [`AXIS_COORDS`], `s` fastest.
///
/// The whole product and not a diagonal, because the crossing is exactly the lanes where the three
/// axes disagree about whether they are inside.
fn axis_coords() -> Vec<[f32; 4]> {
    let mut out = Vec::new();
    for &r in &AXIS_COORDS {
        for &t in &AXIS_COORDS {
            for &s in &AXIS_COORDS {
                out.push([s, t, r, 0.0]);
            }
        }
    }
    out
}

/// The coordinate buffer of an axis family: a `float4` a lane, which is what the 3D slot kernel
/// reads.
fn axis_bytes() -> Vec<u8> {
    axis_coords()
        .iter()
        .flatten()
        .copied()
        .flat_map(f32::to_le_bytes)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The 2D chain the mip-filter family samples

/// Texel `(x, y)` of level `level`: `1000 * level + 1 + x + SIDE * y`.
///
/// Distinct across the whole chain, so one `float` names both the level a sample read and the
/// texel inside it, and every value is exact in `float`.
fn level_texel(level: u32, x: u32, y: u32) -> f32 {
    (1000 * level + 1 + x + SIDE * y) as f32
}

/// The texels of level `level`, row by row.
fn level_bytes(level: u32) -> Vec<u8> {
    let side = SIDE >> level;
    (0..side)
        .flat_map(|y| (0..side).map(move |x| (x, y)))
        .flat_map(|(x, y)| level_texel(level, x, y).to_le_bytes())
        .collect()
}

/// Every level's texels, level 0 first.
fn chain_bytes() -> Vec<Vec<u8>> {
    (0..LEVELS).map(level_bytes).collect()
}

/// The uploads that load `chain`, one a level.
fn chain_uploads<'a>(chain: &'a [Vec<u8>]) -> Vec<Upload<'a>> {
    chain
        .iter()
        .enumerate()
        .map(|(level, bytes)| {
            let side = SIDE >> level as u32;
            Upload {
                level: level as u32,
                slice: 0,
                origin: [0, 0, 0],
                size: [side, side, 1],
                bytes,
                row_pitch: side * 4,
                rows: side,
                image_pitch: 0,
            }
        })
        .collect()
}

/// The mipmapped 2D texture, loaded with `uploads`.
fn mipped_2d<'a>(uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
    TextureRequest::read_2d(PixelFormat::R32Float.as_raw(), SIDE, SIDE)
        .levels(LEVELS)
        .loaded(uploads)
}

/// The coordinates the mip-filter family samples at: a quarter of a texel past each texel's lower
/// corner.
///
/// **Not a texel centre.** At a centre a nearest and a linear filter inside one level read the
/// same texel, so the `(min, mag)` pair this family crosses would be invisible; a quarter off
/// centre a linear filter weights two texels 3/4 and 1/4 on each axis and a nearest one reads the
/// lower, which is what parts the two.
fn quarters() -> Vec<[f32; 2]> {
    (0..SIDE)
        .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
        .map(|(x, y)| {
            [
                (x as f32 + 0.25) / SIDE as f32,
                (y as f32 + 0.25) / SIDE as f32,
            ]
        })
        .collect()
}

/// The coordinate buffer of the mip-filter family: a `float2` a lane.
fn quarter_bytes() -> Vec<u8> {
    quarters()
        .iter()
        .flatten()
        .copied()
        .flat_map(f32::to_le_bytes)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The depth texture the comparison family samples

/// The depth texture's texel `(x, y)`: `(1 + x + SIDE * y) / 32`, a dyadic fraction exact in
/// `float` and inside depth's own `[0, 1]` with room for a reference either side of it.
fn depth_texel(x: u32, y: u32) -> f32 {
    (1 + x + SIDE * y) as f32 / 32.0
}

/// The depth texture's texels, row by row.
fn depth_bytes() -> Vec<u8> {
    (0..SIDE)
        .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
        .flat_map(|(x, y)| depth_texel(x, y).to_le_bytes())
        .collect()
}

/// The two coordinates the comparison family samples at, both with the **same four** texels in a
/// linear footprint — `(1, 1)`, `(2, 1)`, `(1, 2)` and `(2, 2)`, whose depths are 6, 7, 10 and 11
/// thirty-seconds — and different weights on them.
///
/// `0.5` is those texels' shared corner, where each weight is a quarter, so a linear comparison
/// reads 0, 1/4, 1/2, 3/4 or 1 and the four texels are interchangeable. `0.4375` is `1.75 / 4`,
/// three quarters of the way from texel 1's centre to texel 2's, where the weights are 9/16, 3/16,
/// 3/16 and 1/16 — four distinct weights, so the answer names **which** texels passed and not only
/// how many.
const COMPARE_AT: [f32; 2] = [0.5, 0.4375];

/// The references the comparison family walks, as thirty-seconds of depth: below all four texels'
/// depths, exactly on each of them, between each adjacent pair, and above all four.
///
/// On a texel's own depth is the only place `Less` parts from `LessEqual` and `Greater` from
/// `GreaterEqual`; between two of them is where a linear footprint's four comparisons disagree,
/// which is the whole of this family's question.
fn compare_references() -> Vec<f32> {
    [5.0, 6.0, 6.5, 7.0, 8.5, 10.0, 10.5, 11.0, 12.0]
        .iter()
        .map(|thirty_seconds| thirty_seconds / 32.0)
        .collect()
}

/// The lanes of the comparison family: each of [`COMPARE_AT`] against each reference.
fn compare_lanes() -> Vec<([f32; 2], f32)> {
    let mut out = Vec::new();
    for &at in &COMPARE_AT {
        for reference in compare_references() {
            out.push(([at, at], reference));
        }
    }
    out
}

/// The coordinate buffer of the comparison family: `(s, t, reference, 0)` a lane, which is what
/// `rrp_mc_compare` reads.
fn compare_bytes() -> Vec<u8> {
    compare_lanes()
        .iter()
        .flat_map(|&([s, t], reference)| [s, t, reference, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The rows

/// The address-mode triples the `axes` family crosses, `(s, t, r)`.
///
/// The first three have all three modes equal: they are the **controls**, and an implementation
/// that applied one axis's mode to all three answers them exactly, which is what makes the other
/// six rows a measurement rather than a sweep. The next three are the rotations of
/// `(Repeat, MirrorRepeat, ClampToEdge)`, so a collapse onto `s` and one onto `t` give different
/// wrong answers; the last three rotate `MirrorClampToEdge` — the one mode MSL cannot spell in a
/// constexpr sampler — against `Repeat` and `ClampToEdge`.
const TRIPLES: [(&str, [SamplerAddressMode; 3]); 9] = [
    (
        "all-repeat",
        [
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
        ],
    ),
    (
        "all-edge",
        [
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
        ],
    ),
    (
        "all-mirror-repeat",
        [
            SamplerAddressMode::MirrorRepeat,
            SamplerAddressMode::MirrorRepeat,
            SamplerAddressMode::MirrorRepeat,
        ],
    ),
    (
        "repeat-mirror-edge",
        [
            SamplerAddressMode::Repeat,
            SamplerAddressMode::MirrorRepeat,
            SamplerAddressMode::ClampToEdge,
        ],
    ),
    (
        "mirror-edge-repeat",
        [
            SamplerAddressMode::MirrorRepeat,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::Repeat,
        ],
    ),
    (
        "edge-repeat-mirror",
        [
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::MirrorRepeat,
        ],
    ),
    (
        "mirror-clamp-repeat-edge",
        [
            SamplerAddressMode::MirrorClampToEdge,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::ClampToEdge,
        ],
    ),
    (
        "repeat-edge-mirror-clamp",
        [
            SamplerAddressMode::Repeat,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::MirrorClampToEdge,
        ],
    ),
    (
        "edge-mirror-clamp-repeat",
        [
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::MirrorClampToEdge,
            SamplerAddressMode::Repeat,
        ],
    ),
];

/// The triples the border family crosses: a border-reading mode on one axis at a time, on all
/// three, and crossed with the other mode that reads a border on this oracle.
///
/// `border-all` is the control a collapse onto one axis cannot move. The two `zero` rows put
/// `ClampToZero` and `ClampToBorderColor` on **different** axes of one sampler, which is a pair no
/// per-value sweep can state: `sampler_filtering` measured that both read the border colour, and
/// these rows ask whether they still do when the other is the axis out of range.
const BORDER_TRIPLES: [(&str, [SamplerAddressMode; 3]); 6] = [
    (
        "border-s",
        [
            SamplerAddressMode::ClampToBorderColor,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::ClampToEdge,
        ],
    ),
    (
        "border-t",
        [
            SamplerAddressMode::Repeat,
            SamplerAddressMode::ClampToBorderColor,
            SamplerAddressMode::ClampToEdge,
        ],
    ),
    (
        "border-r",
        [
            SamplerAddressMode::Repeat,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToBorderColor,
        ],
    ),
    (
        "border-all",
        [
            SamplerAddressMode::ClampToBorderColor,
            SamplerAddressMode::ClampToBorderColor,
            SamplerAddressMode::ClampToBorderColor,
        ],
    ),
    (
        "zero-s-border-r",
        [
            SamplerAddressMode::ClampToZero,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::ClampToBorderColor,
        ],
    ),
    (
        "border-s-zero-t",
        [
            SamplerAddressMode::ClampToBorderColor,
            SamplerAddressMode::ClampToZero,
            SamplerAddressMode::Repeat,
        ],
    ),
];

/// Every `MTLSamplerBorderColor`, one case each.
const BORDERS: [(&str, SamplerBorderColor); 3] = [
    ("transparent-black", SamplerBorderColor::TransparentBlack),
    ("opaque-black", SamplerBorderColor::OpaqueBlack),
    ("opaque-white", SamplerBorderColor::OpaqueWhite),
];

/// The `(min, mag)` filter pairs the mip-filter family crosses.
///
/// The first two agree and are the controls: a device handed the wrong one of a pair answers them
/// exactly. The last two differ, and which of the two a sample reads is the crossing — a question
/// with no answer at all when they agree, which is why `sampler_filtering`'s mip sweep, whose
/// filters are both nearest, could not ask it.
const FILTER_PAIRS: [(&str, SamplerMinMagFilter, SamplerMinMagFilter); 4] = [
    (
        "nearest-nearest",
        SamplerMinMagFilter::Nearest,
        SamplerMinMagFilter::Nearest,
    ),
    (
        "linear-linear",
        SamplerMinMagFilter::Linear,
        SamplerMinMagFilter::Linear,
    ),
    (
        "linear-min-nearest-mag",
        SamplerMinMagFilter::Linear,
        SamplerMinMagFilter::Nearest,
    ),
    (
        "nearest-min-linear-mag",
        SamplerMinMagFilter::Nearest,
        SamplerMinMagFilter::Linear,
    ),
];

/// The `lodMinClamp` values each filter pair is crossed with.
///
/// The kernel samples at `level(0.0)` and Metal raises that to the clamp, so the clamp **is** the
/// level: 0 is a level of zero, where Vulkan reads the magnifying filter; a quarter is the
/// smallest fractional level in the set, where a linear mip filter blends two levels and a nearest
/// one still reads level 0; 1 is a whole level; 1.5 is fractional between the last two; 2 is the
/// last level there is.
const PAIR_CLAMPS: [f32; 5] = [0.0, 0.25, 1.0, 1.5, 2.0];

/// Every `MTLSamplerMipFilter`, one case each.
const MIP_FILTERS: [(&str, SamplerMipFilter); 3] = [
    ("not-mipmapped", SamplerMipFilter::NotMipmapped),
    ("nearest", SamplerMipFilter::Nearest),
    ("linear", SamplerMipFilter::Linear),
];

/// Every `MTLCompareFunction` a sampler owns, crossed with [`COMPARE_FILTERS`].
const COMPARE_FUNCTIONS: [(&str, CompareFunction); 8] = [
    ("never", CompareFunction::Never),
    ("less", CompareFunction::Less),
    ("equal", CompareFunction::Equal),
    ("less-equal", CompareFunction::LessEqual),
    ("greater", CompareFunction::Greater),
    ("not-equal", CompareFunction::NotEqual),
    ("greater-equal", CompareFunction::GreaterEqual),
    ("always", CompareFunction::Always),
];

/// The two filters each comparison is crossed with: a footprint of one texel and a footprint of
/// four.
const COMPARE_FILTERS: [(&str, SamplerMinMagFilter); 2] = [
    ("nearest", SamplerMinMagFilter::Nearest),
    ("linear", SamplerMinMagFilter::Linear),
];

// ---------------------------------------------------------------------------------------------
// The layouts

/// The layout a slot family's output buffer is read under: one row a lane, whose elements are both
/// `float4`s — the constexpr control and the runtime sampler's answer — so a diff names the lane
/// and the column.
fn slot_layout(lanes: u32) -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(lanes * SLOT_LANE),
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: SLOT_RESULTS * 4,
        row_pitch: SLOT_LANE,
        rows: lanes,
        image_pitch: lanes * SLOT_LANE,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

/// The layout the comparison family's output buffer is read under: one row a lane, whose elements
/// are the five comparisons' first channels and the zeros the fixture writes beside them.
fn compare_layout(lanes: u32) -> ByteLayout {
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(lanes * COMPARE_LANE),
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: COMPARE_RESULTS * 4,
        row_pitch: COMPARE_LANE,
        rows: lanes,
        image_pitch: lanes * COMPARE_LANE,
        images: 1,
        gaps: Gaps::Ignored,
    }
}

// ---------------------------------------------------------------------------------------------
// The cases

/// Which crossing a case puts to Metal.
#[derive(Clone, Copy, Debug)]
enum Family {
    /// [`TRIPLES`]'s nine rows of three address modes at once, over the 3D texture.
    Axes,
    /// One `MTLSamplerBorderColor`, [`BORDER_TRIPLES`]'s six rows of a border-reading mode on one
    /// axis of three.
    AxisBorder(SamplerBorderColor),
    /// One `MTLSamplerMipFilter`, [`FILTER_PAIRS`] crossed with [`PAIR_CLAMPS`].
    MipFilters(SamplerMipFilter),
    /// [`COMPARE_FUNCTIONS`] crossed with [`COMPARE_FILTERS`], over the depth texture.
    CompareFilters,
}

/// One case of this module.
#[derive(Clone, Copy, Debug)]
pub struct SamplerCrossings {
    name: &'static str,
    family: Family,
}

/// Every case of this module.
pub fn all() -> Vec<SamplerCrossings> {
    let mut cases = vec![SamplerCrossings {
        name: "sampler-axis-mode-triples",
        family: Family::Axes,
    }];
    for (colour, border) in BORDERS {
        cases.push(SamplerCrossings {
            name: match colour {
                "transparent-black" => "sampler-axis-border-transparent-black",
                "opaque-black" => "sampler-axis-border-opaque-black",
                _ => "sampler-axis-border-opaque-white",
            },
            family: Family::AxisBorder(border),
        });
    }
    for (mip, filter) in MIP_FILTERS {
        cases.push(SamplerCrossings {
            name: match mip {
                "not-mipmapped" => "sampler-mip-filter-pairs-not-mipmapped",
                "nearest" => "sampler-mip-filter-pairs-nearest",
                _ => "sampler-mip-filter-pairs-linear",
            },
            family: Family::MipFilters(filter),
        });
    }
    cases.push(SamplerCrossings {
        name: "sampler-compare-filter-pairs",
        family: Family::CompareFilters,
    });
    cases
}

/// Every case of this module, boxed for the suite's list.
pub fn boxed() -> Vec<Box<dyn Case>> {
    all().into_iter().map(|case| Box::new(case) as _).collect()
}

/// The site-name spelling of a LOD clamp.
fn clamp_name(clamp: f32) -> String {
    format!("{clamp}").replace('.', "-")
}

impl SamplerCrossings {
    /// The fixture and entry point the case runs.
    fn fixture(&self) -> Fixture {
        match self.family {
            Family::Axes | Family::AxisBorder(_) => Fixture {
                name: SLOTS,
                entry: SLOT_3D,
            },
            Family::MipFilters(_) => Fixture {
                name: SLOTS,
                entry: SLOT_2D,
            },
            Family::CompareFilters => COMPARES,
        }
    }

    /// The sampler state each row of the case makes, under the name its sites carry.
    fn rows(&self) -> Vec<(String, SamplerSpec)> {
        match self.family {
            Family::Axes => TRIPLES
                .iter()
                .map(|&(name, modes)| (name.to_owned(), axis_spec(modes, None)))
                .collect(),
            Family::AxisBorder(border) => BORDER_TRIPLES
                .iter()
                .map(|&(name, modes)| (name.to_owned(), axis_spec(modes, Some(border))))
                .collect(),
            Family::MipFilters(mip) => FILTER_PAIRS
                .iter()
                .flat_map(|&(pair, min, mag)| {
                    PAIR_CLAMPS.iter().map(move |&clamp| {
                        let mut spec = SamplerSpec::new().mipped(mip).clamped_from(clamp);
                        spec.min_filter = min;
                        spec.mag_filter = mag;
                        spec.s_address_mode = SamplerAddressMode::ClampToEdge;
                        spec.t_address_mode = SamplerAddressMode::ClampToEdge;
                        (format!("{pair}-clamp-{}", clamp_name(clamp)), spec)
                    })
                })
                .collect(),
            Family::CompareFilters => COMPARE_FILTERS
                .iter()
                .flat_map(|&(filter_name, filter)| {
                    COMPARE_FUNCTIONS.iter().map(move |&(name, function)| {
                        let mut spec = SamplerSpec::filtered(filter).comparing(function);
                        spec.s_address_mode = SamplerAddressMode::ClampToEdge;
                        spec.t_address_mode = SamplerAddressMode::ClampToEdge;
                        (format!("{filter_name}-{name}"), spec)
                    })
                })
                .collect(),
        }
    }

    /// The lanes the case dispatches, and the bytes its coordinate buffer holds.
    fn lanes(&self) -> (u32, Vec<u8>) {
        match self.family {
            Family::Axes | Family::AxisBorder(_) => (axis_coords().len() as u32, axis_bytes()),
            Family::MipFilters(_) => (quarters().len() as u32, quarter_bytes()),
            Family::CompareFilters => (compare_lanes().len() as u32, compare_bytes()),
        }
    }

    /// The threads a group the case dispatches, a divisor of its lane count.
    fn group(&self) -> u32 {
        match self.family {
            Family::Axes | Family::AxisBorder(_) => AXIS_GROUP,
            Family::MipFilters(_) => MIP_GROUP,
            Family::CompareFilters => COMPARE_GROUP,
        }
    }

    /// The bytes one lane of the case's kernel writes.
    fn lane_bytes(&self) -> u32 {
        match self.family {
            Family::CompareFilters => COMPARE_LANE,
            _ => SLOT_LANE,
        }
    }

    /// The bytes one row's output buffer takes.
    fn output_bytes(&self) -> usize {
        let (lanes, _) = self.lanes();
        lanes as usize * self.lane_bytes() as usize
    }

    /// The layout the row's output buffer is read under.
    fn layout(&self) -> ByteLayout {
        let (lanes, _) = self.lanes();
        match self.family {
            Family::CompareFilters => compare_layout(lanes),
            _ => slot_layout(lanes),
        }
    }

    /// The texels the case's texture holds, as the input fingerprint names them.
    fn texture_inputs(&self) -> Vec<Input> {
        match self.family {
            Family::Axes => vec![Input::bytes("texture:3d", &bytes_3d())],
            Family::AxisBorder(_) => vec![Input::bytes("texture:3d-rgba", &bytes_3d_rgba())],
            Family::MipFilters(_) => chain_bytes()
                .iter()
                .enumerate()
                .map(|(level, bytes)| Input::bytes(format!("texture:level-{level}"), bytes))
                .collect(),
            Family::CompareFilters => vec![Input::bytes("texture:depth", &depth_bytes())],
        }
    }

    /// The texture the case makes, on `side`.
    fn texture(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<Handle, Abort> {
        match self.family {
            Family::Axes => {
                let bytes = bytes_3d();
                let uploads = [upload_3d(&bytes, 4)];
                made(
                    sites,
                    "texture",
                    side.texture(&texture_3d(PixelFormat::R32Float, &uploads)),
                )
            }
            Family::AxisBorder(_) => {
                let bytes = bytes_3d_rgba();
                let uploads = [upload_3d(&bytes, 4)];
                made(
                    sites,
                    "texture",
                    side.texture(&texture_3d(PixelFormat::RGBA8Unorm, &uploads)),
                )
            }
            Family::MipFilters(_) => {
                let chain = chain_bytes();
                let uploads = chain_uploads(&chain);
                made(sites, "texture", side.texture(&mipped_2d(&uploads)))
            }
            Family::CompareFilters => {
                let spec = TextureSpec::new(
                    PixelFormat::Depth32Float,
                    [SIDE, SIDE],
                    SIDE * 4,
                    TextureRole::Read,
                );
                made(
                    sites,
                    "texture",
                    side.texture_2d_in_role(&spec, &depth_bytes()),
                )
            }
        }
    }
}

/// The nearest-filtering state with `modes` on `(s, t, r)` and, where the family states one,
/// `border` as its border colour.
///
/// **Nearest on both filters** so the runtime column names the texel it read rather than a blend
/// of two: this family's question is which texel each axis's mode addresses, and a linear filter
/// would mix two of them on every axis and hide it.
fn axis_spec(modes: [SamplerAddressMode; 3], border: Option<SamplerBorderColor>) -> SamplerSpec {
    let mut spec = SamplerSpec::new();
    spec.s_address_mode = modes[0];
    spec.t_address_mode = modes[1];
    spec.r_address_mode = modes[2];
    if let Some(border) = border {
        spec.border_color = border;
    }
    spec
}

impl Case for SamplerCrossings {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{}{extension}", fixture.name),
                entry: fixture.entry.to_owned(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn allowances(&self) -> Vec<Allowance> {
        // Every state here is one a device's own sampler holds: nearest or linear filters, the
        // five address modes Vulkan has an enumerator for, a border colour, a level clamp and a
        // comparison. Nothing is emulated and nothing is announced, so a notice from any row is a
        // difference at the notices site and reads as divergent at once.
        Vec::new()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let (_, coords) = self.lanes();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("entry", fixture.entry.as_bytes()),
            Input::bytes("buffer:coords", &coords),
            Input::bytes("buffer:out", &vec![FILL; self.output_bytes()]),
        ];
        inputs.extend(self.texture_inputs());
        for (name, spec) in self.rows() {
            inputs.push(Input::bytes(format!("sampler:{name}"), &spec.words()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let fixture = self.fixture();
        let loaded = fixture.loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, fixture.entry),
        )?;
        let texture = self.texture(side, sites)?;
        let (lanes, coords) = self.lanes();
        let coordinates = made(sites, "buffer/coords", side.buffer(&coords))?;

        for (name, spec) in self.rows() {
            let at = |site: &str| format!("{name}/{site}");
            let sampler = made(sites, &at("sampler"), side.sampler(&spec))?;
            let output = made(
                sites,
                &at("buffer/out"),
                side.buffer(&vec![FILL; self.output_bytes()]),
            )?;
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch_threads(
                    pipeline,
                    &Bindings::new(&[texture], &[coordinates, output]).sampled(&[sampler]),
                    &[],
                    [lanes, 1, 1],
                    [self.group(), 1, 1],
                ),
            )?;
            call(
                sites,
                &at("status"),
                side.wait(command_buffer),
                Clone::clone,
            )?;
            let bytes = call(
                sites,
                &at("read-back"),
                side.read_buffer(output, 0, self.output_bytes()),
                |_| Value::Absent,
            )?;
            sites.push(
                Node::branch(name.clone(), Vec::new())
                    .with(Node::leaf("sampled", Observed::bytes(bytes, self.layout()))),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The bytes the committed recording of `case` holds at row `row`, over every outcome of every
    /// answer, or `None` where the tree holds no recording of the case yet.
    ///
    /// A recording that exists but holds no bytes at the row is a failure rather than a skip: that
    /// is a renamed site, and a test that shrugged at it would stop measuring and still pass.
    fn recorded(case: &str, row: &str) -> Option<Vec<Vec<u8>>> {
        let recording = crate::recording::read(case).ok()?;
        let mut out = Vec::new();
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                let held = outcome
                    .transcript
                    .root
                    .child(row)
                    .and_then(|node| node.child("sampled"))
                    .map(|node| &node.observed.value);
                match held {
                    Some(Value::Bytes(bytes)) => out.push(bytes.clone()),
                    _ => panic!("{case}: the recording holds no bytes at {row}/sampled"),
                }
            }
        }
        assert!(!out.is_empty(), "{case}: a recording with no outcome");
        Some(out)
    }

    /// The four `float`s at `at`.
    fn word4(bytes: &[u8], at: usize) -> [f32; 4] {
        let mut out = [0f32; 4];
        for (lane, held) in out.iter_mut().enumerate() {
            let from = at + lane * 4;
            *held = f32::from_le_bytes([
                bytes[from],
                bytes[from + 1],
                bytes[from + 2],
                bytes[from + 3],
            ]);
        }
        out
    }

    /// The runtime sampler's `float4` at lane `lane` of a slot family's bytes: the second of the
    /// two results the kernel writes.
    fn runtime_sample(bytes: &[u8], lane: usize) -> [f32; 4] {
        word4(bytes, lane * SLOT_LANE as usize + 16)
    }

    /// The lane index of the axis grid at `(s, t, r)`, as [`axis_coords`] orders it.
    const fn axis_lane(s: usize, t: usize, r: usize) -> usize {
        r * 25 + t * 5 + s
    }

    /// The comparison family's first result (no texel offset) at lane `lane`.
    fn compare_sample(bytes: &[u8], lane: usize) -> f32 {
        word4(bytes, lane * COMPARE_LANE as usize)[0]
    }

    // -----------------------------------------------------------------------------------------
    // The axis families

    /// Metal addresses the three axes **independently**: every one of the 125 lanes of every
    /// triple is the texel whose `x`, `y` and `z` each come from that axis's own mode, read off the
    /// row's own lanes where only one axis is outside.
    ///
    /// This is the family's whole claim, and it is the claim a collapsed implementation breaks. It
    /// needs no control row: a row's per-axis selection is measured from its own `(i, 0, 0)`,
    /// `(0, j, 0)` and `(0, 0, k)` lanes — `AXIS_COORDS[0]` is inside the texture on every axis —
    /// and the remaining 98 lanes are predicted from those fifteen.
    #[test]
    fn the_recorded_axes_address_independently() {
        let Some(outcomes) = recorded("sampler-axis-mode-triples", TRIPLES[0].0) else {
            return;
        };
        assert!(!outcomes.is_empty());
        for (row, _) in TRIPLES {
            for bytes in recorded("sampler-axis-mode-triples", row).expect("a recorded row") {
                // Each axis's texel index, from the lanes where it alone is outside.
                let xs: Vec<f32> = (0..5)
                    .map(|i| runtime_sample(&bytes, axis_lane(i, 0, 0))[0] - 21.0)
                    .collect();
                let ys: Vec<f32> = (0..5)
                    .map(|j| (runtime_sample(&bytes, axis_lane(0, j, 0))[0] - 18.0) / 4.0)
                    .collect();
                let zs: Vec<f32> = (0..5)
                    .map(|k| (runtime_sample(&bytes, axis_lane(0, 0, k))[0] - 6.0) / 16.0)
                    .collect();
                for (axis, held) in [("x", &xs), ("y", &ys), ("z", &zs)] {
                    for &index in held.iter() {
                        assert!(
                            index >= 0.0 && index < SIDE as f32 && index.fract() == 0.0,
                            "{row}: the {axis} axis read the index {index}, which is no texel of \
                             a side of {SIDE}"
                        );
                    }
                }
                for (k, &r) in AXIS_COORDS.iter().enumerate() {
                    for (j, &t) in AXIS_COORDS.iter().enumerate() {
                        for (i, &s) in AXIS_COORDS.iter().enumerate() {
                            let want = 1.0 + xs[i] + SIDE as f32 * ys[j] + 16.0 * zs[k];
                            let got = runtime_sample(&bytes, axis_lane(i, j, k))[0];
                            assert_eq!(
                                got, want,
                                "{row}: lane (s={s}, t={t}, r={r}) read {got} where the three \
                                 axes' own modes name {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The nine triples are nine different answers, so no row of the family is another's mirror.
    #[test]
    fn the_nine_mode_triples_are_nine_answers() {
        let mut seen = BTreeSet::new();
        for (row, _) in TRIPLES {
            let Some(outcomes) = recorded("sampler-axis-mode-triples", row) else {
                return;
            };
            assert!(
                seen.insert(outcomes[0].clone()),
                "{row} answered as another row"
            );
        }
        assert_eq!(seen.len(), TRIPLES.len());
    }

    /// A border-reading mode reads the border **exactly** where its own axis is the one outside,
    /// whatever the other two axes are doing.
    ///
    /// That is the crossing no per-value sweep states: `sampler_filtering`'s border rows read a
    /// border on `r` with `s` and `t` repeating and inside, so they never ask whether a
    /// border-reading `s` fires when `t` is the axis out of range.
    #[test]
    fn the_recorded_border_reads_only_where_its_own_axis_is_outside() {
        for (colour, border) in BORDERS {
            let name = case_name_of(border);
            let want = border_channels(border);
            for (row, modes) in BORDER_TRIPLES {
                let Some(outcomes) = recorded(name, row) else {
                    return;
                };
                for bytes in outcomes {
                    for (k, &r) in AXIS_COORDS.iter().enumerate() {
                        for (j, &t) in AXIS_COORDS.iter().enumerate() {
                            for (i, &s) in AXIS_COORDS.iter().enumerate() {
                                let outside = [i, j, k];
                                let bordered = modes.iter().zip(outside).any(|(mode, at)| {
                                    at != 0
                                        && matches!(
                                            *mode,
                                            SamplerAddressMode::ClampToBorderColor
                                                | SamplerAddressMode::ClampToZero
                                        )
                                });
                                let got = runtime_sample(&bytes, axis_lane(i, j, k));
                                if bordered {
                                    assert_eq!(
                                        got, want,
                                        "{name}/{row}: lane (s={s}, t={t}, r={r}) read {got:?} \
                                         where a border-reading axis is outside and the {colour} \
                                         border is {want:?}"
                                    );
                                } else {
                                    assert_ne!(
                                        got, want,
                                        "{name}/{row}: lane (s={s}, t={t}, r={r}) read the \
                                         {colour} border where no border-reading axis is outside"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// `ClampToZero` reads the border colour, exactly as `ClampToBorderColor` does, **per axis**.
    ///
    /// `sampler_filtering` measured that of a sampler whose border-reading mode was on one axis
    /// with the other two repeating; these rows put the two modes on **different axes of one
    /// sampler** and the answer is the same colour on both, so the two values of
    /// `MTLSamplerAddressMode` stay **indistinguishable on this oracle** — named here rather than
    /// swept as though they were two answers.
    #[test]
    fn the_recorded_clamp_to_zero_reads_the_border_colour() {
        for (colour, border) in BORDERS {
            let name = case_name_of(border);
            let want = border_channels(border);
            let Some(outcomes) = recorded(name, "zero-s-border-r") else {
                return;
            };
            for bytes in outcomes {
                // s outside, t and r inside: the ClampToZero axis alone decides.
                let zero = runtime_sample(&bytes, axis_lane(1, 0, 0));
                // r outside, s and t inside: the ClampToBorderColor axis alone decides.
                let bordered = runtime_sample(&bytes, axis_lane(0, 0, 1));
                assert_eq!(
                    zero, want,
                    "{name}: a ClampToZero s axis outside read {zero:?}, not the {colour} border"
                );
                assert_eq!(
                    bordered, want,
                    "{name}: a ClampToBorderColor r axis outside read {bordered:?}, not the \
                     {colour} border"
                );
            }
        }
    }

    /// The three border colours are three answers under every one of the six triples.
    #[test]
    fn the_three_border_colours_are_three_answers() {
        for (row, _) in BORDER_TRIPLES {
            let mut seen = BTreeSet::new();
            for (_, border) in BORDERS {
                let Some(outcomes) = recorded(case_name_of(border), row) else {
                    return;
                };
                assert!(
                    seen.insert(outcomes[0].clone()),
                    "{row}: two border colours answered alike"
                );
            }
            assert_eq!(seen.len(), BORDERS.len());
        }
    }

    /// The case name that carries `border`.
    fn case_name_of(border: SamplerBorderColor) -> &'static str {
        match border {
            SamplerBorderColor::TransparentBlack => "sampler-axis-border-transparent-black",
            SamplerBorderColor::OpaqueBlack => "sampler-axis-border-opaque-black",
            _ => "sampler-axis-border-opaque-white",
        }
    }

    /// The four channels an `RGBA8Unorm` sample of `border`'s border texel reads.
    fn border_channels(border: SamplerBorderColor) -> [f32; 4] {
        match border {
            SamplerBorderColor::TransparentBlack => [0.0, 0.0, 0.0, 0.0],
            SamplerBorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
            _ => [1.0, 1.0, 1.0, 1.0],
        }
    }

    // -----------------------------------------------------------------------------------------
    // The mip-filter family

    /// The case name that carries `mip`.
    fn mip_case(mip: SamplerMipFilter) -> &'static str {
        match mip {
            SamplerMipFilter::NotMipmapped => "sampler-mip-filter-pairs-not-mipmapped",
            SamplerMipFilter::Nearest => "sampler-mip-filter-pairs-nearest",
            _ => "sampler-mip-filter-pairs-linear",
        }
    }

    /// **The family's finding, and the one only the pair can state: a `lodMinClamp` of zero reads
    /// the magnifying filter, and any clamp above zero reads the minifying one.**
    ///
    /// Held of the recorded bytes as an identity between rows: the mixed pair at clamp 0 answers
    /// exactly as the pair whose **both** filters are its magnifying one, and at clamp 0.25 exactly
    /// as the pair whose both filters are its minifying one. Two rows whose filters agree cannot
    /// tell those apart, which is why `sampler_filtering`'s mip sweep — both filters nearest —
    /// could not ask.
    #[test]
    fn the_recorded_clamp_of_zero_reads_the_magnifying_filter() {
        for (_, mip) in MIP_FILTERS {
            let name = mip_case(mip);
            let Some(_) = recorded(name, "nearest-nearest-clamp-0") else {
                return;
            };
            let row = |pair: &str, clamp: f32| {
                recorded(name, &format!("{pair}-clamp-{}", clamp_name(clamp)))
                    .expect("a recorded row")[0]
                    .clone()
            };
            let nearest = |clamp| row("nearest-nearest", clamp);
            let linear = |clamp| row("linear-linear", clamp);
            // Linear minification, nearest magnification.
            assert_eq!(
                row("linear-min-nearest-mag", 0.0),
                nearest(0.0),
                "{name}: at a clamp of 0 a linear-min, nearest-mag sampler did not read the \
                 nearest magnifying filter"
            );
            assert_eq!(
                row("linear-min-nearest-mag", 0.25),
                linear(0.25),
                "{name}: at a clamp of 0.25 a linear-min, nearest-mag sampler did not read the \
                 linear minifying filter"
            );
            // The other way round, so neither identity is an accident of which filter is which.
            assert_eq!(
                row("nearest-min-linear-mag", 0.0),
                linear(0.0),
                "{name}: at a clamp of 0 a nearest-min, linear-mag sampler did not read the \
                 linear magnifying filter"
            );
            assert_eq!(
                row("nearest-min-linear-mag", 0.25),
                nearest(0.25),
                "{name}: at a clamp of 0.25 a nearest-min, linear-mag sampler did not read the \
                 nearest minifying filter"
            );
            // And the flip is a real change: the two mixed rows part at 0 and part again at 0.25.
            assert_ne!(
                row("linear-min-nearest-mag", 0.0),
                row("linear-min-nearest-mag", 0.25),
                "{name}: a mixed pair answered a clamp of 0 and of 0.25 alike, so the crossing \
                 this case is about did not happen"
            );
        }
    }

    /// A `NotMipmapped` sampler ignores `lodMinClamp` for the **level** it reads and still honours
    /// it for **which of the two filters** reads that level.
    ///
    /// Two halves of one fact, and the second is the surprise: `sampler_filtering` recorded that
    /// `NotMipmapped` "ignores `lodMinClamp` outright", which holds of the level and not of the
    /// filter. An implementation that took the first half for the whole would answer every mixed
    /// row with the magnifying filter.
    #[test]
    fn the_recorded_not_mipmapped_clamp_moves_the_filter_and_not_the_level() {
        let name = mip_case(SamplerMipFilter::NotMipmapped);
        let Some(_) = recorded(name, "nearest-nearest-clamp-0") else {
            return;
        };
        let row = |pair: &str, clamp: f32| {
            recorded(name, &format!("{pair}-clamp-{}", clamp_name(clamp))).expect("a recorded row")
                [0]
            .clone()
        };
        // The level: every clamp reads level 0's texels, whose numbers are below 1000.
        for (pair, _, _) in FILTER_PAIRS {
            for clamp in PAIR_CLAMPS {
                let bytes = row(pair, clamp);
                for lane in 0..quarters().len() {
                    let held = runtime_sample(&bytes, lane)[0];
                    assert!(
                        held < 1000.0,
                        "{name}/{pair} at clamp {clamp}: lane {lane} read {held}, which is not a \
                         texel of level 0"
                    );
                }
            }
        }
        // A pair whose filters agree answers every clamp alike: the level never moved.
        for (pair, min, mag) in FILTER_PAIRS {
            if min != mag {
                continue;
            }
            for clamp in PAIR_CLAMPS {
                assert_eq!(
                    row(pair, clamp),
                    row(pair, 0.0),
                    "{name}/{pair}: a clamp of {clamp} moved a sampler without a mip filter whose \
                     filters agree"
                );
            }
        }
        // A pair whose filters differ answers a clamp of 0 one way and every clamp above it the
        // other: the filter moved where the level did not.
        for (pair, min, mag) in FILTER_PAIRS {
            if min == mag {
                continue;
            }
            assert_ne!(
                row(pair, 0.25),
                row(pair, 0.0),
                "{name}/{pair}: a clamp of 0.25 read the same filter as a clamp of 0"
            );
            for clamp in PAIR_CLAMPS {
                if clamp == 0.0 {
                    continue;
                }
                assert_eq!(
                    row(pair, clamp),
                    row(pair, 0.25),
                    "{name}/{pair}: the clamps above 0 did not all read the minifying filter"
                );
            }
        }
    }

    /// Every one of [`PAIR_CLAMPS`]'s five values changes the answer **somewhere** in the family,
    /// and the cells where two of them cannot differ are named rather than swept.
    ///
    /// Under a nearest mip filter 0 and 0.25 read the same level (Metal's nearest rule reads
    /// `ceil(c + 0.5) - 1`) and so do 1 and 1.5; under none the level never moves at all. The
    /// linear mip filter is what parts all five, and the mixed filter pairs are what part 0 from
    /// 0.25 under all three. So no value here is indistinguishable **in the family**, which is the
    /// claim, and this test is what holds it.
    #[test]
    fn every_clamp_changes_an_answer_somewhere() {
        for clamp in PAIR_CLAMPS {
            let mut moved = false;
            for (_, mip) in MIP_FILTERS {
                let name = mip_case(mip);
                let Some(_) = recorded(name, "nearest-nearest-clamp-0") else {
                    return;
                };
                for (pair, _, _) in FILTER_PAIRS {
                    let at = |clamp: f32| {
                        recorded(name, &format!("{pair}-clamp-{}", clamp_name(clamp)))
                            .expect("a recorded row")[0]
                            .clone()
                    };
                    if PAIR_CLAMPS
                        .iter()
                        .any(|&other| other != clamp && at(other) != at(clamp))
                    {
                        moved = true;
                    }
                }
            }
            assert!(
                moved,
                "the clamp {clamp} answered as every other clamp in every cell of the family"
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // The comparison family

    /// **Metal filters the comparison's results and not the depth.** A linear row reads values
    /// strictly between 0 and 1 — the weights of the texels that passed — where a nearest row reads
    /// only 0 and 1.
    ///
    /// An implementation that filtered the depth and then compared would read 0 or 1 on **every**
    /// row, and its nearest rows would still be right, which is what makes this the pair's
    /// falsification and not the filter's or the comparison's.
    #[test]
    fn the_recorded_comparison_filters_the_results_and_not_the_depth() {
        let name = "sampler-compare-filter-pairs";
        let Some(_) = recorded(name, "nearest-less") else {
            return;
        };
        let lanes = compare_lanes().len();
        let mut fractions = 0;
        for (function, _) in COMPARE_FUNCTIONS {
            for bytes in recorded(name, &format!("nearest-{function}")).expect("a recorded row") {
                for lane in 0..lanes {
                    let held = compare_sample(&bytes, lane);
                    assert!(
                        held == 0.0 || held == 1.0,
                        "{name}/nearest-{function}: lane {lane} read {held}, and a footprint of \
                         one texel holds one comparison, which is 0 or 1"
                    );
                }
            }
            for bytes in recorded(name, &format!("linear-{function}")).expect("a recorded row") {
                for lane in 0..lanes {
                    let held = compare_sample(&bytes, lane);
                    assert!(
                        (0.0..=1.0).contains(&held),
                        "{name}/linear-{function}: lane {lane} read {held}, which is no weighted \
                         mean of ones and zeros"
                    );
                    if held > 0.0 && held < 1.0 {
                        fractions += 1;
                    }
                }
            }
        }
        assert!(
            fractions >= 40,
            "only {fractions} of the linear rows' lanes read a fraction, so the four texels of \
             the footprint never disagreed and the crossing did not happen"
        );
    }

    /// `Never` and `Always` are the two values of `MTLCompareFunction` **no filter can move**, and
    /// the recording says so rather than leaving a reader to assume the sweep found eight answers.
    ///
    /// Each answers 0 and 1 at every lane whatever the reference, so filtering their results
    /// filters a constant. They are kept as rows because an implementation that compared a filtered
    /// depth would still answer them correctly, which is exactly what makes the other six the
    /// measurement — and because a row that stopped reading a constant would be a defect worth
    /// catching.
    #[test]
    fn never_and_always_are_the_two_values_no_filter_moves() {
        let name = "sampler-compare-filter-pairs";
        let Some(_) = recorded(name, "nearest-never") else {
            return;
        };
        for (function, want) in [("never", 0.0f32), ("always", 1.0f32)] {
            let nearest = recorded(name, &format!("nearest-{function}")).expect("a row");
            let linear = recorded(name, &format!("linear-{function}")).expect("a row");
            assert_eq!(
                nearest[0], linear[0],
                "{name}: the filter moved {function}, which compares nothing"
            );
            for lane in 0..compare_lanes().len() {
                assert_eq!(
                    compare_sample(&linear[0], lane),
                    want,
                    "{name}/linear-{function}: lane {lane} is not the constant {want}"
                );
            }
        }
    }

    /// The other six comparisons **are** moved by the filter, at the lanes where the four texels of
    /// the footprint disagree — so fourteen of the sixteen rows are distinguishable answers.
    #[test]
    fn the_filter_moves_the_other_six_comparisons() {
        let name = "sampler-compare-filter-pairs";
        let Some(_) = recorded(name, "nearest-less") else {
            return;
        };
        for (function, _) in COMPARE_FUNCTIONS {
            if function == "never" || function == "always" {
                continue;
            }
            let nearest = recorded(name, &format!("nearest-{function}")).expect("a row");
            let linear = recorded(name, &format!("linear-{function}")).expect("a row");
            assert_ne!(
                nearest[0], linear[0],
                "{name}: the nearest and the linear {function} rows answered alike, so this cell \
                 compares the filter against nothing"
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // The case set

    /// Every case names its own rows, so no two rows of a case share a site and no recording can
    /// hold one row's bytes under another's name.
    #[test]
    fn every_case_names_its_rows_once() {
        for case in all() {
            let mut seen = BTreeSet::new();
            for (row, _) in case.rows() {
                assert!(
                    seen.insert(row.clone()),
                    "{}: two rows named {row}",
                    case.name
                );
            }
            assert!(!seen.is_empty());
        }
    }

    /// Every case's lane count is a whole multiple of its group size, so no dispatch here is split
    /// and no case declares a notice about one.
    #[test]
    fn every_dispatch_divides_into_whole_groups() {
        for case in all() {
            let (lanes, bytes) = case.lanes();
            assert_eq!(
                lanes % case.group(),
                0,
                "{}: {lanes} lanes do not divide into groups of {}",
                case.name,
                case.group()
            );
            assert_eq!(
                bytes.len() % 4,
                0,
                "{}: the coordinate buffer is not whole words",
                case.name
            );
        }
    }

    /// Every row of the axis families states three address modes, and six of the nine triples
    /// state three **different** ones — which is what no other case of the suite does.
    #[test]
    fn six_triples_hold_three_unequal_modes() {
        let unequal = TRIPLES
            .iter()
            .filter(|(_, modes)| {
                modes[0] != modes[1] && modes[1] != modes[2] && modes[0] != modes[2]
            })
            .count();
        assert_eq!(unequal, 6, "the triples lost their unequal rows");
        let equal = TRIPLES
            .iter()
            .filter(|(_, modes)| modes[0] == modes[1] && modes[1] == modes[2])
            .count();
        assert_eq!(equal, 3, "the triples lost their controls");
    }
}
