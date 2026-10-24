//! What a runtime sampler's mip filter, LOD clamp, third address mode, comparison and reduction
//! make Metal read.
//!
//! `MTLSamplerDescriptor` has sixteen items and the suite reached seven of them. The nine it did
//! not reach are the whole of the descriptor's **filtering** behaviour, and this module puts five
//! of them to Metal through fixtures already in the tree: `mipFilter`, `lodMinClamp`,
//! `rAddressMode`, `compareFunction` and `reductionMode`, the last recorded as the one thing Metal
//! does with it on this oracle. Two more families close two enums the depth axis read short of
//! their own values, `MTLSamplerBorderColor` at 1 of 3 and `MTLSamplerMinMagFilter` at 2 of 3.
//!
//! Every row's observable is **a texel Metal filtered**, never a field read back. A sampler state
//! is not observable; what it makes a sample return is.
//!
//! # The six families
//!
//! **`mip`** — the owned fixture `msf_runtime_samplers`' `msf_slot2d_linear_edge` samples a 2D
//! texture twice a lane: through its own constexpr `filter::linear, address::clamp_to_edge`
//! sampler, which has no mip filter, and through the runtime sampler at `[[sampler(0)]]`, both at
//! `level(0.0)`. The texture is `SIDE` by `SIDE` with `LEVELS` levels whose texels are
//! `level_texel`'s distinct numbers, so a sample **names the level and the texel it read**. One
//! case a mip filter, `CLAMPS` rows a case, each row one `lodMinClamp`. The constexpr column is
//! the control: it is pinned to level 0 by the fixture's own text, so a row where the runtime
//! column parts from it is a row where the clamp moved the level, and a row where the two agree is
//! one where it did not. That is the whole non-vacuity question, answered inside the recorded
//! bytes by `tests::the_recorded_clamps_moved_the_level` and
//! `tests::the_three_mip_filters_are_not_one_answer`.
//!
//! **What the three filters answered**, as lane 0 (texel `(0, 0)`, whose levels hold 1, 1001 and
//! 2001):
//!
//! | `lodMinClamp` | NotMipmapped | Nearest | Linear |
//! |---|---|---|---|
//! | 0 | 1 | 1 | 1 |
//! | 0.25 | 1 | 1 | 251 |
//! | 0.5 | 1 | 1 | 501 |
//! | 1 | 1 | 1001 | 1001 |
//! | 1.5 | 1 | 1001 | 1501 |
//! | 2 | 1 | 2001 | 2001 |
//! | 3 | 1 | 2001 | 2001 |
//! | `FLT_MAX` | 1 | 2001 | 2001 |
//!
//! Three readings worth naming, because none of them is a rule anybody wrote down here:
//! **`NotMipmapped` ignores `lodMinClamp` outright** — all eight rows read the level the kernel
//! named; **`Nearest` picks `ceil(clamp + 0.5) - 1`**, so 0.5 still reads level 0 and 1.5 reads
//! level 1; and **`Linear` blends exactly**, 251 being three quarters of level 0 and a quarter of
//! level 1 to the bit. Past the last level every filter holds at level 2, `FLT_MAX` included.
//!
//! The runtime sampler's two filters are both **nearest** in this family, deliberately. With an
//! explicit level Metal's choice between the minifying and the magnifying filter is a second
//! question — one this family would otherwise confound with the first — and two nearest filters
//! answer it identically whichever Metal picks.
//!
//! **`r-address`** — `msf_slot3d_linear_repeat` samples a 3D texture, its coordinate read from a
//! buffer, through a constexpr sampler that repeats on **all three** axes and through the runtime
//! sampler. The runtime sampler repeats on `s` and `t` and takes the row's mode on `r`, so the
//! constexpr column is again the control: a mode that is not `Repeat` parts from it at every `r`
//! outside `[0, 1]`. All six `MTLSamplerAddressMode` values are rows.
//!
//! The border colour here is **opaque white**, and it was chosen to tell `ClampToZero` from
//! `ClampToBorderColor`. **It does not, and the recording is why:** on this oracle both modes read
//! `1.0`, the opaque-white border's red, not `0.0`. Metal reads the border colour on a
//! clamp-to-zero axis as well, which is what the runtime's own `sampler.rs` already says it
//! measured, so **no border colour can part those two values** and they are indistinguishable
//! here by construction. The opaque-white border stays, because it is what makes that a measured
//! fact rather than an untested assumption: with the default transparent black both modes would
//! read 0 and a reader could not tell a border from a zero.
//!
//! **What the six modes answered**, at `s = t = 0.375` (texel `(1, 1)` of a slice, so a slice `z`
//! reads `6 + 16 z`) with the border reading `1`:
//!
//! | `r` | edge | mirror-clamp | repeat | mirror-repeat | zero | border |
//! |---|---|---|---|---|---|---|
//! | `1.0` | 54 | 54 | 6 | 54 | 1 | 1 |
//! | `1.25` | 54 | 54 | 22 | 38 | 1 | 1 |
//! | `-2.3` | 6 | 54 | 38 | 22 | 1 | 1 |
//! | `3.7` | 54 | 54 | 38 | 22 | 1 | 1 |
//!
//! Five distinct answers of six modes, every lane inside `[0, 1)` agreeing as the control
//! requires. The lane one unit in the last place **below** `+0.0` reads slice 0 under every mode
//! rather than the border, which is `float`'s own subnormal boundary showing through and the same
//! behaviour `nearest_positions` measured on the `s` axis.
//!
//! **`compare`** — `rrp_mirror_clamp`'s `rrp_mc_compare` takes five `sample_compare`s of a
//! `depth2d<float>` through the runtime sampler, at one texel offset each. One case per
//! `MTLCompareFunction`: all eight, which is the half of that enum a sampler owns. The references
//! walk either side of each texel's own depth and sit **on** it, so `Less` and `LessEqual` part at
//! equality and `Greater` and `GreaterEqual` part there too — the one place four of these eight
//! values are distinguishable at all.
//!
//! **`reduction`** — the same 2D fixture with a **linear** runtime sampler at a coordinate between
//! four texel centres, so the footprint really is four texels with four different numbers in it,
//! which is what a minimum or a maximum reduction would reduce. All three
//! `MTLSamplerReductionMode` values are rows, and the recorded answer is that **Metal reduces by
//! the weighted average whatever the mode**: `supportsSamplerReductionMode` is false on this
//! oracle. So these three rows are a **mirror of each other by Metal's own measured behaviour**,
//! not three distinguishable answers, and `tests::the_three_reductions_are_one_answer` asserts
//! exactly that of the recorded bytes rather than leaving a reader to assume a sweep happened. The
//! case still earns its place: it pins the agreement, so an implementation that *did* apply a
//! minimum reduction here would diverge from Metal and be caught.
//!
//! **It is the one family here that cannot be falsified, and that is the finding.** Dropping
//! `reduction_mode` from our leg's descriptor and replaying leaves the case `exact`, because
//! neither Metal nor the runtime applies the mode on this device. So as a *comparison of
//! behaviour on this oracle* these three rows compare nothing, and they are not claimed to: what
//! they hold is the agreement, for a venue where one side might not.
//!
//! **`border`** — the same 3D fixture over an `RGBA8Unorm` texture, sampled through each of the
//! two address modes that read a border with each of the three `MTLSamplerBorderColor` values.
//! Four channels and not one: Metal fills a one-channel format's sample as `(r, 0, 0, 1)`, so
//! transparent black and opaque black both read `0` in the only channel `R32Float` has and that
//! format cannot part them. Three distinct answers of the three colours under **both** modes, and
//! `ClampToZero` reads exactly what `ClampToBorderColor` reads for every one of the three.
//!
//! **`cubic`** — one row, a `Cubic` min and mag filter, which is the third value of
//! `MTLSamplerMinMagFilter` and no device's own. The coordinates are a quarter of a texel past
//! each texel's lower corner, where a sixteen-texel kernel's weights all differ rather than
//! collapsing onto the texel a symmetric kernel would read at a centre. Every one of the sixteen
//! samples compares with Metal bit for bit; the case's verdict is `announced` and not `exact`
//! because the runtime emulates the footprint and says its arithmetic is its own, which the case
//! declares and which loosens nothing.
//!
//! # What is not here, and why
//!
//! - **`lodMaxClamp`** cannot be given a distinguishing value by any kernel in the tree. Every
//!   runtime-sampler kernel samples at an explicit `level(0.0)`; Metal raises that level to
//!   `lodMinClamp` and then lowers it to `lodMaxClamp`, so with the lower clamp at or above the
//!   level the upper clamp is the only one that can bind, and the lower clamp always decides.
//!   `lodMinClamp > lodMaxClamp` is what would make the upper clamp decide, and Metal's validation
//!   refuses it. Closing this needs one kernel that takes its level from a buffer.
//! - **`maxAnisotropy`** needs a footprint with an aspect ratio, which needs a gradient or a
//!   fragment's derivatives; at an explicit level there is none, so every value inside Metal's
//!   1-to-16 range reads alike. Outside 1 to 16 is what Metal's validation refuses, so that stays
//!   out too.
//! - **`lodAverage`** is a fragment-stage fact — whether the level is averaged over the quad — and
//!   no fragment function in the tree samples through a runtime sampler.
//!
//! All three are nonetheless **set** on both legs ([`crate::case::SamplerSpec`] carries every
//! field), so they are the descriptor's defaults as the case states them and not fields one leg
//! never touched. The items axis counts that as reached; this prose is what stops that number
//! being read as depth.
//!
//! # Nothing here can fault a GPU
//!
//! One compute pass a row, textures of at most `SIDE` by `SIDE` by `DEPTH`, read-only, two
//! buffers a row, one thread a lane, no loop. Every coordinate the `mip`, `compare`, `reduction`
//! and `cubic` families sample at is inside its texture; the `r-address` and `border` families'
//! coordinates reach outside on purpose, which is what an address mode and a border are for.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side,
    TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::replay::notice_allowance;
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
    SamplerMipFilter, SamplerReductionMode, TextureType, TextureUsage,
};
use metal2vulkan_runtime::Degradation;

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

/// The 2D and 3D textures' side: a power of two, so the mip chain is exact and every level's
/// texels are whole texels of the one above.
const SIDE: u32 = 4;

/// The 3D texture's depth, which the `r` axis wraps over.
const DEPTH: u32 = 4;

/// The 2D texture's levels: 4 by 4, 2 by 2 and 1 by 1.
const LEVELS: u32 = 3;

/// The `float4` results one lane of a slot kernel writes: the constexpr sampler's and the runtime
/// sampler's.
const SLOT_RESULTS: u32 = 2;

/// The bytes one lane of a slot kernel writes.
const SLOT_LANE: u32 = SLOT_RESULTS * 16;

/// The `float4` results one lane of `rrp_mc_compare` writes: the comparison with no offset, then
/// with the offsets -3, 2, -7 and 5.
const COMPARE_RESULTS: u32 = 5;

/// The bytes one lane of `rrp_mc_compare` writes.
const COMPARE_LANE: u32 = COMPARE_RESULTS * 16;

/// The byte every output buffer holds before a row runs, so a word no thread wrote is a recorded
/// value rather than whatever the allocator left.
const FILL: u8 = 0xab;

/// The threads a group, which nothing a sampler does depends on; every family's lane count is a
/// whole multiple of it.
const GROUP: u32 = 4;

// ---------------------------------------------------------------------------------------------
// The 2D texture: a mip chain whose levels hold distinct numbers

/// Texel `(x, y)` of level `level`: `1000 * level + 1 + x + SIDE * y`.
///
/// Distinct across the whole chain, so one `float` names both the level a sample read and the
/// texel inside it. Every value is a small integer and exact in `float`, so a nearest sample
/// returns it bit for bit and a linear one between four of them is an exact quarter sum.
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

/// Every level's texels, level 0 first: the bytes the uploads borrow from.
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

/// The mipmapped 2D texture the `mip` and `reduction` families sample, loaded with `uploads`.
fn mipped_2d<'a>(uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
    TextureRequest::read_2d(PixelFormat::R32Float.as_raw(), SIDE, SIDE)
        .levels(LEVELS)
        .loaded(uploads)
}

// ---------------------------------------------------------------------------------------------
// The mip family

/// The LOD clamps each `mip` case sweeps, as `lodMinClamp`.
///
/// Metal's default (0), a quarter and a half of the way to level 1, exactly level 1, half way on
/// to level 2, exactly level 2 (the last level there is), one past the last level and the largest
/// `float` there is — which is also `lodMaxClamp`'s default, so the state stays one Metal's
/// validation takes.
const CLAMPS: [f32; 8] = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, f32::MAX];

/// The coordinates the `mip` family samples at: the centre of each texel of level 0, `y` slowest.
///
/// A texel centre is where a linear filter returns the texel itself, so the constexpr control
/// column is exact, and it is where a nearest filter is furthest from an edge.
fn centres() -> Vec<[f32; 2]> {
    (0..SIDE)
        .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
        .map(|(x, y)| {
            [
                (x as f32 + 0.5) / SIDE as f32,
                (y as f32 + 0.5) / SIDE as f32,
            ]
        })
        .collect()
}

/// The coordinates the `cubic` family samples at: a quarter of a texel past each texel's lower
/// corner, so the footprint is never centred on a texel.
///
/// A cubic filter weights sixteen texels, and at a texel centre a symmetric kernel can collapse to
/// the texel itself; a quarter off-centre is where the sixteen weights differ and the filter's own
/// shape shows.
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

/// The coordinate buffer of a 2D slot family.
fn coords_2d(at: &[[f32; 2]]) -> Vec<u8> {
    at.iter()
        .flatten()
        .copied()
        .flat_map(f32::to_le_bytes)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The r-address family

/// The 3D texture's texel `(x, y, z)`: `1 + x + SIDE * y + SIDE * SIDE * z`, every value exact in
/// `float`.
fn texel_3d(x: u32, y: u32, z: u32) -> f32 {
    (1 + x + SIDE * y + SIDE * SIDE * z) as f32
}

/// The 3D texture's texels, `x` fastest then `y` then `z`.
fn bytes_3d() -> Vec<u8> {
    (0..DEPTH)
        .flat_map(|z| (0..SIDE).flat_map(move |y| (0..SIDE).map(move |x| (x, y, z))))
        .flat_map(|(x, y, z)| texel_3d(x, y, z).to_le_bytes())
        .collect()
}

/// The 3D texture, loaded with `uploads`.
fn texture_3d<'a>(uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
    TextureRequest {
        format: PixelFormat::R32Float.as_raw(),
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

/// The four-channel 3D texture the `border` family samples, whose every channel comes from the
/// texture or from the border, so all three border colours are distinguishable.
///
/// `R32Float` cannot tell them apart: Metal fills a one-channel format's sample as
/// `(r, 0, 0, 1)`, so transparent black and opaque black both read `0` in the one channel there
/// is. A four-channel format reads `(0, 0, 0, 0)`, `(0, 0, 0, 1)` and `(1, 1, 1, 1)`.
fn bytes_3d_rgba() -> Vec<u8> {
    (0..DEPTH)
        .flat_map(|z| (0..SIDE).flat_map(move |y| (0..SIDE).map(move |x| (x, y, z))))
        .flat_map(|(x, y, z)| {
            [
                (16 + 16 * x) as u8,
                (16 + 16 * y) as u8,
                (16 + 16 * z) as u8,
                128,
            ]
        })
        .collect()
}

/// The `border` family's 3D texture, loaded with `uploads`.
fn texture_3d_rgba<'a>(uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
    let mut request = texture_3d(uploads);
    request.format = PixelFormat::RGBA8Unorm.as_raw();
    request
}

/// The `border` family's one upload, whose pitches are its four bytes a texel.
fn upload_3d_rgba(bytes: &[u8]) -> Upload<'_> {
    let mut upload = upload_3d(bytes);
    upload.row_pitch = SIDE * 4;
    upload.image_pitch = SIDE * SIDE * 4;
    upload
}

/// The 3D texture's one upload.
fn upload_3d(bytes: &[u8]) -> Upload<'_> {
    Upload {
        level: 0,
        slice: 0,
        origin: [0, 0, 0],
        size: [SIDE, SIDE, DEPTH],
        bytes,
        row_pitch: SIDE * 4,
        rows: SIDE * DEPTH,
        image_pitch: SIDE * SIDE * 4,
    }
}

/// The `r` coordinates the address family sweeps: exactly 0 and exactly 1, one unit in the last
/// place either side of each, the two texel centres nearest each end, and far outside in both
/// directions.
///
/// `-2.3` and `3.7` are the far ones, and they are not symmetric on purpose: a device that folded
/// the axis the wrong way round would read the same texel at `-x` and at `x` and a symmetric pair
/// could not tell.
fn rs() -> Vec<f32> {
    vec![
        0f32.next_down(),
        0.0,
        0f32.next_up(),
        0.5 / DEPTH as f32,
        (DEPTH as f32 - 0.5) / DEPTH as f32,
        1f32.next_down(),
        1.0,
        1f32.next_up(),
        -2.3,
        3.7,
        1.25,
        -0.25,
    ]
}

/// The coordinate buffer of the `r-address` family: `(0.375, 0.375, r, 0)` a lane, whose `s` and
/// `t` are the centre of texel `(1, 1)` and whose `r` is the swept value.
fn coords_3d() -> Vec<u8> {
    rs().iter()
        .flat_map(|&r| [1.5 / SIDE as f32, 1.5 / SIDE as f32, r, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The compare family

/// The depth texture's texel `(x, y)`: `(1 + x + SIDE * y) / 32`, a dyadic fraction exact in
/// `float`.
///
/// The `1 +` and the 32 are both load-bearing. Every texel's depth has to leave room for a
/// reference [`DELTA`] **below** it that is still inside depth's own `[0, 1]`, so no lane asks what
/// Metal does with a reference outside that range — a separate question, and not this one.
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

/// How far either side of a texel's own depth a reference sits: half the gap between two texels'
/// depths, so a reference beside one texel is never another texel's own depth.
const DELTA: f32 = 1.0 / 64.0;

/// The `(coordinate, reference)` lanes the compare family runs: per texel of the depth texture,
/// its centre with a reference below its depth, on it, and above it.
///
/// On it is the lane that tells `Less` from `LessEqual` and `Greater` from `GreaterEqual`, which
/// is the only place those four values of `MTLCompareFunction` are distinguishable.
fn compare_lanes() -> Vec<([f32; 2], f32)> {
    let mut out = Vec::new();
    for y in 0..SIDE {
        for x in 0..SIDE {
            let at = [
                (x as f32 + 0.5) / SIDE as f32,
                (y as f32 + 0.5) / SIDE as f32,
            ];
            let depth = depth_texel(x, y);
            for reference in [depth - DELTA, depth, depth + DELTA] {
                out.push((at, reference));
            }
        }
    }
    out
}

/// The coordinate buffer of the compare family: `(s, t, reference, 0)` a lane, which is what
/// `rrp_mc_compare` reads.
fn coords_compare() -> Vec<u8> {
    compare_lanes()
        .iter()
        .flat_map(|&([s, t], reference)| [s, t, reference, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The reduction family

/// The coordinate the reduction family samples at: the common corner of texels `(0, 0)`, `(1, 0)`,
/// `(0, 1)` and `(1, 1)` of level 0, repeated for each lane of the group.
///
/// A linear filter there weights those four texels a quarter each, which is a footprint a minimum
/// or a maximum reduction would reduce to one of them rather than to their mean. Four lanes, all
/// at the same coordinate, so a group's worth of threads answers the one question four times and a
/// lane that answered differently from its neighbours would show.
fn corners() -> Vec<[f32; 2]> {
    vec![[1.0 / SIDE as f32, 1.0 / SIDE as f32]; GROUP as usize]
}

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

/// The layout the compare family's output buffer is read under: one row a lane, whose elements are
/// the five comparisons' first channels and the zeros the fixture writes beside them.
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

/// Which family a case belongs to, and what it sweeps.
#[derive(Clone, Copy, Debug)]
enum Family {
    /// One mip filter, [`CLAMPS`] rows of `lodMinClamp`.
    Mip(SamplerMipFilter),
    /// Six rows, one an `MTLSamplerAddressMode` on the `r` axis.
    RAddress,
    /// One `MTLCompareFunction`, every lane of [`compare_lanes`].
    Compare(CompareFunction),
    /// Three rows, one an `MTLSamplerReductionMode`.
    Reduction,
    /// Six rows: each `MTLSamplerBorderColor` on each of the two address modes that read a
    /// border.
    Border,
    /// One row: a `Cubic` min and mag filter, the third value of `MTLSamplerMinMagFilter`.
    Cubic,
}

/// Every `MTLSamplerAddressMode`, which the `r-address` family's rows are.
const MODES: [(&str, SamplerAddressMode); 6] = [
    ("clamp-to-edge", SamplerAddressMode::ClampToEdge),
    (
        "mirror-clamp-to-edge",
        SamplerAddressMode::MirrorClampToEdge,
    ),
    ("repeat", SamplerAddressMode::Repeat),
    ("mirror-repeat", SamplerAddressMode::MirrorRepeat),
    ("clamp-to-zero", SamplerAddressMode::ClampToZero),
    (
        "clamp-to-border-color",
        SamplerAddressMode::ClampToBorderColor,
    ),
];

/// Every `MTLSamplerBorderColor`, spelled out one value a row rather than imported in a brace
/// list, which is how the option-space axis reads a value as put to Metal.
const BORDERS: [(&str, SamplerBorderColor); 3] = [
    ("transparent-black", SamplerBorderColor::TransparentBlack),
    ("opaque-black", SamplerBorderColor::OpaqueBlack),
    ("opaque-white", SamplerBorderColor::OpaqueWhite),
];

/// The two address modes that read a border on this oracle: `ClampToBorderColor`, which is for it,
/// and `ClampToZero`, which the recordings show reads it too.
const BORDER_MODES: [(&str, SamplerAddressMode); 2] = [
    ("zero", SamplerAddressMode::ClampToZero),
    ("border", SamplerAddressMode::ClampToBorderColor),
];

/// Every `MTLSamplerReductionMode`, which the `reduction` family's rows are.
const REDUCTIONS: [(&str, SamplerReductionMode); 3] = [
    ("weighted-average", SamplerReductionMode::WeightedAverage),
    ("minimum", SamplerReductionMode::Minimum),
    ("maximum", SamplerReductionMode::Maximum),
];

/// One case of the family.
pub struct SamplerFiltering {
    name: &'static str,
    family: Family,
}

/// Every case: three mip filters, the `r` axis, eight comparisons and the reductions.
pub fn all() -> Vec<SamplerFiltering> {
    let mut cases = vec![
        SamplerFiltering {
            name: "sampler-mip-not-mipmapped",
            family: Family::Mip(SamplerMipFilter::NotMipmapped),
        },
        SamplerFiltering {
            name: "sampler-mip-nearest",
            family: Family::Mip(SamplerMipFilter::Nearest),
        },
        SamplerFiltering {
            name: "sampler-mip-linear",
            family: Family::Mip(SamplerMipFilter::Linear),
        },
        SamplerFiltering {
            name: "sampler-r-address-modes",
            family: Family::RAddress,
        },
        SamplerFiltering {
            name: "sampler-reduction-modes",
            family: Family::Reduction,
        },
        SamplerFiltering {
            name: "sampler-border-colours",
            family: Family::Border,
        },
        SamplerFiltering {
            name: "sampler-cubic-filter",
            family: Family::Cubic,
        },
    ];
    for (name, function) in COMPARE_FUNCTIONS {
        cases.push(SamplerFiltering {
            name,
            family: Family::Compare(function),
        });
    }
    cases
}

/// Every `MTLCompareFunction`, with the case name each one's row carries.
const COMPARE_FUNCTIONS: [(&str, CompareFunction); 8] = [
    ("sampler-compare-never", CompareFunction::Never),
    ("sampler-compare-less", CompareFunction::Less),
    ("sampler-compare-equal", CompareFunction::Equal),
    ("sampler-compare-less-equal", CompareFunction::LessEqual),
    ("sampler-compare-greater", CompareFunction::Greater),
    ("sampler-compare-not-equal", CompareFunction::NotEqual),
    (
        "sampler-compare-greater-equal",
        CompareFunction::GreaterEqual,
    ),
    ("sampler-compare-always", CompareFunction::Always),
];

impl SamplerFiltering {
    /// The fixture and entry point the case runs.
    fn fixture(&self) -> Fixture {
        match self.family {
            Family::Mip(_) | Family::Reduction | Family::Cubic => Fixture {
                name: SLOTS,
                entry: SLOT_2D,
            },
            Family::RAddress | Family::Border => Fixture {
                name: SLOTS,
                entry: SLOT_3D,
            },
            Family::Compare(_) => COMPARES,
        }
    }

    /// The sampler state each row of the case makes, under the name its sites carry.
    fn rows(&self) -> Vec<(String, SamplerSpec)> {
        match self.family {
            Family::Mip(mip) => CLAMPS
                .iter()
                .map(|&clamp| {
                    (
                        format!("clamp-{}", clamp_name(clamp)),
                        SamplerSpec::new().mipped(mip).clamped_from(clamp),
                    )
                })
                .collect(),
            Family::RAddress => MODES
                .iter()
                .map(|&(name, mode)| {
                    let mut spec = SamplerSpec::new()
                        .bordered(SamplerBorderColor::OpaqueWhite)
                        .addressed(SamplerAddressMode::Repeat);
                    spec.r_address_mode = mode;
                    (name.to_owned(), spec)
                })
                .collect(),
            Family::Compare(function) => {
                vec![("compare".to_owned(), SamplerSpec::new().comparing(function))]
            }
            Family::Reduction => REDUCTIONS
                .iter()
                .map(|&(name, reduction)| {
                    let mut spec = SamplerSpec::filtered(SamplerMinMagFilter::Linear);
                    spec.reduction_mode = reduction;
                    (name.to_owned(), spec)
                })
                .collect(),
            Family::Border => BORDER_MODES
                .iter()
                .flat_map(|&(mode_name, mode)| {
                    BORDERS.iter().map(move |&(colour, border)| {
                        let mut spec = SamplerSpec::new()
                            .bordered(border)
                            .addressed(SamplerAddressMode::Repeat);
                        spec.r_address_mode = mode;
                        (format!("{mode_name}-{colour}"), spec)
                    })
                })
                .collect(),
            Family::Cubic => vec![(
                "cubic".to_owned(),
                SamplerSpec::filtered(SamplerMinMagFilter::Cubic),
            )],
        }
    }

    /// The lanes the case dispatches, and the bytes its coordinate buffer holds.
    fn lanes(&self) -> (u32, Vec<u8>) {
        match self.family {
            Family::Mip(_) => (SIDE * SIDE, coords_2d(&centres())),
            Family::Cubic => (SIDE * SIDE, coords_2d(&quarters())),
            Family::RAddress | Family::Border => (rs().len() as u32, coords_3d()),
            Family::Compare(_) => (compare_lanes().len() as u32, coords_compare()),
            Family::Reduction => (GROUP, coords_2d(&corners())),
        }
    }

    /// The bytes one row's output buffer takes.
    fn output_bytes(&self) -> usize {
        let (lanes, _) = self.lanes();
        let lane = match self.family {
            Family::Compare(_) => COMPARE_LANE,
            _ => SLOT_LANE,
        };
        lanes as usize * lane as usize
    }

    /// The layout the row's output buffer is read under.
    fn layout(&self) -> ByteLayout {
        let (lanes, _) = self.lanes();
        match self.family {
            Family::Compare(_) => compare_layout(lanes),
            _ => slot_layout(lanes),
        }
    }

    /// The texels the case's texture holds, as the input fingerprint names them.
    fn texture_inputs(&self) -> Vec<Input> {
        match self.family {
            Family::Mip(_) | Family::Reduction => chain_bytes()
                .iter()
                .enumerate()
                .map(|(level, bytes)| Input::bytes(format!("texture:level-{level}"), bytes))
                .collect(),
            Family::RAddress => vec![Input::bytes("texture:3d", &bytes_3d())],
            Family::Border => vec![Input::bytes("texture:3d-rgba", &bytes_3d_rgba())],
            Family::Cubic => chain_bytes()
                .iter()
                .enumerate()
                .map(|(level, bytes)| Input::bytes(format!("texture:level-{level}"), bytes))
                .collect(),
            Family::Compare(_) => vec![Input::bytes("texture:depth", &depth_bytes())],
        }
    }

    /// The texture the case makes, on `side`.
    fn texture(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<Handle, Abort> {
        match self.family {
            Family::Mip(_) | Family::Reduction | Family::Cubic => {
                let chain = chain_bytes();
                let uploads = chain_uploads(&chain);
                made(sites, "texture", side.texture(&mipped_2d(&uploads)))
            }
            Family::RAddress => {
                let bytes = bytes_3d();
                let uploads = [upload_3d(&bytes)];
                made(sites, "texture", side.texture(&texture_3d(&uploads)))
            }
            Family::Border => {
                let bytes = bytes_3d_rgba();
                let uploads = [upload_3d_rgba(&bytes)];
                made(sites, "texture", side.texture(&texture_3d_rgba(&uploads)))
            }
            Family::Compare(_) => {
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

/// The site-name spelling of a LOD clamp: a short, exact name a reader can tie to [`CLAMPS`].
fn clamp_name(clamp: f32) -> String {
    if clamp == f32::MAX {
        "float-max".to_owned()
    } else {
        format!("{clamp}").replace('.', "-")
    }
}

impl Case for SamplerFiltering {
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
        // Metal announces nothing, so a notice of ours is a difference at the notices site. One
        // family here provokes one: a `Cubic` min and mag filter, which no device's own sampler
        // has, so the runtime emulates Metal's bicubic footprint and announces that its arithmetic
        // is its own (`bicubic-filter-arithmetic`).
        //
        // It is a **declaration and loosens nothing**: measured on the M3 Max, every one of the
        // sixteen samples of `sampler-cubic-filter` compares with Metal bit for bit, and the
        // notices site is the whole of the difference. The other families provoke no notice and
        // declare none, so a notice any of them starts announcing reads as divergent at once.
        //
        // Three more are declared, and the sets differ row by row because which rows announce
        // what is a measurement rather than a rule:
        //
        // - the `Shared` buffer every row writes its samples into
        //   ([`crate::replay::shared_memory_notice`]), for all fifteen rows (the RTX 5080);
        // - `float-sample-filter`, where the device filters a float texture its own way, for
        //   `sampler-mip-linear` and `sampler-r-address-modes` (llvmpipe and the 5080);
        // - `device-texel-denormals`, where the device's texel decode keeps a denormal Metal
        //   flushes, for `sampler-mip-linear` and `sampler-reduction-modes` (llvmpipe and the
        //   5080).
        let mut named = vec![crate::replay::shared_memory_notice()];
        if matches!(self.family, Family::Cubic) {
            named.push(notice_allowance(
                Degradation::BicubicFilterArithmetic.code(),
            ));
        }
        if matches!(self.name, "sampler-mip-linear" | "sampler-r-address-modes") {
            named.push(notice_allowance(Degradation::FloatSampleFilter.code()));
        }
        if matches!(self.name, "sampler-mip-linear" | "sampler-reduction-modes") {
            named.push(notice_allowance(Degradation::DeviceTexelDenormals.code()));
        }
        named
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
                    [GROUP, 1, 1],
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
    use std::collections::{BTreeMap, BTreeSet};

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

    /// The runtime sampler's `float4` at lane `lane` of a slot family's output bytes: the second
    /// of the two results the kernel writes.
    fn runtime_sample(bytes: &[u8], lane: usize) -> [f32; 4] {
        word4(bytes, lane * SLOT_LANE as usize + 16)
    }

    /// The constexpr control's `float4` at lane `lane`: the first of the two, which the fixture's
    /// own text pins to level 0.
    fn control_sample(bytes: &[u8], lane: usize) -> [f32; 4] {
        word4(bytes, lane * SLOT_LANE as usize)
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

    /// Every mip case's rows' recorded bytes, by `(case, row)`.
    type MipRows = BTreeMap<(&'static str, String), Vec<Vec<u8>>>;

    /// Every mip case's rows' recorded bytes, or `None` where any of the three is unrecorded.
    fn mip_rows() -> Option<MipRows> {
        let mut out = BTreeMap::new();
        for case in all() {
            if !matches!(case.family, Family::Mip(_)) {
                continue;
            }
            for (row, _) in case.rows() {
                out.insert((case.name(), row.clone()), recorded(case.name(), &row)?);
            }
        }
        Some(out)
    }

    /// **The non-vacuity evidence for `mipFilter` and `lodMinClamp`, read off Metal's own
    /// recorded bytes.**
    ///
    /// The kernel samples at `level(0.0)` and the constexpr control column is pinned to level 0,
    /// so a row whose runtime column differs from its control column is a row where the sampler
    /// state — and nothing else, since the coordinate, the texture and the kernel are the same
    /// across every row — moved the level Metal read. The test asserts such a row exists, and
    /// prints the clamp each mip filter first parted at, so the measurement is in the output
    /// rather than only in a pass.
    #[test]
    fn the_recorded_clamps_moved_the_level() {
        let Some(rows) = mip_rows() else {
            println!("sampler-mip-*: no recording in the tree yet, so nothing to cross-check");
            return;
        };
        let mut moved: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for ((case, row), outcomes) in &rows {
            for bytes in outcomes {
                let parted = (0..(SIDE * SIDE) as usize)
                    .any(|lane| runtime_sample(bytes, lane)[0] != control_sample(bytes, lane)[0]);
                if parted {
                    moved.entry(case).or_default().push(row.clone());
                }
            }
        }
        for (case, rows) in &moved {
            println!("{case}: the level moved at {rows:?}");
        }
        assert!(
            !moved.is_empty(),
            "no clamp of any mip filter moved the level Metal read: the sweep compares nothing"
        );
    }

    /// Of the three mip filters, at least one reads a level another does not at the same clamp.
    ///
    /// This is the `mipFilter` half of the question the test above answers for `lodMinClamp`: a
    /// clamp could move the level under every filter alike, which would make the filter itself the
    /// vacuous axis.
    #[test]
    fn the_three_mip_filters_are_not_one_answer() {
        let Some(rows) = mip_rows() else {
            println!("sampler-mip-*: no recording in the tree yet");
            return;
        };
        let mut apart = Vec::new();
        for (row, _) in all()
            .iter()
            .find(|case| matches!(case.family, Family::Mip(_)))
            .expect("a mip case")
            .rows()
        {
            let answers: BTreeSet<Vec<u8>> = [
                "sampler-mip-not-mipmapped",
                "sampler-mip-nearest",
                "sampler-mip-linear",
            ]
            .iter()
            .filter_map(|case| rows.get(&(*case, row.clone())))
            .flat_map(|outcomes| outcomes.first().cloned())
            .collect();
            if answers.len() > 1 {
                apart.push((row, answers.len()));
            }
        }
        println!("the mip filters part at {apart:?}");
        assert!(
            !apart.is_empty(),
            "every mip filter read the same bytes at every clamp: the filter axis compares nothing"
        );
    }

    /// **The non-vacuity evidence for `rAddressMode`.** The six modes are not six runs of one
    /// answer, and the ones that do coincide are named.
    #[test]
    fn the_six_r_modes_are_not_one_answer() {
        let case = all()
            .into_iter()
            .find(|case| matches!(case.family, Family::RAddress))
            .expect("the r-address case");
        let mut answers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (row, _) in case.rows() {
            let Some(outcomes) = recorded(case.name(), &row) else {
                println!("{}: no recording in the tree yet", case.name());
                return;
            };
            answers.insert(row, outcomes[0].clone());
        }
        let distinct: BTreeSet<&Vec<u8>> = answers.values().collect();
        let mut alike = Vec::new();
        for (one, bytes) in &answers {
            for (other, same) in &answers {
                if one < other && bytes == same {
                    alike.push((one.clone(), other.clone()));
                }
            }
        }
        println!(
            "{}: {} distinct answers of {} modes; alike: {alike:?}",
            case.name(),
            distinct.len(),
            answers.len()
        );
        assert!(
            distinct.len() > 1,
            "every r address mode read the same bytes: the sweep compares nothing"
        );
    }

    /// **The non-vacuity evidence for `compareFunction`.** The eight functions are not eight runs
    /// of one answer, and the pairs that coincide are named — there should be none, since the
    /// references sit on each texel's depth as well as either side of it.
    #[test]
    fn the_eight_comparisons_are_not_one_answer() {
        let mut answers: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
        for (name, _) in COMPARE_FUNCTIONS {
            let Some(outcomes) = recorded(name, "compare") else {
                println!("{name}: no recording in the tree yet");
                return;
            };
            answers.insert(name, outcomes[0].clone());
        }
        let distinct: BTreeSet<&Vec<u8>> = answers.values().collect();
        let mut alike = Vec::new();
        for (one, bytes) in &answers {
            for (other, same) in &answers {
                if one < other && bytes == same {
                    alike.push((*one, *other));
                }
            }
        }
        println!(
            "{} distinct answers of 8 comparisons; alike: {alike:?}",
            distinct.len()
        );
        assert!(
            distinct.len() > 1,
            "every comparison read the same bytes: the sweep compares nothing"
        );
    }

    /// **The non-vacuity evidence for `borderColor`.** The three colours are three answers on a
    /// four-channel format, and whether the two border-reading address modes agree is recorded
    /// rather than assumed.
    #[test]
    fn the_three_border_colours_are_three_answers() {
        let case = all()
            .into_iter()
            .find(|case| matches!(case.family, Family::Border))
            .expect("the border case");
        let mut answers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (row, _) in case.rows() {
            let Some(outcomes) = recorded(case.name(), &row) else {
                println!("{}: no recording in the tree yet", case.name());
                return;
            };
            answers.insert(row, outcomes[0].clone());
        }
        assert_eq!(answers.len(), BORDERS.len() * BORDER_MODES.len());
        for (mode, _) in BORDER_MODES {
            let colours: BTreeSet<&Vec<u8>> = BORDERS
                .iter()
                .filter_map(|(colour, _)| answers.get(&format!("{mode}-{colour}")))
                .collect();
            println!("{mode}: {} distinct answers of 3 colours", colours.len());
            assert!(
                colours.len() > 1,
                "{mode}: every border colour read the same bytes"
            );
        }
        let mut alike = Vec::new();
        for (colour, _) in BORDERS {
            let zero = answers.get(&format!("zero-{colour}"));
            let border = answers.get(&format!("border-{colour}"));
            if zero == border {
                alike.push(colour);
            }
        }
        println!("clamp-to-zero reads the same as clamp-to-border-color at {alike:?}");
    }

    /// **The `reductionMode` rows are one answer, and that is the measurement.**
    ///
    /// `supportsSamplerReductionMode` is false on this oracle, so Metal reduces a linear
    /// footprint by the weighted average whatever the mode. The three rows are therefore a mirror
    /// of each other by Metal's own behaviour, and this asserts it rather than leaving a reader to
    /// read three rows as three compared answers. An implementation that applied a minimum here
    /// would diverge from the recording, which is what the rows are for.
    #[test]
    fn the_three_reductions_are_one_answer() {
        let case = all()
            .into_iter()
            .find(|case| matches!(case.family, Family::Reduction))
            .expect("the reduction case");
        let mut answers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (row, _) in case.rows() {
            let Some(outcomes) = recorded(case.name(), &row) else {
                println!("{}: no recording in the tree yet", case.name());
                return;
            };
            answers.insert(row, outcomes[0].clone());
        }
        let average = answers
            .get("weighted-average")
            .expect("the weighted-average row");
        let mean = (0..4).map(|at| level_texel(0, at % 2, at / 2)).sum::<f32>() / 4.0;
        for (row, bytes) in &answers {
            assert_eq!(
                bytes, average,
                "{row} read a different footprint from the weighted average, which this oracle \
                 says it cannot: supportsSamplerReductionMode is false here"
            );
            let read = runtime_sample(bytes, 0)[0];
            assert_eq!(
                read, mean,
                "{row}: the linear footprint read {read}, not the four texels' mean {mean}"
            );
        }
    }

    /// Every case names a distinct row set, and every row of a case a distinct sampler state.
    ///
    /// A sweep whose two rows make the same state is a sweep that compares one answer twice, and
    /// the input fingerprint would not say so: two identical rows fingerprint identically and the
    /// recording would hold one answer under two names.
    #[test]
    fn every_row_of_a_case_is_its_own_sampler_state() {
        let mut names = BTreeSet::new();
        for case in all() {
            assert!(names.insert(case.name()), "{} twice", case.name());
            let rows = case.rows();
            let sites: BTreeSet<String> = rows.iter().map(|(name, _)| name.clone()).collect();
            assert_eq!(sites.len(), rows.len(), "{}: a repeated site", case.name());
            let words: BTreeSet<Vec<u8>> = rows.iter().map(|(_, spec)| spec.words()).collect();
            assert_eq!(
                words.len(),
                rows.len(),
                "{}: two rows fingerprint alike",
                case.name()
            );
        }
    }

    /// Each case's layout covers its output buffer exactly, and its lane count is whole groups.
    #[test]
    fn the_layouts_and_the_dispatches_agree_with_the_buffers() {
        for case in all() {
            let (lanes, coords) = case.lanes();
            assert!(lanes > 0, "{}: no lane", case.name());
            assert_eq!(lanes % GROUP, 0, "{}: a short group", case.name());
            assert_eq!(
                case.layout().extent(),
                case.output_bytes(),
                "{}: the layout is not the buffer",
                case.name()
            );
            let per_lane = match case.family {
                Family::Compare(_) | Family::RAddress | Family::Border => 16,
                Family::Mip(_) | Family::Reduction | Family::Cubic => 8,
            };
            assert_eq!(
                coords.len(),
                lanes as usize * per_lane,
                "{}: the coordinate buffer is not one coordinate a lane",
                case.name()
            );
        }
    }

    /// The mip chain's texels are distinct across every level, so a sample names the level it read.
    ///
    /// Without this the `mip` family could not tell a clamp that moved the level from one that did
    /// not: two levels holding the same number would read alike whatever Metal did.
    #[test]
    fn the_mip_chain_names_every_level_apart() {
        let mut seen = BTreeSet::new();
        for level in 0..LEVELS {
            let side = SIDE >> level;
            for y in 0..side {
                for x in 0..side {
                    let texel = level_texel(level, x, y);
                    assert!(
                        seen.insert(texel.to_bits()),
                        "level {level} texel ({x}, {y}) repeats {texel}"
                    );
                }
            }
            assert_eq!(level_bytes(level).len() as u32, side * side * 4);
        }
        assert_eq!(chain_bytes().len(), LEVELS as usize);
    }

    /// The LOD clamps reach a level the kernel never named, which is what makes the sweep a
    /// measurement rather than eight runs of one question.
    ///
    /// The kernel samples at `level(0.0)`. A clamp at or above 1 raises that to a level whose
    /// texels are a different number from level 0's, so the runtime sampler's column **must** part
    /// from the constexpr control at those rows under any mip filter that reads another level. The
    /// clamps also reach the last level there is, one past it, and `float`'s largest value.
    #[test]
    fn the_clamps_reach_a_level_the_case_never_named() {
        let past: Vec<f32> = CLAMPS
            .iter()
            .copied()
            .filter(|&clamp| clamp >= 1.0)
            .collect();
        assert!(past.len() >= 4, "too few clamps past level 0: {past:?}");
        assert!(
            CLAMPS.contains(&(LEVELS as f32 - 1.0)),
            "no clamp at the last level"
        );
        assert!(
            CLAMPS.iter().any(|&clamp| clamp > LEVELS as f32 - 1.0),
            "no clamp past the last level"
        );
        assert!(CLAMPS.contains(&f32::MAX), "no clamp at float's largest");
        // Every clamp is at most `lodMaxClamp`'s default, so Metal's validation takes every state
        // the family makes; an inverted pair is what it refuses.
        for clamp in CLAMPS {
            assert!(
                clamp <= SamplerSpec::new().lod_max_clamp,
                "{clamp} inverted"
            );
        }
    }

    /// The `r` sweep reaches 0 and 1 exactly, one unit in the last place either side of each, and
    /// far outside in both directions — the coordinates the six address modes differ at.
    #[test]
    fn the_r_sweep_reaches_both_edges_and_past_them() {
        let rs = rs();
        for wanted in [0.0f32, 1.0, 0f32.next_down(), 1f32.next_up()] {
            assert!(
                rs.iter().any(|&r| r.to_bits() == wanted.to_bits()),
                "no lane at {wanted}"
            );
        }
        assert!(rs.iter().any(|&r| r < -1.0), "nothing far below");
        assert!(rs.iter().any(|&r| r > 2.0), "nothing far above");
        // The two far coordinates are not negatives of each other, so a device that folded the
        // axis the wrong way round cannot pass by reading the same texel at both ends. The pair
        // one unit in the last place either side of 0 *is* symmetric, on purpose: those two ask
        // which side of the edge Metal puts a signed zero's neighbours on.
        let far: Vec<f32> = rs
            .iter()
            .copied()
            .filter(|&r| r < -1.0 || r > 2.0)
            .collect();
        assert_eq!(far.len(), 2, "not two far coordinates: {far:?}");
        assert_ne!(far[0], -far[1], "the far pair is symmetric");
    }

    /// The compare family's references sit below, on and above each texel's own depth, which is
    /// the only arrangement that tells all eight comparisons apart.
    #[test]
    fn the_compare_references_bracket_every_texel() {
        let lanes = compare_lanes();
        assert_eq!(lanes.len(), (SIDE * SIDE * 3) as usize);
        for y in 0..SIDE {
            for x in 0..SIDE {
                let depth = depth_texel(x, y);
                let mut found = [false; 3];
                for &(_, reference) in &lanes {
                    if reference.to_bits() == (depth - DELTA).to_bits() {
                        found[0] = true;
                    } else if reference.to_bits() == depth.to_bits() {
                        found[1] = true;
                    } else if reference.to_bits() == (depth + DELTA).to_bits() {
                        found[2] = true;
                    }
                }
                assert_eq!(found, [true; 3], "texel ({x}, {y}) is not bracketed");
                // Every reference stays inside depth's own range, so nothing here asks what Metal
                // does with a comparison outside it — a separate question, and not this one.
                assert!(depth - DELTA >= 0.0 && depth + DELTA <= 1.0, "{depth} edge");
            }
        }
        // All eight values of the enum are cases, which is the half of `MTLCompareFunction` a
        // sampler owns.
        let functions: BTreeSet<u32> = COMPARE_FUNCTIONS
            .iter()
            .map(|(_, function)| function.as_raw())
            .collect();
        assert_eq!(functions.len(), 8);
    }

    /// The reduction family's footprint really is four texels with four different numbers in it.
    ///
    /// A reduction over a footprint of one texel reduces to that texel whatever the mode, so the
    /// coordinate has to be the common corner of four and the filter has to be linear; otherwise
    /// the three rows could not differ even on a device that applied the mode.
    #[test]
    fn the_reduction_footprint_is_four_different_texels() {
        let at = corners()[0];
        assert_eq!(at, [1.0 / SIDE as f32, 1.0 / SIDE as f32]);
        let four: BTreeSet<u32> = [(0, 0), (1, 0), (0, 1), (1, 1)]
            .iter()
            .map(|&(x, y)| level_texel(0, x, y).to_bits())
            .collect();
        assert_eq!(four.len(), 4, "the footprint's texels are not all distinct");
        for case in all() {
            if !matches!(case.family, Family::Reduction) {
                continue;
            }
            for (_, spec) in case.rows() {
                assert_eq!(spec.min_filter, SamplerMinMagFilter::Linear);
                assert_eq!(spec.mag_filter, SamplerMinMagFilter::Linear);
            }
        }
    }

    /// The `r-address` family's border colour is opaque white, which is what tells `ClampToZero`
    /// from `ClampToBorderColor`.
    ///
    /// With the default transparent black the two modes read the same number on every format, so
    /// two of the six rows would be a sweep comparing nothing twice.
    #[test]
    fn the_border_colour_tells_the_two_border_modes_apart() {
        for case in all() {
            if !matches!(case.family, Family::RAddress) {
                continue;
            }
            let rows = case.rows();
            assert_eq!(rows.len(), MODES.len());
            for (_, spec) in &rows {
                assert_eq!(spec.border_color, SamplerBorderColor::OpaqueWhite);
                assert_eq!(spec.s_address_mode, SamplerAddressMode::Repeat);
                assert_eq!(spec.t_address_mode, SamplerAddressMode::Repeat);
            }
            let modes: BTreeSet<u32> = rows
                .iter()
                .map(|(_, spec)| spec.r_address_mode.as_raw())
                .collect();
            assert_eq!(modes.len(), 6, "not every address mode is a row");
        }
    }
}
