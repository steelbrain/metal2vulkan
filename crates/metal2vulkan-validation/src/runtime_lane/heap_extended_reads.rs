//! Reads of textures an argument buffer holds, against the same textures at a slot, on the device
//! the cfg selects, under the validation layer: the extended-range, padded and bfloat formats, and
//! Apple's RGBX formats and their RGBA twins.
//!
//! Metal reads a texture an argument buffer holds as the same texture bound to a slot. The
//! function decodes the raw codes of an extended-range, padded 10-bit or bfloat texture under
//! its slot's variant; an argument buffer's texture is known at run time alone, so the binder holds
//! one of a form a heap reads (`ExtendedRange::HEAP_READ`: `BGRA10_XR`, `BGR10_XR`, `R10Unorm_X6`,
//! `RG10Unorm_X12` and the three bfloat formats) raw in the unsigned integer class of its shape,
//! and a float read decodes it by the form its handle names. An RGBX texture holds its RGBA twin's
//! bytes, and every read of it sees alpha 1, at a slot through its view's component mapping. This
//! lane asks whether such a texture's reads (`float` and `half` results, or `uint` or `int` ones
//! for an integer format) and its size and level queries through an argument buffer are, bit for
//! bit, the same texture's through a slot, whose reads are Metal's (the bfloat probes; for RGBX).
//!
//! Each texture is 8 x 4 with its full chain of four levels, every level the probe's
//! pseudo-random bytes; the kernel reads every texel of each level, two texels past it, and a
//! texel of a level past the chain, and queries the width, height and level count at the read's
//! level (the last one at the last level). A heap result that is not the slot's is printed on a
//! `HEAP-READ-DIFFERS` line, and a format whose heap reads inside the texture are all zero where
//! the slot's are not on a `HEAP-READ-ZEROS` line. One difference is announced, and printed on a
//! `HEAP-READ-ANNOUNCED` line and counted apart: an 8-bit sRGB texture's
//! (`metal::format::srgb_raw_twin`) read through a heap pipeline whose stage records
//! `PrecisionNote::SrgbTexelDecode`, which the device announced, decodes on the device where a slot
//! of a format the device holds as its UNORM twin decodes the codes by the device's transfer (a slot
//! of a format the device samples decodes on the device too, the same words); every other
//! difference fails. The extended-range
//! `_sRGB` twins, which a heap does not hold (`ExtendedRange::HEAP_READ`), are not read so yet:
//! their heap dispatch is printed on a `HEAP-READ-UNCOVERED` line, refused or not, and not
//! counted. A texture the
//! runtime does not make (the padded formats', whose plain creation is refused) is printed on a
//! `HEAP-READ-REFUSED` line and counted apart.
//!
//! A sample or gather of such a texture through an argument buffer is the same texture's through a
//! slot too, where the function emulates it (`sampler::heap_raw`): a 2D sample or gather through a
//! constexpr sampler without a mip filter or anisotropy (a `half` gather's of a bfloat form alone),
//! or through a runtime sampler whose bound state says the same. A difference is printed on a
//! `HEAP-SAMPLE-DIFFERS` line. A command whose function samples such a texture otherwise (a mip
//! filter, a `half` gather, a runtime sampler with a mip filter) is refused at bind, printed on a
//! `HEAP-SAMPLE-REFUSED` line.

use super::degrade_notices::on_device;
use super::host_memory::{not_importable, Mapping, GUEST_PAGE};
use super::Messages;
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::error::{BindError, Feature, Unsupported};
use metal2vulkan::library::Library;
use metal2vulkan::metal::format::{srgb_raw_twin, ExtendedRange};
use metal2vulkan::metal::{
    PixelFormat, ResourceUsage, SamplerAddressMode, SamplerMinMagFilter, SamplerMipFilter,
    TextureUsage,
};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, PrecisionNote, Stage, StageSource,
};
use metal2vulkan::Budget;
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Degradation, Device, Error, HostMemory,
    PipelineOptions, Region, ResourceOptions, SamplerState, Size, Texture, TextureDescriptor,
};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Level 0's width and height; the chain has four levels.
const W: u64 = 8;
const H: u64 = 4;
const LEVELS: u32 = 4;

/// How a kernel reads a texture: its `T` in `texture2d<T, access::read>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Texel {
    Float,
    Half,
    Uint,
    Int,
}

/// A constexpr sampler's words (AIR's `air.sampler_state` layout): no mip filter, normalized
/// coordinates, the greatest level clamp, comparison `never`; each axis addressed by `mode` (0
/// clamp to zero or to the border, 1 clamp to the edge, 2 repeat, 3 mirrored repeat), both
/// filters linear where `linear`, the border color `border` (0 transparent black, 2 opaque
/// white), and a nearest mip filter where `mip`.
const fn sampler_words(mode: u64, linear: bool, border: u64, mip: bool) -> u64 {
    let filters = if linear { 1 << 9 | 1 << 11 } else { 0 };
    let mip = if mip { 1 << 13 } else { 0 };
    0x007b_ff00_0008_0000 | mode | mode << 3 | mode << 6 | filters | mip | border << 56
}

/// The constexpr samplers a heap's emulation answers ([`heap_extended_samples_are_the_slot_s`]):
/// nearest and linear clamped to the edge, linear repeating, nearest mirrored, and linear
/// clamped to an opaque-white border.
const SAMPLERS: [(&str, u64); 5] = [
    ("nearest edge", sampler_words(1, false, 0, false)),
    ("linear edge", sampler_words(1, true, 0, false)),
    ("linear repeat", sampler_words(2, true, 0, false)),
    ("nearest mirror", sampler_words(3, false, 0, false)),
    ("linear white border", sampler_words(0, true, 2, false)),
];

/// The extended-range formats, the padded 10-bit ones, their `_sRGB` twins, and the bfloat
/// formats, each with its texel's bytes.
const EXTENDED: [(PixelFormat, u64); 11] = [
    (PixelFormat::BGRA10XR, 8),
    (PixelFormat::BGRA10XRSRGB, 8),
    (PixelFormat::BGR10XR, 4),
    (PixelFormat::BGR10XRSRGB, 4),
    (PixelFormat::R10UnormX6, 2),
    (PixelFormat::R10UnormX6SRGB, 2),
    (PixelFormat::RG10UnormX12, 4),
    (PixelFormat::RG10UnormX12SRGB, 4),
    (PixelFormat::R16Bfloat, 2),
    (PixelFormat::RG16Bfloat, 4),
    (PixelFormat::RGBA16Bfloat, 8),
];

/// The RGBX formats and their RGBA twins, each with its texel's bytes and whether it reads as
/// floats (`None`), or as unsigned (`Some(false)`) or signed (`Some(true)`) integers.
const RGBX: [(PixelFormat, u64, Option<bool>); 22] = {
    use PixelFormat as P;
    [
        (P::RGBX8Unorm, 4, None),
        (P::RGBA8Unorm, 4, None),
        (P::RGBX8UnormSRGB, 4, None),
        (P::RGBA8UnormSRGB, 4, None),
        (P::RGBX8Snorm, 4, None),
        (P::RGBA8Snorm, 4, None),
        (P::RGBX8Uint, 4, Some(false)),
        (P::RGBA8Uint, 4, Some(false)),
        (P::RGBX8Sint, 4, Some(true)),
        (P::RGBA8Sint, 4, Some(true)),
        (P::RGBX16Uint, 8, Some(false)),
        (P::RGBA16Uint, 8, Some(false)),
        (P::RGBX16Sint, 8, Some(true)),
        (P::RGBA16Sint, 8, Some(true)),
        (P::RGBX16Float, 8, None),
        (P::RGBA16Float, 8, None),
        (P::RGBX32Uint, 16, Some(false)),
        (P::RGBA32Uint, 16, Some(false)),
        (P::RGBX32Sint, 16, Some(true)),
        (P::RGBA32Sint, 16, Some(true)),
        (P::RGBX32Float, 16, None),
        (P::RGBA32Float, 16, None),
    ]
};

/// The reads of a format that reads as `integer` ([`RGBX`]).
fn texels_of(integer: Option<bool>) -> &'static [Texel] {
    match integer {
        None => &[Texel::Float, Texel::Half],
        Some(false) => &[Texel::Uint],
        Some(true) => &[Texel::Int],
    }
}

/// Synthetic AIR written for this test: kernel `k` reads the `texture2d<T, access::read>` (`T`
/// as `texel` says) at texture slot 0, or where `heap` the one its argument buffer (buffer 2,
/// `{ texture2d<T, access::read> t; }`) holds, at buffer 0's lane `gid` (a `uint4`: the texel's x
/// and y, its level, and the queries' level), writing the texel (a `half` one widened to `float`)
/// to buffer 1's 16 bytes at `2 gid` and the width, height and level count at the queries' level
/// to its `uint4` `2 gid + 1`.
fn kernel(heap: bool, texel: Texel) -> String {
    let (element, ty, vec4) = match texel {
        Texel::Float => ("float", "v4f32", "<4 x float>"),
        Texel::Half => ("half", "v4f16", "<4 x half>"),
        Texel::Uint => ("uint", "u.v4i32", "<4 x i32>"),
        Texel::Int => ("int", "s.v4i32", "<4 x i32>"),
    };
    let texture = format!("texture2d<{element}, read>");
    let (holder, load) = match heap {
        true => (
            "ptr addrspace(2) %m",
            "  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0\n  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8\n",
        ),
        false => ("ptr addrspace(1) %t", ""),
    };
    let held = match heap {
        true => format!(
            r#"!5 = !{{i32 2, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !8, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!8 = !{{i32 0, i32 8, i32 0, !"{texture}", !"t", !"air.indirect_argument", !9}}
!9 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}"#
        ),
        false => format!(
            r#"!5 = !{{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}"#
        ),
    };
    let (widen, stored) = match texel {
        Texel::Half => (
            "  %hv = extractvalue { <4 x half>, i8 } %r, 0\n  %v = fpext <4 x half> %hv to <4 x float>\n",
            "<4 x float>",
        ),
        _ => ("", vec4),
    };
    let widen = match widen {
        "" => format!("  %v = extractvalue {{ {vec4}, i8 }} %r, 0\n"),
        w => w.to_owned(),
    };
    format!(
        r#"%struct.M = type {{ ptr addrspace(1) }}

define void @k(ptr addrspace(1) %c, ptr addrspace(1) %out, {holder}, i32 %gid) {{
  %i = zext i32 %gid to i64
  %p = getelementptr inbounds <4 x i32>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x i32>, ptr addrspace(1) %p, align 16
  %at = shufflevector <4 x i32> %c4, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %lod = extractelement <4 x i32> %c4, i64 2
  %ql = extractelement <4 x i32> %c4, i64 3
{load}  %rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{ty}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %at, <2 x i32> zeroinitializer, i32 %lod, i32 1)
{widen}  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 %ql)
  %h = call i32 @air.get_height_texture_2d(ptr addrspace(1) %t, i32 %ql)
  %n = call i32 @air.get_num_mip_levels_texture_2d(ptr addrspace(1) %t)
  %d0 = insertelement <4 x i32> zeroinitializer, i32 %w, i64 0
  %d1 = insertelement <4 x i32> %d0, i32 %h, i64 1
  %d2 = insertelement <4 x i32> %d1, i32 %n, i64 2
  %j = shl i64 %i, 1
  %q = getelementptr inbounds {stored}, ptr addrspace(1) %out, i64 %j
  store {stored} %v, ptr addrspace(1) %q, align 16
  %j1 = or i64 %j, 1
  %qd = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i64 %j1
  store <4 x i32> %d2, ptr addrspace(1) %qd, align 16
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ {vec4}, i8 }} @air.read_texture_2d.{ty}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_2d(ptr addrspace(1), i32)
declare i32 @air.get_num_mip_levels_texture_2d(ptr addrspace(1))

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !7}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"c"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}}
{held}
!7 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The probe's bytes of `n` texels of the format of raw value `raw` at level `level`, `bytes` a
/// texel: a 64-bit linear congruential state seeded by both, each byte its top 8 bits.
fn texels(raw: u32, level: u32, n: u64, bytes: u64) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(raw) ^ (u64::from(level) << 32);
    (0..n * bytes)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        })
        .collect()
}

/// A level's width and height.
fn extent(level: u32) -> (u32, u32) {
    ((W as u32 >> level).max(1), (H as u32 >> level).max(1))
}

/// The lanes: each level's every texel, then two past it, at that level, querying it; then a
/// texel of the level past the chain, querying the last level.
fn lanes() -> Vec<[u32; 4]> {
    let mut lanes = Vec::new();
    for level in 0..LEVELS {
        let (w, h) = extent(level);
        lanes.extend((0..w * h).map(|i| [i % w, i / w, level, level]));
        lanes.extend([[w, 0, level, level], [0, h + 2, level, level]]);
    }
    lanes.push([0, 0, LEVELS, LEVELS - 1]);
    lanes
}

/// Whether lane `lane` reads inside the texture.
fn inside(lane: [u32; 4]) -> bool {
    let [x, y, level, _] = lane;
    let (w, h) = extent(level.min(LEVELS - 1));
    level < LEVELS && x < w && y < h
}

/// The pipeline of kernel `k` of [`kernel`].
fn pipeline(device: &Device, heap: bool, texel: Texel) -> ComputePipelineState {
    pipeline_of(device, &kernel(heap, texel))
}

/// The sampler a sample kernel samples through: a constexpr sampler of these words, or the
/// runtime sampler at sampler slot 0.
#[derive(Clone, Copy, Debug)]
enum Through {
    Constexpr(u64),
    Slot,
}

/// What a sample kernel does: a sample, or a gather of a component (0 red to 3 alpha).
#[derive(Clone, Copy, Debug)]
enum Op {
    Sample,
    Gather(u32),
}

/// Synthetic AIR written for this test: kernel `k` samples or gathers (`op`) the
/// `texture2d<T, access::sample>` (`T` `half` where `half`, else `float`) at texture slot 0, or
/// where `heap` the one its argument buffer (buffer 2, `{ texture2d<T, access::sample> t; }`)
/// holds, through `through`, at buffer 0's lane `gid` (a `uint4` whose first two words are the
/// coordinate's floats), writing the result (a `half` one widened to `float`) to buffer 1's 16
/// bytes at `2 gid`.
fn sample_kernel(heap: bool, through: Through, op: Op, half: bool) -> String {
    let (element, suffix, vec4) = match half {
        true => ("half", "v4f16", "<4 x half>"),
        false => ("float", "v4f32", "<4 x float>"),
    };
    let texture = format!("texture2d<{element}, sample>");
    let (holder, load) = match heap {
        true => (
            "ptr addrspace(2) %m",
            "  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0\n  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8\n",
        ),
        false => ("ptr addrspace(1) %t", ""),
    };
    let held = match heap {
        true => format!(
            r#"!5 = !{{i32 2, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !8, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!8 = !{{i32 0, i32 8, i32 0, !"{texture}", !"t", !"air.indirect_argument", !9}}
!9 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}"#
        ),
        false => format!(
            r#"!5 = !{{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}"#
        ),
    };
    let (states, sampler_param, sampler, sampler_meta, gid_index, args) = match through {
        Through::Constexpr(words) => (
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8\n"
            ),
            "",
            "@__air_sampler_state",
            "!air.sampler_states = !{!10}\n!10 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n".to_owned(),
            3,
            "!3, !4, !5, !7",
        ),
        Through::Slot => (
            String::new(),
            "ptr addrspace(2) %s, ",
            "%s",
            "!6 = !{i32 3, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}\n".to_owned(),
            4,
            "!3, !4, !5, !6, !7",
        ),
    };
    let (call, decl) = match op {
        Op::Sample => (
            format!("@air.sample_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)"),
            format!("declare {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)"),
        ),
        Op::Gather(component) => (
            format!("@air.gather_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 {component}, i32 0)"),
            format!("declare {{ {vec4}, i8 }} @air.gather_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)"),
        ),
    };
    let widen = match half {
        true => "  %hv = extractvalue { <4 x half>, i8 } %r, 0\n  %v = fpext <4 x half> %hv to <4 x float>\n",
        false => "  %v = extractvalue { <4 x float>, i8 } %r, 0\n",
    };
    format!(
        r#"%struct.M = type {{ ptr addrspace(1) }}
{states}
define void @k(ptr addrspace(1) %c, ptr addrspace(1) %out, {holder}, {sampler_param}i32 %gid) {{
  %i = zext i32 %gid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> poison, <2 x i32> <i32 0, i32 1>
{load}  %r = call {{ {vec4}, i8 }} {call}
{widen}  %j = shl i64 %i, 1
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  ret void
}}

{decl}

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
{sampler_meta}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args}}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
{held}
!7 = !{{i32 {gid_index}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The sample kernels' lanes: each a coordinate (two floats) of a 14 x 14 grid inside the
/// texture, on its edges and past them, at texel centres and between them.
fn sample_lanes() -> Vec<[u32; 4]> {
    const AT: [f32; 14] = [
        -0.3, -0.0625, 0.0, 0.03125, 0.1, 0.2, 0.333, 0.5, 0.55, 0.7, 0.9375, 1.0, 1.1, 1.6,
    ];
    AT.iter()
        .flat_map(|&v| AT.iter().map(move |&u| [u.to_bits(), v.to_bits(), 0, 0]))
        .collect()
}

/// The pipeline of AIR `text`'s kernel `k`.
fn pipeline_of(device: &Device, text: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble_text(text).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("k").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// An 8 x 4 texture of `format` (texels of `bytes` bytes) with its four levels, each level the
/// probe's bytes, as [`compare`] makes it: the runtime's refusal where it makes none.
fn made(device: &Device, format: PixelFormat, bytes: u64) -> Result<Texture, Error> {
    let mut descriptor = TextureDescriptor::texture_2d(format, W, H, true);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor)?;
    assert_eq!(texture.mipmap_level_count(), u64::from(LEVELS));
    for level in 0..LEVELS {
        let (w, h) = extent(level);
        let (w, h) = (u64::from(w), u64::from(h));
        texture
            .replace_region(
                Region::new_2d(0, 0, w, h),
                u64::from(level),
                0,
                &texels(format.as_raw(), level, w * h, bytes),
                w * bytes,
                0,
            )
            .unwrap();
    }
    Ok(texture)
}

/// The precision notes the stage of kernel `k` of [`kernel`] records as the device compiles it
/// under the empty variant key: those the runtime announces as it makes the pipeline state.
fn stage_notes(device: &Device, heap: bool, texel: Texel) -> BTreeSet<PrecisionNote> {
    let compile = device.compile_device();
    let budget = Budget::standard();
    let bitcode = llvm_assemble_text(&kernel(heap, texel)).unwrap();
    let library = Library::from_air(&bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function("k").unwrap());
    let inputs = VariantKey::new().lowering_inputs(Stage::Kernel);
    let compiled = compile
        .compile_function(&function, &inputs, &budget)
        .unwrap();
    let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
    let stage = compile
        .compile_compute_pipeline(&descriptor, &budget)
        .unwrap()
        .stage;
    stage.precision_notes.iter().copied().collect()
}

/// The device's typed refusal in `error`, if it is one.
fn refusal(error: &Error) -> Option<&Unsupported> {
    match error {
        Error::Unsupported(u) | Error::Bind(BindError::Unsupported(u)) => Some(u),
        _ => None,
    }
}

/// Each lane's texel and queries (eight words) of `pipeline` reading `texture` at a slot, or
/// where `heap` through an argument buffer.
fn dispatch(
    device: &Device,
    (pipeline, heap): (&ComputePipelineState, bool),
    texture: &Texture,
) -> Result<Vec<[u32; 8]>, Error> {
    dispatch_with(device, (pipeline, heap), (texture, None), &lanes())
}

/// Each lane's eight output words of `pipeline` reading `texture` at a slot, or where `heap`
/// through an argument buffer, with `sampler` at sampler slot 0 where there is one, buffer 0
/// holding `lanes`.
fn dispatch_with(
    device: &Device,
    (pipeline, heap): (&ComputePipelineState, bool),
    (texture, sampler): (&Texture, Option<&SamplerState>),
    lanes: &[[u32; 4]],
) -> Result<Vec<[u32; 8]>, Error> {
    let n = lanes.len() as u64;
    let input: Vec<u8> = lanes
        .iter()
        .flatten()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let input = buffer_with(device, &input).unwrap();
    let out = buffer_with(device, &vec![0x5a; (n * 32) as usize]).unwrap();
    let argument = buffer_with(device, &texture.gpu_resource_id().0.to_le_bytes()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    let encoded = (|| {
        encoder.set_compute_pipeline_state(pipeline)?;
        match heap {
            true => {
                encoder.use_resource(texture.as_resource(), ResourceUsage::READ)?;
                encoder.set_buffer(Some(&argument), 0, 2)?;
            }
            false => encoder.set_texture(Some(texture), 0)?,
        }
        if let Some(sampler) = sampler {
            encoder.set_sampler_state(Some(sampler), 0)?;
        }
        encoder.set_buffer(Some(&input), 0, 0)?;
        encoder.set_buffer(Some(&out), 0, 1)?;
        encoder.dispatch_threads(Size::new(n, 1, 1), Size::new(n.min(64), 1, 1))
    })();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    encoded?;
    let bytes = read_buffer(&out, 0, n * 32).unwrap();
    Ok(bytes
        .as_chunks::<32>()
        .0
        .iter()
        .map(|v| {
            std::array::from_fn(|k| u32::from_le_bytes(v[4 * k..4 * k + 4].try_into().unwrap()))
        })
        .collect())
}

/// What one test's formats came to.
#[derive(Default)]
struct Tally {
    checked: usize,
    differ: usize,
    refused: usize,
    /// Lanes that differ under an announcement (the module docs).
    announced: usize,
    /// The kinds the device announced meanwhile.
    kinds: BTreeSet<Degradation>,
}

/// Compare every format of `formats` (each with its texel's bytes and its reads) through an
/// argument buffer with a slot on `device`: a format `covered` says the heap reads is counted,
/// any other printed alone.
fn compare(
    (device, messages): (&Device, &Messages),
    formats: &[(PixelFormat, u64, &[Texel])],
    covered: impl Fn(PixelFormat) -> bool,
) -> Tally {
    let mut pipelines: BTreeMap<(bool, Texel), ComputePipelineState> = BTreeMap::new();
    // The kinds the device announced, and each heap pipeline's stage's notes.
    let mut announced: BTreeSet<Degradation> = BTreeSet::new();
    let mut notes: BTreeMap<Texel, BTreeSet<PrecisionNote>> = BTreeMap::new();
    let lanes = lanes();
    let mut tally = Tally::default();
    for &(format, bytes, reads) in formats {
        // A texture the runtime does not make (the padded formats' plain creation, the refusal) is
        // printed and counted apart: a refusal, not an answer.
        let texture = match made(device, format, bytes) {
            Ok(texture) => texture,
            Err(error) => match refusal(&error) {
                Some(u) => {
                    println!(
                        "HEAP-READ-REFUSED {}: {format:?} texture: {:?} {:?}",
                        device.name(),
                        u.feature,
                        u.needs
                    );
                    tally.refused += 1;
                    continue;
                }
                None => panic!("{format:?}: {error}"),
            },
        };
        for &texel in reads {
            let mut state = |heap: bool| {
                pipelines
                    .entry((heap, texel))
                    .or_insert_with(|| pipeline(device, heap, texel))
                    .clone()
            };
            let (at_slot, at_heap) = (state(false), state(true));
            announced.extend(
                messages
                    .take_notices()
                    .iter()
                    .filter_map(|m| m.degradation.as_ref())
                    .map(|n| n.degradation),
            );
            let noted = notes
                .entry(texel)
                .or_insert_with(|| stage_notes(device, true, texel));
            // The device's decode of an 8-bit sRGB texture's texels, announced.
            let srgb = PrecisionNote::SrgbTexelDecode;
            let decodes = srgb_raw_twin(format).is_some()
                && noted.contains(&srgb)
                && announced.contains(&Degradation::from_precision_note(srgb));
            let slot = dispatch(device, (&at_slot, false), &texture)
                .unwrap_or_else(|e| panic!("{format:?} {texel:?} at a slot: {e}"));
            let held = match dispatch(device, (&at_heap, true), &texture) {
                Ok(held) => held,
                Err(error) => match (covered(format), refusal(&error)) {
                    (false, Some(u)) => {
                        println!(
                            "HEAP-READ-UNCOVERED {}: {format:?} {texel:?}: refused {:?} {:?}",
                            device.name(),
                            u.feature,
                            u.needs
                        );
                        continue;
                    }
                    _ => panic!("{format:?} {texel:?} through an argument buffer: {error}"),
                },
            };
            let wrong: Vec<usize> = (0..lanes.len()).filter(|&i| held[i] != slot[i]).collect();
            if !covered(format) {
                println!(
                    "HEAP-READ-UNCOVERED {}: {format:?} {texel:?}: answered, {} of {} lanes not \
                     the slot's",
                    device.name(),
                    wrong.len(),
                    lanes.len()
                );
                continue;
            }
            tally.checked += lanes.len();
            let texel_words = |r: &[u32; 8]| r[..4] != [0; 4];
            let (some_slot, some_heap) = lanes.iter().zip(slot.iter().zip(&held)).fold(
                (false, false),
                |(s, h), (&lane, (a, b))| {
                    let inside = inside(lane);
                    (
                        s || (inside && texel_words(a)),
                        h || (inside && texel_words(b)),
                    )
                },
            );
            if some_slot && !some_heap {
                println!(
                    "HEAP-READ-ZEROS {}: {format:?} {texel:?}: every heap read inside the texture \
                     is zero, the slot's not",
                    device.name()
                );
            }
            if let Some(&i) = wrong.first() {
                let mark = match decodes {
                    true => {
                        tally.announced += wrong.len();
                        "HEAP-READ-ANNOUNCED"
                    }
                    false => {
                        tally.differ += wrong.len();
                        "HEAP-READ-DIFFERS"
                    }
                };
                println!(
                    "{mark} {}: {format:?} {texel:?}: {} of {} lanes, first at {:?}: {:08x?}, \
                     the slot's {:08x?}",
                    device.name(),
                    wrong.len(),
                    lanes.len(),
                    lanes[i],
                    held[i],
                    slot[i]
                );
            }
        }
    }
    announced.extend(
        messages
            .take_notices()
            .iter()
            .filter_map(|m| m.degradation.as_ref())
            .map(|n| n.degradation),
    );
    tally.kinds = announced;
    tally
}

/// Each read and query of each texture of a form a heap reads, through an argument buffer, is
/// the same texture's through a slot, bit for bit, for `float` and `half` results.
#[test]
fn heap_extended_reads_are_the_slot_s() {
    on_device(|device, messages| {
        let floats: &[Texel] = &[Texel::Float, Texel::Half];
        let formats: Vec<(PixelFormat, u64, &[Texel])> = EXTENDED
            .iter()
            .map(|&(format, bytes)| (format, bytes, floats))
            .collect();
        let tally = compare((device, messages), &formats, |format| {
            ExtendedRange::of(format)
                .and_then(ExtendedRange::heap_read)
                .is_some()
        });
        println!(
            "HEAP-READ extended {}: {} lanes checked, {} differ, {} announced, {} textures \
             refused",
            device.name(),
            tally.checked,
            tally.differ,
            tally.announced,
            tally.refused
        );
        assert!(tally.checked > 0);
        assert_eq!(tally.differ, 0, "heap reads differ from the slot's");
    });
}

/// The device's heap set holds each texture of a form a heap reads as a binding plan's arrays do
/// (`bind::heap_element`: its raw view in the unsigned integer class, the neutral in the float
/// one), so placing one announces nothing (no `UnnamedTextureUnviewable`, which a consumer
/// writing every live texture's elements through `heap_element` would meet), and its `float`
/// reads through an argument buffer are the slot's.
#[test]
fn the_heap_set_holds_the_forms_a_heap_reads_unannounced() {
    on_device(|device, messages| {
        let floats: &[Texel] = &[Texel::Float];
        let formats: Vec<(PixelFormat, u64, &[Texel])> = EXTENDED
            .iter()
            .filter(|&&(format, _)| {
                ExtendedRange::of(format)
                    .and_then(ExtendedRange::heap_read)
                    .is_some()
            })
            .map(|&(format, bytes)| (format, bytes, floats))
            .collect();
        let tally = compare((device, messages), &formats, |_| true);
        println!(
            "HEAP-READ held {}: {} lanes checked, {} differ, {} textures refused, kinds {:?}",
            device.name(),
            tally.checked,
            tally.differ,
            tally.refused,
            tally.kinds
        );
        assert!(tally.checked > 0);
        assert_eq!(tally.differ + tally.announced, 0);
        assert!(
            !tally.kinds.contains(&Degradation::UnnamedTextureUnviewable),
            "{:?}",
            tally.kinds
        );
    });
}

/// The kinds among `messages`' notices delivered so far, each with its notice's object.
fn notices(messages: &Messages) -> Vec<(Degradation, String)> {
    messages
        .take_notices()
        .iter()
        .filter_map(|m| m.degradation.as_ref())
        .map(|n| (n.degradation, n.object.clone()))
        .collect()
}

/// The texture formats a heap holds by their raw texels alone that the runtime makes (the
/// padded ones' creation is refused), with their texels' bytes.
const RAW_HELD: [(PixelFormat, u64); 5] = [
    (PixelFormat::BGRA10XR, 8),
    (PixelFormat::BGR10XR, 4),
    (PixelFormat::R16Bfloat, 2),
    (PixelFormat::RG16Bfloat, 4),
    (PixelFormat::RGBA16Bfloat, 8),
];

/// Each sample and gather of each texture a heap holds by its raw texels alone, through an
/// argument buffer, is the same texture's through a slot, bit for bit, whose answers are
/// Metal's: `float` and `half` samples and `float` gathers of the red and alpha components through
/// each constexpr sampler of [`SAMPLERS`], and a linear sample through a runtime sampler (clamped
/// to the edge, no mip filter) at sampler slot 0, at [`sample_lanes`].
#[test]
fn heap_extended_samples_are_the_slot_s() {
    on_device(|device, messages| {
        let lanes = sample_lanes();
        let mut runtime = SamplerDescriptor::new();
        runtime.min_filter = SamplerMinMagFilter::Linear;
        runtime.mag_filter = SamplerMinMagFilter::Linear;
        runtime.s_address_mode = SamplerAddressMode::ClampToEdge;
        runtime.t_address_mode = SamplerAddressMode::ClampToEdge;
        let runtime = device.new_sampler_state(&runtime).unwrap();
        let mut cases: Vec<(String, Through, Op, bool)> = Vec::new();
        for (name, words) in SAMPLERS {
            for (op, half) in [
                (Op::Sample, false),
                (Op::Sample, true),
                (Op::Gather(0), false),
                (Op::Gather(3), false),
            ] {
                cases.push((name.to_owned(), Through::Constexpr(words), op, half));
            }
        }
        cases.push((
            "runtime linear edge".to_owned(),
            Through::Slot,
            Op::Sample,
            false,
        ));
        cases.push((
            "runtime linear edge".to_owned(),
            Through::Slot,
            Op::Sample,
            true,
        ));
        let (mut checked, mut differ) = (0, 0);
        let _ = messages.take_notices();
        for (name, through, op, half) in &cases {
            let at_slot = pipeline_of(device, &sample_kernel(false, *through, *op, *half));
            let at_heap = pipeline_of(device, &sample_kernel(true, *through, *op, *half));
            let sampler = matches!(through, Through::Slot).then_some(&runtime);
            for &(format, bytes) in &RAW_HELD {
                let texture = made(device, format, bytes).unwrap();
                let slot = dispatch_with(device, (&at_slot, false), (&texture, sampler), &lanes)
                    .unwrap_or_else(|e| panic!("{format:?} {name} {op:?} at a slot: {e}"));
                let held = dispatch_with(device, (&at_heap, true), (&texture, sampler), &lanes)
                    .unwrap_or_else(|e| panic!("{format:?} {name} {op:?} through a heap: {e}"));
                checked += lanes.len();
                let wrong: Vec<usize> = (0..lanes.len()).filter(|&i| held[i] != slot[i]).collect();
                if let Some(&i) = wrong.first() {
                    differ += wrong.len();
                    println!(
                        "HEAP-SAMPLE-DIFFERS {}: {format:?} {name} {op:?} half {half}: {} of {} \
                         lanes, first at {:?}: {:08x?}, the slot's {:08x?}",
                        device.name(),
                        wrong.len(),
                        lanes.len(),
                        lanes[i],
                        &held[i][..4],
                        &slot[i][..4]
                    );
                }
            }
        }
        let announced = notices(messages);
        println!(
            "HEAP-SAMPLE extended {}: {} cases, {checked} lanes checked, {differ} differ, \
             announced {announced:?}",
            device.name(),
            cases.len()
        );
        assert!(checked > 0);
        assert_eq!(differ, 0, "heap samples differ from the slot's");
    });
}

/// A command whose function samples a texture a heap holds by its raw texels alone in a way the
/// function does not emulate, reached through an argument buffer (made resident), is refused at
/// bind (`Unsupported(ArgumentBufferTextures)`, `RuntimeFact::HeapRawSamples`): a nearest mip
/// filter's sample, a `half` gather, and a sample through a runtime sampler whose mip filter is
/// linear. The same command over an ordinary `RGBA8Unorm` texture runs, and so does one that
/// only reads the raw texture.
#[test]
fn a_heap_sample_the_function_does_not_emulate_refuses_a_raw_texture() {
    on_device(|device, _| {
        let mut mipmapped = SamplerDescriptor::new();
        mipmapped.min_filter = SamplerMinMagFilter::Linear;
        mipmapped.mag_filter = SamplerMinMagFilter::Linear;
        mipmapped.mip_filter = SamplerMipFilter::Linear;
        let mipmapped = device.new_sampler_state(&mipmapped).unwrap();
        let plain = made(device, PixelFormat::RGBA8Unorm, 4).unwrap();
        let raw = made(device, PixelFormat::BGRA10XR, 8).unwrap();
        let lanes = sample_lanes();
        let read = pipeline(device, true, Texel::Float);
        dispatch(device, (&read, true), &raw).unwrap();
        let cases = [
            (
                "nearest mip filter",
                Through::Constexpr(sampler_words(1, false, 0, true)),
                Op::Sample,
                false,
                None,
            ),
            (
                "half gather",
                Through::Constexpr(sampler_words(1, false, 0, false)),
                Op::Gather(0),
                true,
                None,
            ),
            (
                "runtime linear mip filter",
                Through::Slot,
                Op::Sample,
                false,
                Some(&mipmapped),
            ),
        ];
        for (name, through, op, half, sampler) in cases {
            let at_heap = pipeline_of(device, &sample_kernel(true, through, op, half));
            dispatch_with(device, (&at_heap, true), (&plain, sampler), &lanes)
                .unwrap_or_else(|e| panic!("{name} of an RGBA8Unorm texture: {e}"));
            let refused =
                dispatch_with(device, (&at_heap, true), (&raw, sampler), &lanes).expect_err(name);
            println!("HEAP-SAMPLE-REFUSED {}: {name}: {refused:?}", device.name());
            assert!(
                matches!(
                    refusal(&refused).map(|u| &u.feature),
                    Some(Feature::ArgumentBufferTextures)
                ),
                "{name}: {refused:?}"
            );
        }
        drop((plain, raw));
    });
}

/// A `BGR10_XR` texture over a buffer (as the consumer makes a surface's first plane: over a
/// no-copy buffer, the consumer's, where the device imports the pages, and over an ordinary
/// one), live beside commands that sample through an argument buffer. A command that makes the
/// view resident (`useResource:`) and samples it is answered as its slot answers (a nearest
/// constexpr sampler, which the function emulates; twice) or refused at bind
/// (`Unsupported(ArgumentBufferTextures)`, a nearest mip filter), and announces nothing: its
/// shadow, where the device shadows the view, is copied in. A command that samples an ordinary
/// `RGBA8Unorm` texture through the argument buffer, binding the heap set that holds the view
/// without naming it or making it resident, runs, and where the view is a shadow announces
/// `UnnamedShadowedTexture` naming it, once per device (its function may read the shadow stale);
/// where the view is not, it announces nothing. Each view is let go before the next is made, so a
/// command meets one view at a time. What making a view announces is its own (`ShadowedTexture`
/// where the device shadows it, once per device: a later view's shadow is then unknown, and its
/// command announces nothing, the notice spent).
#[test]
fn a_raw_texture_over_a_buffer_announces_its_shadow_only_where_not_named() {
    on_device(|device, messages| {
        let lanes = sample_lanes();
        let nearest = Through::Constexpr(sampler_words(1, false, 0, false));
        let mip = Through::Constexpr(sampler_words(1, false, 0, true));
        let emulated = pipeline_of(device, &sample_kernel(true, nearest, Op::Sample, false));
        let refused = pipeline_of(device, &sample_kernel(true, mip, Op::Sample, false));
        let at_slot = pipeline_of(device, &sample_kernel(false, nearest, Op::Sample, false));
        let plain = made(device, PixelFormat::RGBA8Unorm, 4).unwrap();
        let unnamed = |phase: &str| {
            for (name, pipeline) in [("nearest", &emulated), ("nearest mip filter", &refused)] {
                dispatch_with(device, (pipeline, true), (&plain, None), &lanes)
                    .unwrap_or_else(|e| panic!("{name} of an RGBA8Unorm texture {phase}: {e}"));
            }
        };
        unnamed("before the views live");
        let before = notices(messages);
        println!(
            "HEAP-SAMPLE-VIEWS {}: before the views {before:?}",
            device.name()
        );
        // Four rows of 256 bytes, each 8 texels of 4 bytes and the rest of the row.
        let bytes = texels(PixelFormat::BGR10XR.as_raw(), 0, 4 * 64, 4);
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGR10XR, W, H, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::shared();
        let memory = Mapping::anonymous(GUEST_PAGE);
        assert!(bytes.len() <= GUEST_PAGE);
        // SAFETY: the mapping holds `GUEST_PAGE` bytes, at least `bytes.len()`, and nothing
        // reads or writes them meanwhile.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), memory.pointer().as_ptr(), bytes.len())
        };
        // Whether the device's shadow notices are spent (each is announced once per device).
        let (mut made_spent, mut unnamed_spent) = (false, false);
        for name in ["no-copy", "ordinary"] {
            let buffer = match name {
                // SAFETY: the mapping outlives the buffer and its view, dropped first.
                "no-copy" => match unsafe {
                    device.new_buffer_with_bytes_no_copy(
                        HostMemory::new(memory.pointer()),
                        GUEST_PAGE as u64,
                        ResourceOptions::shared(),
                        None,
                    )
                } {
                    Err(error) if not_importable(&error) => {
                        println!("HEAP-SAMPLE-VIEW {} {name}: not imported", device.name());
                        continue;
                    }
                    buffer => buffer.unwrap(),
                },
                _ => buffer_with(device, &bytes).unwrap(),
            };
            let view = buffer.new_texture(&descriptor, 0, 256).unwrap();
            let making = notices(messages);
            let made_shadow = making
                .iter()
                .any(|(k, _)| *k == Degradation::ShadowedTexture);
            // `None` where the notice was spent by an earlier view.
            let shadow = match made_spent {
                true => None,
                false => Some(made_shadow),
            };
            made_spent |= made_shadow;
            let slot = dispatch_with(device, (&at_slot, false), (&view, None), &lanes).unwrap();
            let _ = notices(messages);
            let mut held = Vec::new();
            for _ in 0..2 {
                held.push(
                    dispatch_with(device, (&emulated, true), (&view, None), &lanes)
                        .unwrap_or_else(|e| panic!("the {name} view's sample through a heap: {e}")),
                );
            }
            let refusal_of = dispatch_with(device, (&refused, true), (&view, None), &lanes)
                .expect_err("the view's mip-filtered sample through a heap");
            let resident = notices(messages);
            unnamed("beside the view");
            let beside = notices(messages);
            unnamed("beside the view again");
            let again = notices(messages);
            let wrong = (0..lanes.len())
                .filter(|&i| held.iter().any(|h| h[i] != slot[i]))
                .count();
            println!(
                "HEAP-SAMPLE-VIEW {} {name}: making it {making:?} (shadow {shadow:?}); resident: \
                 {wrong} of {} lanes differ from the slot's, the mip-filtered one {refusal_of:?}, \
                 announced {resident:?}; beside it {beside:?}, again {again:?}",
                device.name(),
                lanes.len()
            );
            assert_eq!(
                wrong, 0,
                "the {name} view's heap sample differs from its slot's"
            );
            assert!(
                matches!(
                    refusal(&refusal_of).map(|u| &u.feature),
                    Some(Feature::ArgumentBufferTextures)
                ),
                "{refusal_of:?}"
            );
            let shadows = |n: &[(Degradation, String)]| {
                n.iter()
                    .filter(|(k, _)| *k == Degradation::UnnamedShadowedTexture)
                    .cloned()
                    .collect::<Vec<_>>()
            };
            assert_eq!(shadows(&resident), [], "the {name} view resident");
            assert_eq!(shadows(&again), [], "the {name} view unnamed again");
            let named = (
                Degradation::UnnamedShadowedTexture,
                format!("texture at resource ID {}", view.gpu_resource_id().0),
            );
            let expected = match (shadow, unnamed_spent) {
                (Some(true), false) => vec![named],
                _ => Vec::new(),
            };
            assert_eq!(beside, expected, "the {name} view unnamed");
            unnamed_spent |= !shadows(&beside).is_empty();
            drop((view, buffer));
        }
        drop(plain);
        drop(memory);
    });
}

/// Each read and query of each RGBX texture and its RGBA twin, through an argument buffer, is
/// the same texture's through a slot, bit for bit: `float` and `half` reads of a float format,
/// `uint` or `int` reads of an integer one (the consumer's report of integer RGBX textures
/// reading zeros through argument buffers); an 8-bit sRGB one's where the device decodes it,
/// announced (the module docs).
#[test]
fn heap_rgbx_reads_are_the_slot_s() {
    on_device(|device, messages| {
        let formats: Vec<(PixelFormat, u64, &[Texel])> = RGBX
            .iter()
            .map(|&(format, bytes, integer)| (format, bytes, texels_of(integer)))
            .collect();
        let tally = compare((device, messages), &formats, |_| true);
        println!(
            "HEAP-READ rgbx {}: {} lanes checked, {} differ, {} announced, {} textures refused",
            device.name(),
            tally.checked,
            tally.differ,
            tally.announced,
            tally.refused
        );
        assert!(tally.checked > 0);
        assert_eq!(tally.differ, 0, "heap reads differ from the slot's");
    });
}
