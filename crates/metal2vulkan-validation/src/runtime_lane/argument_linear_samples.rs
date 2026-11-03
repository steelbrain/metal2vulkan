//! A device test of a heap texture's linear sample on the device the cfg selects, under the
//! validation layer: a texture an argument buffer holds, sampled linearly at level 0 through a
//! constexpr sampler, a runtime sampler slot, or the sampler the argument buffer holds beside it,
//! answers every lane as the same texture bound to a slot and sampled through the constexpr sampler
//! or the runtime sampler of the same state.
//!
//! Metal filters a heap texture as a slot's, and through an argument buffer's sampler as through
//! a constexpr or a runtime sampler of the same state (Apple M4). Both samples are the device's
//! own sampler's: on MoltenVK, Metal's own; on a device whose sampler is not Metal's, the device's
//! filter, which the lane judges within [`DEVICE_FILTER`] ([`device_filters`]).
//!
//! The textures are 8x8 (2D) and 4x4x4 (3D) of five formats of measured classes (8- and 16-bit
//! normalized, half-float, 32-bit float, and a packed small float), their texels deterministic
//! noise; each is sampled at coordinates across and past its edges, with `float` and `half`
//! results, through linear samplers of every address mode, a border colour, and (2D) pixel
//! coordinates; but a 3D texture's mirror-clamped axes, which no probe measured.

use super::degrade_notices::counts as notice_counts;
use super::depth_compares::{pipeline_of, run_held, Held, Through};
use super::driver_deviations::stage_notes;
use super::extended_range_targets::half_bits;
use super::{layer_reported_nothing_at_all, validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkProperty;
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, TextureType,
};
use metal2vulkan::pipeline::PrecisionNote;
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    ComputePipelineState, Degradation, Device, Region, SamplerState, Texture, TextureDescriptor,
};
use std::collections::BTreeMap;

/// A constexpr sampler's words: linear filters, no mip filter, normalized coordinates clamped to
/// the edge.
pub(super) const LINEAR_EDGE: u64 = 0x007b_ff00_0008_0a49;

/// The formats sampled: one of each measured class's family.
const FORMATS: [PixelFormat; 5] = [
    PixelFormat::RGBA8Unorm,
    PixelFormat::RGBA16Unorm,
    PixelFormat::RGBA16Float,
    PixelFormat::RGBA32Float,
    PixelFormat::RG11B10Float,
];

/// The runtime samplers' states, each of linear filters and no mip filter: an address mode on
/// every axis, the border colour it reads, and whether its coordinates are normalized.
const STATES: [(SamplerAddressMode, SamplerBorderColor, bool); 7] = [
    (
        SamplerAddressMode::ClampToEdge,
        SamplerBorderColor::TransparentBlack,
        true,
    ),
    (
        SamplerAddressMode::Repeat,
        SamplerBorderColor::TransparentBlack,
        true,
    ),
    (
        SamplerAddressMode::MirrorRepeat,
        SamplerBorderColor::TransparentBlack,
        true,
    ),
    (
        SamplerAddressMode::MirrorClampToEdge,
        SamplerBorderColor::TransparentBlack,
        true,
    ),
    (
        SamplerAddressMode::ClampToZero,
        SamplerBorderColor::TransparentBlack,
        true,
    ),
    (
        SamplerAddressMode::ClampToBorderColor,
        SamplerBorderColor::OpaqueWhite,
        true,
    ),
    (
        SamplerAddressMode::ClampToEdge,
        SamplerBorderColor::TransparentBlack,
        false,
    ),
];

/// Whether the device's own sampler answers a filtered sample where Metal's would differ
/// (`VkProperty::DeviceSampling`: a device whose sampler is not Metal's). Its answers are then
/// Metal's only within the bound `PrecisionNote::DeviceSampling` records, and a lane that compares
/// them judges each channel within [`DEVICE_FILTER`]; on a device that samples through Metal's own
/// sampler, bit for bit.
pub(super) fn device_filters(device: &Device) -> bool {
    device
        .compile_device()
        .profile()
        .enabled()
        .properties
        .contains(&VkProperty::DeviceSampling)
}

/// The bound a lane judges the device's filter by: two codes of an 8-bit channel (2/255), above
/// the largest linear filter's difference `PrecisionNote::DeviceSampling` records (1.832 codes,
/// an sRGB texture's on NVIDIA).
pub(super) const DEVICE_FILTER: f32 = 2.0 / 255.0;

/// Whether `got` answers as `want` does: bit for bit, or within [`DEVICE_FILTER`] where `within`.
pub(super) fn alike(got: f32, want: f32, within: bool) -> bool {
    got.to_bits() == want.to_bits() || (within && (got - want).abs() <= DEVICE_FILTER)
}

/// A texture's side: 8 in 2D, 4 in 3D.
fn side(three: bool) -> u64 {
    if three {
        4
    } else {
        8
    }
}

/// Synthetic AIR written for this test: kernel `k` samples a `texture2d` (a `texture3d` where
/// `three`) of `float` (`half` where `half`) at level 0 and buffer 0's lane `gid` (a `float4`
/// whose first two or three components are the coordinate) through `through`, writing the
/// `float4` (the `half4` widened) to buffer 1's lane `gid`. Where `heap`, the texture is the one
/// its argument buffer (buffer 2, `{ texture t; sampler s; }`) holds, else texture slot 0's; a
/// runtime sampler is slot 0's.
pub(super) fn kernel(heap: bool, through: Through, three: bool, half: bool) -> String {
    let (global, states) = match through {
        Through::Constexpr(words) => (
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8\n"
            ),
            "!air.sampler_states = !{!11}\n!11 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        ),
        _ => (String::new(), ""),
    };
    let (param, operand, args, gid) = match through {
        Through::Constexpr(_) => ("", "@__air_sampler_state", "!3, !4, !5, !7", 3),
        Through::Runtime => (", ptr addrspace(2) %rs", "%rs", "!3, !4, !5, !6, !7", 4),
        Through::Argument => ("", "%s", "!3, !4, !5, !7", 3),
    };
    let (dim, n) = if three { ("3d", 3) } else { ("2d", 2) };
    let (element, ty) = if half {
        ("half", "v4f16")
    } else {
        ("float", "v4f32")
    };
    let texture = format!("texture{dim}<{element}, sample>");
    let lanes = (0..n)
        .map(|i| format!("i32 {i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let (holder, load) = match heap {
        true => (
            "ptr addrspace(2) %m",
            format!(
                "  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0\n  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8\n{}",
                match through {
                    Through::Argument => "  %ps = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 1\n  %s = load ptr addrspace(2), ptr addrspace(2) %ps, align 8\n",
                    _ => "",
                }
            ),
        ),
        false => ("ptr addrspace(1) %t", String::new()),
    };
    let held = match heap {
        true => format!(
            r#"!5 = !{{i32 2, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !8, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!8 = !{{i32 0, i32 8, i32 0, !"{texture}", !"t", !"air.indirect_argument", !9, i32 8, i32 8, i32 0, !"sampler", !"s", !"air.indirect_argument", !10}}
!9 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}
!10 = !{{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}"#
        ),
        false => format!(
            r#"!5 = !{{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}"#
        ),
    };
    let widen = match half {
        true => "  %h = extractvalue { <4 x half>, i8 } %r, 0\n  %v = fpext <4 x half> %h to <4 x float>\n",
        false => "  %v = extractvalue { <4 x float>, i8 } %r, 0\n",
    };
    format!(
        r#"%struct.M = type {{ ptr addrspace(1), ptr addrspace(2) }}
{global}
define void @k(ptr addrspace(1) %c, ptr addrspace(1) %out, {holder}{param}, i32 %gid) {{
  %i = zext i32 %gid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i
  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16
  %uv = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <{n} x i32> <{lanes}>
{load}  %r = call {{ <4 x {element}>, i8 }} @air.sample_texture_{dim}.{ty}(ptr addrspace(1) %t, ptr addrspace(2) {operand}, <{n} x float> %uv, i1 true, <{n} x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)
{widen}  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  ret void
}}

declare {{ <4 x {element}>, i8 }} @air.sample_texture_{dim}.{ty}(ptr addrspace(1), ptr addrspace(2), <{n} x float>, i1, <{n} x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args}}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"c"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
{held}
!6 = !{{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"rs"}}
!7 = !{{i32 {gid}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A deterministic noise source (a 64-bit linear congruential generator's high bits).
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    /// A finite float's bits: a sign where `signed`, an exponent field in `exponents`, and
    /// `fraction` bits of fraction (an 8-bit exponent field above 23 of them, else 5 bits).
    fn float(&mut self, signed: bool, exponents: (u32, u32), fraction: u32) -> u32 {
        let width = if fraction == 23 { 8 } else { 5 };
        let r = self.next();
        let exponent = exponents.0 + r % (exponents.1 - exponents.0 + 1);
        let sign = u32::from(signed) & (r >> 31);
        let bits = self.next() & ((1 << fraction) - 1);
        sign << (width + fraction) | exponent << fraction | bits
    }
}

/// The texels of a texture of `format` with `count` texels, as its bytes.
fn texels(format: PixelFormat, count: u64, noise: &mut Noise) -> Vec<u8> {
    let mut bytes = Vec::new();
    for _ in 0..count {
        match format {
            PixelFormat::RGBA8Unorm => bytes.extend(noise.next().to_le_bytes()),
            PixelFormat::RGBA16Unorm => {
                bytes.extend(noise.next().to_le_bytes());
                bytes.extend(noise.next().to_le_bytes());
            }
            // Halves of magnitude 2^-5 to 2^1, either sign.
            PixelFormat::RGBA16Float => {
                for _ in 0..4 {
                    bytes.extend((noise.float(true, (10, 16), 10) as u16).to_le_bytes());
                }
            }
            // Floats of magnitude 2^-7 to 2^2, either sign.
            PixelFormat::RGBA32Float => {
                for _ in 0..4 {
                    bytes.extend(noise.float(true, (120, 129), 23).to_le_bytes());
                }
            }
            // Unsigned 11-, 11- and 10-bit floats of magnitude 2^-5 to 2^2.
            PixelFormat::RG11B10Float => {
                let r = noise.float(false, (10, 17), 6);
                let g = noise.float(false, (10, 17), 6);
                let b = noise.float(false, (10, 17), 5);
                bytes.extend((r | g << 11 | b << 22).to_le_bytes());
            }
            _ => unreachable!("an unlisted format"),
        }
    }
    bytes
}

/// The bytes of one texel of `format`.
fn texel_bytes(format: PixelFormat) -> u64 {
    match format {
        PixelFormat::RGBA8Unorm | PixelFormat::RG11B10Float => 4,
        PixelFormat::RGBA16Unorm | PixelFormat::RGBA16Float => 8,
        _ => 16,
    }
}

/// A texture of `format`, 2D or (where `three`) 3D, its texels noise.
fn texture(device: &Device, format: PixelFormat, three: bool, noise: &mut Noise) -> Texture {
    let n = side(three);
    let mut d = TextureDescriptor::texture_2d(format, n, n, false);
    let depth = if three { n } else { 1 };
    if three {
        d.texture_type = TextureType::Type3D;
        d.depth = n;
    }
    let texture = device.new_texture(&d).unwrap();
    let row = n * texel_bytes(format);
    let bytes = texels(format, n * n * depth, noise);
    let image = if three { row * n } else { 0 };
    texture
        .replace_region(
            Region::new_3d(0, 0, 0, n, n, depth),
            0,
            0,
            &bytes,
            row,
            image,
        )
        .unwrap();
    texture
}

/// The coordinates sampled, each a job's first components: a grid across the texture and past
/// each edge by about a third of it, off the texel centres; scaled to texels where `pixel`.
fn jobs(three: bool, pixel: bool) -> Vec<[f32; 4]> {
    let steps = if three { 11 } else { 29 };
    let at = |k: u32| -0.3 + 1.6 * (k as f32 + 0.37) / steps as f32;
    let scale = if pixel { side(three) as f32 } else { 1.0 };
    let mut jobs = Vec::new();
    for z in 0..if three { steps } else { 1 } {
        for y in 0..steps {
            for x in 0..steps {
                let w = if three { at(z) } else { 0.0 };
                jobs.push([at(x) * scale, at(y) * scale, w * scale, 0.0]);
            }
        }
    }
    jobs
}

/// A runtime sampler of linear filters and no mip filter in state `state`.
fn sampler(
    device: &Device,
    (address, border, normalized): (SamplerAddressMode, SamplerBorderColor, bool),
) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = SamplerMinMagFilter::Linear;
    d.mag_filter = SamplerMinMagFilter::Linear;
    d.s_address_mode = address;
    d.t_address_mode = address;
    d.r_address_mode = address;
    d.border_color = border;
    d.normalized_coordinates = normalized;
    device.new_sampler_state(&d).unwrap()
}

/// The kernels of one shape and result: the slot's and the heap's through a constexpr sampler,
/// and through a runtime sampler slot, and the heap's through its argument buffer's sampler; and
/// whether each heap kernel records `PrecisionNote::HalfTexelConversion` (in that order).
struct Kernels {
    slot_constexpr: ComputePipelineState,
    heap_constexpr: ComputePipelineState,
    slot_runtime: ComputePipelineState,
    heap_runtime: ComputePipelineState,
    heap_argument: ComputePipelineState,
    half_texels: [bool; 3],
}

impl Kernels {
    fn new(device: &Device, three: bool, half: bool) -> Kernels {
        let bitcode = |heap: bool, through: Through| {
            llvm_assemble_text(&kernel(heap, through, three, half)).unwrap()
        };
        let make = |heap: bool, through: Through| pipeline_of(device, &bitcode(heap, through), "k");
        let noted = |through: Through| {
            stage_notes(device, &bitcode(true, through), "k")
                .contains(&PrecisionNote::HalfTexelConversion)
        };
        let constexpr = Through::Constexpr(LINEAR_EDGE);
        Kernels {
            slot_constexpr: make(false, constexpr),
            heap_constexpr: make(true, constexpr),
            slot_runtime: make(false, Through::Runtime),
            heap_runtime: make(true, Through::Runtime),
            heap_argument: make(true, Through::Argument),
            half_texels: [constexpr, Through::Runtime, Through::Argument].map(noted),
        }
    }
}

/// Whether the `half` results `got` and `want` (each widened to `float`) lie within one unit in
/// the last place of each other: `PrecisionNote::HalfTexelConversion`'s bound, a `half` result
/// the device's `float` texel rounded to nearest even where Metal rounds a 32-bit float texel's
/// sample toward zero (`docs/FAST-PATHS.md`).
fn within_half_ulp(got: f32, want: f32) -> bool {
    let rank = |x: f32| {
        let h = half_bits(x);
        match h & 0x8000 {
            0 => i32::from(h),
            _ => -i32::from(h & 0x7FFF),
        }
    };
    !got.is_nan() && !want.is_nan() && (rank(got) - rank(want)).abs() <= 1
}

/// Every linear sample of a heap texture (2D and 3D, `float` and `half` results; through a
/// constexpr sampler, a runtime sampler slot, and its argument buffer's sampler) answers each lane
/// bit for bit as the slot texture's sample through the same state (module docs), but within
/// [`DEVICE_FILTER`] where the device's own sampler filters (`device_filters`), and a `half`
/// result within one unit in its last place where the heap kernel records
/// `PrecisionNote::HalfTexelConversion` ([`within_half_ulp`]): through the argument buffer's
/// sampler on the Apple M3 Max through MoltenVK 1.4.2, a 32-bit float texture's `half` sample is
/// the `float` one rounded to nearest even, where the slot's is Metal's, rounded toward zero.
#[test]
fn heap_linear_samples_answer_as_slot_samples() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let within = device_filters(&device);
        let mut noise = Noise(0x5eed_d499);
        let mut compared = 0usize;
        // The `half` lanes a unit in the last place apart where the heap kernel records it.
        let mut half_ulps = 0usize;
        // The differing lanes by group (format, shape, result, sampler), with the first few.
        let mut wrong: BTreeMap<String, (usize, Vec<String>)> = BTreeMap::new();
        for three in [false, true] {
            let kernels = [false, true].map(|half| (half, Kernels::new(&device, three, half)));
            for format in FORMATS {
                let texture = texture(&device, format, three, &mut noise);
                let mut judge =
                    |what: String, noted: bool, jobs: &[[f32; 4]], slot: &[f32], heap: &[f32]| {
                        for (j, job) in jobs.iter().enumerate() {
                            for lane in 0..4 {
                                let (want, got) = (slot[4 * j + lane], heap[4 * j + lane]);
                                compared += 1;
                                if noted && !alike(got, want, within) && within_half_ulp(got, want)
                                {
                                    half_ulps += 1;
                                } else if !alike(got, want, within) {
                                    let shape = if three { "3D" } else { "2D" };
                                    let group = wrong
                                        .entry(format!("{format:?} {shape} {what}"))
                                        .or_default();
                                    group.0 += 1;
                                    if group.1.len() < 3 {
                                        group.1.push(format!(
                                            "at {:?} lane {lane}: {got:e}, the slot's {want:e}",
                                            &job[..if three { 3 } else { 2 }],
                                        ));
                                    }
                                }
                            }
                        }
                    };
                for (half, k) in &kernels {
                    let result = if *half { "half" } else { "float" };
                    let normalized = jobs(three, false);
                    let slot = run_held(
                        &device,
                        (&texture, Held::Slot, None),
                        &k.slot_constexpr,
                        &normalized,
                        1,
                    );
                    let heap = run_held(
                        &device,
                        (&texture, Held::Argument(None), None),
                        &k.heap_constexpr,
                        &normalized,
                        1,
                    );
                    judge(
                        format!("{result} constexpr sampler"),
                        *half && k.half_texels[0],
                        &normalized,
                        &slot,
                        &heap,
                    );
                    for state in STATES {
                        // Metal reads pixel coordinates of 1D and 2D textures alone; no probe
                        // measured a 3D texture's mirror-clamped axis.
                        if three && (!state.2 || state.0 == SamplerAddressMode::MirrorClampToEdge) {
                            continue;
                        }
                        let s = sampler(&device, state);
                        let jobs = jobs(three, !state.2);
                        let slot = run_held(
                            &device,
                            (&texture, Held::Slot, Some(&s)),
                            &k.slot_runtime,
                            &jobs,
                            1,
                        );
                        for (what, held, runtime, state_of, noted) in [
                            (
                                "runtime sampler",
                                Held::Argument(None),
                                Some(&s),
                                &k.heap_runtime,
                                k.half_texels[1],
                            ),
                            (
                                "argument buffer's sampler",
                                Held::Argument(Some(&s)),
                                None,
                                &k.heap_argument,
                                k.half_texels[2],
                            ),
                        ] {
                            let heap =
                                run_held(&device, (&texture, held, runtime), state_of, &jobs, 1);
                            judge(
                                format!("{result} {what} {state:?}"),
                                *half && noted,
                                &jobs,
                                &slot,
                                &heap,
                            );
                        }
                    }
                }
            }
        }
        let differ: usize = wrong.values().map(|g| g.0).sum();
        let groups: Vec<String> = wrong
            .iter()
            .map(|(group, (n, first))| format!("{group}: {n} lanes\n  {}", first.join("\n  ")))
            .collect();
        eprintln!(
            "{name}: {differ} of {compared} lanes differ, {half_ulps} more `half` lanes a unit in \
             the last place apart under a recorded half-texel-conversion"
        );
        assert!(
            wrong.is_empty(),
            "{name}: {differ} of {compared} lanes differ, in {} groups:\n{}",
            groups.len(),
            groups.join("\n")
        );
        let announced = notice_counts(&messages.take_notices());
        let note = Degradation::from_precision_note(PrecisionNote::HalfTexelConversion);
        assert!(
            half_ulps == 0 || announced.contains_key(&note),
            "{name}: the half-texel-conversion answers, unannounced: {announced:?}"
        );
    }
    layer_reported_nothing_at_all(&messages);
}
