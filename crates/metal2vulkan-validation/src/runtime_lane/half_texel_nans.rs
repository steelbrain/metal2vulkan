//! The NaNs of float texels read, sampled and gathered with `half` and `float` results, on the
//! device the cfg selects, under the validation layer, under the lane's exact policy and under the
//! default one (native device sampling and maths, what a consumer runs).
//!
//! What plain Metal answers (our own probe: an Apple M3 Max under macOS 26.7,
//! safe and fast math alike, every NaN lane of the eight formats below): a lane is a NaN where a
//! tap of nonzero weight is one (a linear sample at a texel's centre reads that texel alone), or
//! where the taps of nonzero weight hold both infinities. Its bits:
//!
//! - with a `half` result, the texel's own bits for a read, a nearest sample or a gather of a
//!   16-bit float texel (`R16Float`, `RG16Float`, `RGBA16Float`), its sign, payload and
//!   signalling bit kept; `0x7e00` everywhere else: a 32-bit float texel's NaN, a small float's
//!   (`RG11B10Float`), and every linear sample's;
//! - with a `float` result, a 32-bit float texel's own bits for a read, a nearest sample or a
//!   gather; `0x7fc00000` everywhere else.
//!
//! `RGB9E5Float` decodes every code to a finite value.
//!
//! The kernels are this lane's own AIR: each lane `x` reads texel `(x, 0)` of an [`N`] by 1
//! texture holding [`texels`], samples it at the texel's centre (nearest or linear), a quarter
//! texel right of it (linear: weights 3/4 and 1/4 on texels `x` and `x + 1`), or gathers red at
//! the corner between texels `x` and `x + 1`, through a constexpr sampler or the runtime sampler
//! at slot 0, and stores the `half4` or `float4` to buffer 1. Every lane Metal answers a NaN is
//! compared bit for bit, and every other lane must not be a NaN; a lane that differs is printed
//! on a `HALF-TEXEL-DIFFERS` line. Under the default policy a `float` result is the device's own
//! texel (native device sampling and texel conversion): its NaN lanes are printed on
//! `HALF-TEXEL-RAW` lines, the device's decode itself.
//!
//! A NaN's bits are the device's (`device-nan-encoding`, `native-nan-encoding`): Metal's only where
//! the device answers them (a 16-bit float texel's one-texel accesses where the device widens a
//! half exactly, NVIDIA), and whether a lane is a NaN at all is the device's filter's (a
//! zero-weight tap of an infinity or a NaN may make one); both are measured and printed, not
//! asserted.
//!
//! Under every policy a `half` lane whose value is a finite `float` beyond `half`'s range (the
//! largest `float`, alone or filtered) is infinity of its sign, or 65504 where the device's
//! conversion rounds toward zero (the native conversion's recorded bound); a lane that is not is
//! printed on a `HALF-TEXEL-OVERFLOW` line, and the test fails. On NVIDIA and lavapipe the
//! device folds native half rounding's conversion pair (no `NoContraction`), so such a value
//! reaches the store's bits unrounded; previously it stored about `1.0` and carried into the other
//! lane of its pair (a NaN `0x7e00` read `0x7e02`).

use super::degrade_notices::{counts, on_device_with};
use super::{at_plan, buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, DeviceOptions, PipelineOptions, Region,
    ResourceOptions, SamplerState, Size, Texture, TextureDescriptor,
};

/// The halves of the 16-bit float formats: quiet and signalling NaNs of both signs and payload
/// edges, then infinities, denormals, zeros and normals (our probe's).
const V: [u16; 24] = [
    0x7e00, 0x7c01, 0x7dff, 0x7fff, 0x7e01, 0x7d55, 0xfe00, 0xfc01, 0xfdff, 0xffff, 0xfe01, 0x7f00,
    0x7c00, 0xfc00, 0x0001, 0x8001, 0x03ff, 0x3c00, 0x0000, 0x8000, 0x7bff, 0xfbff, 0x3555, 0xff00,
];

/// The words of the 32-bit float formats: NaNs of every class, then infinities, denormals, zeros
/// and normals.
const F: [u32; 24] = [
    0x7fc00000, 0x7f800001, 0x7fbfffff, 0xffc00001, 0xff800001, 0x7fffffff, 0x7fc12345, 0xffffffff,
    0x7f800000, 0x00000001, 0x80000001, 0x3f800000, 0x7fe00000, 0xffc00000, 0x7f802000, 0x00000000,
    0x7fc02000, 0x7fa00000, 0xff800000, 0x80000000, 0x7f7fffff, 0x3eaaaaab, 0x7fd00000, 0xfff00000,
];

/// `RG11B10Float`'s 11-bit red and green codes: NaNs, infinity, normals, denormals, zero.
const R11: [u32; 12] = [
    0x7C1, 0x7FF, 0x7E0, 0x7D5, 0x7C0, 0x3C0, 0x000, 0x001, 0x7BF, 0x400, 0x5AA, 0x7F0,
];

/// `RG11B10Float`'s 10-bit blue codes.
const B10: [u32; 12] = [
    0x3E1, 0x3FF, 0x3F0, 0x3EA, 0x3E0, 0x1E0, 0x000, 0x001, 0x3DF, 0x200, 0x2AA, 0x3F8,
];

/// `RGB9E5Float`'s shared exponents, the largest among them.
const E5: [u32; 8] = [31, 30, 15, 0, 31, 16, 24, 31];

/// The texture's width: one texel per lane.
const N: u64 = V.len() as u64;

/// How a format's texels are written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Half,
    Float,
    SmallFloat,
    SharedExponent,
}

/// A format the lane reads.
#[derive(Clone, Copy, Debug)]
struct Format {
    format: PixelFormat,
    channels: usize,
    kind: Kind,
}

const FORMATS: [Format; 8] = [
    Format {
        format: PixelFormat::R16Float,
        channels: 1,
        kind: Kind::Half,
    },
    Format {
        format: PixelFormat::RG16Float,
        channels: 2,
        kind: Kind::Half,
    },
    Format {
        format: PixelFormat::RGBA16Float,
        channels: 4,
        kind: Kind::Half,
    },
    Format {
        format: PixelFormat::R32Float,
        channels: 1,
        kind: Kind::Float,
    },
    Format {
        format: PixelFormat::RG32Float,
        channels: 2,
        kind: Kind::Float,
    },
    Format {
        format: PixelFormat::RGBA32Float,
        channels: 4,
        kind: Kind::Float,
    },
    Format {
        format: PixelFormat::RG11B10Float,
        channels: 3,
        kind: Kind::SmallFloat,
    },
    Format {
        format: PixelFormat::RGB9E5Float,
        channels: 3,
        kind: Kind::SharedExponent,
    },
];

impl Format {
    /// A texel's bytes.
    fn bytes(self) -> u64 {
        match self.kind {
            Kind::Half => 2 * self.channels as u64,
            Kind::Float => 4 * self.channels as u64,
            Kind::SmallFloat | Kind::SharedExponent => 4,
        }
    }
}

/// `RG11B10Float`'s texel `x`.
fn small_float(x: usize) -> u32 {
    R11[x % 12] | R11[(x + 5) % 12] << 11 | B10[(x + 3) % 12] << 22
}

/// `RGB9E5Float`'s texel `x`.
fn shared_exponent(x: usize) -> u32 {
    let (r, g, b) = (
        (x as u32 * 37) & 511,
        (x as u32 * 101 + 7) & 511,
        (x as u32 * 13 + 300) & 511,
    );
    r | g << 9 | b << 18 | E5[x % 8] << 27
}

/// The texture's bytes.
fn texels(f: Format) -> Vec<u8> {
    (0..V.len())
        .flat_map(|x| match f.kind {
            Kind::Half => (0..f.channels)
                .flat_map(|c| V[(x + 5 * c) % V.len()].to_le_bytes())
                .collect::<Vec<_>>(),
            Kind::Float => (0..f.channels)
                .flat_map(|c| F[(x + 5 * c) % F.len()].to_le_bytes())
                .collect(),
            Kind::SmallFloat => small_float(x).to_le_bytes().to_vec(),
            Kind::SharedExponent => shared_exponent(x).to_le_bytes().to_vec(),
        })
        .collect()
}

/// What channel `c` of texel `x` of `f` holds, as far as a NaN goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// A NaN; a 16-bit float's bits, or a 32-bit float's word.
    Nan(u32),
    PlusInfinity,
    MinusInfinity,
    Finite,
}

fn class(f: Format, x: usize, c: usize) -> Class {
    if c >= f.channels {
        return Class::Finite;
    }
    let (sign, exp_all_ones, mantissa, bits) = match f.kind {
        Kind::Half => {
            let h = u32::from(V[(x + 5 * c) % V.len()]);
            (h >> 15, (h >> 10) & 31 == 31, h & 0x3ff, h)
        }
        Kind::Float => {
            let w = F[(x + 5 * c) % F.len()];
            (w >> 31, (w >> 23) & 255 == 255, w & 0x7f_ffff, w)
        }
        Kind::SmallFloat => {
            let code = [R11[x % 12], R11[(x + 5) % 12], B10[(x + 3) % 12]][c];
            let m = if c < 2 { 6 } else { 5 };
            (0, code >> m == 31, code & ((1 << m) - 1), code)
        }
        Kind::SharedExponent => return Class::Finite,
    };
    match (exp_all_ones, mantissa, sign) {
        (true, 0, 0) => Class::PlusInfinity,
        (true, 0, _) => Class::MinusInfinity,
        (true, _, _) => Class::Nan(bits),
        _ => Class::Finite,
    }
}

/// What a kernel does with its texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// A read of a `texture2d<_, read>`.
    Read,
    /// A sample at the texel's centre through a nearest constexpr sampler.
    Nearest,
    /// A sample at the texel's centre through a linear constexpr sampler.
    Linear,
    /// A sample a quarter texel right of the centre through a linear constexpr sampler.
    Quarter,
    /// A gather of red through a nearest constexpr sampler.
    Gather,
    /// A sample at the texel's centre through the runtime sampler at slot 0, nearest.
    SlotNearest,
    /// A sample a quarter texel right of the centre through the runtime sampler at slot 0,
    /// linear.
    SlotQuarter,
}

const OPS: [Op; 7] = [
    Op::Read,
    Op::Nearest,
    Op::Linear,
    Op::Quarter,
    Op::Gather,
    Op::SlotNearest,
    Op::SlotQuarter,
];

impl Op {
    /// The taps of nonzero weight of lane `x`'s channel (a gather's lane is its own texel's).
    fn taps(self, x: usize, lane: usize) -> Vec<usize> {
        let next = (x + 1).min(V.len() - 1);
        match self {
            Op::Read | Op::Nearest | Op::Linear | Op::SlotNearest => vec![x],
            Op::Quarter | Op::SlotQuarter => vec![x, next],
            Op::Gather => vec![[x, next, next, x][lane]],
        }
    }

    /// Whether the access reads one texel's value unfiltered.
    fn one_texel(self) -> bool {
        matches!(self, Op::Read | Op::Nearest | Op::Gather | Op::SlotNearest)
    }
}

/// Metal's answer for lane `lane` of `op`'s lane `x` of `f` where it is a NaN (the module docs):
/// its bits, as a `half`'s or a `float`'s; `None` where it is not one.
fn metal_nan(f: Format, op: Op, x: usize, lane: usize, half: bool) -> Option<u32> {
    let channel = if op == Op::Gather { 0 } else { lane };
    let classes: Vec<Class> = op
        .taps(x, lane)
        .into_iter()
        .map(|t| class(f, t, channel))
        .collect();
    let nan = classes.iter().find_map(|c| match c {
        Class::Nan(bits) => Some(*bits),
        _ => None,
    });
    let opposed = classes.contains(&Class::PlusInfinity) && classes.contains(&Class::MinusInfinity);
    match (nan, op.one_texel(), half) {
        (Some(bits), true, true) if f.kind == Kind::Half => Some(bits),
        (Some(bits), true, false) if f.kind == Kind::Float => Some(bits),
        (Some(_), _, true) => Some(0x7e00),
        (Some(_), _, false) => Some(0x7fc0_0000),
        (None, _, true) if opposed => Some(0x7e00),
        (None, _, false) if opposed => Some(0x7fc0_0000),
        (None, _, _) => None,
    }
}

/// A nearest constexpr sampler's words (AIR's `air.sampler_state` layout): no mip filter,
/// normalized coordinates, every axis clamped to the edge.
const NEAREST: u64 = 0x007b_ff00_0008_0000 | 1 | 1 << 3 | 1 << 6;

/// [`NEAREST`] with linear minification and magnification.
const LINEAR: u64 = NEAREST | 1 << 9 | 1 << 11;

/// Synthetic AIR written for this lane (the module docs): kernel `k` of `op` with a `half4`
/// (`half`) or `float4` result, buffer 0 holding a lane's coordinate (a read's `x` as an `int`,
/// a sample's or gather's two `float`s), buffer 1 the results, texture slot 0 the texture and
/// sampler slot 0 the runtime sampler.
fn kernel(op: Op, half: bool) -> String {
    let (elem, sfx, size) = match half {
        true => ("half", "f16", 8),
        false => ("float", "f32", 16),
    };
    let vt = format!("<4 x {elem}>");
    let (texture_type, access_meta) = match op {
        Op::Read => (format!("texture2d<{elem}, read>"), "air.read"),
        _ => (format!("texture2d<{elem}, sample>"), "air.sample"),
    };
    let constexpr = match op {
        Op::Nearest | Op::Gather => Some(NEAREST),
        Op::Linear | Op::Quarter => Some(LINEAR),
        _ => None,
    };
    let states = match constexpr {
        Some(w) => format!(
            "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n"
        ),
        None => String::new(),
    };
    let sampler_meta = match constexpr {
        Some(_) => "!air.sampler_states = !{!10}\n!10 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        None => "",
    };
    let slot = matches!(op, Op::SlotNearest | Op::SlotQuarter);
    let sampler_param = if slot { "ptr addrspace(2) %s, " } else { "" };
    let sampler = if slot { "%s" } else { "@__air_sampler_state" };
    let uv = "  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16\n  %uv = shufflevector <4 x float> %c4, <4 x float> poison, <2 x i32> <i32 0, i32 1>\n";
    let (at, call, decl) = match op {
        Op::Read => (
            "  %x = load i32, ptr addrspace(1) %p, align 16\n  %at = insertelement <2 x i32> zeroinitializer, i32 %x, i64 0\n  %rs = call ptr addrspace(2) @air.get_read_sampler()\n".to_owned(),
            format!("@air.read_texture_2d.v4{sfx}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %at, <2 x i32> zeroinitializer, i32 0, i32 1)"),
            format!("declare ptr addrspace(2) @air.get_read_sampler()\ndeclare {{ {vt}, i8 }} @air.read_texture_2d.v4{sfx}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)"),
        ),
        Op::Gather => (
            uv.to_owned(),
            format!("@air.gather_texture_2d.v4{sfx}(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)"),
            format!("declare {{ {vt}, i8 }} @air.gather_texture_2d.v4{sfx}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)"),
        ),
        _ => (
            uv.to_owned(),
            format!("@air.sample_texture_2d.v4{sfx}(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)"),
            format!("declare {{ {vt}, i8 }} @air.sample_texture_2d.v4{sfx}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)"),
        ),
    };
    let lane_type = if op == Op::Read { "uint4" } else { "float4" };
    let (args, gid_index) = match slot {
        true => ("!3, !4, !5, !6, !7", 4),
        false => ("!3, !4, !5, !7", 3),
    };
    format!(
        r#"; Owned synthetic kernel.
source_filename = "half_texel_nans.metal"
{states}
define void @k(ptr addrspace(1) %c, ptr addrspace(1) %out, ptr addrspace(1) %t, {sampler_param}i32 %gid) {{
  %i = zext i32 %gid to i64
  %p = getelementptr inbounds <4 x i32>, ptr addrspace(1) %c, i64 %i
{at}  %r = call {{ {vt}, i8 }} {call}
  %v = extractvalue {{ {vt}, i8 }} %r, 0
  %q = getelementptr inbounds {vt}, ptr addrspace(1) %out, i64 %i
  store {vt} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

{decl}

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
{sampler_meta}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args}}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"{lane_type}", !"air.arg_name", !"c"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{elem}4", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"{access_meta}", !"air.arg_type_name", !"{texture_type}", !"air.arg_name", !"t"}}
!6 = !{{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!7 = !{{i32 {gid_index}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The pipeline state of `op`'s kernel on `device`.
fn pipeline_of(device: &Device, op: Op, half: bool) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(&kernel(op, half)).unwrap())
        .unwrap();
    let function = library.new_function_with_name("k").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A lane's coordinate for `op` (the module docs).
fn lane(op: Op, x: usize) -> [u32; 4] {
    let n = N as f32;
    let u = match op {
        Op::Read => return [x as u32, 0, 0, 0],
        Op::Gather => (x as f32 + 1.0) / n,
        Op::Quarter | Op::SlotQuarter => (x as f32 + 0.75) / n,
        _ => (x as f32 + 0.5) / n,
    };
    [u.to_bits(), 0.5f32.to_bits(), 0, 0]
}

/// The texture of `f`, holding [`texels`].
fn made(device: &Device, f: Format) -> Texture {
    let bytes = texels(f);
    let mut d = TextureDescriptor::texture_2d(f.format, N, 1, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    texture
        .replace_region(Region::new_2d(0, 0, N, 1), 0, 0, &bytes, N * f.bytes(), 0)
        .unwrap();
    texture
}

/// Each lane's four words of `op` on `texture` (a `half`'s bits zero-extended).
fn dispatch(
    device: &Device,
    (pipeline, op, half): (&ComputePipelineState, Op, bool),
    texture: &Texture,
    sampler: &SamplerState,
) -> Vec<[u32; 4]> {
    let input: Vec<u8> = (0..V.len())
        .flat_map(|x| lane(op, x))
        .flat_map(u32::to_le_bytes)
        .collect();
    let input = buffer_with(device, &input).unwrap();
    let out = buffer_with(device, &vec![0x5a; (N * 16) as usize]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(pipeline).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    if matches!(op, Op::SlotNearest | Op::SlotQuarter) {
        encoder.set_sampler_state(Some(sampler), 0).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(N, 1, 1), Size::new(N, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let bytes = read_buffer(&out, 0, N * 16).unwrap();
    let size = if half { 2 } else { 4 };
    bytes
        .chunks_exact(4 * size)
        .take(V.len())
        .map(|v| {
            std::array::from_fn(|k| {
                let w = &v[size * k..size * k + size];
                match half {
                    true => u32::from(u16::from_le_bytes([w[0], w[1]])),
                    false => u32::from_le_bytes(w.try_into().unwrap()),
                }
            })
        })
        .collect()
}

/// The sign bit (`0x8000` or `0`) of lane `lane` of `op`'s lane `x` of `f` where its value is a
/// finite `float` that a `half` cannot hold (from `65520` up, which rounds to infinity), its
/// taps weighted as the access weighs them; `None` elsewhere.
fn overflows(f: Format, op: Op, x: usize, lane: usize) -> Option<u32> {
    let channel = if op == Op::Gather { 0 } else { lane };
    if f.kind != Kind::Float || channel >= f.channels {
        return None;
    }
    let taps = op.taps(x, lane);
    let weights: &[f64] = match taps.len() {
        2 => &[0.75, 0.25],
        _ => &[1.0],
    };
    let mut v = 0.0;
    for (t, w) in taps.into_iter().zip(weights) {
        let tap = f64::from(f32::from_bits(F[(t + 5 * channel) % F.len()]));
        if !tap.is_finite() {
            return None;
        }
        v += w * tap;
    }
    (v.abs() >= 65520.0).then_some(if v < 0.0 { 0x8000 } else { 0 })
}

/// Whether `w` is a NaN's bits, as a `half`'s or a `float`'s.
fn is_nan(w: u32, half: bool) -> bool {
    match half {
        true => w & 0x7c00 == 0x7c00 && w & 0x3ff != 0,
        false => w & 0x7f80_0000 == 0x7f80_0000 && w & 0x7f_ffff != 0,
    }
}

/// The lanes of every access of [`OPS`], `half` and `float`, to every format of [`FORMATS`] on
/// `device` that differ from Metal's NaN answers, printed; under the default policy (`raw`) also
/// every NaN lane of a `float` result, the device's decode. Returns the count of differing lanes
/// of `half` results and of `float` results, and of the `half` lanes beyond `half`'s range that
/// are neither infinity nor 65504.
fn measure(device: &Device, leg: &str, raw: bool) -> (usize, usize, usize) {
    let sampler = |filter| {
        let mut d = SamplerDescriptor::new();
        d.min_filter = filter;
        d.mag_filter = filter;
        d.s_address_mode = SamplerAddressMode::ClampToEdge;
        d.t_address_mode = SamplerAddressMode::ClampToEdge;
        device.new_sampler_state(&d).unwrap()
    };
    let (nearest, linear) = (
        sampler(SamplerMinMagFilter::Nearest),
        sampler(SamplerMinMagFilter::Linear),
    );
    let name = device.name();
    let mut totals = (0, 0, 0);
    for half in [true, false] {
        let result = if half { "half" } else { "float" };
        for op in OPS {
            let pipeline = pipeline_of(device, op, half);
            let s = if op == Op::SlotQuarter {
                &linear
            } else {
                &nearest
            };
            for f in FORMATS {
                let texture = made(device, f);
                let got = dispatch(device, (&pipeline, op, half), &texture, s);
                let (mut nans, mut differ) = (0, 0);
                for (x, lanes) in got.iter().enumerate() {
                    for (k, &g) in lanes.iter().enumerate() {
                        let want = metal_nan(f, op, x, k, half);
                        nans += usize::from(want.is_some());
                        let bad = match want {
                            Some(w) => g != w,
                            None => is_nan(g, half),
                        };
                        let shown = |w: Option<u32>| match (w, half) {
                            (Some(w), true) => format!("{w:04x}"),
                            (Some(w), false) => format!("{w:08x}"),
                            (None, _) => "no NaN".to_owned(),
                        };
                        if bad {
                            differ += 1;
                            eprintln!(
                                "HALF-TEXEL-DIFFERS {leg} {result} {op:?} {:?} texel {x} lane {k}: {} (Metal {})",
                                f.format,
                                shown(Some(g)),
                                shown(want)
                            );
                        }
                        if let (true, Some(sign)) = (half, overflows(f, op, x, k)) {
                            if g != sign | 0x7c00 && g != sign | 0x7bff {
                                totals.2 += 1;
                                eprintln!(
                                    "HALF-TEXEL-OVERFLOW {leg} {op:?} {:?} texel {x} lane {k}: {g:04x}",
                                    f.format
                                );
                            }
                        }
                        if raw && !half && is_nan(g, false) {
                            eprintln!(
                                "HALF-TEXEL-RAW {name} {op:?} {:?} texel {x} lane {k}: {g:08x}",
                                f.format
                            );
                        }
                    }
                }
                eprintln!(
                    "HALF-TEXEL-NANS {name} {leg} {result} {op:?} {:?}: {nans} NaN lanes, {differ} differ",
                    f.format
                );
                match half {
                    true => totals.0 += differ,
                    false => totals.1 += differ,
                }
            }
        }
    }
    totals
}

/// Each device's NaN answers for float texels' reads, samples and gathers, `half` and `float`,
/// under the lane's exact policy and the default one: measured and printed, the differing lanes
/// counted.
#[test]
fn float_texel_nans_by_device_and_policy() {
    for (leg, default) in [("exact", false), ("default", true)] {
        on_device_with(
            |options| {
                if default {
                    options.policy = DeviceOptions::default().policy;
                }
                at_plan(options)
            },
            |device, messages| {
                let (h, f, overflows) = measure(device, leg, default);
                let announced = counts(&messages.take_notices());
                eprintln!(
                    "HALF-TEXEL-TOTAL {} {leg}: {h} half lanes differ, {f} float lanes differ, {overflows} half lanes beyond half's range neither infinity nor 65504; announced {:?}",
                    device.name(),
                    announced.keys().map(|k| k.code()).collect::<Vec<_>>()
                );
                assert_eq!(
                    overflows,
                    0,
                    "a half beyond half's range is infinity or 65504 (the HALF-TEXEL-OVERFLOW lines)"
                );
            },
        );
    }
}
