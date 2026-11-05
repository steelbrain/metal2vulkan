//! A `float` read, nearest sample and gather of a half-float texture's NaN texels, on the device
//! the cfg selects, under the validation layer.
//!
//! What plain Metal answers (an Apple M4 under macOS 27.0.1): with a `float`
//! result, every NaN of an `R16Float`, `RG16Float` or `RGBA16Float` texel, whatever its sign and
//! payload, is `0x7fc00000`, and every other half its exact widening, denormals kept; a 32-bit
//! float texel's words are as stored, a signalling or negative NaN's included. That holds for a
//! read through `access::read` and `access::read_write`, a texture buffer's read, a nearest
//! sample and a gather. NVIDIA and RADV hand back the half widened with its NaN's sign and
//! payload, ANV every NaN as `0xffc00000`, and lavapipe a signalling NaN quieted (the format
//! scout's rows). The module hands back the device's texel: on MoltenVK, Metal's own; on a device
//! whose texture unit is not Metal's (no `VkProperty::MetalSampler`), the device's NaN, which
//! its precision note records (`PrecisionNote::DeviceTexelConversion`), and the lane takes any NaN
//! for Metal's there. The note's bound also leaves a 32-bit float texel's NaN in a gather to the
//! device (NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU gathers every one as `0x7fffffff`, where
//! its reads and samples hand back the stored words), which the lane takes for Metal's too.
//!
//! The kernels are this lane's own AIR: each reads texel `(x, 0)` of an [`N`] by 1 texture whose
//! texel `x` holds [`word`]'s channels (a texture buffer's element `x`), samples it at the texel's
//! centre, or gathers at the corner between texels `x` and `x + 1`, and stores the `float4` to
//! buffer 1 at `x`. `held` reads the texture an argument buffer holds (a heap element, whose
//! handle carries the class). Each access's every lane is compared with Metal's; a lane that
//! differs is printed on a `HALF-READ-DIFFERS` line, and the test fails.

use super::degrade_notices::on_device;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkProperty;
use metal2vulkan::metal::{
    PixelFormat, ResourceUsage, SamplerAddressMode, SamplerMinMagFilter, TextureUsage,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region, ResourceOptions,
    SamplerState, Size, Texture, TextureDescriptor,
};

/// The halves read: quiet and signalling NaNs of both signs and payload edges, then infinities,
/// denormals, zeros and normals (our probe's).
const V: [u16; 24] = [
    0x7e00, 0x7c01, 0x7dff, 0x7fff, 0x7e01, 0x7d55, 0xfe00, 0xfc01, 0xfdff, 0xffff, 0xfe01, 0x7f00,
    0x7c00, 0xfc00, 0x0001, 0x8001, 0x03ff, 0x3c00, 0x0000, 0x8000, 0x7bff, 0xfbff, 0x3555, 0xff00,
];

/// The 32-bit words read: NaNs of every class, then infinities, denormals, zeros and normals.
const F: [u32; 24] = [
    0x7fc00000, 0x7f800001, 0x7fbfffff, 0xffc00001, 0xff800001, 0x7fffffff, 0x7fc12345, 0xffffffff,
    0x7f800000, 0x00000001, 0x80000001, 0x3f800000, 0x7fe00000, 0xffc00000, 0x7f802000, 0x00000000,
    0x7fc02000, 0x7fa00000, 0xff800000, 0x80000000, 0x7f7fffff, 0x3eaaaaab, 0x7fd00000, 0xfff00000,
];

/// The texture's width: one texel per value.
const N: u64 = V.len() as u64;

/// A format the lane reads: its channels and whether they are halves.
#[derive(Clone, Copy, Debug)]
struct Format {
    format: PixelFormat,
    channels: usize,
    half: bool,
}

const FORMATS: [Format; 6] = [
    Format {
        format: PixelFormat::R16Float,
        channels: 1,
        half: true,
    },
    Format {
        format: PixelFormat::RG16Float,
        channels: 2,
        half: true,
    },
    Format {
        format: PixelFormat::RGBA16Float,
        channels: 4,
        half: true,
    },
    Format {
        format: PixelFormat::R32Float,
        channels: 1,
        half: false,
    },
    Format {
        format: PixelFormat::RG32Float,
        channels: 2,
        half: false,
    },
    Format {
        format: PixelFormat::RGBA32Float,
        channels: 4,
        half: false,
    },
];

impl Format {
    /// A texel's bytes.
    fn bytes(self) -> u64 {
        (self.channels * if self.half { 2 } else { 4 }) as u64
    }
}

/// Channel `c` of texel `x` of `f`: a half's or a 32-bit word's bits, each value at every channel
/// position across the texels.
fn word(f: Format, x: usize, c: usize) -> u32 {
    let at = (x + 5 * c) % V.len();
    match f.half {
        true => u32::from(V[at]),
        false => F[at],
    }
}

/// The texture's bytes.
fn texels(f: Format) -> Vec<u8> {
    (0..V.len())
        .flat_map(|x| (0..f.channels).map(move |c| (x, c)))
        .flat_map(|(x, c)| {
            let w = word(f, x, c);
            match f.half {
                true => (w as u16).to_le_bytes().to_vec(),
                false => w.to_le_bytes().to_vec(),
            }
        })
        .collect()
}

/// The exact `f32` bits of half `h`.
fn widened(h: u16) -> u32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    match exp {
        0x1f => sign | 0x7f80_0000 | mant << 13,
        0 => sign | (mant as f32 * 2f32.powi(-24)).to_bits(),
        e => sign | (e + 112) << 23 | mant << 13,
    }
}

/// Metal's `float` lane for channel `c` of texel `x` of `f` (the module docs): a missing green or
/// blue channel 0, a missing alpha 1.
fn metal(f: Format, x: usize, c: usize) -> u32 {
    if c >= f.channels {
        return if c == 3 { 0x3f80_0000 } else { 0 };
    }
    let w = word(f, x, c);
    match f.half {
        true if f32::from_bits(widened(w as u16)).is_nan() => 0x7fc0_0000,
        true => widened(w as u16),
        false => w,
    }
}

/// What a kernel does with its texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// A read of a `texture2d<float, read>`.
    Read,
    /// A read of a `texture2d<float, read_write>` (a storage image).
    ReadWrite,
    /// A read of a `texture_buffer<float, read>`.
    Buffer,
    /// A read of the `texture2d<float, read>` an argument buffer holds.
    Held,
    /// A sample at the texel's centre through a nearest constexpr sampler.
    SampleConstexpr,
    /// A sample at the texel's centre through the nearest runtime sampler at sampler slot 0.
    SampleSlot,
    /// A gather of red through a nearest constexpr sampler.
    Gather,
}

const OPS: [Op; 7] = [
    Op::Read,
    Op::ReadWrite,
    Op::Buffer,
    Op::Held,
    Op::SampleConstexpr,
    Op::SampleSlot,
    Op::Gather,
];

/// A nearest constexpr sampler's words (AIR's `air.sampler_state` layout): no mip filter,
/// normalized coordinates, every axis clamped to the edge.
const NEAREST: u64 = 0x007b_ff00_0008_0000 | 1 | 1 << 3 | 1 << 6;

/// Synthetic AIR written for this lane (the module docs): kernel `k` of `op`, buffer 0 holding a
/// lane's coordinate (a read's `x` as an `int`, a sample's or gather's two `float`s), buffer 1
/// the `float4` results, texture slot 0 (or buffer 2's argument buffer) the texture.
fn kernel(op: Op) -> String {
    let (texture_type, access_meta, access_code) = match op {
        Op::Read | Op::Held => ("texture2d<float, read>", "air.read", 1),
        Op::ReadWrite => ("texture2d<float, read_write>", "air.read_write", 3),
        Op::Buffer => ("texture_buffer<float, read>", "air.read", 1),
        _ => ("texture2d<float, sample>", "air.sample", 0),
    };
    let constexpr = matches!(op, Op::SampleConstexpr | Op::Gather);
    let states = match constexpr {
        true => format!(
            "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {NEAREST}, i64 0], align 8\n"
        ),
        false => String::new(),
    };
    let sampler_meta = match constexpr {
        true => "!air.sampler_states = !{!10}\n!10 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        false => "",
    };
    let (texture_param, texture_load) = match op {
        Op::Held => (
            "ptr addrspace(2) %held",
            "  %t = load ptr addrspace(1), ptr addrspace(2) %held, align 8\n",
        ),
        _ => ("ptr addrspace(1) %t", ""),
    };
    let sampler_param = match op {
        Op::SampleSlot => "ptr addrspace(2) %s, ",
        _ => "",
    };
    let (at, call, decl) = match op {
        Op::Read | Op::ReadWrite | Op::Held => (
            "  %x = load i32, ptr addrspace(1) %p, align 16\n  %at = insertelement <2 x i32> zeroinitializer, i32 %x, i64 0\n  %rs = call ptr addrspace(2) @air.get_read_sampler()\n".to_owned(),
            format!("@air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %at, <2 x i32> zeroinitializer, i32 0, i32 {access_code})"),
            "declare ptr addrspace(2) @air.get_read_sampler()\ndeclare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)".to_owned(),
        ),
        Op::Buffer => (
            "  %x = load i32, ptr addrspace(1) %p, align 16\n  %rs = call ptr addrspace(2) @air.get_read_sampler()\n".to_owned(),
            "@air.read_texture_buffer_1d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, i32 %x, i32 1)".to_owned(),
            "declare ptr addrspace(2) @air.get_read_sampler()\ndeclare { <4 x float>, i8 } @air.read_texture_buffer_1d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, i32)".to_owned(),
        ),
        Op::SampleConstexpr | Op::SampleSlot => {
            let sampler = match op {
                Op::SampleSlot => "%s",
                _ => "@__air_sampler_state",
            };
            (
                "  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16\n  %uv = shufflevector <4 x float> %c4, <4 x float> poison, <2 x i32> <i32 0, i32 1>\n".to_owned(),
                format!("@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)"),
                "declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)".to_owned(),
            )
        }
        Op::Gather => (
            "  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16\n  %uv = shufflevector <4 x float> %c4, <4 x float> poison, <2 x i32> <i32 0, i32 1>\n".to_owned(),
            "@air.gather_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)".to_owned(),
            "declare { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)".to_owned(),
        ),
    };
    let lane_type = match op {
        Op::Read | Op::ReadWrite | Op::Held | Op::Buffer => "uint4",
        _ => "float4",
    };
    let texture_meta = match op {
        Op::Held => "!5 = !{i32 2, !\"air.indirect_buffer\", !\"air.buffer_size\", i32 8, !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.struct_type_info\", !8, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"Held\", !\"air.arg_name\", !\"held\"}\n!8 = !{i32 0, i32 8, i32 0, !\"texture2d<float, read>\", !\"t\", !\"air.indirect_argument\", !9}\n!9 = !{i32 0, !\"air.texture\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.arg_type_name\", !\"texture2d<float, read>\", !\"air.arg_name\", !\"t\"}".to_owned(),
        _ => format!("!5 = !{{i32 2, !\"air.texture\", !\"air.location_index\", i32 0, i32 1, !\"{access_meta}\", !\"air.arg_type_name\", !\"{texture_type}\", !\"air.arg_name\", !\"t\"}}"),
    };
    let (args, gid_index) = match op {
        Op::SampleSlot => ("!3, !4, !5, !6, !7", 4),
        _ => ("!3, !4, !5, !7", 3),
    };
    format!(
        r#"; Owned synthetic kernel.
source_filename = "half_read_nans.metal"
{states}
define void @k(ptr addrspace(1) %c, ptr addrspace(1) %out, {texture_param}, {sampler_param}i32 %gid) {{
  %i = zext i32 %gid to i64
  %p = getelementptr inbounds <4 x i32>, ptr addrspace(1) %c, i64 %i
{texture_load}{at}  %r = call {{ <4 x float>, i8 }} {call}
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
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
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"{lane_type}", !"air.arg_name", !"c"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
{texture_meta}
!6 = !{{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!7 = !{{i32 {gid_index}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The pipeline state of `op`'s kernel on `device`.
fn pipeline_of(device: &Device, op: Op) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(&kernel(op)).unwrap())
        .unwrap();
    let function = library.new_function_with_name("k").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A lane's coordinate for `op`: a read's texel, a sample's texel centre, a gather's corner
/// between texels `x` and `x + 1` (on the row's middle).
fn lane(op: Op, x: usize) -> [u32; 4] {
    let n = N as f32;
    match op {
        Op::Read | Op::ReadWrite | Op::Held | Op::Buffer => [x as u32, 0, 0, 0],
        Op::Gather => [((x as f32 + 1.0) / n).to_bits(), 0.5f32.to_bits(), 0, 0],
        _ => [((x as f32 + 0.5) / n).to_bits(), 0.5f32.to_bits(), 0, 0],
    }
}

/// Metal's four lanes of `op` at lane `x` of `f`: a gather's red of texels `x`, `x + 1` (clamped
/// to the edge) in Metal's order, each texel's own channels elsewhere.
fn metal_lanes(f: Format, op: Op, x: usize) -> [u32; 4] {
    match op {
        Op::Gather => {
            let next = (x + 1).min(V.len() - 1);
            [x, next, next, x].map(|t| metal(f, t, 0))
        }
        _ => std::array::from_fn(|c| metal(f, x, c)),
    }
}

/// The texture of `f` that `op` reads, holding [`texels`].
fn made(device: &Device, f: Format, op: Op) -> Texture {
    let bytes = texels(f);
    let usage = match op {
        Op::ReadWrite => TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        _ => TextureUsage::SHADER_READ,
    };
    if op == Op::Buffer {
        let buffer = buffer_with(device, &bytes).unwrap();
        let d = TextureDescriptor::texture_buffer(f.format, N, ResourceOptions::shared(), usage);
        return buffer.new_texture(&d, 0, N * f.bytes()).unwrap();
    }
    let mut d = TextureDescriptor::texture_2d(f.format, N, 1, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    texture
        .replace_region(Region::new_2d(0, 0, N, 1), 0, 0, &bytes, N * f.bytes(), 0)
        .unwrap();
    texture
}

/// Each lane's four `float` words of `op` on `texture`.
fn dispatch(
    device: &Device,
    (pipeline, op): (&ComputePipelineState, Op),
    texture: &Texture,
    sampler: &SamplerState,
) -> Vec<[u32; 4]> {
    let input: Vec<u8> = (0..V.len())
        .flat_map(|x| lane(op, x))
        .flat_map(u32::to_le_bytes)
        .collect();
    let input = buffer_with(device, &input).unwrap();
    let out = buffer_with(device, &vec![0x5a; (N * 16) as usize]).unwrap();
    let held = buffer_with(device, &texture.gpu_resource_id().0.to_le_bytes()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(pipeline).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    match op {
        Op::Held => {
            encoder.set_buffer(Some(&held), 0, 2).unwrap();
            encoder
                .use_resource(texture.as_resource(), ResourceUsage::READ)
                .unwrap();
        }
        _ => encoder.set_texture(Some(texture), 0).unwrap(),
    }
    if op == Op::SampleSlot {
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
    read_buffer(&out, 0, N * 16)
        .unwrap()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|v| {
            std::array::from_fn(|k| u32::from_le_bytes(v[4 * k..4 * k + 4].try_into().unwrap()))
        })
        .collect()
}

/// Every access of [`OPS`] to every format of [`FORMATS`] hands back Metal's words: a half-float
/// texel's NaN as `0x7fc00000` (a NaN of the device's encoding where its texture unit is not
/// Metal's), a 32-bit float texel's words as stored (a gathered NaN of the device's encoding
/// there).
#[test]
fn half_float_reads_hand_back_metals_nans() {
    let mut report = Vec::new();
    let mut differ_total = 0;
    on_device(|device, _| {
        // The device's texel, a half-float NaN its own, where its texture unit is not Metal's.
        let own = !device
            .compile_device()
            .profile()
            .enabled()
            .properties
            .contains(&VkProperty::MetalSampler);
        let mut nearest = SamplerDescriptor::new();
        nearest.min_filter = SamplerMinMagFilter::Nearest;
        nearest.mag_filter = SamplerMinMagFilter::Nearest;
        nearest.s_address_mode = SamplerAddressMode::ClampToEdge;
        nearest.t_address_mode = SamplerAddressMode::ClampToEdge;
        let nearest = device.new_sampler_state(&nearest).unwrap();
        for op in OPS {
            let pipeline = pipeline_of(device, op);
            for f in FORMATS {
                let texture = made(device, f, op);
                let got = dispatch(device, (&pipeline, op), &texture, &nearest);
                let mut differ = Vec::new();
                for (x, lanes) in got.iter().enumerate() {
                    let want = metal_lanes(f, op, x);
                    for (k, (&g, &w)) in lanes.iter().zip(&want).enumerate() {
                        let nans = f32::from_bits(g).is_nan() && f32::from_bits(w).is_nan();
                        // The device's NaN: a half-float texel's, and a 32-bit float texel's in a
                        // gather (`PrecisionNote::DeviceTexelConversion`'s bound).
                        let device_nan = own && nans && (f.half || matches!(op, Op::Gather));
                        if g != w && !device_nan {
                            differ.push(format!("texel {x} lane {k}: {g:08x} (Metal {w:08x})"));
                        }
                    }
                }
                for d in &differ {
                    eprintln!("HALF-READ-DIFFERS {op:?} {:?}: {d}", f.format);
                }
                differ_total += differ.len();
                report.push(format!(
                    "{op:?} {:?}: {} lanes differ",
                    f.format,
                    differ.len()
                ));
            }
        }
    });
    eprintln!("half read NaNs: {report:#?}");
    assert_eq!(differ_total, 0, "lanes differ from Metal's: {report:#?}");
}
