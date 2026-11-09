//! A texture of another numeric class at a storage slot, a texture buffer and a slot the
//! function only queries, and the saturation of every integer texel's write, on the device the cfg
//! selects, under the validation layer.
//!
//! Plain Metal reads such a texture in its format's own class and hands back the words, a 16-bit
//! result an integer word's low 16 bits or a float texel's `half` bits; writes the declared lanes
//! as that class stores them, an integer channel saturated to its range; and answers its real
//! size (our probe on the M3 Max). Each test prints one line per case that differs and a tally;
//! that the new path ran is shown by the answers themselves (the neutral the slot bound before
//! reads zero texels, drops every write and queries a zero size) and by no case announcing
//! `cross-class-texture-neutral`.
//!
//! - [`cross_class_storage_and_buffer_reads_are_the_native_words`]: an `access::read_write`
//!   read and a texture buffer read of each format of [`FORMATS`] through each type of another
//!   class give the native slot's words (a `short` or `ushort` result of a float format the
//!   native `half` read's bits), but a `half` NaN lane, whose bits are the device's where either
//!   stage records `native-half-conversion`, `native-nan-encoding` or `device-nan-encoding`
//!   (counted apart).
//! - [`integer_writes_store_metals_saturated_words`]: a write of `float`, `half`, `int`, `uint`,
//!   `short` and `ushort` texels to each integer format of [`FORMATS`], through an
//!   `access::write` slot and a texture buffer, stores Metal's words: each lane's bits (a 16-bit
//!   lane extended by the format's signedness) saturated to its channel's range, through a slot
//!   of the format's own class as through one of another class.
//! - [`integer_writes_into_float_formats_store_the_bits_as_floats`]: an `int` or `uint` write to
//!   a float format stores what a `float` write of those bits stores, the device's conversion
//!   onto a half-float channel, announced as `device-color-conversion` (Metal rounds it to
//!   nearest even); a `short` or `ushort` write what a `half` write of those bits
//!   stores; a NaN channel's bits the device's (counted apart), where the device announced
//!   `native-half-conversion`, `native-nan-encoding` or `device-nan-encoding` (the pipeline it
//!   specialized to the bound format records the write's note, which the stage compiled without
//!   that format does not).
//! - [`cross_class_queries_are_the_textures_size`]: a slot of another class that the function
//!   only queries answers the texture's width and height.

use super::degrade_notices::on_device;
use super::driver_deviations::half_nan;
use super::extended_range_targets::half_value;
use super::{buffer_with, read_buffer, Messages};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::format::sampled_type;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, PrecisionNote, SampledType, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Degradation, Device, ExecutedPipeline,
    PipelineOptions, Region, ResourceOptions, Size, Texture, TextureDescriptor,
};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// A 2D texture's width and height (one level), and a texture buffer's texels.
const W: u64 = 8;
const H: u64 = 4;
const TEXELS: u64 = 64;

/// The formats, each with its texel's bytes and channels: float-class formats of each kind and
/// signed and unsigned integer formats of each width.
const FORMATS: [(PixelFormat, u64, usize); 21] = {
    use PixelFormat as P;
    [
        (P::R8Unorm, 1, 1),
        (P::RGBA8Unorm, 4, 4),
        (P::BGRA8Unorm, 4, 4),
        (P::RGBA8Snorm, 4, 4),
        (P::RGB10A2Unorm, 4, 4),
        (P::R16Float, 2, 1),
        (P::RGBA16Unorm, 8, 4),
        (P::RGBA16Float, 8, 4),
        (P::R32Float, 4, 1),
        (P::RGBA32Float, 16, 4),
        (P::R8Sint, 1, 1),
        (P::RGBA8Sint, 4, 4),
        (P::RG16Sint, 4, 2),
        (P::RGBA16Sint, 8, 4),
        (P::R32Sint, 4, 1),
        (P::RGBA32Sint, 16, 4),
        (P::R8Uint, 1, 1),
        (P::RGBA8Uint, 4, 4),
        (P::RGB10A2Uint, 4, 4),
        (P::RGBA16Uint, 8, 4),
        (P::RGBA32Uint, 16, 4),
    ]
};

/// The 32-bit words a write's lanes take (the probe's): each channel's edges and
/// past them, both signs, and float bits.
const WORDS: [u32; 32] = [
    0x0000_0000,
    0x3F00_0000,
    0x3E80_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x4000_0000,
    0x437F_0000,
    0x4396_0000,
    0x477F_FF00,
    0x7FC0_0000,
    0x8000_0000,
    0x7F80_0000,
    0x0000_0001,
    0x0000_012C,
    0x0000_00FF,
    0x0000_0100,
    0x0000_7FFF,
    0x0000_8000,
    0x0000_FFFF,
    0x0001_0000,
    0x7FFF_FFFF,
    0xFFFF_FFFF,
    0xFFFF_FF80,
    0x3C00_3C00,
    0x4E6E_6B28,
    0xC2C8_0000,
    0x3F7F_FFFF,
    0x3B80_8081,
    0x0000_0080,
    0xFFFF_8000,
    0x0000_03FF,
    0x1234_5678,
];

/// The 16-bit lanes a write takes (the probe's).
const HALVES: [u16; 24] = [
    0x0000, 0x3800, 0x3C00, 0xBC00, 0x5BF8, 0x7C00, 0x7E00, 0x0001, 0x00FF, 0x0100, 0x012C, 0x7FFF,
    0x8000, 0xFFFF, 0x3400, 0xFC00, 0x03FF, 0x0080, 0xFF80, 0x4000, 0x57F8, 0x1234, 0xABCD, 0x7BFF,
];

/// A texel type a kernel declares at its texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Ty {
    Float,
    Half,
    Int,
    Uint,
    Short,
    Ushort,
}

impl Ty {
    const ALL: [Ty; 6] = [
        Ty::Float,
        Ty::Half,
        Ty::Int,
        Ty::Uint,
        Ty::Short,
        Ty::Ushort,
    ];

    /// The MSL element, the LLVM vector, its bytes, and the intrinsics' suffix.
    fn spelling(self) -> (&'static str, &'static str, u64, &'static str) {
        match self {
            Ty::Float => ("float", "<4 x float>", 16, "v4f32"),
            Ty::Half => ("half", "<4 x half>", 8, "v4f16"),
            Ty::Int => ("int", "<4 x i32>", 16, "s.v4i32"),
            Ty::Uint => ("uint", "<4 x i32>", 16, "u.v4i32"),
            Ty::Short => ("short", "<4 x i16>", 8, "s.v4i16"),
            Ty::Ushort => ("ushort", "<4 x i16>", 8, "u.v4i16"),
        }
    }

    /// Its numeric class.
    fn class(self) -> SampledType {
        match self {
            Ty::Float | Ty::Half => SampledType::Float,
            Ty::Int | Ty::Short => SampledType::Sint,
            Ty::Uint | Ty::Ushort => SampledType::Uint,
        }
    }

    /// Whether its lanes are 16-bit.
    fn narrow(self) -> bool {
        matches!(self, Ty::Half | Ty::Short | Ty::Ushort)
    }

    /// The type of a native slot giving this type's answer over a texture of class `class`: a
    /// float format's `half` read for a 16-bit integer result, else the class's 32-bit type.
    fn native(self, class: SampledType) -> Ty {
        match (self, class) {
            (Ty::Short | Ty::Ushort, SampledType::Float) => Ty::Half,
            (_, SampledType::Float) => Ty::Float,
            (_, SampledType::Sint) => Ty::Int,
            _ => Ty::Uint,
        }
    }
}

/// What a kernel does with its texture at its thread `gid`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    /// A read of an `access::read_write` 2D texture at texel `(gid % 8, gid / 8)`.
    RwRead,
    /// A read of an `access::read` texture buffer at texel `gid`.
    BufRead,
    /// A write of buffer 0's texel `gid` to an `access::write` 2D texture at `(gid % 8, gid / 8)`.
    Write,
    /// A write of buffer 0's texel `gid` to an `access::write` texture buffer at texel `gid`.
    BufWrite,
    /// The width and height of a sampled 2D texture the function does nothing else with.
    Query,
}

impl Kind {
    /// Whether its texture is a texture buffer.
    fn buffer(self) -> bool {
        matches!(self, Kind::BufRead | Kind::BufWrite)
    }

    /// Its thread count: every texel.
    fn threads(self) -> u64 {
        match self.buffer() {
            true => TEXELS,
            false => W * H,
        }
    }
}

/// Synthetic AIR written for this test: kernel `k` with `ty` texels at texture slot 0 as `kind`
/// says, buffer 0 the written texels, buffer 1 the results (16 bytes a thread: a read's texel, a
/// 16-bit one in the first 8; a query's width and height).
fn kernel(kind: Kind, ty: Ty) -> String {
    let (element, vec4, bytes, suffix) = ty.spelling();
    let (texture, access) = match kind {
        Kind::RwRead => (format!("texture2d<{element}, read_write>"), "read_write"),
        Kind::BufRead => (format!("texture_buffer<{element}, read>"), "read"),
        Kind::Write => (format!("texture2d<{element}, write>"), "write"),
        Kind::BufWrite => (format!("texture_buffer<{element}, write>"), "write"),
        Kind::Query => (format!("texture2d<{element}>"), "sample"),
    };
    let at = "  %x = and i32 %gid, 7\n  %y = lshr i32 %gid, 3\n  %a0 = insertelement <2 x i32> zeroinitializer, i32 %x, i64 0\n  %at = insertelement <2 x i32> %a0, i32 %y, i64 1\n";
    let store = format!(
        "  %q = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i64 %i\n  store {vec4} %v, ptr addrspace(1) %q, align {bytes}\n"
    );
    let load = format!(
        "  %p = getelementptr inbounds {vec4}, ptr addrspace(1) %src, i64 %i\n  %v = load {vec4}, ptr addrspace(1) %p, align {bytes}\n"
    );
    let (body, decl) = match kind {
        Kind::RwRead => (
            format!(
                "{at}  %rs = call ptr addrspace(2) @air.get_read_sampler()\n  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %at, <2 x i32> zeroinitializer, i32 0, i32 3)\n  %v = extractvalue {{ {vec4}, i8 }} %r, 0\n{store}"
            ),
            format!(
                "declare ptr addrspace(2) @air.get_read_sampler()\ndeclare {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)"
            ),
        ),
        Kind::BufRead => (
            format!(
                "  %rs = call ptr addrspace(2) @air.get_read_sampler()\n  %r = call {{ {vec4}, i8 }} @air.read_texture_buffer_1d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, i32 %gid, i32 1)\n  %v = extractvalue {{ {vec4}, i8 }} %r, 0\n{store}"
            ),
            format!(
                "declare ptr addrspace(2) @air.get_read_sampler()\ndeclare {{ {vec4}, i8 }} @air.read_texture_buffer_1d.{suffix}(ptr addrspace(1), ptr addrspace(2), i32, i32)"
            ),
        ),
        Kind::Write => (
            format!(
                "{at}{load}  call void @air.write_texture_2d.{suffix}(ptr addrspace(1) %t, <2 x i32> %at, {vec4} %v, i32 0, i32 2)\n"
            ),
            format!(
                "declare void @air.write_texture_2d.{suffix}(ptr addrspace(1), <2 x i32>, {vec4}, i32, i32)"
            ),
        ),
        Kind::BufWrite => (
            format!(
                "{load}  call void @air.write_texture_buffer_1d.{suffix}(ptr addrspace(1) %t, i32 %gid, {vec4} %v, i32 2)\n"
            ),
            format!(
                "declare void @air.write_texture_buffer_1d.{suffix}(ptr addrspace(1), i32, {vec4}, i32)"
            ),
        ),
        Kind::Query => (
            "  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)\n  %h = call i32 @air.get_height_texture_2d(ptr addrspace(1) %t, i32 0)\n  %z0 = insertelement <4 x i32> zeroinitializer, i32 %w, i64 0\n  %z1 = insertelement <4 x i32> %z0, i32 %h, i64 1\n  %q = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i64 %i\n  store <4 x i32> %z1, ptr addrspace(1) %q, align 16\n".to_owned(),
            "declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)\ndeclare i32 @air.get_height_texture_2d(ptr addrspace(1), i32)".to_owned(),
        ),
    };
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %src, ptr addrspace(1) %out, i32 %gid) {{
  %i = zext i32 %gid to i64
{body}  ret void
}}

{decl}

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{access}", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 {bytes}, !"air.arg_type_align_size", i32 {bytes}, !"air.arg_type_name", !"{element}4", !"air.arg_name", !"src"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The pipeline state of kernel `k` doing `kind` with `ty` texels.
fn pipeline_of(device: &Device, kind: Kind, ty: Ty) -> ComputePipelineState {
    let bitcode = llvm_assemble_text(&kernel(kind, ty)).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("k").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// `n` bytes of a 64-bit linear congruential state seeded by `seed`, each its top 8 bits.
fn random_bytes(seed: u32, n: u64) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(seed);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        })
        .collect()
}

/// A texture of `format` (`bytes` a texel) for `kind`, holding `contents` where given: an 8 x 4
/// 2D texture made for shader reads and writes, or a texture buffer of 64 texels over a shared
/// buffer it reads back from. `None` where the runtime refuses it (printed).
fn made(
    device: &Device,
    (format, bytes): (PixelFormat, u64),
    kind: Kind,
    contents: Option<&[u8]>,
) -> Option<(Texture, Option<metal2vulkan_runtime::Buffer>)> {
    let usage = match kind {
        Kind::Query => TextureUsage::SHADER_READ,
        _ => TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
    };
    if kind.buffer() {
        let row = TEXELS * bytes;
        let zero = vec![0u8; row as usize];
        let backing = buffer_with(device, contents.unwrap_or(&zero)).unwrap();
        let d = TextureDescriptor::texture_buffer(format, TEXELS, ResourceOptions::shared(), usage);
        return match backing.new_texture(&d, 0, row) {
            Ok(t) => Some((t, Some(backing))),
            Err(e) => {
                println!(
                    "CROSS-CLASS-SKIP {}: a texture buffer of {format:?}: {e}",
                    device.name()
                );
                None
            }
        };
    }
    let mut d = TextureDescriptor::texture_2d(format, W, H, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::shared();
    let texture = match device.new_texture(&d) {
        Ok(t) => t,
        Err(e) => {
            println!(
                "CROSS-CLASS-SKIP {}: a texture of {format:?}: {e}",
                device.name()
            );
            return None;
        }
    };
    if let Some(contents) = contents {
        texture
            .replace_region(Region::new_2d(0, 0, W, H), 0, 0, contents, W * bytes, 0)
            .unwrap();
    }
    Some((texture, None))
}

/// The bytes `texture` holds: its backing buffer's, or the 2D texture's.
fn stored(
    (texture, backing): (&Texture, Option<&metal2vulkan_runtime::Buffer>),
    bytes: u64,
) -> Vec<u8> {
    match backing {
        Some(b) => read_buffer(b, 0, TEXELS * bytes).unwrap(),
        None => {
            let mut out = vec![0u8; (W * H * bytes) as usize];
            texture
                .get_bytes(&mut out, W * bytes, 0, Region::new_2d(0, 0, W, H), 0, 0)
                .unwrap();
            out
        }
    }
}

/// Run `pipeline` (`kind`) over `texture` at slot 0, buffer 0 holding `src`: each thread's four
/// output words (a 16-bit result its first two, the fill `0x5a5a5a5a` after), or the device's
/// refusal of the binding.
fn dispatch(
    device: &Device,
    (pipeline, kind): (&ComputePipelineState, Kind),
    texture: &Texture,
    src: &[u8],
) -> Result<Vec<[u32; 4]>, String> {
    dispatch_noted(device, (pipeline, kind), texture, src).map(|(words, _)| words)
}

/// [`dispatch`], and the precision notes of the stage the dispatch ran, specialized ones included:
/// the variant the runtime compiled for the bound texture (its format's storage choice, its
/// cross-class input), as `CommandBuffer::executions` records it.
fn dispatch_noted(
    device: &Device,
    (pipeline, kind): (&ComputePipelineState, Kind),
    texture: &Texture,
    src: &[u8],
) -> Result<(Vec<[u32; 4]>, BTreeSet<PrecisionNote>), String> {
    let n = kind.threads();
    let src = buffer_with(device, src).unwrap();
    let out = buffer_with(device, &vec![0x5a; (n * 16) as usize]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(pipeline).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&src), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(n, 1, 1), Size::new(n.min(32), 1, 1))
        .map_err(|e| e.to_string())?;
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let executions = command_buffer.executions();
    let [execution] = executions.as_slice() else {
        panic!("{} executions of one dispatch", executions.len());
    };
    let ExecutedPipeline::Compute(executed) = execution.pipeline() else {
        panic!("a dispatch of a render pipeline");
    };
    let notes = (executed.stage.precision_notes.iter().copied())
        .chain(executed.stage.specialized_notes.iter().map(|s| s.note))
        .collect();
    let bytes = read_buffer(&out, 0, n * 16).unwrap();
    let words = bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|v| {
            std::array::from_fn(|k| u32::from_le_bytes(v[4 * k..4 * k + 4].try_into().unwrap()))
        })
        .collect();
    Ok((words, notes))
}

/// What Metal answers for a `ty` result where the native slot's result is `native`: its words,
/// or a 16-bit result's low halves packed in the first two words, the fill after.
fn expected(native: [u32; 4], ty: Ty) -> [u32; 4] {
    match ty.narrow() {
        false => native,
        true => {
            let low = |k: usize| native[k] & 0xffff;
            [
                low(0) | low(1) << 16,
                low(2) | low(3) << 16,
                0x5a5a_5a5a,
                0x5a5a_5a5a,
            ]
        }
    }
}

/// The NaN notes a stage may record: the device's `half` conversion and widening
/// (`PrecisionNote::NativeHalfConversion`, which may quiet a signalling NaN), a NaN result's bits
/// (`PrecisionNote::NativeNanEncoding`) and a texel's stored or decoded NaN
/// (`PrecisionNote::DeviceNanEncoding`).
const NAN_NOTES: [PrecisionNote; 3] = [
    PrecisionNote::NativeHalfConversion,
    PrecisionNote::NativeNanEncoding,
    PrecisionNote::DeviceNanEncoding,
];

/// Whether `got`'s words answer `want`'s for a `ty` result: the same words, but where `nan_bits`
/// a 16-bit result's lane (the low and high halves of its first two words) that is a `half` NaN
/// in both may hold other bits.
fn read_holds(got: &[u32; 4], want: &[u32; 4], ty: Ty, nan_bits: bool) -> bool {
    if got == want {
        return true;
    }
    if !(nan_bits && ty.narrow()) || got[2..] != want[2..] {
        return false;
    }
    got[..2].iter().zip(&want[..2]).all(|(&g, &w)| {
        [0, 16].into_iter().all(|shift| {
            let (g, w) = ((g >> shift) as u16, (w >> shift) as u16);
            g == w || (half_nan(g) && half_nan(w))
        })
    })
}

/// Whether the stored texel `got` answers `want` in a float format whose channels are `half`s
/// (`halves`) or `float`s: the same bytes, but a channel that is a NaN in both may hold other
/// bits (the device's, which the test requires an announced NaN note for).
fn stored_holds(got: &[u8], want: &[u8], halves: bool) -> bool {
    if got == want {
        return true;
    }
    match halves {
        true => got
            .as_chunks::<2>()
            .0
            .iter()
            .zip(want.as_chunks::<2>().0)
            .all(|(g, w)| {
                let (g, w) = (u16::from_le_bytes(*g), u16::from_le_bytes(*w));
                g == w || (half_nan(g) && half_nan(w))
            }),
        false => got
            .as_chunks::<4>()
            .0
            .iter()
            .zip(want.as_chunks::<4>().0)
            .all(|(g, w)| {
                let (g, w) = (f32::from_le_bytes(*g), f32::from_le_bytes(*w));
                g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan())
            }),
    }
}

/// The kinds of `cross-class-texture-neutral` and its kin the device announced since the last
/// take.
fn neutral_announced(messages: &Messages) -> BTreeSet<Degradation> {
    messages
        .take_notices()
        .iter()
        .filter_map(|m| m.degradation.as_ref())
        .map(|n| n.degradation)
        .filter(|k| *k == Degradation::CrossClassTextureNeutral)
        .collect()
}

/// Each `access::read_write` read and each texture buffer read of each format of [`FORMATS`]
/// through each type of another class gives the native slot's words, a 16-bit result their low
/// 16 bits (a float format's: the native `half` read's bits); none announces the neutral.
#[test]
fn cross_class_storage_and_buffer_reads_are_the_native_words() {
    on_device(|device, messages| {
        let mut pipelines: BTreeMap<(Kind, Ty), ComputePipelineState> = BTreeMap::new();
        let (mut checked, mut differ, mut skipped, mut nonzero) = (0usize, 0usize, 0usize, 0usize);
        let mut nans = 0usize;
        let _ = messages.take_notices();
        for &(format, bytes, _) in &FORMATS {
            let class = sampled_type(format).expect("a format of a class");
            for kind in [Kind::RwRead, Kind::BufRead] {
                let contents = random_bytes(format.as_raw(), kind.threads() * bytes);
                let Some((texture, _backing)) =
                    made(device, (format, bytes), kind, Some(&contents))
                else {
                    skipped += 1;
                    continue;
                };
                let mut state = |ty: Ty| {
                    pipelines
                        .entry((kind, ty))
                        .or_insert_with(|| pipeline_of(device, kind, ty))
                        .clone()
                };
                for ty in Ty::ALL.into_iter().filter(|ty| ty.class() != class) {
                    let native = state(ty.native(class));
                    let (want, native_notes) =
                        match dispatch_noted(device, (&native, kind), &texture, &[0; 16]) {
                            Ok(w) => w,
                            Err(e) => {
                                println!(
                                    "CROSS-CLASS-SKIP {}: {kind:?} of {format:?}: {e}",
                                    device.name()
                                );
                                skipped += 1;
                                continue;
                            }
                        };
                    let cross = state(ty);
                    let (got, cross_notes) =
                        dispatch_noted(device, (&cross, kind), &texture, &[0; 16])
                            .unwrap_or_else(|e| panic!("{kind:?} {ty:?} over {format:?}: {e}"));
                    checked += got.len();
                    nonzero += got.iter().filter(|w| w[..2] != [0, 0]).count();
                    // A float format's 16-bit integer result is the native `half` result's bits,
                    // packed as it is.
                    let expected = |native: [u32; 4], t: Ty| match t.native(class).narrow() {
                        true => native,
                        false => expected(native, t),
                    };
                    // Whether either stage the runtime ran leaves a NaN's bits to the device: it
                    // records one of [`NAN_NOTES`].
                    let bits = NAN_NOTES
                        .iter()
                        .any(|n| native_notes.contains(n) || cross_notes.contains(n));
                    let unlike = (0..got.len()).filter(|&i| got[i] != expected(want[i], ty));
                    nans += unlike.count();
                    let wrong: Vec<usize> = (0..got.len())
                        .filter(|&i| !read_holds(&got[i], &expected(want[i], ty), ty, bits))
                        .collect();
                    nans -= wrong.len();
                    if let Some(&i) = wrong.first() {
                        differ += wrong.len();
                        println!(
                            "CROSS-CLASS-DIFFERS {}: {kind:?} {ty:?} over {format:?}: {} of {} \
                             texels, first {i}: {:08x?}, expected {:08x?}",
                            device.name(),
                            wrong.len(),
                            got.len(),
                            got[i],
                            expected(want[i], ty)
                        );
                    }
                }
            }
        }
        let neutral = neutral_announced(messages);
        println!(
            "CROSS-CLASS storage reads {}: {checked} texels checked, {differ} differ, \
             {nans} hold NaN halves of the device's bits, {nonzero} nonzero, {skipped} skipped, \
             announced {neutral:?}",
            device.name()
        );
        assert!(checked > 0 && nonzero > 0);
        assert_eq!(
            differ, 0,
            "cross-class storage reads differ from the native slot's"
        );
        assert!(neutral.is_empty(), "an exact read bound the neutral");
    });
}

/// The lanes a write of `ty` texels takes for texel `n`: [`WORDS`] for a 32-bit type, [`HALVES`]
/// for a 16-bit one, each lane's word or 16 bits.
fn lane(ty: Ty, n: u64, c: usize) -> u32 {
    let k = (n as usize * 4 + c) * 7;
    match ty.narrow() {
        true => u32::from(HALVES[k % HALVES.len()]),
        false => WORDS[k % WORDS.len()],
    }
}

/// The bytes of buffer 0 for a write of `ty` texels with lanes `lanes(n, c)`.
fn source(ty: Ty, n: u64, lanes: impl Fn(u64, usize) -> u32) -> Vec<u8> {
    (0..n)
        .flat_map(|t| (0..4).map(move |c| (t, c)))
        .flat_map(|(t, c)| match ty.narrow() {
            true => (lanes(t, c) as u16).to_le_bytes().to_vec(),
            false => lanes(t, c).to_le_bytes().to_vec(),
        })
        .collect()
}

/// Each channel's width of integer format `format` (red to alpha), its signedness, and the
/// value channel `c` of `texel` (its bytes) holds, as that signedness reads it.
fn integer_channel(format: PixelFormat, texel: &[u8], c: usize) -> (u32, bool, i64) {
    use PixelFormat as P;
    let signed = sampled_type(format) == Some(SampledType::Sint);
    if format == P::RGB10A2Uint {
        let w = u32::from_le_bytes(texel[..4].try_into().unwrap());
        let (at, bits) = [(0, 10), (10, 10), (20, 10), (30, 2)][c];
        return (bits, false, i64::from((w >> at) & ((1 << bits) - 1)));
    }
    let size = match format {
        P::R8Sint | P::RGBA8Sint | P::R8Uint | P::RGBA8Uint => 1,
        P::RG16Sint | P::RGBA16Sint | P::RGBA16Uint => 2,
        _ => 4,
    };
    let b = &texel[c * size..(c + 1) * size];
    let v = match (size, signed) {
        (1, true) => i64::from(b[0] as i8),
        (1, false) => i64::from(b[0]),
        (2, true) => i64::from(i16::from_le_bytes([b[0], b[1]])),
        (2, false) => i64::from(u16::from_le_bytes([b[0], b[1]])),
        (_, true) => i64::from(i32::from_le_bytes(b.try_into().unwrap())),
        (_, false) => i64::from(u32::from_le_bytes(b.try_into().unwrap())),
    };
    (8 * size as u32, signed, v)
}

/// What plain Metal stores in a channel of `bits` bits of an integer format (`signed`) from a
/// lane (a 32-bit word, or a 16-bit lane where `sixteen`): the lane extended by the format's
/// signedness, saturated to the channel's range.
fn metal_store(lane: u32, sixteen: bool, bits: u32, signed: bool) -> i64 {
    let v: i64 = match (sixteen, signed) {
        (true, true) => i64::from(lane as u16 as i16),
        (true, false) => i64::from(lane & 0xffff),
        (false, true) => i64::from(lane as i32),
        (false, false) => i64::from(lane),
    };
    match signed {
        true => v.clamp(-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1),
        false => v.clamp(0, (1i64 << bits) - 1),
    }
}

/// Each write of `float`, `half`, `int`, `uint`, `short` and `ushort` texels to each integer
/// format of [`FORMATS`], through an `access::write` 2D slot and a texture buffer, stores
/// Metal's words in each of the format's channels: the lane's bits (a 16-bit lane extended by
/// the format's signedness) saturated to the channel's range, through a slot of the format's own
/// class as through one of another; none announces the neutral.
#[test]
fn integer_writes_store_metals_saturated_words() {
    on_device(|device, messages| {
        let mut pipelines: BTreeMap<(Kind, Ty), ComputePipelineState> = BTreeMap::new();
        let (mut checked, mut differ, mut skipped, mut saturated) = (0usize, 0usize, 0usize, 0);
        let _ = messages.take_notices();
        for &(format, bytes, channels) in &FORMATS {
            if sampled_type(format) == Some(SampledType::Float) {
                continue;
            }
            for kind in [Kind::Write, Kind::BufWrite] {
                for ty in Ty::ALL {
                    let Some((texture, backing)) = made(device, (format, bytes), kind, None) else {
                        skipped += 1;
                        continue;
                    };
                    let state = pipelines
                        .entry((kind, ty))
                        .or_insert_with(|| pipeline_of(device, kind, ty))
                        .clone();
                    let n = kind.threads();
                    let src = source(ty, n, |t, c| lane(ty, t, c));
                    if let Err(e) = dispatch(device, (&state, kind), &texture, &src) {
                        println!(
                            "CROSS-CLASS-SKIP {}: {kind:?} {ty:?} of {format:?}: {e}",
                            device.name()
                        );
                        skipped += 1;
                        continue;
                    }
                    let held = stored((&texture, backing.as_ref()), bytes);
                    let mut wrong = Vec::new();
                    for t in 0..n {
                        let texel = &held[(t * bytes) as usize..((t + 1) * bytes) as usize];
                        for c in 0..channels {
                            let (bits, signed, got) = integer_channel(format, texel, c);
                            let l = lane(ty, t, c);
                            let want = metal_store(l, ty.narrow(), bits, signed);
                            checked += 1;
                            let wide = match (ty.narrow(), signed) {
                                (true, true) => i64::from(l as u16 as i16),
                                (true, false) => i64::from(l),
                                (false, true) => i64::from(l as i32),
                                (false, false) => i64::from(l),
                            };
                            saturated += usize::from(want != wide);
                            if got != want {
                                wrong.push((t, c, l, got, want));
                            }
                        }
                    }
                    if let Some(&(t, c, l, got, want)) = wrong.first() {
                        differ += wrong.len();
                        println!(
                            "CROSS-CLASS-WRITE-DIFFERS {}: {kind:?} {ty:?} into {format:?}: {} \
                             channels, first texel {t} channel {c}: lane {l:#x} stored {got}, \
                             Metal's {want}",
                            device.name(),
                            wrong.len()
                        );
                    }
                }
            }
        }
        let neutral = neutral_announced(messages);
        println!(
            "CROSS-CLASS integer writes {}: {checked} channels checked, {saturated} saturated, \
             {differ} differ, {skipped} skipped, announced {neutral:?}",
            device.name()
        );
        assert!(checked > 0 && saturated > 0);
        assert_eq!(
            differ, 0,
            "integer writes differ from Metal's saturated words"
        );
        assert!(neutral.is_empty(), "an exact write bound the neutral");
    });
}

/// Each `int` or `uint` write to a float format of [`FORMATS`] stores what a `float` write of
/// the lanes' bits stores, the device's conversion onto a half-float channel (where Metal rounds
/// to nearest even, so the stage records `device-color-conversion`), and each `short` or
/// `ushort` write what a `half` write of the same bits stores, through an `access::write` 2D slot
/// and a texture buffer; a 16-bit lane's NaN in a 32-bit float channel as Metal stores it,
/// `0x7fc00000`. None announces the neutral.
#[test]
fn integer_writes_into_float_formats_store_the_bits_as_floats() {
    on_device(|device, messages| {
        use PixelFormat as P;
        let mut pipelines: BTreeMap<(Kind, Ty), ComputePipelineState> = BTreeMap::new();
        let (mut checked, mut differ, mut skipped) = (0usize, 0usize, 0usize);
        let mut nans = 0usize;
        // A 32-bit lane written into a half-float format, which the device's conversion rounds.
        let mut rounded = false;
        let _ = messages.take_notices();
        for &(format, bytes, _) in &FORMATS {
            if sampled_type(format) != Some(SampledType::Float) {
                continue;
            }
            let halves = matches!(format, P::R16Float | P::RGBA16Float);
            for kind in [Kind::Write, Kind::BufWrite] {
                for ty in [Ty::Int, Ty::Uint, Ty::Short, Ty::Ushort] {
                    // The native write giving the answer, and its lanes: an `int` or `uint`
                    // write's the `float` write of its bits, the device's conversion; a 16-bit
                    // lane's NaN into a 32-bit float channel Metal's `0x7fc00000` (the rule for a
                    // `half` write), so those take the `float` write of the widened value, which
                    // stores its bits.
                    let wide = matches!(format, P::R32Float | P::RGBA32Float);
                    let (native, lanes): (Ty, Box<dyn Fn(u64, usize) -> u32>) =
                        match (ty.narrow(), halves) {
                            (true, _) if wide => (
                                Ty::Float,
                                Box::new(move |t, c| {
                                    let f = half_value(lane(ty, t, c) as u16);
                                    match f.is_nan() {
                                        true => 0x7fc0_0000,
                                        false => f.to_bits(),
                                    }
                                }),
                            ),
                            (true, _) => (Ty::Half, Box::new(move |t, c| lane(ty, t, c))),
                            (false, _) => (Ty::Float, Box::new(move |t, c| lane(ty, t, c))),
                        };
                    rounded |= halves && !ty.narrow();
                    let n = kind.threads();
                    let mut run = |t: Ty, src: &[u8]| -> Option<Vec<u8>> {
                        let (texture, backing) = made(device, (format, bytes), kind, None)?;
                        let state = pipelines
                            .entry((kind, t))
                            .or_insert_with(|| pipeline_of(device, kind, t))
                            .clone();
                        match dispatch(device, (&state, kind), &texture, src) {
                            Ok(_) => Some(stored((&texture, backing.as_ref()), bytes)),
                            Err(e) => {
                                println!(
                                    "CROSS-CLASS-SKIP {}: {kind:?} {t:?} of {format:?}: {e}",
                                    device.name()
                                );
                                None
                            }
                        }
                    };
                    let Some(want) = run(native, &source(native, n, &lanes)) else {
                        skipped += 1;
                        continue;
                    };
                    let Some(got) = run(ty, &source(ty, n, |t, c| lane(ty, t, c))) else {
                        skipped += 1;
                        continue;
                    };
                    checked += n as usize;
                    let texel = |t: u64| (t * bytes) as usize..((t + 1) * bytes) as usize;
                    nans += (0..n).filter(|&t| got[texel(t)] != want[texel(t)]).count();
                    let wrong: Vec<u64> = (0..n)
                        .filter(|&t| !stored_holds(&got[texel(t)], &want[texel(t)], halves))
                        .collect();
                    nans -= wrong.len();
                    if let Some(&t) = wrong.first() {
                        differ += wrong.len();
                        let r = texel(t);
                        println!(
                            "CROSS-CLASS-WRITE-DIFFERS {}: {kind:?} {ty:?} into {format:?}: {} of \
                             {n} texels, first {t}: lanes {:08x?} stored {:02x?}, the {native:?} \
                             write's {:02x?}",
                            device.name(),
                            wrong.len(),
                            (0..4).map(|c| lane(ty, t, c)).collect::<Vec<_>>(),
                            &got[r.clone()],
                            &want[r]
                        );
                    }
                }
            }
        }
        let announced: BTreeSet<Degradation> = messages
            .take_notices()
            .iter()
            .filter_map(|m| m.degradation.as_ref())
            .map(|n| n.degradation)
            .collect();
        println!(
            "CROSS-CLASS float-format writes {}: {checked} texels checked, {differ} differ, \
             {nans} hold NaNs of the device's bits, {skipped} skipped, announced {announced:?}",
            device.name()
        );
        assert!(checked > 0);
        assert_eq!(differ, 0, "integer writes into float formats differ");
        assert!(
            !announced.contains(&Degradation::CrossClassTextureNeutral),
            "an exact write bound the neutral"
        );
        assert!(
            !rounded || announced.contains(&Degradation::DeviceColorConversion),
            "a 32-bit lane rounded onto a half-float channel without device-color-conversion, \
             announced {announced:?}"
        );
        let nan_kinds = NAN_NOTES.map(Degradation::from_precision_note);
        assert!(
            nans == 0 || nan_kinds.iter().any(|k| announced.contains(k)),
            "{nans} texels hold the device's NaN, announced {announced:?}"
        );
    });
}

/// A slot of each type of another class that the function only queries, over each format of
/// [`FORMATS`], answers the texture's width and height (the neutral it bound before answered
/// zero); none announces the neutral.
#[test]
fn cross_class_queries_are_the_textures_size() {
    on_device(|device, messages| {
        let mut pipelines: BTreeMap<Ty, ComputePipelineState> = BTreeMap::new();
        let (mut checked, mut differ) = (0usize, 0usize);
        let _ = messages.take_notices();
        for &(format, bytes, _) in &FORMATS {
            let class = sampled_type(format).expect("a format of a class");
            let Some((texture, _)) = made(device, (format, bytes), Kind::Query, None) else {
                continue;
            };
            for ty in Ty::ALL.into_iter().filter(|ty| ty.class() != class) {
                let state = pipelines
                    .entry(ty)
                    .or_insert_with(|| pipeline_of(device, Kind::Query, ty))
                    .clone();
                let got = dispatch(device, (&state, Kind::Query), &texture, &[0; 16])
                    .unwrap_or_else(|e| panic!("query {ty:?} over {format:?}: {e}"));
                checked += 1;
                if got[0][..2] != [W as u32, H as u32] {
                    differ += 1;
                    println!(
                        "CROSS-CLASS-QUERY-DIFFERS {}: {ty:?} over {format:?}: {:?}",
                        device.name(),
                        &got[0][..2]
                    );
                }
            }
        }
        let neutral = neutral_announced(messages);
        println!(
            "CROSS-CLASS queries {}: {checked} checked, {differ} differ, announced {neutral:?}",
            device.name()
        );
        assert!(checked > 0);
        assert_eq!(
            differ, 0,
            "cross-class queries differ from the texture's size"
        );
        assert!(neutral.is_empty(), "an exact query bound the neutral");
    });
}
