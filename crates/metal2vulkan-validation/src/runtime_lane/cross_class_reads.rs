//! Reads of a texture through a slot of another numeric class, on the device the cfg selects, under
//! the validation layer.
//!
//! Plain Metal reads, samples and gathers a texture bound at a slot whose declared texel type is
//! of another class than its format's (float or half, int, uint) in the format's own class, and
//! hands back the words as the declared type: a 32-bit word as is, a `half` result of an integer
//! format each word's low 16 bits, an integer format's linear sample its nearest one (our probe,
//! 536 of 536 cross-class results). The function reads such a texture under its slot's
//! `cross_class` variant (`VariantKey::cross_class`), which declares the
//! image in the bound class and hands back the words.
//!
//! This lane asks, for each format of [`FORMATS`] and each declared type of another class,
//! whether a read (every texel, and three past the texture), a sample through a nearest and a
//! linear constexpr sampler and a linear runtime sampler, and a gather of the red and alpha
//! components (at [`sample_lanes`]) give, word for word, the same access's result through a slot
//! of the format's own class, a `half` result the low 16 bits of each. A lane that differs is
//! printed on a `CROSS-CLASS-DIFFERS` line. That the new path is what ran is shown twice: the
//! device's compile layer settles each case's bind on the variant naming the slot with the
//! format's class (printed on a `CROSS-CLASS-VARIANT` line per kernel and class, and asserted;
//! the plan of the variant must name it again), and no exact case announces
//! `cross-class-texture-neutral`, the slot's neutral (whose zero texels would differ besides).
//!
//! A `ushort` result is read too: an integer format's words' low 16 bits, a float format's `half`
//! result's bits.
//!
//! A `half` result's lane whose bits are a NaN is the device's where the stage leaves a `half`
//! NaN's bits to it: where the module rounds through the device's conversion
//! (`PrecisionNote::NativeHalfConversion`, whose widening may quiet a signalling NaN, as lavapipe's
//! does) or widens a `half` by the device (`NativeNanEncoding`). Such a lane is any NaN, counted
//! apart (`CROSS-CLASS-NAN` lines); every other lane stays the native slot's.
//!
//! A slot whose access has no exact answer (a `uint` read of a depth texture here) binds the
//! declared type's neutral, and the device announces `cross-class-texture-neutral` once
//! ([`CROSS_CLASS`]).
//!
//! A texture an argument buffer holds is read the same way (the probe's argument buffer half, the
//! same 536 results): the function's read of such a texture reads a member of another numeric class
//! through the heap's array of that class, where the handle's class index names one, and hands back
//! the words. The lane reads each format of [`FORMATS`] and `RGBX8Uint` through an argument buffer
//! in each foreign declared type, with its width and height, against the native slot's read (every
//! lane, and the queries, the slot's; a read that went to the neutral would give zero texels and a
//! 1 x 1 size). A sample through an argument buffer has no answer for such a member yet: it reads
//! the neutral, announced the same way. One difference may be announced, and is printed on a
//! `CROSS-CLASS-ANNOUNCED` line and counted apart: an 8-bit sRGB texture's read is the device's
//! decode, where the stage records `PrecisionNote::SrgbTexelDecode` and the device announced it
//! (as `heap_extended_reads` allows the same texture's own-class read).
//!
//! The words of an 8-bit sRGB texture's gather are also checked against Metal's own answer, not
//! only against the native slot's: Metal gathers an `RGBA8Unorm_sRGB` texture through a `float`,
//! `int`, or `uint` slot as the table's decode of each footprint texel's code, the same bits
//! through each (the probe's texture and positions, 240 of 240 words through each class on an M4
//! and an M3). The device decodes the texels: through Metal's own texture unit Metal's words,
//! elsewhere the device's own, the same through each class, announced.

use super::degrade_notices::on_device_with;
use super::driver_deviations::half_nan;
use super::{at_plan, buffer_with, read_buffer, Messages};
use crate::air_library::AirLibrary;
use crate::candidate::engine::variants::compile_compute_for_shape;
use metal2vulkan::bind::{
    BindState, BoundBuffer, BoundTexture, SamplerDescriptor, StageTable, TextureFacts,
};
use metal2vulkan::cache::VariantKey;
use metal2vulkan::library::Library;
use metal2vulkan::metal::format::{sampled_type, srgb_raw_twin};
use metal2vulkan::metal::{
    PixelFormat, ResourceUsage, SamplerAddressMode, SamplerMinMagFilter, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, PrecisionNote, SampledType, SlotClass, SlotRef,
    Stage, StageSource,
};
use metal2vulkan::Budget;
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Degradation, Device, PipelineOptions, Region,
    ResourceOptions, SamplerState, Size, Texture, TextureDescriptor,
};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

/// The kind a bind of the declared type's neutral announces
/// ([`an_unanswered_cross_class_access_binds_the_neutral_announced`]).
pub(super) const CROSS_CLASS: [Degradation; 1] = [Degradation::CrossClassTextureNeutral];

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The texture's width and height (one level).
const W: u32 = 8;
const H: u32 = 4;

/// Texture slot 0 of the kernel.
const TEXTURE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// The formats read through each foreign class, each with its texel's bytes: float-class
/// formats of each kind (normalized, 8-bit sRGB, packed, half and float), and signed and
/// unsigned integer formats of each width.
const FORMATS: [(PixelFormat, u64); 24] = {
    use PixelFormat as P;
    [
        (P::R8Unorm, 1),
        (P::RGBA8Unorm, 4),
        (P::RGBA8UnormSRGB, 4),
        (P::BGRA8Unorm, 4),
        (P::RGBA8Snorm, 4),
        (P::RGB10A2Unorm, 4),
        (P::RG11B10Float, 4),
        (P::RGB9E5Float, 4),
        (P::R16Float, 2),
        (P::RGBA16Unorm, 8),
        (P::RGBA16Float, 8),
        (P::R32Float, 4),
        (P::RGBA32Float, 16),
        (P::R8Sint, 1),
        (P::RGBA8Sint, 4),
        (P::RG16Sint, 4),
        (P::RGBA16Sint, 8),
        (P::R32Sint, 4),
        (P::RGBA32Sint, 16),
        (P::R8Uint, 1),
        (P::RGBA8Uint, 4),
        (P::RGB10A2Uint, 4),
        (P::RGBA16Uint, 8),
        (P::RGBA32Uint, 16),
    ]
};

/// A texel type a kernel declares at its texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Ty {
    Float,
    Half,
    Int,
    Uint,
    Ushort,
}

impl Ty {
    /// The types the lane reads an argument buffer's texture through, every one of 32-bit or
    /// `half` lanes.
    const EXACT: [Ty; 4] = [Ty::Float, Ty::Half, Ty::Int, Ty::Uint];

    /// The types the lane reads a slot's texture through: [`EXACT`](Self::EXACT) and `ushort`.
    const SLOT: [Ty; 5] = [Ty::Float, Ty::Half, Ty::Int, Ty::Uint, Ty::Ushort];

    /// The MSL element, the LLVM vector, its store alignment, and the intrinsics' suffix.
    fn spelling(self) -> (&'static str, &'static str, u32, &'static str) {
        match self {
            Ty::Float => ("float", "<4 x float>", 16, "v4f32"),
            Ty::Half => ("half", "<4 x half>", 8, "v4f16"),
            Ty::Int => ("int", "<4 x i32>", 16, "s.v4i32"),
            Ty::Uint => ("uint", "<4 x i32>", 16, "u.v4i32"),
            Ty::Ushort => ("ushort", "<4 x i16>", 8, "u.v4i16"),
        }
    }

    /// Its numeric class.
    fn class(self) -> SampledType {
        match self {
            Ty::Float | Ty::Half => SampledType::Float,
            Ty::Int => SampledType::Sint,
            Ty::Uint | Ty::Ushort => SampledType::Uint,
        }
    }

    /// The 32-bit type of class `class`: the native slot's.
    fn native(class: SampledType) -> Ty {
        match class {
            SampledType::Float => Ty::Float,
            SampledType::Sint => Ty::Int,
            _ => Ty::Uint,
        }
    }

    /// The type of the native slot whose answer this type's result over a texture of class
    /// `class` takes: a float format's `half` result for a `ushort` one (Metal's 16-bit integer
    /// result of a float texel is its `half` bits), else [`native`](Self::native).
    fn answered_by(self, class: SampledType) -> Ty {
        match (self, class) {
            (Ty::Ushort, SampledType::Float) => Ty::Half,
            _ => Ty::native(class),
        }
    }

    /// Whether its lanes are 16-bit.
    fn narrow(self) -> bool {
        matches!(self, Ty::Half | Ty::Ushort)
    }
}

/// A constexpr sampler's words (AIR's `air.sampler_state` layout): no mip filter, normalized
/// coordinates, every axis clamped to the edge, both filters linear where `linear`.
const fn sampler_words(linear: bool) -> u64 {
    let filters = if linear { 1 << 9 | 1 << 11 } else { 0 };
    0x007b_ff00_0008_0000 | 1 | 1 << 3 | 1 << 6 | filters
}

/// The sampler a sample or gather goes through: a constexpr sampler of these words, or the
/// runtime sampler at sampler slot 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Through {
    Constexpr(u64),
    Slot,
}

/// What a kernel does with its texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Op {
    /// A read of an `access::read` texture at buffer 0's lane (`x`, `y`).
    Read,
    /// A sample at buffer 0's lane (two floats), at level 0.
    Sample(Through),
    /// A gather of a component (0 red to 3 alpha) at buffer 0's lane (two floats).
    Gather(Through, u32),
}

/// Synthetic AIR written for this test: kernel `k` accesses (`op`) the `texture2d<T>` (`T` as
/// `ty` says) at texture slot 0, at buffer 0's lane `gid` (a `uint4`: a read's texel `x` and
/// `y`, or a sample's or gather's coordinate's two floats), storing the result as its declared
/// type to buffer 1's 16 bytes at `gid` (a 16-bit one's first 8).
fn kernel(op: Op, ty: Ty) -> String {
    kernel_at(op, ty, false)
}

/// [`kernel`], or where `held` one accessing the texture its argument buffer (buffer 2,
/// `{ texture2d<T> t; }`) holds, storing the result to buffer 1's `uint4` at `2 gid` and the
/// texture's width and height to the one at `2 gid + 1`.
fn kernel_at(op: Op, ty: Ty, held: bool) -> String {
    let (element, vec4, align, suffix) = ty.spelling();
    let (access, through) = match op {
        Op::Read => ("read", None),
        Op::Sample(through) | Op::Gather(through, _) => ("sample", Some(through)),
    };
    let texture = format!("texture2d<{element}, {access}>");
    let (states, sampler_param, sampler, sampler_meta, gid_index, args) = match through {
        Some(Through::Constexpr(words)) => (
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8\n"
            ),
            "",
            "@__air_sampler_state",
            "!air.sampler_states = !{!10}\n!10 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
            3,
            "!3, !4, !5, !7",
        ),
        Some(Through::Slot) => (
            String::new(),
            "ptr addrspace(2) %s, ",
            "%s",
            "",
            4,
            "!3, !4, !5, !6, !7",
        ),
        None => (String::new(), "", "", "", 3, "!3, !4, !5, !7"),
    };
    let (at, call, decl) = match op {
        Op::Read => (
            "  %c4 = load <4 x i32>, ptr addrspace(1) %p, align 16\n  %at = shufflevector <4 x i32> %c4, <4 x i32> poison, <2 x i32> <i32 0, i32 1>\n  %rs = call ptr addrspace(2) @air.get_read_sampler()\n",
            format!("@air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %at, <2 x i32> zeroinitializer, i32 0, i32 1)"),
            format!("declare ptr addrspace(2) @air.get_read_sampler()\ndeclare {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)"),
        ),
        Op::Sample(_) => (
            "  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16\n  %uv = shufflevector <4 x float> %c4, <4 x float> poison, <2 x i32> <i32 0, i32 1>\n",
            format!("@air.sample_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)"),
            format!("declare {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)"),
        ),
        Op::Gather(_, component) => (
            "  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16\n  %uv = shufflevector <4 x float> %c4, <4 x float> poison, <2 x i32> <i32 0, i32 1>\n",
            format!("@air.gather_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 {component}, i32 0)"),
            format!("declare {{ {vec4}, i8 }} @air.gather_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)"),
        ),
    };
    let lane_type = match op {
        Op::Read => "uint4",
        _ => "float4",
    };
    let (holder, load, at_out, sizes, sizes_decl, texture_meta) = match held {
        false => (
            "ptr addrspace(1) %t",
            "",
            "%i",
            "",
            "",
            format!(
                r#"!5 = !{{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{access}", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}"#
            ),
        ),
        true => (
            "ptr addrspace(2) %m",
            "  %pt = getelementptr inbounds %struct.M, ptr addrspace(2) %m, i64 0, i32 0\n  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8\n",
            "%o",
            "  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)\n  %h = call i32 @air.get_height_texture_2d(ptr addrspace(1) %t, i32 0)\n  %z0 = insertelement <4 x i32> zeroinitializer, i32 %w, i64 0\n  %z1 = insertelement <4 x i32> %z0, i32 %h, i64 1\n  %o1 = add i64 %o, 1\n  %qz = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i64 %o1\n  store <4 x i32> %z1, ptr addrspace(1) %qz, align 16\n",
            "declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)\ndeclare i32 @air.get_height_texture_2d(ptr addrspace(1), i32)\n",
            format!(
                r#"!5 = !{{i32 2, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !8, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"M", !"air.arg_name", !"m"}}
!8 = !{{i32 0, i32 8, i32 0, !"{texture}", !"t", !"air.indirect_argument", !9}}
!9 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{access}", !"air.arg_type_name", !"{texture}", !"air.arg_name", !"t"}}"#
            ),
        ),
    };
    let held_type = match held {
        true => "%struct.M = type { ptr addrspace(1) }\n",
        false => "",
    };
    format!(
        r#"{held_type}{states}
define void @k(ptr addrspace(1) %c, ptr addrspace(1) %out, {holder}, {sampler_param}i32 %gid) {{
{load}  %i = zext i32 %gid to i64
  %o = shl i64 %i, 1
  %p = getelementptr inbounds <4 x i32>, ptr addrspace(1) %c, i64 %i
{at}  %r = call {{ {vec4}, i8 }} {call}
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i64 {at_out}
  store {vec4} %v, ptr addrspace(1) %q, align {align}
{sizes}  ret void
}}

{decl}
{sizes_decl}
!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
{sampler_meta}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args}}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"{lane_type}", !"air.arg_name", !"c"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}}
{texture_meta}
!6 = !{{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
!7 = !{{i32 {gid_index}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// The probe's bytes of `n` texels of the format of raw value `raw`, `bytes` a texel: a 64-bit
/// linear congruential state seeded by it, each byte its top 8 bits.
fn texels(raw: u32, n: u64, bytes: u64) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(raw);
    (0..n * bytes)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        })
        .collect()
}

/// A read's lanes: every texel, then three past the texture.
fn read_lanes() -> Vec<[u32; 4]> {
    let mut lanes: Vec<[u32; 4]> = (0..W * H).map(|i| [i % W, i / W, 0, 0]).collect();
    lanes.extend([[W, 0, 0, 0], [0, H + 2, 0, 0], [W + 9, H + 9, 0, 0]]);
    lanes
}

/// A sample's or gather's lanes: each a coordinate (two floats) of a 14 x 14 grid inside the
/// texture, on its edges and past them, at texel centres and between them.
fn sample_lanes() -> Vec<[u32; 4]> {
    const AT: [f32; 14] = [
        -0.3, -0.0625, 0.0, 0.03125, 0.1, 0.2, 0.333, 0.5, 0.55, 0.7, 0.9375, 1.0, 1.1, 1.6,
    ];
    AT.iter()
        .flat_map(|&v| AT.iter().map(move |&u| [u.to_bits(), v.to_bits(), 0, 0]))
        .collect()
}

/// The pipeline state of AIR `text`'s kernel `k`.
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

/// The variant key the device's compile layer settles a bind of a `format` texture at kernel
/// `text`'s slot 0 on (its plan's own key, which the variant's plan names again), and the precision
/// notes of that variant's stage.
fn settled_key(
    device: &Device,
    text: &str,
    format: PixelFormat,
) -> (VariantKey, BTreeSet<PrecisionNote>) {
    let compile = device.compile_device();
    let bitcode = llvm_assemble_text(text).unwrap();
    let library = Library::from_air(&bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function("k").unwrap());
    let mut table = StageTable::default();
    let texture = BoundTexture::Texture {
        facts: TextureFacts::new(TextureType::Type2D, format),
        resource_id: None,
    };
    table.textures.insert(0, texture);
    for index in [0, 1] {
        let buffer = BoundBuffer::Buffer {
            length: 4096,
            offset: 0,
            gpu_address: 0x1_0000,
            device_address: None,
            attribute_stride: None,
        };
        table.buffers.insert(index, buffer);
    }
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, table);
    let budget = Budget::standard();
    let (pipeline, key) =
        compile_compute_for_shape(compile, &function, &state.shape(), None, None, &budget)
            .unwrap_or_else(|e| panic!("{format:?}: {e:?}"));
    (key, pipeline.stage.precision_notes.into_iter().collect())
}

/// Whether a stage whose notes are `notes` leaves a `half` NaN's bits to the device: it rounds
/// through the device's conversion and may widen a `half` by it (`PrecisionNote::NativeHalfConversion`),
/// or widens one by the device (`NativeNanEncoding`).
fn leaves_half_nan_bits(notes: &BTreeSet<PrecisionNote>) -> bool {
    notes.contains(&PrecisionNote::NativeHalfConversion)
        || notes.contains(&PrecisionNote::NativeNanEncoding)
}

/// Whether the words `got` of a `ty` result answer the `want` the native slot's result gives:
/// the same words, but where `nan_bits` (the stage leaves a `half` NaN's bits to the device,
/// [`leaves_half_nan_bits`]) a `half` result's lane (the low and high halves of its first two
/// words) that is a NaN in both may hold other bits.
fn holds(got: &[u32], want: &[u32], ty: Ty, nan_bits: bool) -> bool {
    if got == want {
        return true;
    }
    if !(nan_bits && ty == Ty::Half) || got[2..] != want[2..] {
        return false;
    }
    got[..2].iter().zip(&want[..2]).all(|(&g, &w)| {
        [0, 16].into_iter().all(|shift| {
            let (g, w) = ((g >> shift) as u16, (w >> shift) as u16);
            g == w || (half_nan(g) && half_nan(w))
        })
    })
}

/// An 8 x 4 texture of `format` (texels of `bytes` bytes), one level, the probe's bytes.
fn made(device: &Device, format: PixelFormat, bytes: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, u64::from(W), u64::from(H), false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("{format:?}: {e}"));
    let (w, h) = (u64::from(W), u64::from(H));
    texture
        .replace_region(
            Region::new_2d(0, 0, w, h),
            0,
            0,
            &texels(format.as_raw(), w * h, bytes),
            w * bytes,
            0,
        )
        .unwrap();
    texture
}

/// Each lane's four output words of `pipeline` accessing `texture` at slot 0, with `sampler` at
/// sampler slot 0 where there is one, buffer 0 holding `lanes`. A 16-bit result is the first two
/// words; the rest keep the fill, `0x5a5a5a5a`.
fn dispatch(
    device: &Device,
    pipeline: &ComputePipelineState,
    (texture, sampler): (&Texture, Option<&SamplerState>),
    lanes: &[[u32; 4]],
) -> Vec<[u32; 4]> {
    dispatch_words(device, (pipeline, false), (texture, sampler), lanes)
}

/// [`dispatch`] of a [`kernel_at`] pipeline where `held` the texture its argument buffer holds
/// (made resident), each lane's `N` output words.
fn dispatch_words<const N: usize>(
    device: &Device,
    (pipeline, held): (&ComputePipelineState, bool),
    (texture, sampler): (&Texture, Option<&SamplerState>),
    lanes: &[[u32; 4]],
) -> Vec<[u32; N]> {
    let n = lanes.len() as u64;
    let stride = 4 * N as u64;
    let input: Vec<u8> = lanes
        .iter()
        .flatten()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let input = buffer_with(device, &input).unwrap();
    let out = buffer_with(device, &vec![0x5a; (n * stride) as usize]).unwrap();
    let argument = buffer_with(device, &texture.gpu_resource_id().0.to_le_bytes()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(pipeline).unwrap();
    match held {
        true => {
            encoder
                .use_resource(texture.as_resource(), ResourceUsage::READ)
                .unwrap();
            encoder.set_buffer(Some(&argument), 0, 2).unwrap();
        }
        false => encoder.set_texture(Some(texture), 0).unwrap(),
    }
    if let Some(sampler) = sampler {
        encoder.set_sampler_state(Some(sampler), 0).unwrap();
    }
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(n, 1, 1), Size::new(n.min(64), 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let bytes = read_buffer(&out, 0, n * stride).unwrap();
    bytes
        .chunks_exact(stride as usize)
        .map(|v| {
            std::array::from_fn(|k| u32::from_le_bytes(v[4 * k..4 * k + 4].try_into().unwrap()))
        })
        .collect()
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

/// Whether the native result answering a `ty` result over a texture of class `class` is 16-bit
/// ([`Ty::answered_by`]).
fn native_type_narrow(ty: Ty, class: SampledType) -> bool {
    ty.answered_by(class).narrow()
}

/// What one test's cases came to.
#[derive(Default)]
struct Tally {
    /// Lanes compared.
    checked: usize,
    /// Lanes not the native slot's.
    differ: usize,
    /// Lanes the native slot's but a `half` NaN's bits, which the stage leaves to the device.
    nans: usize,
    /// The (op, declared type, class) whose settled variant was checked.
    variants: BTreeSet<(Op, Ty, SampledType)>,
}

/// Compare every format of [`FORMATS`] through each foreign declared type with its native slot,
/// for each op of `ops` at `lanes`, `sampler` at sampler slot 0 for a runtime sampler's.
fn compare(
    (device, messages): (&Device, &Messages),
    ops: &[Op],
    lanes: &[[u32; 4]],
    sampler: Option<&SamplerState>,
) -> Tally {
    let mut tally = Tally::default();
    let mut pipelines: BTreeMap<(Op, Ty), ComputePipelineState> = BTreeMap::new();
    let mut nan_bits: BTreeMap<(Op, Ty, SampledType), bool> = BTreeMap::new();
    let _ = messages.take_notices();
    for &(format, bytes) in &FORMATS {
        let class = sampled_type(format).expect("a format of a class");
        let texture = made(device, format, bytes);
        for &op in ops {
            let through = match op {
                Op::Sample(Through::Slot) | Op::Gather(Through::Slot, _) => sampler,
                _ => None,
            };
            let mut state = |ty: Ty| {
                pipelines
                    .entry((op, ty))
                    .or_insert_with(|| pipeline_of(device, &kernel(op, ty)))
                    .clone()
            };
            let mut answers: BTreeMap<Ty, Vec<[u32; 4]>> = BTreeMap::new();
            for ty in Ty::SLOT.into_iter().filter(|ty| ty.class() != class) {
                let native = ty.answered_by(class);
                let want = answers.entry(native).or_insert_with(|| {
                    let own = state(native);
                    dispatch(device, &own, (&texture, through), lanes)
                });
                let cross = state(ty);
                if tally.variants.insert((op, ty, class)) {
                    let (key, notes) = settled_key(device, &kernel(op, ty), format);
                    nan_bits.insert((op, ty, class), leaves_half_nan_bits(&notes));
                    println!(
                        "CROSS-CLASS-VARIANT {}: {op:?} {ty:?} over {format:?}: cross_class {:?}",
                        device.name(),
                        key.cross_class
                    );
                    assert_eq!(
                        key.cross_class,
                        BTreeMap::from([(TEXTURE, class)]),
                        "{op:?} {ty:?} over {format:?}: the settled variant"
                    );
                }
                let got = dispatch(device, &cross, (&texture, through), lanes);
                tally.checked += lanes.len();
                // A 16-bit native result (a float format's `half` one) is the answer as it is.
                let expected = |native: [u32; 4], ty: Ty| match native_type_narrow(ty, class) {
                    true => native,
                    false => expected(native, ty),
                };
                let device_nans = nan_bits[&(op, ty, class)];
                let wrong: Vec<usize> = (0..lanes.len())
                    .filter(|&i| !holds(&got[i], &expected(want[i], ty), ty, device_nans))
                    .collect();
                let nans = (0..lanes.len())
                    .filter(|&i| got[i] != expected(want[i], ty))
                    .count()
                    - wrong.len();
                if nans > 0 {
                    tally.nans += nans;
                    println!(
                        "CROSS-CLASS-NAN {}: {op:?} {ty:?} over {format:?}: {nans} of {} lanes \
                         hold a half NaN of the device's bits",
                        device.name(),
                        lanes.len()
                    );
                }
                if let Some(&i) = wrong.first() {
                    tally.differ += wrong.len();
                    println!(
                        "CROSS-CLASS-DIFFERS {}: {op:?} {ty:?} over {format:?}: {} of {} lanes, \
                         first at {:08x?}: {:08x?}, the native slot's {:08x?} (expected \
                         {:08x?})",
                        device.name(),
                        wrong.len(),
                        lanes.len(),
                        lanes[i],
                        got[i],
                        want[i],
                        expected(want[i], ty)
                    );
                }
            }
        }
    }
    let kinds: BTreeSet<Degradation> = messages
        .take_notices()
        .iter()
        .filter_map(|m| m.degradation.as_ref())
        .map(|n| n.degradation)
        .collect();
    assert!(
        !kinds.contains(&Degradation::CrossClassTextureNeutral),
        "an exact case bound the neutral: {kinds:?}"
    );
    tally
}

/// Print and check `tally` of test `what`.
fn report(device: &Device, what: &str, tally: &Tally) {
    println!(
        "CROSS-CLASS {what} {}: {} lanes checked, {} differ, {} more hold a half NaN of the \
         device's bits, {} variants settled",
        device.name(),
        tally.checked,
        tally.differ,
        tally.nans,
        tally.variants.len()
    );
    assert!(tally.checked > 0);
    assert_eq!(
        tally.differ, 0,
        "{what}: cross-class results differ from the native slot's"
    );
}

/// Each read of each format of [`FORMATS`] through each foreign declared type is the native
/// slot's read's words, a `half` result their low 16 bits: every texel and three past the
/// texture.
#[test]
fn cross_class_reads_are_the_native_words() {
    on_device_with(at_plan, |device, messages| {
        let tally = compare((device, messages), &[Op::Read], &read_lanes(), None);
        report(device, "reads", &tally);
    });
}

/// Each sample of each format of [`FORMATS`] through each foreign declared type is the native
/// slot's sample's words, a `half` result their low 16 bits: through a nearest and a linear
/// constexpr sampler (an integer format's linear sample is its nearest one).
#[test]
fn cross_class_samples_are_the_native_words() {
    on_device_with(at_plan, |device, messages| {
        let ops = [
            Op::Sample(Through::Constexpr(sampler_words(false))),
            Op::Sample(Through::Constexpr(sampler_words(true))),
        ];
        let tally = compare((device, messages), &ops, &sample_lanes(), None);
        report(device, "samples", &tally);
    });
}

/// Each sample through a linear runtime sampler (clamped to the edge, no mip filter) at sampler
/// slot 0, and each gather of the red and alpha components through a nearest constexpr sampler,
/// of each format of [`FORMATS`] through each foreign declared type is the native slot's words.
#[test]
fn cross_class_runtime_samples_and_gathers_are_the_native_words() {
    on_device_with(at_plan, |device, messages| {
        let mut linear = SamplerDescriptor::new();
        linear.min_filter = SamplerMinMagFilter::Linear;
        linear.mag_filter = SamplerMinMagFilter::Linear;
        linear.s_address_mode = SamplerAddressMode::ClampToEdge;
        linear.t_address_mode = SamplerAddressMode::ClampToEdge;
        let linear = device.new_sampler_state(&linear).unwrap();
        let nearest = Through::Constexpr(sampler_words(false));
        let ops = [
            Op::Sample(Through::Slot),
            Op::Gather(nearest, 0),
            Op::Gather(nearest, 3),
        ];
        let tally = compare((device, messages), &ops, &sample_lanes(), Some(&linear));
        report(device, "runtime samples and gathers", &tally);
    });
}

/// The probe's gather positions (our probe's, run on an M4 and an M3):
/// on each row, at its texels' centres and three eighths of a texel below them, on columns -1,
/// 0, 2, 4, 7, and 8, at their centres and a quarter of a texel left of them; the first 60. No
/// subtexel rounding moves a footprint: every coordinate is exact in `f32`, and each lands a
/// quarter texel or more from a footprint's edge, or on a texel's centre.
fn probe_gather_lanes() -> Vec<[u32; 4]> {
    let mut lanes = Vec::new();
    for ky in 0..H {
        for oy in [0.5, 0.875] {
            for kx in [-1i32, 0, 2, 4, 7, 8] {
                for ox in [0.25, 0.5] {
                    let u = (f64::from(kx) + ox) / f64::from(W);
                    let v = (f64::from(ky) + oy) / f64::from(H);
                    lanes.push([(u as f32).to_bits(), (v as f32).to_bits(), 0, 0]);
                }
            }
        }
    }
    lanes.truncate(60);
    lanes
}

/// Metal's gather of the red component of `bytes`, an `RGBA8Unorm_sRGB` texture's texels, at
/// `lane` through a clamp-to-edge sampler: the footprint's texels `(i0, j1)`, `(i1, j1)`,
/// `(i1, j0)`, `(i0, j0)`, each code decoded to `step / 4095`, correctly rounded (`4095` is
/// `2^12 - 1`, so the quotient's bits repeat the step's twelve and no `f64` result lies on an
/// `f32` tie).
fn metal_srgb_gather(bytes: &[u8], lane: [u32; 4]) -> [u32; 4] {
    let corner = |at: u32, n: u32| (f64::from(f32::from_bits(at)) * f64::from(n) - 0.5).floor();
    let clamp = |i: f64, n: u32| i.clamp(0.0, f64::from(n - 1)) as usize;
    let (x, y) = (corner(lane[0], W), corner(lane[1], H));
    let (i0, i1) = (clamp(x, W), clamp(x + 1.0, W));
    let (j0, j1) = (clamp(y, H), clamp(y + 1.0, H));
    let decode = |i: usize, j: usize| {
        let code = u32::from(bytes[4 * (j * W as usize + i)]);
        ((f64::from(super::srgb_step(code)) / 4095.0) as f32).to_bits()
    };
    [
        decode(i0, j1),
        decode(i1, j1),
        decode(i1, j0),
        decode(i0, j0),
    ]
}

/// A gather of an `RGBA8Unorm_sRGB` texture's red component through a slot of each class is the
/// device's decode of the footprint texels, the same bits through a `float`, an `int`, and a `uint`
/// slot. Through Metal's own texture unit (`VkProperty::MetalSampler`) each word is Metal's, the
/// footprint texel's code decoded through Metal's table (our probe's texture and positions: 240 of
/// 240 words through each on an M4 and an M3), and nothing announces the device's decode;
/// elsewhere the gathers announce `srgb-texel-decode`, and the words that differ from Metal's are
/// counted.
#[test]
fn an_srgb_gather_through_each_class_is_the_devices_decode() {
    on_device_with(at_plan, |device, messages| {
        let format = PixelFormat::RGBA8UnormSRGB;
        let (w, h) = (u64::from(W), u64::from(H));
        let bytes = texels(format.as_raw(), w * h, 4);
        let texture = made(device, format, 4);
        let lanes = probe_gather_lanes();
        let want: Vec<[u32; 4]> = lanes
            .iter()
            .map(|&l| metal_srgb_gather(&bytes, l))
            .collect();
        let op = Op::Gather(Through::Constexpr(sampler_words(false)), 0);
        let _ = messages.take_notices();
        let mut differ = 0;
        let mut through: Vec<Vec<[u32; 4]>> = Vec::new();
        for ty in [Ty::Float, Ty::Int, Ty::Uint] {
            let pipeline = pipeline_of(device, &kernel(op, ty));
            let got = dispatch(device, &pipeline, (&texture, None), &lanes);
            through.push(got.clone());
            let wrong: Vec<usize> = (0..lanes.len()).filter(|&i| got[i] != want[i]).collect();
            println!(
                "CROSS-CLASS-SRGB-GATHER {}: {ty:?}: {} words, {} differ from Metal's table{}",
                device.name(),
                4 * lanes.len(),
                wrong
                    .iter()
                    .map(|&i| (0..4).filter(|&k| got[i][k] != want[i][k]).count())
                    .sum::<usize>(),
                wrong
                    .first()
                    .map(|&i| format!(
                        ", first at {:08x?}: {:08x?}, Metal's {:08x?}",
                        lanes[i], got[i], want[i]
                    ))
                    .unwrap_or_default()
            );
            differ += wrong.len();
        }
        let kinds: BTreeSet<Degradation> = messages
            .take_notices()
            .iter()
            .filter_map(|m| m.degradation.as_ref())
            .map(|n| n.degradation)
            .collect();
        let srgb = Degradation::from_precision_note(PrecisionNote::SrgbTexelDecode);
        let metals = device
            .compile_device()
            .profile()
            .enabled()
            .properties
            .contains(&metal2vulkan::device::VkProperty::MetalSampler);
        assert!(
            through.iter().all(|got| *got == through[0]),
            "sRGB gathers differ between the classes"
        );
        assert_eq!(
            kinds.contains(&srgb),
            !metals,
            "the device's decode announced: {kinds:?}"
        );
        if metals {
            assert_eq!(differ, 0, "sRGB gathers differ from Metal's table decode");
        }
    });
}

/// A `uint` read of a `Depth32Float` texture, which has no exact answer yet (a depth format's; a
/// `ushort` sample of a color texture is answered since), binds the declared type's neutral (zero
/// texels) at each of two dispatches, and the device announces `cross-class-texture-neutral` once.
#[test]
fn an_unanswered_cross_class_access_binds_the_neutral_announced() {
    on_device_with(at_plan, |device, messages| {
        let pipeline = pipeline_of(device, &kernel(Op::Read, Ty::Uint));
        let mut descriptor = TextureDescriptor::texture_2d(
            PixelFormat::Depth32Float,
            u64::from(W),
            u64::from(H),
            false,
        );
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::private();
        let texture = device.new_texture(&descriptor).unwrap();
        let lanes = read_lanes();
        let _ = messages.take_notices();
        for _ in 0..2 {
            let got = dispatch(device, &pipeline, (&texture, None), &lanes);
            assert!(
                got.iter().all(|w| w[..3] == [0, 0, 0]),
                "{:08x?}",
                &got[..4]
            );
        }
        let announced: Vec<Degradation> = messages
            .take_notices()
            .iter()
            .filter_map(|m| m.degradation.as_ref())
            .map(|n| n.degradation)
            .filter(|kind| CROSS_CLASS.contains(kind))
            .collect();
        println!(
            "CROSS-CLASS neutral {}: announced {announced:?}",
            device.name()
        );
        assert_eq!(announced, CROSS_CLASS);
    });
}

/// The formats read through an argument buffer: [`FORMATS`], and `RGBX8Uint` (an integer format
/// whose alpha Metal reads as 1).
fn held_formats() -> Vec<(PixelFormat, u64)> {
    let mut formats = FORMATS.to_vec();
    formats.push((PixelFormat::RGBX8Uint, 4));
    formats
}

/// The precision notes of the device's compile of kernel `text`'s stage.
fn stage_notes(device: &Device, text: &str) -> BTreeSet<PrecisionNote> {
    let compile = device.compile_device();
    let budget = Budget::standard();
    let bitcode = llvm_assemble_text(text).unwrap();
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

/// Each read of each format of [`held_formats`] an argument buffer holds, through each foreign
/// declared type, is the native slot's read's words (a `half` result their low 16 bits), every
/// texel and three past the texture, and its width and height queries are the texture's, but
/// for an 8-bit sRGB texture's announced device decode (the module docs); no such read announces
/// `cross-class-texture-neutral`.
#[test]
fn argument_buffer_cross_class_reads_are_the_native_words() {
    on_device_with(at_plan, |device, messages| {
        let lanes = read_lanes();
        let mut pipelines: BTreeMap<(Ty, bool), ComputePipelineState> = BTreeMap::new();
        let mut state = |ty: Ty, held: bool| {
            pipelines
                .entry((ty, held))
                .or_insert_with(|| pipeline_of(device, &kernel_at(Op::Read, ty, held)))
                .clone()
        };
        let mut notes: BTreeMap<Ty, BTreeSet<PrecisionNote>> = BTreeMap::new();
        let mut kinds: BTreeSet<Degradation> = BTreeSet::new();
        let (mut checked, mut differ, mut announced, mut words) = (0usize, 0, 0, 0);
        let mut nans = 0;
        let _ = messages.take_notices();
        for (format, bytes) in held_formats() {
            let class = sampled_type(format).expect("a format of a class");
            let texture = made(device, format, bytes);
            let native = state(Ty::native(class), false);
            let want = dispatch(device, &native, (&texture, None), &lanes);
            for ty in Ty::EXACT.into_iter().filter(|ty| ty.class() != class) {
                let held = state(ty, true);
                let got: Vec<[u32; 8]> =
                    dispatch_words(device, (&held, true), (&texture, None), &lanes);
                kinds.extend(
                    messages
                        .take_notices()
                        .iter()
                        .filter_map(|m| m.degradation.as_ref())
                        .map(|n| n.degradation),
                );
                // The device's decode of an 8-bit sRGB texture's texels, announced.
                let srgb = PrecisionNote::SrgbTexelDecode;
                let noted = notes
                    .entry(ty)
                    .or_insert_with(|| stage_notes(device, &kernel_at(Op::Read, ty, true)));
                let decodes = srgb_raw_twin(format).is_some()
                    && noted.contains(&srgb)
                    && kinds.contains(&Degradation::from_precision_note(srgb));
                checked += lanes.len();
                let answer = |i: usize| -> [u32; 6] {
                    let e = expected(want[i], ty);
                    [e[0], e[1], e[2], e[3], W, H]
                };
                let device_nans = leaves_half_nan_bits(noted);
                let wrong: Vec<usize> = (0..lanes.len())
                    .filter(|&i| !holds(&got[i][..6], &answer(i), ty, device_nans))
                    .collect();
                nans += (0..lanes.len())
                    .filter(|&i| got[i][..6] != answer(i))
                    .count()
                    - wrong.len();
                words += (0..lanes.len()).filter(|&i| got[i][..4] != [0; 4]).count();
                if let Some(&i) = wrong.first() {
                    let mark = match decodes {
                        true => {
                            announced += wrong.len();
                            "CROSS-CLASS-ANNOUNCED"
                        }
                        false => {
                            differ += wrong.len();
                            "CROSS-CLASS-DIFFERS"
                        }
                    };
                    println!(
                        "{mark} {}: argument-buffer Read {ty:?} over {format:?}: {} of {} lanes, \
                         first at {:08x?}: {:08x?}, expected {:08x?}",
                        device.name(),
                        wrong.len(),
                        lanes.len(),
                        lanes[i],
                        &got[i][..6],
                        answer(i)
                    );
                }
            }
        }
        println!(
            "CROSS-CLASS argument-buffer reads {}: {checked} lanes checked, {differ} differ, \
             {announced} announced, {nans} hold a half NaN of the device's bits, {words} lanes of \
             nonzero words, announced {kinds:?}",
            device.name()
        );
        assert!(checked > 0);
        assert!(words > 0, "every read the neutral's");
        assert_eq!(differ, 0, "argument-buffer cross-class reads differ");
        assert!(
            !kinds.contains(&Degradation::CrossClassTextureNeutral),
            "an exact read bound the neutral: {kinds:?}"
        );
    });
}

/// A `float` sample of an `RGBA8Uint` texture an argument buffer holds, which has no exact answer
/// yet, reads the float class's neutral at each of two dispatches, and the device announces
/// `cross-class-texture-neutral` once.
#[test]
fn an_argument_buffer_cross_class_sample_is_announced() {
    on_device_with(at_plan, |device, messages| {
        let op = Op::Sample(Through::Constexpr(sampler_words(false)));
        let pipeline = pipeline_of(device, &kernel_at(op, Ty::Float, true));
        let texture = made(device, PixelFormat::RGBA8Uint, 4);
        let lanes = sample_lanes();
        let _ = messages.take_notices();
        for _ in 0..2 {
            let got: Vec<[u32; 8]> =
                dispatch_words(device, (&pipeline, true), (&texture, None), &lanes);
            println!(
                "CROSS-CLASS argument-buffer sample {}: first lane {:08x?}",
                device.name(),
                &got[0][..6]
            );
            assert!(got.iter().all(|w| w[4..6] == [W, H]), "{:08x?}", got[0]);
        }
        let announced: Vec<Degradation> = messages
            .take_notices()
            .iter()
            .filter_map(|m| m.degradation.as_ref())
            .map(|n| n.degradation)
            .filter(|kind| CROSS_CLASS.contains(kind))
            .collect();
        println!(
            "CROSS-CLASS argument-buffer neutral {}: announced {announced:?}",
            device.name()
        );
        assert_eq!(announced, CROSS_CLASS);
    });
}
