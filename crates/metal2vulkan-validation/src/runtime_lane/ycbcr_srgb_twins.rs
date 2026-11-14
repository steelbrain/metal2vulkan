//! Device tests of the 4:4:4 two-plane YCbCr `_sRGB` twins, `YCBCR8_444_2P_sRGB` and
//! `YCBCR10_444_2P_sRGB` (523 and 527; `runtime/src/ycbcr.rs`), on the device the cfg selects,
//! under the validation layer, against Metal's own bytes ([`METAL`]): eight colors written by a
//! kernel's `write()`, by a fragment function's `write()` in a pass into a `BGRA8Unorm` target,
//! and cleared and stored by a pass into the texture, over plane 0 of a `444v` or `x444` surface
//! without a matrix. Each surface is a Shared buffer standing for the IOSurface, laid out as the
//! probe's: an 8x2 texture, rows [`ROW`] bytes apart, plane 1 at [`CHROMA_AT`], every other byte
//! random and kept. What Metal was not measured doing is refused, typed (`Feature::Format`): a
//! pass that loads such a texture (the surface keeps its bytes), and a pipeline that blends into
//! one.

use super::degrade_notices::on_device;
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::error::{Error as CompileError, Feature};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, Device, Error, IoSurfacePlane, IoSurfaceYCbCr, Library,
    LoadAction, PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use std::collections::BTreeSet;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The texture's width and height, as the probe's.
pub(super) const W: u64 = 8;
pub(super) const H: u64 = 2;
/// The bytes between rows of either plane.
const ROW: u64 = 64;
/// The chroma plane's first byte.
const CHROMA_AT: u64 = 4096;
/// The surface's bytes: both planes.
pub(super) const LENGTH: u64 = CHROMA_AT + ROW * H;

/// The probe's colors, column `x` the color `x`: black, white, gray 0.5, red, green, blue, and
/// two mixed ones.
pub(super) const COLORS: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.5, 0.5, 0.5],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.2, 0.6, 0.9],
    [0.9, 0.3, 0.1],
];

/// Metal's luma, Cb and Cr codes of each color of [`COLORS`] written by a kernel's and by a
/// fragment function's `write()`, of `YCBCR8_444_2P_sRGB` over `444v` and of
/// `YCBCR10_444_2P_sRGB` over `x444`, both surfaces without a matrix (an Apple M3 Max under macOS
/// 27.0.1, the consumer's probe; a 10-bit code the high 10 bits of its little-endian word, the
/// low 6 bits 0). A pass's clear to each color stores the same codes, but the 10-bit luma of
/// (0.2, 0.6, 0.9): 697 ([`METAL_PASS_LUMA_10`]).
const METAL: [[[u16; 3]; 8]; 2] = [
    [
        [16, 128, 128],
        [235, 128, 128],
        [177, 128, 128],
        [81, 90, 240],
        [145, 54, 34],
        [41, 240, 110],
        [174, 157, 90],
        [162, 88, 174],
    ],
    [
        [64, 512, 512],
        [940, 512, 512],
        [708, 512, 512],
        [326, 361, 960],
        [578, 215, 137],
        [164, 960, 439],
        [696, 630, 360],
        [649, 351, 695],
    ],
];

/// The 10-bit luma of color 6, (0.2, 0.6, 0.9), Metal's pass stores (its writes store 696):
/// Metal's own spread between its paths, whose value by the exact curve is 696.47.
const METAL_PASS_LUMA_10: (usize, u16) = (6, 697);

/// The two formats, each over its surface's OSType, and [`METAL`]'s row of it.
const TWINS: [(PixelFormat, &str, usize); 2] = [
    (PixelFormat::YCBCR84442PSRGB, "444v", 0),
    (PixelFormat::YCBCR104442PSRGB, "x444", 1),
];

/// Synthetic AIR written for this lane: kernel `write_columns` writes buffer 0's `float4` `gid.x`
/// to texel `gid` of texture 0; vertex `corner` places vertex `vid` at buffer 0's `float4` `vid`;
/// fragment `write_fragment` writes buffer 0's `float4` of its pixel's column to its pixel of
/// texture 0 and returns 0 to color 0; fragment `values` returns buffer 0's `float4` 0.
const SHADERS: &str = r#"; Owned synthetic shaders.
define void @write_columns(ptr addrspace(1) %t, ptr addrspace(1) noalias %colors, <2 x i32> %gid) {
entry:
  %x = extractelement <2 x i32> %gid, i64 0
  %j = zext i32 %x to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %colors, i64 %j
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> %v, i32 0, i32 2)
  ret void
}

define <{ <4 x float> }> @corner(i32 %vid, ptr addrspace(2) %p) {
  %i = zext i32 %vid to i64
  %q = getelementptr inbounds <4 x float>, ptr addrspace(2) %p, i64 %i
  %v = load <4 x float>, ptr addrspace(2) %q, align 16
  %o = insertvalue <{ <4 x float> }> undef, <4 x float> %v, 0
  ret <{ <4 x float> }> %o
}

define <4 x float> @write_fragment(<4 x float> %pos, ptr addrspace(1) %t, ptr addrspace(1) %colors) {
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %gid = fptoui <2 x float> %xy to <2 x i32>
  %x = extractelement <2 x i32> %gid, i64 0
  %j = zext i32 %x to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %colors, i64 %j
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> %v, i32 0, i32 2)
  ret <4 x float> zeroinitializer
}

define <4 x float> @values(<4 x float> %pos, ptr addrspace(1) %colors) {
  %v = load <4 x float>, ptr addrspace(1) %colors, align 16
  ret <4 x float> %v
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.kernel = !{!0}
!air.vertex = !{!10}
!air.fragment = !{!20, !30}
!air.version = !{!97}
!air.language_version = !{!98}
!0 = !{ptr @write_columns, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"colors"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!10 = !{ptr @corner, !11, !12}
!11 = !{!13}
!12 = !{!14, !15}
!13 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!14 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!15 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!20 = !{ptr @write_fragment, !21, !22}
!21 = !{!23}
!22 = !{!24, !25, !26}
!23 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!24 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!25 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!26 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"colors"}
!30 = !{ptr @values, !21, !31}
!31 = !{!24, !32}
!32 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"colors"}
!97 = !{i32 2, i32 8, i32 0}
!98 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A small generator of the surfaces' bytes.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u8
    }
}

/// A surface's random bytes, of `seed`.
pub(super) fn surface_bytes(seed: u64) -> Vec<u8> {
    let mut random = Random(seed);
    (0..LENGTH).map(|_| random.next()).collect()
}

/// The texture of `pf` over plane 0 of `surface`, a surface of OSType `ostype` without a matrix
/// laid out by [`ROW`] and [`CHROMA_AT`], of `usage`.
pub(super) fn twin_texture(
    surface: &Buffer,
    pf: PixelFormat,
    ostype: &str,
    usage: TextureUsage,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::shared();
    let code: [u8; 4] = ostype.as_bytes().try_into().unwrap();
    let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(code), 0, [W, H], CHROMA_AT, ROW);
    let at = IoSurfacePlane::new(0, ROW, true).with_ycbcr(facts);
    surface.new_texture_with_iosurface_plane(&d, at)
}

/// The library of [`SHADERS`].
pub(super) fn library(device: &Device) -> Library {
    device
        .new_library_with_air(&llvm_assemble_text(SHADERS).unwrap())
        .unwrap()
}

/// A buffer of [`COLORS`] as `float4`s, alpha 1.
fn colors(device: &Device) -> Buffer {
    let bytes: Vec<u8> = COLORS
        .iter()
        .flat_map(|&[r, g, b]| [r, g, b, 1.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    buffer_with(device, &bytes).unwrap()
}

/// `bytes` with texel `(x, y)`'s codes `[y, cb, cr]` of `bits` bits stored at their places, as
/// `codes` gives them: the luma code in its row of the luma plane, and the Cb and Cr codes at the
/// texel's pair in its row of the chroma plane, a code a byte or the high bits of a little-endian
/// 16-bit word.
fn stored(bytes: &[u8], bits: u32, codes: impl Fn(u64, u64) -> [u16; 3]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    let unit = u64::from(bits.div_ceil(8));
    let mut put = |at: u64, code: u16| {
        let at = at as usize;
        match unit {
            1 => out[at] = code as u8,
            _ => out[at..at + 2].copy_from_slice(&(code << (16 - bits)).to_le_bytes()),
        }
    };
    for y in 0..H {
        for x in 0..W {
            let [luma, cb, cr] = codes(x, y);
            put(y * ROW + x * unit, luma);
            let pair = CHROMA_AT + y * ROW + 2 * x * unit;
            put(pair, cb);
            put(pair + unit, cr);
        }
    }
    out
}

/// Metal's bytes of the probe's writes over `bytes`: column `x` holds [`METAL`]'s codes of color
/// `x` of the format's `row`.
fn metal_writes(bytes: &[u8], bits: u32, row: usize) -> Vec<u8> {
    stored(bytes, bits, |x, _| METAL[row][x as usize])
}

/// Where `after` and `want` (surfaces of codes of `bits` bits) differ: the columns whose codes
/// differ in either plane, each a color of [`COLORS`] in the writes, and whether a byte past every
/// code differs.
fn apart(after: &[u8], want: &[u8], bits: u32) -> (BTreeSet<u64>, bool) {
    let unit = u64::from(bits.div_ceil(8));
    let mut columns = BTreeSet::new();
    let mut past = false;
    for (at, _) in after
        .iter()
        .zip(want)
        .enumerate()
        .filter(|(_, (a, w))| a != w)
    {
        let at = at as u64;
        let (offset, pitch) = match at >= CHROMA_AT {
            true => (at - CHROMA_AT, 2 * unit),
            false => (at, unit),
        };
        let (row, column) = (offset / ROW, offset % ROW / pitch);
        if row < H && column < W {
            columns.insert(column);
        } else {
            past = true;
        }
    }
    (columns, past)
}

/// Commit `command_buffer`, wait for it, and expect it to have completed without an error.
pub(super) fn complete(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A kernel's `write()` of the eight colors into an `_sRGB` twin over a surface stores Metal's
/// codes of each into both planes, and keeps every other byte: of `YCBCR8_444_2P_sRGB` over
/// `444v` and of `YCBCR10_444_2P_sRGB` over `x444`, column `x` written with color `x` in both rows.
#[test]
fn a_kernel_s_writes_to_an_srgb_twin_store_metal_s_codes() {
    on_device(|device, _| {
        let library = library(device);
        let function = library.new_function_with_name("write_columns").unwrap();
        let state = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let values = colors(device);
        for (seed, (pf, ostype, row)) in (0x5230..).zip(TWINS) {
            let bytes = surface_bytes(seed);
            let surface = buffer_with(device, &bytes).unwrap();
            let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let texture = twin_texture(&surface, pf, ostype, usage).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder.set_texture(Some(&texture), 0).unwrap();
            encoder.set_buffer(Some(&values), 0, 0).unwrap();
            encoder
                .dispatch_threads(Size::new(W, H, 1), Size::new(8, 2, 1))
                .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let bits = [8, 10][row];
            let want = metal_writes(&bytes, bits, row);
            let after = read_buffer(&surface, 0, LENGTH).unwrap();
            let (colors, past) = apart(&after, &want, bits);
            assert!(
                colors.is_empty() && !past,
                "{pf:?} over {ostype}: the colors whose codes are not Metal's {colors:?}, a byte \
                 past the codes changed {past}"
            );
        }
    });
}

/// The corners of two triangles covering the whole target, as `float4`s.
fn whole(device: &Device) -> Buffer {
    let corners: Vec<u8> = [
        [-1.0f32, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ]
    .iter()
    .flat_map(|&[x, y]| [x, y, 0.0, 1.0])
    .flat_map(f32::to_le_bytes)
    .collect();
    buffer_with(device, &corners).unwrap()
}

/// A render pipeline state of vertex `corner` and fragment `fragment` into color 0 of `pf`,
/// `adjust` changing its attachment.
fn pipeline(
    device: &Device,
    library: &Library,
    (fragment, pf): (&str, PixelFormat),
    adjust: impl FnOnce(&mut ColorAttachmentDescriptor),
) -> Result<RenderPipelineState, Error> {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("corner"));
    descriptor.fragment = Some(function(fragment));
    let mut attachment = ColorAttachmentDescriptor::new(pf);
    adjust(&mut attachment);
    descriptor.color_attachments[0] = Some(attachment);
    device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// A pass of one color attachment, `target`, loaded by `load`, cleared to `clear`, and stored.
fn pass(target: &Texture, load: LoadAction, clear: [f64; 4]) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = load;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(clear[0], clear[1], clear[2], clear[3]);
    descriptor
}

/// A fragment function's `write()` of the eight colors into an `_sRGB` twin, in a pass into a
/// `BGRA8Unorm` target over the whole texture, stores Metal's codes of each into both planes, and
/// keeps every other byte.
#[test]
fn a_fragment_s_writes_to_an_srgb_twin_store_metal_s_codes() {
    on_device(|device, _| {
        let library = library(device);
        let state = pipeline(
            device,
            &library,
            ("write_fragment", PixelFormat::BGRA8Unorm),
            |_| {},
        )
        .unwrap();
        let queue = device.new_command_queue().unwrap();
        let (values, corners) = (colors(device), whole(device));
        let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, W, H, false);
        d.usage = TextureUsage::RENDER_TARGET;
        let target = device.new_texture(&d).unwrap();
        for (seed, (pf, ostype, row)) in (0x5240..).zip(TWINS) {
            let bytes = surface_bytes(seed);
            let surface = buffer_with(device, &bytes).unwrap();
            let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            let texture = twin_texture(&surface, pf, ostype, usage).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let descriptor = pass(&target, LoadAction::Clear, [0.0; 4]);
            let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
            encoder.set_render_pipeline_state(&state).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            encoder.set_fragment_texture(Some(&texture), 0).unwrap();
            encoder.set_fragment_buffer(Some(&values), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let bits = [8, 10][row];
            let want = metal_writes(&bytes, bits, row);
            let after = read_buffer(&surface, 0, LENGTH).unwrap();
            let (colors, past) = apart(&after, &want, bits);
            assert!(
                colors.is_empty() && !past,
                "{pf:?} over {ostype}: the colors whose codes are not Metal's after the fragment's \
                 writes {colors:?}, a byte past the codes changed {past}"
            );
        }
    });
}

/// A pass that clears an `_sRGB` twin to each of the eight colors and stores it stores Metal's
/// codes of that color at every texel of both planes, and keeps every other byte; the 10-bit luma
/// of (0.2, 0.6, 0.9) within a code of Metal's pass (697, its writes' 696: Metal's own spread).
#[test]
fn a_pass_s_clear_of_an_srgb_twin_stores_metal_s_codes() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        for (seed, (pf, ostype, row)) in (0x5250..).zip(TWINS) {
            let bits = [8, 10][row];
            // The colors whose clears' bytes are not Metal's.
            let mut colors = Vec::new();
            for (i, &[r, g, b]) in COLORS.iter().enumerate() {
                let bytes = surface_bytes(seed + 16 * i as u64);
                let surface = buffer_with(device, &bytes).unwrap();
                let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
                let texture = twin_texture(&surface, pf, ostype, usage).unwrap();
                let clear = [r, g, b, 1.0].map(f64::from);
                let command_buffer = queue.command_buffer().unwrap();
                let descriptor = pass(&texture, LoadAction::Clear, clear);
                let encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
                encoder.end_encoding().unwrap();
                complete(&command_buffer);
                let after = read_buffer(&surface, 0, LENGTH).unwrap();
                // Metal's pass's codes of the color, and ours where they lie within a code of
                // Metal's 10-bit luma.
                let mut metal = METAL[row][i];
                if bits == 10 && i == METAL_PASS_LUMA_10.0 {
                    metal[0] = METAL_PASS_LUMA_10.1;
                }
                let luma = u16::from_le_bytes([after[0], after[1]]) >> 6;
                if bits == 10 && luma.abs_diff(metal[0]) <= 1 {
                    metal[0] = luma;
                }
                let want = stored(&bytes, bits, |_, _| metal);
                if after != want {
                    colors.push(i);
                }
            }
            assert!(
                colors.is_empty(),
                "{pf:?} over {ostype}: the colors whose clears' bytes are not Metal's {colors:?}"
            );
        }
    });
}

/// Whether `result` is a refusal of the format, typed (`Feature::Format`).
fn refused_format<T>(result: Result<T, Error>) -> bool {
    match result {
        Err(Error::Unsupported(u)) | Err(Error::Compile(CompileError::Unsupported(u))) => {
            u.feature == Feature::Format
        }
        _ => false,
    }
}

/// What Metal was not measured doing with an `_sRGB` twin is refused, typed (`Feature::Format`):
/// a pass that loads one, at its start, naming the format, the surface keeping its bytes; and a
/// pipeline that blends into one. The same pipeline unblended is made.
#[test]
fn an_srgb_twin_s_loads_and_blends_are_refused_typed() {
    on_device(|device, _| {
        let library = library(device);
        let queue = device.new_command_queue().unwrap();
        for (seed, (pf, ostype, _)) in (0x5260..).zip(TWINS) {
            let made = pipeline(device, &library, ("values", pf), |_| {});
            assert!(made.is_ok(), "{pf:?}: {:?}", made.err());
            let blended = pipeline(device, &library, ("values", pf), |a| {
                a.blending_enabled = true
            });
            assert!(refused_format(blended), "{pf:?} blended");
            let bytes = surface_bytes(seed);
            let surface = buffer_with(device, &bytes).unwrap();
            let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            let texture = twin_texture(&surface, pf, ostype, usage).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let descriptor = pass(&texture, LoadAction::Load, [0.0; 4]);
            match command_buffer.render_command_encoder(&descriptor) {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::Format, "{pf:?}");
                    let named = u.symbol.map(|s| s.to_string());
                    assert_eq!(named.as_deref(), Some(pf.name()), "{pf:?}");
                }
                other => panic!("{pf:?} loaded: {:?}", other.err()),
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let after = read_buffer(&surface, 0, LENGTH).unwrap();
            assert!(after == bytes, "{pf:?}: the surface's bytes changed");
        }
    });
}
