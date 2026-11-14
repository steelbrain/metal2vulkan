//! Device tests of render passes into two-plane YCbCr textures (`runtime/src/ycbcr.rs`'s
//! **Passes**), on the device the cfg selects, under the validation layer. Each surface is a Shared
//! buffer standing for the IOSurface: the luma plane at byte 0 and the interleaved chroma plane at
//! [`CHROMA_AT`], rows [`ROW`] bytes apart, the bytes between and past the rows' codes random too.
//!
//! - A pass into `YCBCR8_420_2P`, `YCBCR8_422_2P` or `YCBCR8_444_2P` stores every texel into
//!   both planes as Metal stores it (`metal2vulkan::bind::ycbcr_store`, whose codes matched
//!   Metal's bytes of our probe's clears and draws): a clear, a draw over part of a cleared texture
//!   (chroma blocks of drawn and cleared texels among them), and a full-screen draw over a
//!   `DontCare` load, of values out of `[0, 1]`, NaNs, infinities and values beside the grid's ties
//!   among them; every other byte is kept, and nothing is announced. So does one into
//!   `YCBCR10_420_2P`, `YCBCR10_422_2P` or `YCBCR10_444_2P`, each code in the high 10 bits of its
//!   16-bit word, over the surfaces and conversions measured: `x420`, `xf20`, `x422`, `xf22`,
//!   `x444` and `xf44`, without a matrix and under BT.709; and one into a code format,
//!   `RGB8_420_2P` to `RGB10_444_2P`, each texel's green, blue and red as codes.
//! - A pass that loads such a texture loads each texel as the texture reads it and stores every
//!   texel of each 32x32 tile its draws reach, re-encoded, keeping the planes' bytes of the other
//!   tiles (`loaded_passes_store_the_tiles_they_reach_as_metal`); a `DontCare` store, or a store
//!   left `Unknown`, stores nothing; a write mask keeps the masked channels' cleared or loaded
//!   values; a blend blends the pass's floats, within a code of Metal's tile of halves, and
//!   announces it (`masked_blended_and_unstored_passes_are_metal_s`).
//! - What is not measured is refused, typed (`Feature::Format`), leaving the surface's bytes: the
//!   4:2:0 and 4:2:2 `_sRGB` twins' pipelines and passes (the 4:4:4 twins' are
//!   `ycbcr_srgb_twins`').
//! - A packed 10-bit format's pipeline is made, as Metal makes it, and a pass into its texture
//!   is refused, typed and naming the format, leaving the surface's bytes: Metal's pass fails,
//!   its error and the bytes it leaves varying from run to run, so no exact form exists.

use super::degrade_notices::{counts, on_device};
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::{
    ycbcr_read, ycbcr_store, YCbCrCodes, YCbCrMatrix, YCbCrOutput, YCbCrRange,
};
use metal2vulkan::error::{Error as CompileError, Feature};
use metal2vulkan::metal::format::ycbcr_format;
use metal2vulkan::metal::{BlendFactor, ColorWriteMask, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandQueue, Degradation, Device, Error, IoSurfacePlane, IoSurfaceYCbCr,
    Library, LoadAction, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, ScissorRect, StoreAction, SurfaceYCbCrMatrix, Texture, TextureDescriptor,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The kinds this lane's loaded and blended passes announce, for `degrade_notices`' census.
pub(super) const YCBCR_PASSES: [Degradation; 2] =
    [Degradation::YCbCrLoadTiles, Degradation::YCbCrBlendRounding];

/// The texture's width and height, even, so every format's chroma blocks tile it.
const W: u64 = 40;
const H: u64 = 24;
/// The bytes between rows of either plane.
const ROW: u64 = 256;
/// The chroma plane's first byte.
const CHROMA_AT: u64 = ROW * H;
/// The surface's bytes: both planes, the chroma plane as tall as the luma plane at most.
const LENGTH: u64 = 2 * ROW * H;

/// Synthetic AIR written for this lane: vertex `corner` places vertex `vid` at buffer 0's
/// `float4` `vid`; fragment `values` writes to color 0 buffer 0's `float4` of its pixel (row by
/// row, [`W`] a row).
const SHADERS: &str = r#"; Owned synthetic shaders.
define <{ <4 x float> }> @corner(i32 %vid, ptr addrspace(2) %p) {
  %i = zext i32 %vid to i64
  %q = getelementptr inbounds <4 x float>, ptr addrspace(2) %p, i64 %i
  %v = load <4 x float>, ptr addrspace(2) %q, align 16
  %o = insertvalue <{ <4 x float> }> undef, <4 x float> %v, 0
  ret <{ <4 x float> }> %o
}

define <4 x float> @values(<4 x float> %pos, ptr addrspace(1) %v) {
  %x = extractelement <4 x float> %pos, i64 0
  %y = extractelement <4 x float> %pos, i64 1
  %xi = fptoui float %x to i32
  %yi = fptoui float %y to i32
  %row = mul i32 %yi, 40
  %i = add i32 %row, %xi
  %w = zext i32 %i to i64
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %v, i64 %w
  %c = load <4 x float>, ptr addrspace(1) %q, align 16
  ret <4 x float> %c
}

!air.vertex = !{!90}
!air.fragment = !{!80}
!air.version = !{!97}
!air.language_version = !{!98}
!90 = !{ptr @corner, !91, !92}
!91 = !{!93}
!92 = !{!94, !95}
!93 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!94 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!95 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!97 = !{i32 2, i32 8, i32 0}
!98 = !{!"Metal", i32 4, i32 0, i32 0}
!80 = !{ptr @values, !81, !82}
!81 = !{!83}
!82 = !{!84, !85}
!83 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!84 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!85 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"v"}
"#;

/// A small generator of the test's bytes and values.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// A channel a draw writes: on the grid of 1/2048, beside one of its ties (up to three ulps
    /// either side), any multiple of 2^-24 in `[0, 1)`, one out of `[0, 1]`, or a NaN, an
    /// infinity, a zero or a subnormal.
    fn channel(&mut self) -> f32 {
        const SPECIALS: [f32; 8] = [
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -0.0,
            1.0e-45,
            1.000_000_1,
            0.999_999_94,
            1.0e30,
        ];
        match self.next() % 8 {
            0 | 1 => (self.next() % 2049) as f32 / 2048.0,
            2 | 3 => {
                let mut t = ((self.next() % 2048) as f32 + 0.5) / 2048.0;
                for _ in 0..self.next() % 4 {
                    t = match self.next() % 2 {
                        0 => f32::from_bits(t.to_bits() + 1),
                        _ => f32::from_bits(t.to_bits() - 1),
                    };
                }
                t
            }
            4 => (self.next() % (1 << 24)) as f32 / (1 << 24) as f32,
            5 => (self.next() % (1 << 24)) as f32 / (1 << 22) as f32 - 1.5,
            _ => SPECIALS[self.next() as usize % SPECIALS.len()],
        }
    }
}

/// `v` rounded to the nearest `half`, ties to even, as `float` (finite values within the `half`
/// range; NaN, infinities and zero kept).
fn half_nearest(v: f32) -> f32 {
    if !v.is_finite() || v == 0.0 {
        return v;
    }
    // The `half` step at `v`: its exponent's (2^-14 at the least), 10 bits below.
    let exponent = (((v.to_bits() >> 23) & 0xff) as i32 - 127).max(-14);
    let step = 2f32.powi(exponent - 10);
    (v / step).round_ties_even() * step
}

/// The codes of `pf`'s layout.
fn codes_of(pf: PixelFormat) -> YCbCrCodes {
    ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap()
}

/// The texture of `pf` over plane 0 of `surface`, a surface of OSType `ostype` and `matrix` laid
/// out by [`ROW`] and [`CHROMA_AT`], of render target and shader read usage.
fn pass_texture(
    surface: &Buffer,
    pf: PixelFormat,
    ostype: &str,
    matrix: Option<SurfaceYCbCrMatrix>,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let code: [u8; 4] = ostype.as_bytes().try_into().unwrap();
    let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(code), 0, [W, H], CHROMA_AT, ROW);
    let facts = match matrix {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    };
    let at = IoSurfacePlane::new(0, ROW, true).with_ycbcr(facts);
    surface.new_texture_with_iosurface_plane(&d, at)
}

/// A render pipeline state drawing `values` into color 0 of `pf`, `adjust` changing its
/// attachment.
fn pipeline(
    device: &Device,
    library: &Library,
    pf: PixelFormat,
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
    descriptor.fragment = Some(function("values"));
    let mut attachment = ColorAttachmentDescriptor::new(pf);
    adjust(&mut attachment);
    descriptor.color_attachments[0] = Some(attachment);
    device.new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// The corners of two triangles covering the rectangle of texels `[0, x)` across and `[0, y)`
/// down, as `float4`s (Metal's normalized coordinates: up is the first row).
fn rectangle(device: &Device, [x, y]: [u64; 2]) -> Buffer {
    let right = 2.0 * x as f32 / W as f32 - 1.0;
    let bottom = 1.0 - 2.0 * y as f32 / H as f32;
    let corners: Vec<u8> = [
        [-1.0f32, bottom],
        [right, bottom],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [right, bottom],
        [right, 1.0],
    ]
    .iter()
    .flat_map(|&[x, y]| [x, y, 0.0, 1.0])
    .flat_map(|v| v.to_le_bytes())
    .collect();
    buffer_with(device, &corners).unwrap()
}

/// How a test's pass renders.
#[derive(Clone, Copy, Debug)]
enum Pass {
    /// A clear, no draw.
    Clear,
    /// A clear, then a draw of the texels left of column 21 and above row 13.
    ClearThenPart,
    /// A `DontCare` load, then a draw of every texel.
    Whole,
}

/// The pass `pass` into `texture`, cleared to `clear`, drawing `values` through `state`, in one
/// command buffer of `queue`, completed.
fn render(
    queue: &CommandQueue,
    texture: &Texture,
    (pass, clear): (Pass, [f64; 4]),
    (state, values): (&RenderPipelineState, &Buffer),
) {
    let device = &texture.device();
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(texture.clone());
    color.load_action = match pass {
        Pass::Clear | Pass::ClearThenPart => LoadAction::Clear,
        Pass::Whole => LoadAction::DontCare,
    };
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(clear[0], clear[1], clear[2], clear[3]);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
    let drawn = match pass {
        Pass::Clear => None,
        Pass::ClearThenPart => Some([21, 13]),
        Pass::Whole => Some([W, H]),
    };
    if let Some(drawn) = drawn {
        let corners = rectangle(device, drawn);
        encoder.set_render_pipeline_state(state).unwrap();
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder.set_fragment_buffer(Some(values), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// `bytes` after a pass's store of `texels` (each texel's `[r, g, b]`, row by row) as Metal
/// stores it over `conversion` (`metal2vulkan::bind::ycbcr_store`): every texel's luma code at
/// its place in its row of the luma plane, and each chroma pair's Cb and Cr codes at its place in
/// its row of the chroma plane; every other byte as it was.
fn stored_bytes(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
    texels: &[[f32; 3]],
) -> Vec<u8> {
    let (luma, chroma) = ycbcr_store(codes, conversion, texels, [W as u32, H as u32]).unwrap();
    let mut out = bytes.to_vec();
    for (i, &code) in luma.iter().enumerate() {
        let (x, y) = (i as u64 % W, i as u64 / W);
        out[(y * ROW + x) as usize] = code as u8;
    }
    let [across, _] = codes.subsampling.halved();
    let pairs = if across { W / 2 } else { W };
    for (i, &[cb, cr]) in chroma.iter().enumerate() {
        let (x, y) = (i as u64 % pairs, i as u64 / pairs);
        let at = (CHROMA_AT + y * ROW + 2 * x) as usize;
        out[at] = cb as u8;
        out[at + 1] = cr as u8;
    }
    out
}

/// [`stored_bytes`] for a layout of 10-bit codes, each in the high 10 bits of its little-endian
/// 16-bit word, the low 6 bits 0.
fn stored_words(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
    texels: &[[f32; 3]],
) -> Vec<u8> {
    let (luma, chroma) = ycbcr_store(codes, conversion, texels, [W as u32, H as u32]).unwrap();
    let mut out = bytes.to_vec();
    let mut put = |at: u64, code: u16| {
        let at = at as usize;
        out[at..at + 2].copy_from_slice(&(code << 6).to_le_bytes());
    };
    for (i, &code) in luma.iter().enumerate() {
        let (x, y) = (i as u64 % W, i as u64 / W);
        put(y * ROW + 2 * x, code);
    }
    let [across, _] = codes.subsampling.halved();
    let pairs = if across { W / 2 } else { W };
    for (i, &[cb, cr]) in chroma.iter().enumerate() {
        let (x, y) = (i as u64 % pairs, i as u64 / pairs);
        let at = CHROMA_AT + y * ROW + 4 * x;
        put(at, cb);
        put(at + 2, cr);
    }
    out
}

/// The first byte where `after` and `want` differ, and both there, for a failure's message.
fn first_difference(after: &[u8], want: &[u8]) -> Option<(usize, u8, u8)> {
    after
        .iter()
        .zip(want)
        .enumerate()
        .find(|(_, (a, w))| a != w)
        .map(|(i, (&a, &w))| (i, a, w))
}

/// A pass into an 8-bit converted YCbCr texture stores every texel's codes into both planes as
/// Metal stores them: for each format, over both ranges and each matrix, a clear, a draw over part
/// of a cleared texture, and a full-screen draw over a `DontCare` load; every other byte kept, and
/// nothing announced past the pipeline states' own notes. So does one into a 10-bit converted one,
/// over each of its surfaces measured, without a matrix and under BT.709, and one into each code
/// format.
#[test]
fn a_pass_stores_metal_s_codes_into_both_planes() {
    use PixelFormat as P;
    use YCbCrMatrix::{Bt2020, Bt601, Bt709};
    use YCbCrRange::{Full, Video};
    on_device(|device, messages| {
        let library = device
            .new_library_with_air(&llvm_assemble_text(SHADERS).unwrap())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let (b709, b2020) = (
            Some(SurfaceYCbCrMatrix::Itu709),
            Some(SurfaceYCbCrMatrix::Itu2020),
        );
        let mut seed = 0x6a1;
        for (pf, ostype, matrix, conversion) in [
            (P::YCBCR84202P, "420v", None, (Video, Bt601)),
            (P::YCBCR84202P, "420f", b709, (Full, Bt709)),
            (P::YCBCR84222P, "422v", b2020, (Video, Bt2020)),
            (P::YCBCR84222P, "422f", None, (Full, Bt601)),
            (P::YCBCR84442P, "444v", b709, (Video, Bt709)),
            (P::YCBCR84442P, "444f", b2020, (Full, Bt2020)),
            (P::YCBCR104202P, "x420", None, (Video, Bt601)),
            (P::YCBCR104202P, "xf20", b709, (Full, Bt709)),
            (P::YCBCR104222P, "x422", b709, (Video, Bt709)),
            (P::YCBCR104222P, "xf22", None, (Full, Bt601)),
            (P::YCBCR104442P, "x444", None, (Video, Bt601)),
            (P::YCBCR104442P, "xf44", b709, (Full, Bt709)),
            (P::RGB84202P, "420v", None, (Video, Bt601)),
            (P::RGB84222P, "422f", b709, (Full, Bt709)),
            (P::RGB84442P, "444v", None, (Video, Bt601)),
            (P::RGB104202P, "x420", None, (Video, Bt601)),
            (P::RGB104222P, "xf22", b709, (Full, Bt709)),
            (P::RGB104442P, "x444", None, (Video, Bt601)),
        ] {
            let codes = codes_of(pf);
            let state = pipeline(device, &library, pf, |_| {}).unwrap();
            messages.take_notices();
            for pass in [Pass::Clear, Pass::ClearThenPart, Pass::Whole] {
                seed += 1;
                let case = format!("{pf:?} over {ostype}, {matrix:?}, {pass:?}");
                let mut random = Random(seed);
                let bytes: Vec<u8> = (0..LENGTH).map(|_| random.next() as u8).collect();
                let surface = buffer_with(device, &bytes).unwrap();
                let texture = pass_texture(&surface, pf, ostype, matrix).unwrap();
                let values: Vec<[f32; 4]> = (0..W * H)
                    .map(|_| std::array::from_fn(|_| random.channel()))
                    .collect();
                let clear: [f64; 4] =
                    std::array::from_fn(|_| f64::from(random.next() % 6145) / 4096.0 - 0.25);
                let buffer: Vec<u8> = values
                    .iter()
                    .flatten()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let buffer = buffer_with(device, &buffer).unwrap();
                render(&queue, &texture, (pass, clear), (&state, &buffer));
                // Metal takes the clear color through `float`, then `half` rounded to nearest
                // (A 10-bit store keeps a `half`).
                let cleared = [0, 1, 2].map(|c| half_nearest(clear[c] as f32));
                let texels: Vec<[f32; 3]> = (0..W * H)
                    .map(|i| {
                        let (x, y) = (i % W, i / W);
                        let drawn = match pass {
                            Pass::Clear => false,
                            Pass::ClearThenPart => x < 21 && y < 13,
                            Pass::Whole => true,
                        };
                        let [r, g, b, _] = values[i as usize];
                        if drawn {
                            [r, g, b]
                        } else {
                            cleared
                        }
                    })
                    .collect();
                let want = match codes.bits {
                    10 => stored_words(&bytes, codes, conversion, &texels),
                    _ => stored_bytes(&bytes, codes, conversion, &texels),
                };
                let after = read_buffer(&surface, 0, LENGTH).unwrap();
                assert!(
                    after == want,
                    "{case}: the surface's bytes after the pass, first difference (at, got, \
                     want) {:?}",
                    first_difference(&after, &want)
                );
            }
        }
        let notices = counts(&messages.take_notices());
        assert!(notices.is_empty(), "{notices:?}");
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

/// What a pass into a YCbCr texture does that is not measured is refused, typed
/// (`Feature::Format`), and the surface keeps its bytes: the 4:2:0 `_sRGB` twins' pipelines and
/// passes (the 10-bit formats' once too). The packed formats' pipelines are made, unblended, and
/// their passes refused, naming the format.
#[test]
fn ycbcr_passes_outside_the_matched_store_are_refused_typed() {
    use PixelFormat as P;
    on_device(|device, _| {
        let library = device
            .new_library_with_air(&llvm_assemble_text(SHADERS).unwrap())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        assert!(
            refused_format(pipeline(device, &library, P::YCBCR104202PSRGB, |_| {})),
            "10-bit 4:2:0 sRGB's pipeline"
        );
        for pf in [P::YCBCR104202PPACKED, P::YCBCR104442PPACKEDSRGB] {
            let made = pipeline(device, &library, pf, |_| {});
            assert!(made.is_ok(), "{pf:?}: {:?}", made.err());
            let blended = pipeline(device, &library, pf, |a| a.blending_enabled = true);
            assert!(refused_format(blended), "{pf:?} blended");
        }
        let mut random = Random(0x6b0);
        for (pf, ostype, load, store) in [
            (
                P::YCBCR104202PSRGB,
                "x420",
                LoadAction::Clear,
                StoreAction::Store,
            ),
            (
                P::YCBCR104202PPACKED,
                "p420",
                LoadAction::Clear,
                StoreAction::Store,
            ),
            (
                P::YCBCR104442PPACKEDSRGB,
                "pf44",
                LoadAction::DontCare,
                StoreAction::Store,
            ),
        ] {
            let case = format!("{pf:?} over {ostype}, {load:?} and {store:?}");
            let bytes: Vec<u8> = (0..2 * LENGTH).map(|_| random.next() as u8).collect();
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = pass_texture(&surface, pf, ostype, None).unwrap();
            let mut descriptor = RenderPassDescriptor::new();
            let color = &mut descriptor.color_attachments[0];
            color.texture = Some(texture);
            color.load_action = load;
            color.store_action = store;
            let command_buffer = queue.command_buffer().unwrap();
            let encoder = command_buffer.render_command_encoder(&descriptor);
            match encoder {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::Format, "{case}");
                    let named = u.symbol.map(|s| s.to_string());
                    assert_eq!(named.as_deref(), Some(pf.name()), "{case}");
                }
                other => panic!("{case}: {:?}", other.err()),
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let after = read_buffer(&surface, 0, bytes.len() as u64).unwrap();
            assert!(after == bytes, "{case}: the surface's bytes changed");
        }
    });
}

/// The texels a texture of `codes` over the surface `bytes` reads, as `[r, g, b]` floats
/// (`metal2vulkan::bind::ycbcr_read`: a converted value's step `v / 2048`, a code `v / (2^b - 1)`):
/// each texel's luma code and its
/// chroma pairs filtered as Metal reads them (the nearer pair weighted 3 and the farther 1 along
/// each halved axis, the conversion module's filter).
fn loaded_texels(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
) -> Vec<[f32; 3]> {
    let unit = u64::from(codes.bytes);
    let code = |at: u64| -> u32 {
        let at = at as usize;
        match unit {
            1 => u32::from(bytes[at]),
            _ => u32::from(u16::from_le_bytes([bytes[at], bytes[at + 1]])) >> 6,
        }
    };
    let [across, down] = codes.subsampling.halved();
    let [cw, ch] = [(W, across), (H, down)].map(|(n, halved)| if halved { n / 2 } else { n });
    let pair = |x: u64, y: u64| {
        let at = CHROMA_AT + y * ROW + 2 * unit * x;
        [code(at), code(at + unit)]
    };
    // The chroma pairs along one axis at texel `v` and their weights.
    let taps = |v: u64, halved: bool, n: u64| -> Vec<(u64, u32)> {
        match halved {
            false => vec![(v, 4)],
            true => {
                let near = v / 2;
                let far = match v % 2 {
                    1 => (near + 1).min(n - 1),
                    _ => near.saturating_sub(1),
                };
                vec![(near, 3), (far, 1)]
            }
        }
    };
    (0..W * H)
        .map(|i| {
            let (x, y) = (i % W, i / W);
            let mut sums = [0u32; 2];
            for (cx, wx) in taps(x, across, cw) {
                for (cy, wy) in taps(y, down, ch) {
                    let [cb, cr] = pair(cx, cy);
                    sums[0] += wx * wy * cb;
                    sums[1] += wx * wy * cr;
                }
            }
            let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
            let luma = code(y * ROW + unit * x) as u16;
            let scale = match codes.output {
                YCbCrOutput::Codes => ((1u32 << codes.bits) - 1) as f32,
                _ => 2048.0,
            };
            ycbcr_read(codes, conversion, [luma, cb, cr]).map(|v| f32::from(v) / scale)
        })
        .collect()
}

/// `stored` (the surface's bytes after a store of every texel) inside the texels `[x0, x1)`
/// across and every row, chroma pairs of those columns too, and `bytes` elsewhere.
fn within_columns(bytes: &[u8], stored: &[u8], codes: YCbCrCodes, [x0, x1]: [u64; 2]) -> Vec<u8> {
    let unit = u64::from(codes.bytes);
    let [across, _] = codes.subsampling.halved();
    let mut out = bytes.to_vec();
    let mut keep = |offset: u64, rows: u64, columns: [u64; 2], pair: u64| {
        for y in 0..rows {
            let from = (offset + y * ROW + pair * unit * columns[0]) as usize;
            let to = (offset + y * ROW + pair * unit * columns[1]) as usize;
            out[from..to].copy_from_slice(&stored[from..to]);
        }
    };
    keep(0, H, [x0, x1], 1);
    let half = |v: u64| if across { v / 2 } else { v };
    keep(CHROMA_AT, H, [half(x0), half(x1)], 2);
    out
}

/// A pass that loads a converted YCbCr texture stores as Metal's tile does (the M3 Max under
/// macOS 27.0.1, our probe `ycp`: a load and a partial draw of 128 x 96 `420v` and `x420`
/// surfaces): the texels it loads are the texture's reads, its draws write over them, and its
/// store re-encodes every texel of each 32x32 tile a draw reaches, keeping every byte of the
/// others. Here a draw of the texels left of column 21 and above row 13, scissored to them,
/// reaches the tile of columns 0 to 31: those columns store the loaded and drawn texels' codes,
/// and columns 32 to 39 keep their bytes. A full-screen draw over a load stores every texel.
#[test]
fn loaded_passes_store_the_tiles_they_reach_as_metal() {
    use PixelFormat as P;
    use YCbCrMatrix::{Bt601, Bt709};
    use YCbCrRange::{Full, Video};
    on_device(|device, messages| {
        let library = device
            .new_library_with_air(&llvm_assemble_text(SHADERS).unwrap())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let b709 = Some(SurfaceYCbCrMatrix::Itu709);
        let mut seed = 0x7a1;
        for (pf, ostype, matrix, conversion) in [
            (P::YCBCR84202P, "420v", None, (Video, Bt601)),
            (P::YCBCR84222P, "422f", b709, (Full, Bt709)),
            (P::YCBCR84442P, "444v", None, (Video, Bt601)),
            (P::YCBCR104202P, "x420", b709, (Video, Bt709)),
            (P::YCBCR104222P, "xf22", None, (Full, Bt601)),
            (P::YCBCR104442P, "xf44", b709, (Full, Bt709)),
            (P::RGB84202P, "420f", None, (Full, Bt601)),
            (P::RGB104442P, "x444", b709, (Video, Bt709)),
        ] {
            let codes = codes_of(pf);
            let state = pipeline(device, &library, pf, |_| {}).unwrap();
            for drawn in [[21, 13], [W, H]] {
                seed += 1;
                let case = format!("{pf:?} over {ostype}, {matrix:?}, a draw of {drawn:?}");
                let mut random = Random(seed);
                let bytes: Vec<u8> = (0..LENGTH).map(|_| random.next() as u8).collect();
                let surface = buffer_with(device, &bytes).unwrap();
                let texture = pass_texture(&surface, pf, ostype, matrix).unwrap();
                let values: Vec<[f32; 4]> = (0..W * H)
                    .map(|_| std::array::from_fn(|_| random.channel()))
                    .collect();
                let buffer: Vec<u8> = values
                    .iter()
                    .flatten()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let buffer = buffer_with(device, &buffer).unwrap();
                let mut descriptor = RenderPassDescriptor::new();
                let color = &mut descriptor.color_attachments[0];
                color.texture = Some(texture.clone());
                color.load_action = LoadAction::Load;
                color.store_action = StoreAction::Store;
                let command_buffer = queue.command_buffer().unwrap();
                let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
                let corners = rectangle(device, drawn);
                encoder.set_render_pipeline_state(&state).unwrap();
                encoder
                    .set_scissor_rect(ScissorRect {
                        x: 0,
                        y: 0,
                        width: drawn[0],
                        height: drawn[1],
                    })
                    .unwrap();
                encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                encoder.set_fragment_buffer(Some(&buffer), 0, 0).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
                encoder.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None, "{case}");
                let loaded = loaded_texels(&bytes, codes, conversion);
                let texels: Vec<[f32; 3]> = (0..W * H)
                    .map(|i| {
                        let (x, y) = (i % W, i / W);
                        let [r, g, b, _] = values[i as usize];
                        match x < drawn[0] && y < drawn[1] {
                            true => [r, g, b],
                            false => loaded[i as usize],
                        }
                    })
                    .collect();
                let stored = match codes.bits {
                    10 => stored_words(&bytes, codes, conversion, &texels),
                    _ => stored_bytes(&bytes, codes, conversion, &texels),
                };
                // The tiles the draw reaches: columns 0 to 31, or every column.
                let reached = if drawn[0] < W { 32 } else { W };
                let want = within_columns(&bytes, &stored, codes, [0, reached]);
                let after = read_buffer(&surface, 0, LENGTH).unwrap();
                assert!(
                    after == want,
                    "{case}: the surface's bytes after the pass, first difference (at, got, \
                     want) {:?}",
                    first_difference(&after, &want)
                );
            }
        }
        let notices = counts(&messages.take_notices());
        assert!(notices.contains_key(&YCBCR_PASSES[0]), "{notices:?}");
    });
}

/// The rest of a pass's shapes into a converted YCbCr texture, as Metal's tile of halves takes
/// them (the M3 Max under macOS 27.0.1, our probe `ycp`): a `DontCare` store, and a store action
/// left `Unknown` to the pass's end, store nothing; a write mask keeps the masked channels'
/// cleared values (red alone, green and blue, and every channel but alpha); and a draw that
/// blends with source alpha stores within a code of Metal's blend of the cleared halves (an
/// 8-bit format's in half arithmetic, a 10-bit one's in `float` rounded toward zero, which the
/// device's blend of floats can miss by one), announced (`ycbcr-blend-rounding`).
#[test]
fn masked_blended_and_unstored_passes_are_metal_s() {
    use PixelFormat as P;
    use YCbCrMatrix::{Bt601, Bt709};
    use YCbCrRange::{Full, Video};
    on_device(|device, messages| {
        let library = device
            .new_library_with_air(&llvm_assemble_text(SHADERS).unwrap())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let b709 = Some(SurfaceYCbCrMatrix::Itu709);
        let mut seed = 0x7b1;
        let clear = [0.2, 0.5, 0.8, 1.0];
        let cleared = [0, 1, 2].map(|c| half_nearest(clear[c] as f32));
        for (pf, ostype, matrix, conversion) in [
            (P::YCBCR84202P, "420f", b709, (Full, Bt709)),
            (P::YCBCR104442P, "x444", None, (Video, Bt601)),
            (P::RGB84222P, "422v", None, (Video, Bt601)),
            (P::RGB104202P, "xf20", b709, (Full, Bt709)),
        ] {
            let codes = codes_of(pf);
            for case in 0..6 {
                seed += 1;
                let mut random = Random(seed);
                let bytes: Vec<u8> = (0..LENGTH).map(|_| random.next() as u8).collect();
                let surface = buffer_with(device, &bytes).unwrap();
                let texture = pass_texture(&surface, pf, ostype, matrix).unwrap();
                // Values in [0, 1], their alphas too.
                let values: Vec<[f32; 4]> = (0..W * H)
                    .map(|_| std::array::from_fn(|_| (random.next() % 65537) as f32 / 65536.0))
                    .collect();
                let buffer: Vec<u8> = values
                    .iter()
                    .flatten()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let buffer = buffer_with(device, &buffer).unwrap();
                let (mask, blend, store) = match case {
                    0 => (ColorWriteMask::RED, false, StoreAction::Store),
                    1 => (
                        ColorWriteMask::GREEN | ColorWriteMask::BLUE,
                        false,
                        StoreAction::Store,
                    ),
                    2 => (ColorWriteMask::ALPHA, false, StoreAction::Store),
                    3 => (ColorWriteMask::ALL, true, StoreAction::Store),
                    4 => (ColorWriteMask::ALL, false, StoreAction::DontCare),
                    _ => (ColorWriteMask::ALL, false, StoreAction::Unknown),
                };
                let state = pipeline(device, &library, pf, |a| {
                    a.write_mask = mask;
                    a.blending_enabled = blend;
                    a.source_rgb_blend_factor = BlendFactor::SourceAlpha;
                    a.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
                })
                .unwrap();
                let name = format!("{pf:?} over {ostype}, {mask:?}, blend {blend}, {store:?}");
                let mut descriptor = RenderPassDescriptor::new();
                let color = &mut descriptor.color_attachments[0];
                color.texture = Some(texture.clone());
                color.load_action = LoadAction::Clear;
                color.store_action = store;
                color.clear_color = ClearColor::new(clear[0], clear[1], clear[2], clear[3]);
                let command_buffer = queue.command_buffer().unwrap();
                let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
                let corners = rectangle(device, [W, H]);
                encoder.set_render_pipeline_state(&state).unwrap();
                encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
                encoder.set_fragment_buffer(Some(&buffer), 0, 0).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
                encoder.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None, "{name}");
                let after = read_buffer(&surface, 0, LENGTH).unwrap();
                if store != StoreAction::Store {
                    assert!(after == bytes, "{name}: the surface's bytes changed");
                    continue;
                }
                let kept = [
                    ColorWriteMask::RED,
                    ColorWriteMask::GREEN,
                    ColorWriteMask::BLUE,
                ];
                let texels: Vec<[f32; 3]> = values
                    .iter()
                    .map(|v| {
                        std::array::from_fn(|c| match blend {
                            // Metal's blend of the cleared halves: an 8-bit format's in half
                            // arithmetic, a 10-bit one's in `float`.
                            true if codes.bits == 8 => {
                                let a = half_nearest(v[3]);
                                let s = half_nearest(half_nearest(v[c]) * a);
                                half_nearest(cleared[c] * half_nearest(1.0 - a) + s)
                            }
                            true => v[c] * v[3] + cleared[c] * (1.0 - v[3]),
                            false if mask.contains(kept[c]) => v[c],
                            false => cleared[c],
                        })
                    })
                    .collect();
                let want = match codes.bits {
                    10 => stored_words(&bytes, codes, conversion, &texels),
                    _ => stored_bytes(&bytes, codes, conversion, &texels),
                };
                if !blend {
                    assert!(
                        after == want,
                        "{name}: first difference (at, got, want) {:?}",
                        first_difference(&after, &want)
                    );
                    continue;
                }
                // Within a code of Metal's: each byte (8-bit codes) or 16-bit word's code.
                let step = match codes.bits {
                    10 => 2,
                    _ => 1,
                };
                for at in (0..after.len()).step_by(step) {
                    let read = |b: &[u8]| match step {
                        2 => i32::from(u16::from_le_bytes([b[at], b[at + 1]]) >> 6),
                        _ => i32::from(b[at]),
                    };
                    let (got, wanted) = (read(&after), read(&want));
                    assert!((got - wanted).abs() <= 1, "{name}: at {at}: {got} {wanted}");
                }
            }
        }
        let notices = counts(&messages.take_notices());
        assert!(notices.contains_key(&YCBCR_PASSES[1]), "{notices:?}");
    });
}
