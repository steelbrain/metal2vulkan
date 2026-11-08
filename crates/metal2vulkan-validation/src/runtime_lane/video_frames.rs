//! What a guest's frames of 4K HDR video cost on the device the cfg selects: report only, in
//! release builds.
//!
//! The consumer's guest decodes each frame into an IOSurface over its pages, of CoreVideo's
//! `x420` layout: 10-bit 4:2:0 codes of BT.2020 video range, each in the high 10 bits of a
//! little-endian 16-bit word (P010), the luma plane's 3840x2160 words and the chroma plane's
//! 1920x1080 Cb and Cr pairs, both planes' rows 7680 bytes apart, and every plane 32 bytes into
//! its page as CoreVideo lays it out (the consumer's measurement of VideoToolbox's frames on a
//! 2019 MacBook Pro under macOS 26.7.1: at 1920x1080 the planes at 32 and 2088992). Its compositor
//! then samples the frame into a 3840x2160 `BGRA8Unorm` target in one render pass, through one of
//! two forms ([`Form`]): a `YCBCR10_420_2P` texture over plane 0 (the runtime's two-plane YCbCr
//! path, `ycbcr.rs`), or an `R16Unorm` texture over plane 0 and an `RG16Unorm` one over plane 1,
//! which its own shader converts. The second form also runs with both planes 256 bytes into their
//! pages ([`Form::AlignedPlanes`]), where RADV lays a `DRM_FORMAT_MOD_LINEAR` plane: the same frame
//! in place where CoreVideo's 32 bytes make shadows.
//!
//! Each frame here: the CPU writes a whole new frame into the next of [`POOL`] surfaces (the
//! decoder's pool), then one command buffer of that pass, committed and waited for. Its surfaces'
//! pages ([`Pages`]) are private anonymous memory, imported by pointer, or on Linux a sealed
//! `memfd` mapped shared with a dma-buf source, as QEMU's guest RAM is (RADV refuses that memory's
//! pointer and imports dma-bufs). The textures are made once over each surface's buffer, or a new
//! buffer and textures each frame ([`Making`]).
//!
//! Printed per case (`VIDEO-FRAME`): each texture's placement as the runtime answers it before it
//! is made (`Buffer::texture_placement_with_iosurface_plane`), the medians and 90th percentiles of
//! the frame's GPU time (`GPUEndTime` less `GPUStartTime`), its host times (the CPU's write of the
//! frame, the textures' making, the encoding, and the commit to completion), and every work count
//! the frames moved, per frame. Each device settles its in-place checks first, so the placements
//! are those of a session whose verdicts are known. Printed by [`video_plane_placements`]
//! (`VIDEO-PLACEMENT`): where single-plane textures over decoded planes of CoreVideo's 8-bit `420v`
//! and 10-bit `x420` frames lie at each of several offsets into their page, sampled alone or written
//! too, and the words of each `shadowed-texture` notice. Printed by [`video_device_cache_costs`]
//! (`VIDEO-CACHE`): the frame blitted into buffers of the runtime's own at CoreVideo's offsets,
//! made without and with `BufferDescriptor::aligned_byte`. Ignored by default: measurements, not
//! checks. Run them by name with `--run-ignored all`.

#[cfg(target_os = "linux")]
use super::dma_buf_imports::{udmabuf_openable, Calls, Memfd, View};
use super::host_memory::{Mapping, GUEST_PAGE};
use super::no_copy_textures::{floats, staged};
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::DevicePolicy;
use metal2vulkan::metal::{
    PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureSwizzle, TextureSwizzleChannels,
    TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, BufferContents, BufferDescriptor, CommandQueue, Device, HostMemory, IoSurfacePlane,
    IoSurfaceYCbCr, LoadAction, PipelineOptions, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, SamplerState, StoreAction, SurfaceYCbCrMatrix, Texture, TextureDescriptor,
    WorkCounts,
};
use std::time::Instant;

/// The frame's luma plane: 3840x2160 words.
const W: u64 = 3840;
const H: u64 = 2160;
/// Both planes' bytes per row.
const ROW: u64 = 7680;
/// The luma plane's first byte: 32 bytes into the surface's first page, CoreVideo's. The chroma
/// plane's lies right after the luma plane's rows, 32 bytes into a page.
const LUMA_AT: u64 = 32;
/// The luma plane's first byte in [`Form::AlignedPlanes`]: 256 bytes into the first page, a
/// multiple of RADV's step for a `DRM_FORMAT_MOD_LINEAR` plane's offset. The chroma plane, right
/// after the luma plane's rows, lies 256 bytes into a page too.
const ALIGNED_AT: u64 = 256;
/// The surface's bytes, its pages whole, for either plane offset.
const LEN: usize = ((ALIGNED_AT + ROW * H * 3 / 2) as usize).next_multiple_of(GUEST_PAGE);
/// The decoder's surfaces, written in turn.
const POOL: usize = 3;
/// `'x420'`.
const X420: u32 = u32::from_be_bytes(*b"x420");

/// The untimed and the timed frames of each case: fewer on a CPU device, whose 4K passes take
/// a large part of a second.
const WARM: usize = if cfg!(m2v_cpu_icd) { 2 } else { 10 };
const FRAMES: usize = if cfg!(m2v_cpu_icd) { 6 } else { 60 };

/// Synthetic AIR written for this lane: vertex `video_corner` places vertex `vid` at buffer 0's
/// `float4` `vid`; fragment `video_ycbcr` samples texture 0 through sampler 0 at its pixel's centre
/// over the texture's size; fragment `video_planes` samples texture 0 (luma) and texture 1 (chroma)
/// so and converts them by coefficients near BT.2020's video range, each an exact binary fraction
/// (a measurement's arithmetic, not Metal's).
const SHADERS: &str = r#"; Owned synthetic shaders.
define <{ <4 x float> }> @video_corner(i32 %vid, ptr addrspace(2) %p) {
  %i = zext i32 %vid to i64
  %q = getelementptr inbounds <4 x float>, ptr addrspace(2) %p, i64 %i
  %v = load <4 x float>, ptr addrspace(2) %q, align 16
  %o = insertvalue <{ <4 x float> }> undef, <4 x float> %v, 0
  ret <{ <4 x float> }> %o
}

define <4 x float> @video_ycbcr(<4 x float> %pos, ptr addrspace(1) %t, ptr addrspace(2) %s) {
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  %h = tail call i32 @air.get_height_texture_2d(ptr addrspace(1) %t, i32 0)
  %fw = uitofp i32 %w to float
  %fh = uitofp i32 %h to float
  %wh0 = insertelement <2 x float> undef, float %fw, i64 0
  %wh = insertelement <2 x float> %wh0, float %fh, i64 1
  %uv = fdiv <2 x float> %xy, %wh
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %c = extractvalue { <4 x float>, i8 } %r, 0
  ret <4 x float> %c
}

define <4 x float> @video_planes(<4 x float> %pos, ptr addrspace(1) %luma, ptr addrspace(1) %chroma, ptr addrspace(2) %s) {
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %luma, i32 0)
  %h = tail call i32 @air.get_height_texture_2d(ptr addrspace(1) %luma, i32 0)
  %fw = uitofp i32 %w to float
  %fh = uitofp i32 %h to float
  %wh0 = insertelement <2 x float> undef, float %fw, i64 0
  %wh = insertelement <2 x float> %wh0, float %fh, i64 1
  %uv = fdiv <2 x float> %xy, %wh
  %ys = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %luma, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %cs = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %chroma, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %yv = extractvalue { <4 x float>, i8 } %ys, 0
  %cv = extractvalue { <4 x float>, i8 } %cs, 0
  %y0 = extractelement <4 x float> %yv, i64 0
  %cb0 = extractelement <4 x float> %cv, i64 0
  %cr0 = extractelement <4 x float> %cv, i64 1
  %y1 = fmul float %y0, 1.125
  %y = fsub float %y1, 0.0625
  %cb1 = fmul float %cb0, 1.125
  %cb = fsub float %cb1, 0.5625
  %cr1 = fmul float %cr0, 1.125
  %cr = fsub float %cr1, 0.5625
  %rr = fmul float %cr, 1.5
  %r = fadd float %y, %rr
  %gb = fmul float %cb, 0.125
  %gr = fmul float %cr, 0.5625
  %g1 = fsub float %y, %gb
  %g = fsub float %g1, %gr
  %bb = fmul float %cb, 1.875
  %b = fadd float %y, %bb
  %c0 = insertelement <4 x float> <float poison, float poison, float poison, float 1.0>, float %r, i64 0
  %c1 = insertelement <4 x float> %c0, float %g, i64 1
  %c2 = insertelement <4 x float> %c1, float %b, i64 2
  ret <4 x float> %c2
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_2d(ptr addrspace(1), i32)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.vertex = !{!90}
!air.fragment = !{!80, !70}
!air.version = !{!97}
!air.language_version = !{!98}
!90 = !{ptr @video_corner, !91, !92}
!91 = !{!93}
!92 = !{!94, !95}
!93 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!94 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!95 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!97 = !{i32 2, i32 8, i32 0}
!98 = !{!"Metal", i32 4, i32 0, i32 0}
!80 = !{ptr @video_ycbcr, !81, !82}
!81 = !{!83}
!82 = !{!84, !85, !86}
!83 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!84 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!85 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!86 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!70 = !{ptr @video_planes, !81, !72}
!72 = !{!84, !85, !75, !76}
!75 = !{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"c"}
!76 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
"#;

/// How the compositor samples a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// A `YCBCR10_420_2P` texture over plane 0, the surface's facts given, sampled by
    /// `video_ycbcr`.
    YCbCr,
    /// As [`Form::YCbCr`], with a swizzle that swaps red and blue: a swizzled two-plane YCbCr
    /// texture keeps its image of converted texels on every device (`ycbcr_images_kept_swizzle`),
    /// so this is the image's route for the same frame, the conversion before the pass and the
    /// device's filter of the converted texels.
    YCbCrImage,
    /// An `R16Unorm` texture over plane 0 and an `RG16Unorm` one over plane 1, sampled and
    /// converted by `video_planes`.
    Planes,
    /// As [`Form::Planes`], of a frame whose planes lie 256 bytes into their pages
    /// ([`ALIGNED_AT`]).
    AlignedPlanes,
}

impl Form {
    /// The luma plane's first byte in the surface; the chroma plane's follows its rows.
    fn luma_at(self) -> u64 {
        match self {
            Form::YCbCr | Form::YCbCrImage | Form::Planes => LUMA_AT,
            Form::AlignedPlanes => ALIGNED_AT,
        }
    }
}

/// Where a case's surfaces' pages lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pages {
    Anonymous,
    #[cfg(target_os = "linux")]
    GuestRam,
}

/// Whether a case's textures are made once over each surface's buffer, or a new buffer and
/// textures each frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Making {
    Once,
    EachFrame,
}

/// A surface's pages, and how a no-copy buffer is made over them.
enum Surface {
    Anonymous(Mapping),
    #[cfg(target_os = "linux")]
    GuestRam {
        view: View,
        memfd: std::rc::Rc<Memfd>,
        calls: Calls,
    },
}

impl Surface {
    /// The pool's surface `index` of `pages`, of the guest RAM's `memfd` where they are its.
    #[cfg(target_os = "linux")]
    fn new(pages: Pages, memfd: Option<&std::rc::Rc<Memfd>>, index: usize) -> Surface {
        match pages {
            Pages::Anonymous => Surface::Anonymous(Mapping::anonymous(LEN)),
            Pages::GuestRam => {
                let memfd = memfd.expect("the guest RAM's file");
                Surface::GuestRam {
                    view: memfd.view(&[(index * LEN, LEN)]),
                    memfd: std::rc::Rc::clone(memfd),
                    calls: Calls::default(),
                }
            }
        }
    }

    /// The pool's surface of `pages`: anonymous memory.
    #[cfg(not(target_os = "linux"))]
    fn new(pages: Pages, _: Option<&()>, _: usize) -> Surface {
        match pages {
            Pages::Anonymous => Surface::Anonymous(Mapping::anonymous(LEN)),
        }
    }

    /// A no-copy buffer over the surface's [`LEN`] bytes, of default options, as the consumer
    /// makes one over a surface's pages.
    fn buffer(&self, device: &Device) -> Buffer {
        let memory = match self {
            // SAFETY: the mapping outlives every buffer over it (each case drops them first).
            Surface::Anonymous(mapping) => unsafe { HostMemory::new(mapping.pointer()) },
            #[cfg(target_os = "linux")]
            Surface::GuestRam { view, memfd, calls } => {
                // SAFETY: the view outlives every buffer over it, and the source makes dma-bufs
                // of its pages.
                unsafe { HostMemory::with_dma_buf(view.pointer(0), memfd.source(view, calls)) }
            }
        };
        // SAFETY: as above.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                LEN as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap()
    }

    /// Write `bytes` over the surface's first bytes, as the decoder writes a frame.
    fn write(&self, bytes: &[u8]) {
        match self {
            Surface::Anonymous(mapping) => mapping.write(0, bytes),
            #[cfg(target_os = "linux")]
            Surface::GuestRam { view, .. } => view.write(0, bytes),
        }
    }
}

/// A frame's bytes, numbered `n`, its luma plane `luma_at` bytes into the surface: each code in
/// the high 10 bits of its word, the low 6 bits noise, which no form reads as Metal does not.
fn frame_bytes(n: u32, luma_at: u64) -> Vec<u8> {
    let chroma_at = luma_at + ROW * H;
    let mut bytes = vec![0u8; LEN];
    let mut state = 0x9e37_79b9_u32.wrapping_mul(n + 1);
    let mut word = |code: u32| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (((code & 0x3ff) << 6) | (state & 0x3f)) as u16
    };
    for y in 0..H {
        for x in 0..W {
            let code = 64 + ((x + y + u64::from(n) * 7) % 876) as u32;
            let at = (luma_at + y * ROW + x * 2) as usize;
            bytes[at..at + 2].copy_from_slice(&word(code).to_le_bytes());
        }
    }
    for y in 0..H / 2 {
        for x in 0..W / 2 {
            let cb = 64 + ((x * 3 + u64::from(n)) % 896) as u32;
            let cr = 64 + ((y * 5 + u64::from(n)) % 896) as u32;
            let at = (chroma_at + y * ROW + x * 4) as usize;
            bytes[at..at + 2].copy_from_slice(&word(cb).to_le_bytes());
            bytes[at + 2..at + 4].copy_from_slice(&word(cr).to_le_bytes());
        }
    }
    bytes
}

/// A surface's buffer and the textures a form samples over it ([`textures`]).
type Made = (Buffer, Vec<(Texture, String)>);

/// The textures `form` samples over `buffer`, each with its placement as the runtime answered it
/// before making it.
fn textures(buffer: &Buffer, form: Form) -> Vec<(Texture, String)> {
    let (luma_at, chroma_at) = (form.luma_at(), form.luma_at() + ROW * H);
    let planes: Vec<(PixelFormat, u64, u64, IoSurfacePlane)> = match form {
        Form::YCbCr | Form::YCbCrImage => {
            let facts = IoSurfaceYCbCr::new(X420, 0, [W, H], chroma_at, ROW)
                .with_matrix(SurfaceYCbCrMatrix::Itu2020);
            let plane = IoSurfacePlane::new(luma_at, ROW, true).with_ycbcr(facts);
            vec![(PixelFormat::YCBCR104202P, W, H, plane)]
        }
        Form::Planes | Form::AlignedPlanes => vec![
            (
                PixelFormat::R16Unorm,
                W,
                H,
                IoSurfacePlane::new(luma_at, ROW, true),
            ),
            (
                PixelFormat::RG16Unorm,
                W / 2,
                H / 2,
                IoSurfacePlane::new(chroma_at, ROW, true),
            ),
        ],
    };
    planes
        .into_iter()
        .map(|(pf, w, h, plane)| {
            let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = buffer.resource_options();
            if form == Form::YCbCrImage {
                d.swizzle = TextureSwizzleChannels {
                    red: TextureSwizzle::Blue,
                    blue: TextureSwizzle::Red,
                    ..TextureSwizzleChannels::DEFAULT
                };
            }
            let placement = buffer.texture_placement_with_iosurface_plane(&d, plane);
            let texture = buffer.new_texture_with_iosurface_plane(&d, plane).unwrap();
            (texture, format!("{pf:?} {placement:?}"))
        })
        .collect()
}

/// The fragment function that samples `form`.
fn fragment(form: Form) -> &'static str {
    match form {
        Form::YCbCr | Form::YCbCrImage => "video_ycbcr",
        Form::Planes | Form::AlignedPlanes => "video_planes",
    }
}

/// The pipeline drawing `fragment` into a `BGRA8Unorm` target.
fn pipeline(device: &Device, fragment: &str) -> RenderPipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(SHADERS).unwrap())
        .unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut d = RenderPipelineDescriptor::new(function("video_corner"));
    d.fragment = Some(function(fragment));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// The two triangles covering the target, as `float4`s.
fn whole_target(device: &Device) -> Buffer {
    let corners: Vec<u8> = [
        [-1.0f32, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ]
    .iter()
    .flat_map(|&[x, y]| floats([x, y, 0.0, 1.0]))
    .collect();
    staged(device, &corners)
}

/// One frame: where `copy` is given, a blit of [`LEN`] bytes of its first buffer into its second,
/// then a pass drawing `textures` through `pipeline` and `sampler` over the whole of `target`, in
/// one command buffer, committed and waited for. The host's time encoding, its time from the
/// commit until the wait returned, and the command buffer's GPU time, in microseconds.
fn frame(
    queue: &CommandQueue,
    target: &Texture,
    (pipeline, sampler, corners): (&RenderPipelineState, &SamplerState, &Buffer),
    textures: &[(Texture, String)],
    copy: Option<(&Buffer, &Buffer)>,
) -> (f64, f64, f64) {
    let began = Instant::now();
    let command_buffer = queue.command_buffer().unwrap();
    if let Some((from, into)) = copy {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(from, 0, into, 0, LEN as u64)
            .unwrap();
        blit.end_encoding().unwrap();
    }
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::DontCare;
    attachment.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(corners), 0, 0).unwrap();
    encoder
        .set_fragment_sampler_state(Some(sampler), 0)
        .unwrap();
    for (index, (texture, _)) in textures.iter().enumerate() {
        encoder
            .set_fragment_texture(Some(texture), index as u32)
            .unwrap();
    }
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let encoded = since(began);
    let committed = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let waited = since(committed);
    let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
    (encoded, waited, gpu * 1e6)
}

/// Microseconds since `at`.
fn since(at: Instant) -> f64 {
    at.elapsed().as_secs_f64() * 1e6
}

/// The median and 90th percentile of `values`.
fn median_p90(mut values: Vec<f64>) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    (at(50), at(90))
}

/// The `name: number` fields of `counts`' `Debug` form, in order (nested structures' fields
/// named by their own names).
fn count_fields(counts: &WorkCounts) -> Vec<(String, u64)> {
    let text = format!("{counts:?}");
    let mut fields = Vec::new();
    let mut name = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_alphanumeric() || c == '_' {
            name.push(c);
            continue;
        }
        if c == ':' && !name.is_empty() {
            while chars.peek() == Some(&' ') {
                chars.next();
            }
            let mut digits = String::new();
            while let Some(&d) = chars.peek().filter(|d| d.is_ascii_digit()) {
                digits.push(d);
                chars.next();
            }
            if let Ok(value) = digits.parse() {
                fields.push((name.clone(), value));
            }
        }
        name.clear();
    }
    fields
}

/// Every work count that moved from `before` to `after`, per frame of `frames`.
pub(super) fn moved_per_frame(before: &WorkCounts, after: &WorkCounts, frames: usize) -> String {
    let before = count_fields(before);
    let after = count_fields(after);
    if before.len() != after.len() {
        return format!("the counts' Debug forms differ in shape: {before:?} against {after:?}");
    }
    before
        .iter()
        .zip(&after)
        .filter(|((_, b), (_, a))| a != b)
        .map(|((name, b), (_, a))| {
            let per = (*a as f64 - *b as f64) / frames as f64;
            format!("{name} {per:.2}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The longest a device's in-place checks may take here.
const SETTLE: std::time::Duration = std::time::Duration::from_secs(120);

/// The device the cfg selects as a consumer makes it: the default policy, no layer and no
/// execution records, its in-place checks settled.
fn consumer_device(messages: &Messages, cache: &ScratchCache) -> Device {
    consumer_device_with(messages, cache, DevicePolicy::default())
}

/// [`consumer_device`] under `policy`.
fn consumer_device_with(messages: &Messages, cache: &ScratchCache, policy: DevicePolicy) -> Device {
    let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
        options.layers.clear();
        options.policy = policy;
        options.record_executions = false;
    })
    .unwrap();
    let settled = device.settle_in_place_checks(SETTLE);
    println!(
        "VIDEO-DEVICE {}: in-place checks settled {settled}",
        device.name()
    );
    messages.take_notices();
    device
}

/// A sampler of `filter` for minification and magnification.
fn sampler(device: &Device, filter: SamplerMinMagFilter) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = filter;
    d.mag_filter = filter;
    device.new_sampler_state(&d).unwrap()
}

/// The compositor's 3840x2160 `BGRA8Unorm` target.
fn target(device: &Device) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, W, H, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    device.new_texture(&d).unwrap()
}

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn video_frame_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = consumer_device(&messages, &cache);
    let queue = device.new_command_queue().unwrap();
    let sampler = sampler(&device, SamplerMinMagFilter::Linear);
    let corners = whole_target(&device);
    let target = target(&device);
    let pages_kinds = [
        Pages::Anonymous,
        #[cfg(target_os = "linux")]
        Pages::GuestRam,
    ];
    println!(
        "VIDEO-DEVICE {}: unified memory {}",
        device.name(),
        device.properties().has_unified_memory
    );
    for pages in pages_kinds {
        #[cfg(target_os = "linux")]
        let memfd = match pages {
            Pages::GuestRam if !udmabuf_openable("video_frame_costs") => continue,
            Pages::GuestRam => Some(std::rc::Rc::new(Memfd::new(LEN * POOL))),
            Pages::Anonymous => None,
        };
        #[cfg(not(target_os = "linux"))]
        let memfd = None;
        let surfaces: Vec<Surface> = (0..POOL)
            .map(|index| Surface::new(pages, memfd.as_ref(), index))
            .collect();
        for form in [Form::YCbCr, Form::Planes, Form::AlignedPlanes] {
            let frames = [
                frame_bytes(0, form.luma_at()),
                frame_bytes(1, form.luma_at()),
            ];
            let pipeline = pipeline(&device, fragment(form));
            for making in [Making::Once, Making::EachFrame] {
                let mut kept: Vec<Option<Made>> = (0..POOL).map(|_| None).collect();
                let mut placements = String::new();
                let (mut writes, mut makes, mut encodes, mut waits, mut gpus) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let mut before = device.work_counts();
                for k in 0..WARM + FRAMES {
                    if k == WARM {
                        before = device.work_counts();
                    }
                    let index = k % POOL;
                    let began = Instant::now();
                    surfaces[index].write(&frames[k % 2]);
                    let written = since(began);
                    let began = Instant::now();
                    let made = match (&kept[index], making) {
                        (Some(made), Making::Once) => (made.0.clone(), made.1.clone()),
                        _ => {
                            let buffer = surfaces[index].buffer(&device);
                            let textures = textures(&buffer, form);
                            (buffer, textures)
                        }
                    };
                    let making_us = since(began);
                    if k == 0 {
                        placements = made
                            .1
                            .iter()
                            .map(|(_, placement)| placement.as_str())
                            .collect::<Vec<_>>()
                            .join("; ");
                    }
                    let (encoded, waited, gpu) = frame(
                        &queue,
                        &target,
                        (&pipeline, &sampler, &corners),
                        &made.1,
                        None,
                    );
                    if making == Making::Once {
                        kept[index] = Some(made);
                    }
                    if k >= WARM {
                        writes.push(written);
                        makes.push(making_us);
                        encodes.push(encoded);
                        waits.push(waited);
                        gpus.push(gpu);
                    }
                }
                let after = device.work_counts();
                drop(kept);
                let (write, write90) = median_p90(writes);
                let (make, make90) = median_p90(makes);
                let (encode, encode90) = median_p90(encodes);
                let (wait, wait90) = median_p90(waits);
                let (gpu, gpu90) = median_p90(gpus);
                println!(
                    "VIDEO-FRAME on {}: {pages:?} pages, {form:?}, textures made {making:?}: \
                     [{placements}]; GPU {gpu:.0} us (p90 {gpu90:.0}); host: frame write \
                     {write:.0} us (p90 {write90:.0}), making {make:.0} us (p90 {make90:.0}), \
                     encode {encode:.0} us (p90 {encode90:.0}), commit to completion {wait:.0} us \
                     (p90 {wait90:.0}); per frame: {}",
                    device.name(),
                    moved_per_frame(&before, &after, FRAMES),
                );
            }
        }
        drop(surfaces);
    }
    let notices = super::degrade_notices::counts(&messages.take_notices());
    println!("VIDEO-NOTICES on {}: {notices:?}", device.name());
}

/// [`video_frame_costs`]' frame taken apart over anonymous pages, textures made once: each form
/// sampled linear and nearest, with the CPU writing a new frame each time and with the frame left
/// as written (the GPU reading pages the CPU has not touched since), so the sampling's own cost
/// shows apart from the pages' (`VIDEO-PART`); the planes read in place under each
/// `YCbCrInPlacePolicy`, the other forms under the default policy alone.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn video_frame_parts() {
    use metal2vulkan::device::YCbCrInPlacePolicy;
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let mut exact = DevicePolicy::default();
    exact.ycbcr_in_place_reads = YCbCrInPlacePolicy::Exact;
    for (policy, forms) in [
        (
            DevicePolicy::default(),
            &[
                Form::YCbCr,
                Form::YCbCrImage,
                Form::Planes,
                Form::AlignedPlanes,
            ][..],
        ),
        (exact, &[Form::YCbCr][..]),
    ] {
        let device = consumer_device_with(&messages, &cache, policy);
        parts(&device, forms, policy.ycbcr_in_place_reads);
    }
}

/// [`video_frame_parts`] on `device`, whose planes read in place follow `reads`, of `forms`.
fn parts(device: &Device, forms: &[Form], reads: metal2vulkan::device::YCbCrInPlacePolicy) {
    let queue = device.new_command_queue().unwrap();
    let corners = whole_target(device);
    let target = target(device);
    let surfaces: Vec<Surface> = (0..POOL)
        .map(|_| Surface::Anonymous(Mapping::anonymous(LEN)))
        .collect();
    for &form in forms {
        let frames = [
            frame_bytes(0, form.luma_at()),
            frame_bytes(1, form.luma_at()),
        ];
        let pipeline = pipeline(device, fragment(form));
        let made: Vec<Made> = surfaces
            .iter()
            .map(|surface| {
                surface.write(&frames[0]);
                let buffer = surface.buffer(device);
                let textures = textures(&buffer, form);
                (buffer, textures)
            })
            .collect();
        for filter in [SamplerMinMagFilter::Linear, SamplerMinMagFilter::Nearest] {
            let sampler = sampler(device, filter);
            for written in [true, false] {
                let mut gpus = Vec::new();
                let before = device.work_counts();
                for k in 0..WARM + FRAMES {
                    let index = k % POOL;
                    if written {
                        surfaces[index].write(&frames[k % 2]);
                    }
                    let (_, _, gpu) = frame(
                        &queue,
                        &target,
                        (&pipeline, &sampler, &corners),
                        &made[index].1,
                        None,
                    );
                    if k >= WARM {
                        gpus.push(gpu);
                    }
                }
                let after = device.work_counts();
                let (gpu, gpu90) = median_p90(gpus);
                let per = |a: u64, b: u64| (a - b) as f64 / (WARM + FRAMES) as f64;
                println!(
                    "VIDEO-PART on {}: {form:?} (in-place reads {reads:?}) [{}], {filter:?}, \
                     frame written each time {written}: GPU {gpu:.0} us (p90 {gpu90:.0}); per \
                     frame: shadow texels \
                     copied in {:.0}, planes bound in place {:.2}, refreshes {:.2}, whole \
                     conversions {:.2}",
                    device.name(),
                    made[0]
                        .1
                        .iter()
                        .map(|(_, placement)| placement.as_str())
                        .collect::<Vec<_>>()
                        .join("; "),
                    per(
                        after.shadow_texels_copied_in,
                        before.shadow_texels_copied_in
                    ),
                    per(after.ycbcr_planes_bound, before.ycbcr_planes_bound),
                    per(after.ycbcr_refreshes, before.ycbcr_refreshes),
                    per(
                        after.ycbcr_whole_conversions,
                        before.ycbcr_whole_conversions
                    ),
                );
            }
        }
        drop(made);
    }
    drop(surfaces);
}

/// [`video_frame_costs`]' frame over a device cache, as a consumer that lands each decoded frame
/// in a buffer of the runtime's own: [`POOL`] `Private` buffers of [`LEN`] bytes, the planes at
/// CoreVideo's offsets ([`LUMA_AT`] and right after the luma rows), sampled by `video_planes`
/// through textures made once over each, each frame's bytes blitted into its buffer from a
/// `Shared` staging buffer in the frame's command buffer before the pass (`VIDEO-CACHE`). The
/// buffers are made without `BufferDescriptor::aligned_byte`, then laying byte [`LUMA_AT`] on a
/// multiple of 256 (`with_aligned_byte`): where the device's linear-image alignment is past 32
/// bytes the planes of the first are shadows copied in at every frame, those of the second the
/// linear images over the buffer's memory.
#[test]
#[ignore = "a release-mode measurement: run by name"]
fn video_device_cache_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = consumer_device(&messages, &cache);
    let queue = device.new_command_queue().unwrap();
    let sampler = sampler(&device, SamplerMinMagFilter::Linear);
    let corners = whole_target(&device);
    let target = target(&device);
    let pipeline = pipeline(&device, fragment(Form::Planes));
    let staging = [0, 1].map(|n| staged(&device, &frame_bytes(n, LUMA_AT)));
    for aligned in [None, Some(LUMA_AT)] {
        let made: Vec<Made> = (0..POOL)
            .map(|_| {
                let d = BufferDescriptor::new(
                    LEN as u64,
                    ResourceOptions::private(),
                    BufferContents::Zero,
                );
                let d = match aligned {
                    Some(byte) => d.with_aligned_byte(byte),
                    None => d,
                };
                let buffer = device.new_buffer_with_descriptor(d).unwrap();
                let textures = textures(&buffer, Form::Planes);
                (buffer, textures)
            })
            .collect();
        let mut gpus = Vec::new();
        let mut before = device.work_counts();
        for k in 0..WARM + FRAMES {
            if k == WARM {
                before = device.work_counts();
            }
            let index = k % POOL;
            let (_, _, gpu) = frame(
                &queue,
                &target,
                (&pipeline, &sampler, &corners),
                &made[index].1,
                Some((&staging[k % 2], &made[index].0)),
            );
            if k >= WARM {
                gpus.push(gpu);
            }
        }
        let after = device.work_counts();
        let (gpu, gpu90) = median_p90(gpus);
        println!(
            "VIDEO-CACHE on {}: Private caches, aligned byte {aligned:?}: [{}]; GPU {gpu:.0} us \
             (p90 {gpu90:.0}); per frame: {}",
            device.name(),
            made[0]
                .1
                .iter()
                .map(|(_, placement)| placement.as_str())
                .collect::<Vec<_>>()
                .join("; "),
            moved_per_frame(&before, &after, FRAMES),
        );
    }
    let notices = super::degrade_notices::counts(&messages.take_notices());
    println!("VIDEO-NOTICES on {}: {notices:?}", device.name());
}

/// Where single-plane textures over a decoded frame's planes lie (`VIDEO-PLACEMENT`), as the
/// consumer's guest makes them: CoreVideo's 8-bit `420v` planes (`R8Unorm` 3840x2160 and `RG8Unorm`
/// 1920x1080, rows 3840 bytes apart) and 10-bit `x420` planes (`R16Unorm` and `RG16Unorm`, rows
/// 7680 apart), each at 0, 32 (CoreVideo's), 64, 128 and 256 bytes into its page, sampled alone
/// (`ShaderRead`), written too (`ShaderRead | ShaderWrite`) and rendered into
/// (`ShaderRead | RenderTarget`), over `Shared` and `Private` buffers
/// of the runtime's own (each with its address, its contents' offset into a page and the device's
/// linear texture alignments), anonymous pages and, on Linux,
/// a `memfd`'s pages with a dma-buf source and a guest RAM's (`HostRam` over a `memfd`, its source
/// asked for chunks as long as `udmabuf` takes, as the consumer's is), two buffers over it: the
/// placement the runtime answers before the making, and what the making took. Then the words of
/// every distinct `shadowed-texture` notice, which name the driver's refusal.
#[test]
#[ignore = "a measurement: run by name"]
fn video_plane_placements() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = consumer_device(&messages, &cache);
    for (options, what) in [
        (
            ResourceOptions::shared(),
            "a Shared buffer of the runtime's",
        ),
        (
            ResourceOptions::private(),
            "a Private buffer of the runtime's",
        ),
    ] {
        let buffer = device.new_buffer(LEN as u64, options).unwrap();
        let linear: Vec<String> = [
            PixelFormat::R8Unorm,
            PixelFormat::RG8Unorm,
            PixelFormat::R16Unorm,
            PixelFormat::RG16Unorm,
        ]
        .into_iter()
        .map(|pf| {
            let alignment = device
                .minimum_linear_texture_alignment_for_pixel_format(pf)
                .unwrap();
            format!("{pf:?} {alignment}")
        })
        .collect();
        println!(
            "VIDEO-PLACEMENT on {}: {what}: gpuAddress {:#x}, contents {:?}, linear texture \
             alignments [{}]",
            device.name(),
            buffer.gpu_address().0,
            buffer.contents().map(|at| at.as_ptr() as usize % 4096),
            linear.join(", "),
        );
        probe_planes(&device, what, &buffer);
    }
    let anonymous = Surface::new(Pages::Anonymous, None, 0);
    probe_planes(&device, "anonymous pages", &anonymous.buffer(&device));
    drop(anonymous);
    #[cfg(target_os = "linux")]
    if udmabuf_openable("video_plane_placements") {
        let memfd = std::rc::Rc::new(Memfd::new(LEN));
        let surface = Surface::new(Pages::GuestRam, Some(&memfd), 0);
        probe_planes(&device, "dma-buf pages", &surface.buffer(&device));
        drop(surface);
        ram_planes(&device);
    }
    let mut words: Vec<String> = messages
        .take_notices()
        .into_iter()
        .filter(|message| message.id_name == "shadowed-texture")
        .map(|message| message.text)
        .collect();
    words.sort();
    words.dedup();
    for text in words {
        println!("VIDEO-PLACEMENT-NOTICE on {}: {text}", device.name());
    }
}

/// [`video_plane_placements`] over a guest RAM: two buffers of [`LEN`] bytes, the first object
/// over the RAM's chunk and the one after it.
#[cfg(target_os = "linux")]
fn ram_planes(device: &Device) {
    use super::dma_buf_imports::udmabuf_size_limit;
    use metal2vulkan_runtime::{HostRam, RunResolver};
    use std::sync::Arc;
    let memfd = Memfd::new(2 * LEN);
    let calls = Calls::default();
    let source = memfd.source(&memfd.whole, &calls);
    // SAFETY: the file's whole mapping, readable and writable while `memfd` lives, which outlives
    // the device's use of the RAM and every object over it.
    let ram = unsafe { HostRam::new(&[(memfd.whole.pointer(0), 2 * LEN as u64)], None) }
        .unwrap()
        .with_dma_buf(source)
        .with_largest_dma_buf(udmabuf_size_limit());
    let ram = Arc::new(ram);
    let used = device.use_host_ram(&ram).unwrap();
    // The view is the RAM's own mapping: each object's pages are the run of its own addresses.
    let resolver: RunResolver =
        Arc::new(|at: u64, length: u64| Some(std::iter::once(at..at + length).collect()));
    for (index, what) in [
        (0, "a guest RAM's first object"),
        (1, "a guest RAM's second object"),
    ] {
        let before = device.work_counts();
        // SAFETY: the RAM's pages, mapped while `memfd` lives, which outlives the buffer.
        let memory = unsafe { HostMemory::new(memfd.whole.pointer(index * LEN)) }
            .with_runs(Arc::clone(&resolver));
        // SAFETY: as above.
        let buffer = unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                LEN as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap();
        let after = device.work_counts();
        println!(
            "VIDEO-PLACEMENT on {}: {what}: buffers made in a run's import {}, in a chunk's {}",
            device.name(),
            after.run_bound_buffers_made - before.run_bound_buffers_made,
            after.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        );
        probe_planes(device, what, &buffer);
        drop(buffer);
    }
    drop(used);
    drop(ram);
}

/// [`video_plane_placements`]' planes over `buffer`, of pages `what` names.
fn probe_planes(device: &Device, what: &str, buffer: &Buffer) {
    let planes = [
        (PixelFormat::R8Unorm, W, H, W),
        (PixelFormat::RG8Unorm, W / 2, H / 2, W),
        (PixelFormat::R16Unorm, W, H, ROW),
        (PixelFormat::RG16Unorm, W / 2, H / 2, ROW),
    ];
    let usages = [
        TextureUsage::SHADER_READ,
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
    ];
    let first = device.work_counts();
    for (pf, w, h, row) in planes {
        for at in [0, 32, 64, 128, 256] {
            for usage in usages {
                let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
                d.usage = usage;
                d.resource_options = buffer.resource_options();
                let plane = IoSurfacePlane::new(at, row, true);
                let placement = buffer.texture_placement_with_iosurface_plane(&d, plane);
                let before = device.work_counts();
                let made = buffer.new_texture_with_iosurface_plane(&d, plane);
                let after = device.work_counts();
                let took = match &made {
                    Ok(_) if after.host_textures_in_place > before.host_textures_in_place => {
                        "in place".to_owned()
                    }
                    Ok(_) if after.shadowed_textures == before.shadowed_textures => {
                        "an image over the buffer's memory".to_owned()
                    }
                    Ok(_) => format!(
                        "a shadow ({} refused for the pitch)",
                        after.host_texture_shadows_refused_pitch
                            - before.host_texture_shadows_refused_pitch
                    ),
                    Err(error) => format!("refused: {error}"),
                };
                println!(
                    "VIDEO-PLACEMENT on {}: {what}, {pf:?} {w}x{h}, rows {row}, at {at}, \
                     {usage:?}: answered {placement:?}, took {took}",
                    device.name()
                );
                drop(made);
            }
        }
    }
    let last = device.work_counts();
    println!(
        "VIDEO-PLACEMENT on {}: {what}: imports read apart for a texture {}, dma-buf images {}, \
         dma-buf imports {}",
        device.name(),
        last.host_imports_read_apart_texture - first.host_imports_read_apart_texture,
        last.dma_buf_images - first.dma_buf_images,
        last.dma_buf_imports - first.dma_buf_imports,
    );
}
